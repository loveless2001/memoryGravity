#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from train.mg_core import MGConfig, TinyMemoryGravityLM


GLYPH_TOKEN_ID = 1
QUERY_TOKEN_ID = 2
FILLER_LOW_ID = 3


def parse_delays(spec: str) -> list[int]:
    return [int(part) for part in spec.split(",") if part.strip()]


def autocast_context(args, device: torch.device):
    if device.type != "cuda" or args.amp == "none":
        return nullcontext()
    if args.amp == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def make_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    value_vocab_size: int,
    delays: list[int],
    device: torch.device,
    generator: torch.Generator,
):
    if seq_len < max(delays) + 4:
        raise ValueError("seq_len must be at least max(delay) + 4")

    values = torch.randint(
        FILLER_LOW_ID,
        FILLER_LOW_ID + value_vocab_size,
        (batch_size,),
        generator=generator,
    )
    delay_idx = torch.randint(0, len(delays), (batch_size,), generator=generator)
    selected_delays = torch.tensor([delays[i] for i in delay_idx.tolist()], dtype=torch.long)

    idx = torch.randint(
        FILLER_LOW_ID,
        vocab_size,
        (batch_size, seq_len),
        generator=generator,
    )

    glyph_pos = torch.full((batch_size,), 1, dtype=torch.long)
    query_pos = glyph_pos + selected_delays
    target_pos = query_pos + 1

    rows = torch.arange(batch_size)
    idx[rows, glyph_pos] = GLYPH_TOKEN_ID
    idx[rows, glyph_pos + 1] = values
    idx[rows, query_pos] = QUERY_TOKEN_ID
    idx[rows, target_pos] = values

    glyph_mask = (idx == GLYPH_TOKEN_ID).to(torch.float32)
    target_mask = torch.zeros(batch_size, seq_len - 1, dtype=torch.bool)
    target_mask[rows, query_pos] = True

    return {
        "idx": idx.to(device),
        "glyph_mask": glyph_mask.to(device),
        "target_mask": target_mask.to(device),
    }


def query_loss_from_logits(logits: torch.Tensor, idx: torch.Tensor, target_mask: torch.Tensor):
    next_logits = logits[:, :-1, :]
    next_targets = idx[:, 1:]
    masked_logits = next_logits[target_mask]
    masked_targets = next_targets[target_mask]
    loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets)
    return loss, masked_logits, masked_targets


@torch.no_grad()
def evaluate_by_delay(model, args, device, delays: list[int], batches: int, seed_offset: int):
    model.eval()
    metrics = {}
    amp_ctx = autocast_context(args, device)

    for delay in delays:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + seed_offset + delay)
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for _ in range(batches):
            batch = make_batch(
                batch_size=args.eval_batch_size,
                seq_len=args.seq_len,
                vocab_size=args.vocab_size,
                value_vocab_size=args.value_vocab_size,
                delays=[delay],
                device=device,
                generator=generator,
            )
            with amp_ctx:
                out = model(
                    batch["idx"],
                    glyph_mask=batch["glyph_mask"],
                    targets=batch["idx"],
                    return_attn=False,
                    return_mass=False,
                )
            query_loss, masked_logits, masked_targets = query_loss_from_logits(
                out["logits"], batch["idx"], batch["target_mask"]
            )
            total_loss += query_loss.item()
            total_correct += (masked_logits.argmax(dim=-1) == masked_targets).sum().item()
            total_count += masked_targets.numel()

        metrics[delay] = {
            "query_loss": total_loss / max(batches, 1),
            "retrieval_acc": total_correct / max(total_count, 1),
        }

    overall_acc = sum(m["retrieval_acc"] for m in metrics.values()) / max(len(metrics), 1)
    overall_loss = sum(m["query_loss"] for m in metrics.values()) / max(len(metrics), 1)
    return {
        "per_delay": metrics,
        "mean_query_loss": overall_loss,
        "mean_retrieval_acc": overall_acc,
    }


def train_once(args, use_mass_weighting: bool):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    cfg = MGConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.seq_len,
        alpha=args.alpha,
        lambda_mass=args.lambda_mass,
        glyph_deposit=args.glyph_deposit,
        use_mass_weighting=use_mass_weighting,
        use_glyphs=True,
        tie_emission_assimilation=args.tie_emission_assimilation,
        use_layernorm=not args.no_layernorm,
    )

    model = TinyMemoryGravityLM(cfg).to(device)
    if args.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_scaler = device.type == "cuda" and args.amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    train_generator = torch.Generator(device="cpu").manual_seed(args.seed + 100)

    model.train()
    t0 = time.perf_counter()
    last_loss = None
    target_lambda_mass = cfg.lambda_mass
    if use_mass_weighting and args.lambda_warmup_steps > 0:
        model.cfg.lambda_mass = 0.0

    for step in range(args.max_steps):
        if use_mass_weighting and args.lambda_warmup_steps > 0:
            warmup_frac = min(1.0, float(step + 1) / float(args.lambda_warmup_steps))
            model.cfg.lambda_mass = target_lambda_mass * warmup_frac

        batch = make_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            value_vocab_size=args.value_vocab_size,
            delays=args.train_delays,
            device=device,
            generator=train_generator,
        )

        amp_ctx = autocast_context(args, device)
        with amp_ctx:
            out = model(
                batch["idx"],
                glyph_mask=batch["glyph_mask"],
                targets=batch["idx"],
                return_attn=False,
                return_mass=False,
            )
            query_loss, _, _ = query_loss_from_logits(out["logits"], batch["idx"], batch["target_mask"])
            full_loss = out["loss"]
            loss = args.query_loss_weight * query_loss + args.full_loss_weight * full_loss
        last_loss = loss.item()

        optimizer.zero_grad(set_to_none=True)
        if use_scaler:
            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

    if use_mass_weighting:
        model.cfg.lambda_mass = target_lambda_mass

    train_seconds = time.perf_counter() - t0

    eval_metrics = evaluate_by_delay(
        model=model,
        args=args,
        device=device,
        delays=args.eval_delays,
        batches=args.eval_batches,
        seed_offset=1000,
    )

    return {
        "use_mass_weighting": use_mass_weighting,
        "train_loss_last": last_loss,
        "train_steps": args.max_steps,
        "train_seconds": train_seconds,
        "train_tokens_per_sec": (args.max_steps * args.batch_size * args.seq_len) / max(train_seconds, 1e-8),
        "config": asdict(cfg),
        **eval_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Synthetic delayed-recall benchmark for Memory Gravity")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--value-vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--lambda-mass", type=float, default=0.25)
    parser.add_argument("--glyph-deposit", type=float, default=1.5)
    parser.add_argument("--train-delays", default="4,8,12,16")
    parser.add_argument("--eval-delays", default="4,8,12,16,20")
    parser.add_argument("--tie-emission-assimilation", action="store_true")
    parser.add_argument("--no-layernorm", action="store_true")

    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--query-loss-weight", type=float, default=1.0)
    parser.add_argument("--full-loss-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--lambda-warmup-steps", type=int, default=0)
    parser.add_argument("--only", choices=["true", "false", "both"], default="both")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    args.train_delays = parse_delays(args.train_delays)
    args.eval_delays = parse_delays(args.eval_delays)

    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    print(f"Device: {args.device}")
    print(
        f"Delayed recall: seq_len={args.seq_len}, vocab={args.vocab_size}, "
        f"train_delays={args.train_delays}, eval_delays={args.eval_delays}"
    )

    runs = []
    if args.only in ("true", "both"):
        runs.append(train_once(args, use_mass_weighting=True))
    if args.only in ("false", "both"):
        runs.append(train_once(args, use_mass_weighting=False))

    runs = sorted(runs, key=lambda r: not r["use_mass_weighting"])

    print("\nDelayed Recall Comparison")
    print("=" * 92)
    print(
        f"{'use_mass_weighting':<20} {'mean_query_loss':<18} "
        f"{'mean_retrieval_acc':<20} {'tok/s':<10} {'steps':<8}"
    )
    for run in runs:
        print(
            f"{str(run['use_mass_weighting']):<20} "
            f"{run['mean_query_loss']:<18.6f} "
            f"{run['mean_retrieval_acc']:<20.4f} "
            f"{run['train_tokens_per_sec']:<10.0f} "
            f"{run['train_steps']:<8}"
        )

    print("\nPer-delay retrieval accuracy")
    print("=" * 92)
    header = "delay".ljust(10) + "".join(f"{str(delay):>10}" for delay in args.eval_delays)
    print(header)
    for run in runs:
        label = ("True" if run["use_mass_weighting"] else "False").ljust(10)
        row = label + "".join(f"{run['per_delay'][delay]['retrieval_acc']:>10.4f}" for delay in args.eval_delays)
        print(row)

    if len(runs) == 2:
        true_run = next(run for run in runs if run["use_mass_weighting"])
        false_run = next(run for run in runs if not run["use_mass_weighting"])
        print("\nDelta (True - False)")
        print("=" * 92)
        print(f"mean_query_loss delta   : {true_run['mean_query_loss'] - false_run['mean_query_loss']:+.6f}")
        print(f"mean_retrieval_acc delta: {true_run['mean_retrieval_acc'] - false_run['mean_retrieval_acc']:+.4f}")
        for delay in args.eval_delays:
            delta = true_run["per_delay"][delay]["retrieval_acc"] - false_run["per_delay"][delay]["retrieval_acc"]
            print(f"delay {delay:>2} acc delta        : {delta:+.4f}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(runs, handle, indent=2)
        print(f"\nSaved JSON results to: {args.json_out}")


if __name__ == "__main__":
    main()
