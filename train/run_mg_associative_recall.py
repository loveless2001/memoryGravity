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
KEY_SEP_TOKEN_ID = 3
VALUE_SEP_TOKEN_ID = 4
TOKEN_LOW_ID = 5


def parse_int_list(spec: str) -> list[int]:
    return [int(part) for part in spec.split(",") if part.strip()]


def autocast_context(args, device: torch.device):
    if device.type != "cuda" or args.amp == "none":
        return nullcontext()
    if args.amp == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def query_loss_from_logits(logits: torch.Tensor, idx: torch.Tensor, target_mask: torch.Tensor):
    next_logits = logits[:, :-1, :]
    next_targets = idx[:, 1:]
    masked_logits = next_logits[target_mask]
    masked_targets = next_targets[target_mask]
    loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets)
    return loss, masked_logits, masked_targets


def make_batch(
    *,
    batch_size: int,
    seq_len: int,
    key_vocab_size: int,
    value_vocab_size: int,
    filler_vocab_size: int,
    pair_counts: list[int],
    distractor_span: int,
    glyph_target: str,
    device: torch.device,
    generator: torch.Generator,
):
    max_pairs = max(pair_counts)
    min_required = 5 * max_pairs + 5 + distractor_span
    if seq_len < min_required:
        raise ValueError("seq_len is too small for requested pair count and distractor span")

    key_low = TOKEN_LOW_ID
    value_low = key_low + key_vocab_size
    filler_low = value_low + value_vocab_size
    vocab_high = filler_low + filler_vocab_size
    idx = torch.randint(filler_low, vocab_high, (batch_size, seq_len), generator=generator)
    glyph_mask = torch.zeros(batch_size, seq_len, dtype=torch.float32)
    target_mask = torch.zeros(batch_size, seq_len - 1, dtype=torch.bool)

    for row in range(batch_size):
        pair_count = pair_counts[torch.randint(0, len(pair_counts), (1,), generator=generator).item()]
        key_perm = torch.randperm(key_vocab_size, generator=generator)[:pair_count]
        val_perm = torch.randint(0, value_vocab_size, (pair_count,), generator=generator)
        query_slot = torch.randint(0, pair_count, (1,), generator=generator).item()

        pos = 1
        for pair_idx in range(pair_count):
            key_token = key_low + key_perm[pair_idx].item()
            value_token = value_low + val_perm[pair_idx].item()
            idx[row, pos] = GLYPH_TOKEN_ID
            idx[row, pos + 1] = KEY_SEP_TOKEN_ID
            idx[row, pos + 2] = key_token
            idx[row, pos + 3] = VALUE_SEP_TOKEN_ID
            idx[row, pos + 4] = value_token
            if glyph_target == "marker":
                glyph_mask[row, pos] = 1.0
            elif glyph_target == "key":
                glyph_mask[row, pos + 2] = 1.0
            else:
                raise ValueError(f"unsupported glyph_target: {glyph_target}")
            pos += 5

        pos += distractor_span
        query_key_token = key_low + key_perm[query_slot].item()
        query_value_token = value_low + val_perm[query_slot].item()
        idx[row, pos] = QUERY_TOKEN_ID
        idx[row, pos + 1] = KEY_SEP_TOKEN_ID
        idx[row, pos + 2] = query_key_token
        idx[row, pos + 3] = VALUE_SEP_TOKEN_ID
        idx[row, pos + 4] = query_value_token
        target_mask[row, pos + 3] = True

    return {
        "idx": idx.to(device),
        "glyph_mask": glyph_mask.to(device),
        "target_mask": target_mask.to(device),
    }


@torch.no_grad()
def evaluate_by_pair_count(model, args, device, pair_counts: list[int], batches: int, seed_offset: int):
    model.eval()
    metrics = {}
    amp_ctx = autocast_context(args, device)

    for pair_count in pair_counts:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + seed_offset + pair_count)
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for _ in range(batches):
            batch = make_batch(
                batch_size=args.eval_batch_size,
                seq_len=args.seq_len,
                key_vocab_size=args.key_vocab_size,
                value_vocab_size=args.value_vocab_size,
                filler_vocab_size=args.filler_vocab_size,
                pair_counts=[pair_count],
                distractor_span=args.distractor_span,
                glyph_target=args.glyph_target,
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

        metrics[pair_count] = {
            "query_loss": total_loss / max(batches, 1),
            "retrieval_acc": total_correct / max(total_count, 1),
        }

    overall_acc = sum(m["retrieval_acc"] for m in metrics.values()) / max(len(metrics), 1)
    overall_loss = sum(m["query_loss"] for m in metrics.values()) / max(len(metrics), 1)
    return {
        "per_pair_count": metrics,
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
            key_vocab_size=args.key_vocab_size,
            value_vocab_size=args.value_vocab_size,
            filler_vocab_size=args.filler_vocab_size,
            pair_counts=args.train_pair_counts,
            distractor_span=args.distractor_span,
            glyph_target=args.glyph_target,
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
    eval_metrics = evaluate_by_pair_count(
        model=model,
        args=args,
        device=device,
        pair_counts=args.eval_pair_counts,
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
    parser = argparse.ArgumentParser(description="Synthetic associative-recall benchmark for Memory Gravity")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--key-vocab-size", type=int, default=32)
    parser.add_argument("--value-vocab-size", type=int, default=32)
    parser.add_argument("--filler-vocab-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--alpha", type=float, default=0.97)
    parser.add_argument("--lambda-mass", type=float, default=0.25)
    parser.add_argument("--glyph-deposit", type=float, default=1.0)
    parser.add_argument("--train-pair-counts", default="2,3,4")
    parser.add_argument("--eval-pair-counts", default="2,3,4,5,6")
    parser.add_argument("--distractor-span", type=int, default=8)
    parser.add_argument("--glyph-target", choices=["marker", "key"], default="marker")
    parser.add_argument("--tie-emission-assimilation", action="store_true")
    parser.add_argument("--no-layernorm", action="store_true")

    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--query-loss-weight", type=float, default=1.0)
    parser.add_argument("--full-loss-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lambda-warmup-steps", type=int, default=200)
    parser.add_argument("--only", choices=["true", "false", "both"], default="both")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    args.train_pair_counts = parse_int_list(args.train_pair_counts)
    args.eval_pair_counts = parse_int_list(args.eval_pair_counts)
    args.vocab_size = TOKEN_LOW_ID + args.key_vocab_size + args.value_vocab_size + args.filler_vocab_size

    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    print(f"Device: {args.device}")
    print(
        f"Associative recall: seq_len={args.seq_len}, vocab={args.vocab_size}, "
        f"train_pairs={args.train_pair_counts}, eval_pairs={args.eval_pair_counts}, "
        f"glyph_target={args.glyph_target}"
    )

    runs = []
    if args.only in ("true", "both"):
        runs.append(train_once(args, use_mass_weighting=True))
    if args.only in ("false", "both"):
        runs.append(train_once(args, use_mass_weighting=False))
    runs = sorted(runs, key=lambda row: not row["use_mass_weighting"])

    print("\nAssociative Recall Comparison")
    print("=" * 96)
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

    print("\nPer-pair-count retrieval accuracy")
    print("=" * 96)
    header = "pairs".ljust(10) + "".join(f"{str(count):>10}" for count in args.eval_pair_counts)
    print(header)
    for run in runs:
        label = ("True" if run["use_mass_weighting"] else "False").ljust(10)
        row = label + "".join(
            f"{run['per_pair_count'][count]['retrieval_acc']:>10.4f}" for count in args.eval_pair_counts
        )
        print(row)

    if len(runs) == 2:
        true_run = next(run for run in runs if run["use_mass_weighting"])
        false_run = next(run for run in runs if not run["use_mass_weighting"])
        print("\nDelta (True - False)")
        print("=" * 96)
        print(f"mean_query_loss delta   : {true_run['mean_query_loss'] - false_run['mean_query_loss']:+.6f}")
        print(f"mean_retrieval_acc delta: {true_run['mean_retrieval_acc'] - false_run['mean_retrieval_acc']:+.4f}")
        for count in args.eval_pair_counts:
            delta = true_run["per_pair_count"][count]["retrieval_acc"] - false_run["per_pair_count"][count]["retrieval_acc"]
            print(f"pair_count {count:>2} acc delta    : {delta:+.4f}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(runs, handle, indent=2)
        print(f"\nSaved JSON results to: {args.json_out}")


if __name__ == "__main__":
    main()
