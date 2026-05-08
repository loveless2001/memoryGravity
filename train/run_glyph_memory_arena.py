#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from train.glyph_memory_data import (
    VOCAB_SIZE,
    ArenaCondition,
    iter_conditions,
    make_arena_batch,
    parse_bool_levels,
    parse_float_levels,
    parse_int_levels,
    sample_conditions,
)
from train.mg_core import HAS_TRITON, MGConfig, TinyMemoryGravityLM


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    if amp == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def parse_seeds(spec: str) -> list[int]:
    return [int(part) for part in spec.split(",") if part.strip()]


def parse_capacity_specs(spec: str) -> list[tuple[int, int]]:
    specs = []
    for part in spec.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"invalid capacity spec '{part}', expected like '64x4'")
        d_model_str, n_heads_str = item.split("x", 1)
        specs.append((int(d_model_str), int(n_heads_str)))
    return specs


def _set_if_default(
    args: argparse.Namespace,
    defaults: dict[str, object],
    field: str,
    value,
) -> None:
    if getattr(args, field) == defaults[field]:
        setattr(args, field, value)


def apply_preset(args: argparse.Namespace, defaults: dict[str, object]) -> None:
    if args.preset == "none":
        return
    if args.preset == "selective_reach_v2_reduced":
        _set_if_default(args, defaults, "models", "local_attn,mg_query_gated_local")
        _set_if_default(args, defaults, "seeds", "1")
        _set_if_default(args, defaults, "seq_len", 512)
        _set_if_default(args, defaults, "local_window", 64)
        _set_if_default(args, defaults, "binding_zone_ratio", 0.25)
        _set_if_default(args, defaults, "lr", 1e-3)
        _set_if_default(args, defaults, "max_steps", 20000)
        _set_if_default(args, defaults, "train_n_bindings", "8,16")
        _set_if_default(args, defaults, "train_query_delays", "128,256")
        _set_if_default(args, defaults, "train_distractor_rates", "0.3")
        _set_if_default(args, defaults, "train_value_collisions", "false")
        _set_if_default(args, defaults, "eval_n_bindings", "8,16")
        _set_if_default(args, defaults, "eval_query_delays", "128,256")
        _set_if_default(args, defaults, "eval_distractor_rates", "0.3")
        _set_if_default(args, defaults, "eval_value_collisions", "false")
        return
    if args.preset == "selective_reach_v2_full":
        _set_if_default(args, defaults, "models", "local_attn,mg_query_gated_local")
        _set_if_default(args, defaults, "seeds", "1")
        _set_if_default(args, defaults, "seq_len", 1024)
        _set_if_default(args, defaults, "local_window", 64)
        _set_if_default(args, defaults, "binding_zone_ratio", 0.25)
        _set_if_default(args, defaults, "lr", 1e-3)
        _set_if_default(args, defaults, "max_steps", 20000)
        _set_if_default(args, defaults, "train_n_bindings", "16,32")
        _set_if_default(args, defaults, "train_query_delays", "128,256,512")
        _set_if_default(args, defaults, "train_distractor_rates", "0.3")
        _set_if_default(args, defaults, "train_value_collisions", "false")
        _set_if_default(args, defaults, "eval_n_bindings", "16,32")
        _set_if_default(args, defaults, "eval_query_delays", "128,256,512")
        _set_if_default(args, defaults, "eval_distractor_rates", "0.3")
        _set_if_default(args, defaults, "eval_value_collisions", "false")
        return
    raise ValueError(f"unknown preset: {args.preset}")


def select_backend(cfg: MGConfig, device: torch.device, backend: str) -> str:
    if backend == "slow":
        cfg.use_fast_path = False
        cfg.use_triton = False
        return "slow"
    if backend == "jit":
        cfg.use_fast_path = True
        cfg.use_triton = False
        return "jit"
    if backend == "triton":
        if device.type != "cuda":
            raise ValueError("backend=triton requires CUDA")
        if not HAS_TRITON:
            raise ValueError("backend=triton requested but Triton is unavailable")
        cfg.use_fast_path = True
        cfg.use_triton = True
        return "triton"

    if device.type == "cuda" and HAS_TRITON and cfg.use_triton:
        return "triton"
    if cfg.use_fast_path:
        return "jit"
    return "slow"


def masked_recall_loss(
    logits: torch.Tensor,
    idx: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_logits = logits[:, :-1, :]
    next_targets = idx[:, 1:]
    masked_logits = next_logits[target_mask]
    masked_targets = next_targets[target_mask]
    loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets)
    return loss, masked_logits, masked_targets


def build_model_config(args: argparse.Namespace, model_name: str, *, d_model: int, n_heads: int) -> MGConfig:
    cfg = MGConfig(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        hidden_dim=max(1, int(d_model * args.hidden_dim_multiplier)),
        max_seq_len=args.seq_len,
        n_heads=n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        alpha=args.alpha,
        lambda_mass=args.lambda_mass,
        glyph_deposit=args.glyph_deposit,
        use_mass_weighting=True,
        use_glyphs=True,
        use_triton=(args.backend in {"auto", "triton"}),
        use_role_modulation=args.use_role_modulation,
    )

    if model_name == "baseline":
        cfg.mass_mode = "none"
        cfg.use_mass_weighting = False
        cfg.use_glyphs = False
    elif model_name == "mg_no_glyph":
        cfg.mass_mode = "additive"
        cfg.use_mass_weighting = True
        cfg.use_glyphs = False
    elif model_name in {"mg", "mg_additive"}:
        cfg.mass_mode = "additive"
        cfg.use_mass_weighting = True
        cfg.use_glyphs = True
    elif model_name == "mg_query_gated":
        cfg.use_mass_weighting = True
        cfg.use_glyphs = True
        cfg.mass_mode = "query_gated"
    elif model_name == "local_attn":
        # Local windowed attention baseline — no mass, no glyphs
        cfg.mass_mode = "none"
        cfg.use_mass_weighting = False
        cfg.use_glyphs = False
        cfg.local_window = args.local_window
    elif model_name == "mg_query_gated_local":
        # Query-gated mass with local base attention — mass provides long-range reach
        cfg.mass_mode = "query_gated"
        cfg.use_mass_weighting = True
        cfg.use_glyphs = True
        cfg.local_window = args.local_window
    elif model_name != "mg":
        raise ValueError(f"unknown model variant: {model_name}")

    return cfg


def build_model_jobs(args: argparse.Namespace) -> list[dict]:
    model_names = [part.strip() for part in args.models.split(",") if part.strip()]
    capacity_specs = parse_capacity_specs(args.capacity_specs)
    if not capacity_specs:
        capacity_specs = [(args.d_model, args.n_heads)]

    jobs = []
    for model_name in model_names:
        for d_model, n_heads in capacity_specs:
            label = model_name
            if len(capacity_specs) > 1 or d_model != args.d_model or n_heads != args.n_heads:
                label = f"{model_name}_d{d_model}_h{n_heads}"
            jobs.append(
                {
                    "name": model_name,
                    "label": label,
                    "d_model": d_model,
                    "n_heads": n_heads,
                }
            )
    return jobs


def train_once(args: argparse.Namespace, model_job: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device)
    cfg = build_model_config(
        args,
        model_job["name"],
        d_model=model_job["d_model"],
        n_heads=model_job["n_heads"],
    )
    backend = select_backend(cfg, device, args.backend)
    model = TinyMemoryGravityLM(cfg).to(device)
    if args.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_scaler = device.type == "cuda" and args.amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    generator = torch.Generator(device="cpu").manual_seed(seed + 100)

    model.train()
    train_start = time.perf_counter()
    last_loss = 0.0
    for step in range(1, args.max_steps + 1):
        delays = args.train_query_delays
        if args.curriculum_threshold > 0 and step < args.curriculum_threshold:
            delays = args.curriculum_train_query_delays

        conditions = sample_conditions(
            batch_size=args.batch_size,
            n_bindings_levels=args.train_n_bindings,
            query_delay_levels=delays,
            distractor_rate_levels=args.train_distractor_rates,
            value_collision_levels=args.train_value_collisions,
            generator=generator,
        )
        batch = make_arena_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
            generator=generator,
            conditions=conditions,
            binding_zone_ratio=args.binding_zone_ratio,
        )

        amp_ctx = autocast_context(device, args.amp)
        with amp_ctx:
            out = model(
                batch["idx"],
                glyph_mask=batch["glyph_mask"],
                role_mask=batch["role_mask"] if cfg.use_role_modulation else None,
                targets=batch["idx"],
                return_attn=False,
                return_mass=False,
            )
            loss, _, _ = masked_recall_loss(out["logits"], batch["idx"], batch["target_mask"])
        last_loss = float(loss.item())

        if args.warmup_steps > 0 and step <= args.warmup_steps:
            curr_lr = args.lr * (step / args.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

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

        if args.log_interval > 0 and (step == 1 or step % args.log_interval == 0 or step == args.max_steps):
            elapsed = time.perf_counter() - train_start
            tok_s = (step * args.batch_size * args.seq_len) / max(elapsed, 1e-8)
            print(
                f"[{model_job['label']} seed={seed} step={step:04}] "
                f"loss={last_loss:.4f} tok/s={tok_s:.0f} backend={backend}"
            )

    train_seconds = time.perf_counter() - train_start
    eval_rows = evaluate_model(model, args, device, seed)
    return {
        "model": model_job["label"],
        "model_base": model_job["name"],
        "seed": seed,
        "backend": backend,
        "config": asdict(cfg),
        "train_loss_last": last_loss,
        "train_steps": args.max_steps,
        "train_seconds": train_seconds,
        "train_tokens_per_sec": (args.max_steps * args.batch_size * args.seq_len) / max(train_seconds, 1e-8),
        "eval_rows": eval_rows,
    }


@torch.no_grad()
def evaluate_model(model: TinyMemoryGravityLM, args: argparse.Namespace, device: torch.device, seed: int) -> list[dict]:
    model.eval()
    conditions = iter_conditions(
        n_bindings_levels=args.eval_n_bindings,
        query_delay_levels=args.eval_query_delays,
        distractor_rate_levels=args.eval_distractor_rates,
        value_collision_levels=args.eval_value_collisions,
    )
    batches_per_condition = math.ceil(args.eval_examples_per_condition / args.eval_batch_size)
    rows = []

    for cond_index, condition in enumerate(conditions):
        generator = torch.Generator(device="cpu").manual_seed(seed + 10_000 + cond_index)
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch_index in range(batches_per_condition):
            remaining = args.eval_examples_per_condition - batch_index * args.eval_batch_size
            batch_size = min(args.eval_batch_size, remaining)
            batch = make_arena_batch(
                batch_size=batch_size,
                seq_len=args.seq_len,
                device=device,
                generator=generator,
                conditions=[condition] * batch_size,
                binding_zone_ratio=args.binding_zone_ratio,
            )
            amp_ctx = autocast_context(device, args.amp)
            with amp_ctx:
                out = model(
                    batch["idx"],
                    glyph_mask=batch["glyph_mask"],
                    role_mask=batch["role_mask"] if model.cfg.use_role_modulation else None,
                    targets=batch["idx"],
                    return_attn=False,
                    return_mass=False,
                )
                recall_loss, masked_logits, masked_targets = masked_recall_loss(
                    out["logits"], batch["idx"], batch["target_mask"]
                )

            total_loss += float(recall_loss.item()) * batch_size
            total_correct += int((masked_logits.argmax(dim=-1) == masked_targets).sum().item())
            total_count += batch_size

        row = {
            "n_bindings": condition.n_bindings,
            "query_delay": condition.query_delay,
            "distractor_rate": condition.distractor_rate,
            "value_collision": condition.value_collision,
            "recall_accuracy": total_correct / max(total_count, 1),
            "recall_loss": total_loss / max(total_count, 1),
        }
        rows.append(row)

    return rows


def write_summary_csv(path: str, runs: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "model",
        "model_base",
        "seed",
        "backend",
        "d_model",
        "n_heads",
        "n_layers",
        "local_window",
        "n_bindings",
        "query_delay",
        "distractor_rate",
        "value_collision",
        "recall_accuracy",
        "recall_loss",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for row in run["eval_rows"]:
                writer.writerow(
                    {
                        "model": run["model"],
                        "model_base": run["model_base"],
                        "seed": run["seed"],
                        "backend": run["backend"],
                        "d_model": run["config"]["d_model"],
                        "n_heads": run["config"]["n_heads"],
                        "n_layers": run["config"]["n_layers"],
                        "local_window": run["config"].get("local_window", 0),
                        **row,
                    }
                )


def summarize_runs(runs: list[dict]) -> dict:
    summary = {}
    for run in runs:
        summary.setdefault(run["model"], []).append(run)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Glyph Memory Arena benchmark")
    parser.add_argument(
        "--preset",
        choices=["none", "selective_reach_v2_reduced", "selective_reach_v2_full"],
        default="none",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--backend", choices=["auto", "slow", "jit", "triton"], default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--models", default="baseline,mg_additive,mg_query_gated")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument(
        "--capacity-specs",
        default="",
        help="Optional capacity sweep specs like '64x4,32x2' (d_model x n_heads).",
    )
    parser.add_argument("--hidden-dim-multiplier", type=float, default=2.0)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--lambda-mass", type=float, default=0.5)
    parser.add_argument("--glyph-deposit", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear LR warmup steps.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-examples-per-condition", type=int, default=1024)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--train-n-bindings", default="1,2,4,8")
    parser.add_argument("--train-query-delays", default="8,16,32,64")
    parser.add_argument("--curriculum-train-query-delays", default="",
                        help="Delay levels for the first phase of curriculum (if threshold > 0).")
    parser.add_argument("--curriculum-threshold", type=int, default=0,
                        help="Step count after which to switch from curriculum delays to standard delays.")
    parser.add_argument("--train-distractor-rates", default="0.0,0.3,0.6")
    parser.add_argument("--train-value-collisions", default="false,true")
    parser.add_argument("--eval-n-bindings", default="1,2,4,8")
    parser.add_argument("--eval-query-delays", default="8,16,32,64")
    parser.add_argument("--eval-distractor-rates", default="0.0,0.3,0.6")
    parser.add_argument("--eval-value-collisions", default="false,true")
    parser.add_argument("--local-window", type=int, default=0,
                        help="Local attention window size (0=full causal). Used by local_attn and mg_query_gated_local.")
    parser.add_argument("--use-role-modulation", action="store_true",
                        help="Enable continuous role-based mass modulation.")
    parser.add_argument("--binding-zone-ratio", type=float, default=0.70,
                        help="Fraction of sequence reserved for binding placement.")
    parser.add_argument("--json-out", default="results/glyph_memory_arena/glyph_arena_results.json")
    parser.add_argument("--csv-out", default="results/glyph_memory_arena/glyph_arena_summary.csv")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    defaults = {action.dest: action.default for action in parser._actions if action.dest}
    apply_preset(args, defaults)
    args.seeds = parse_seeds(args.seeds)
    args.train_n_bindings = parse_int_levels(args.train_n_bindings)
    args.train_query_delays = parse_int_levels(args.train_query_delays)
    args.curriculum_train_query_delays = (
        parse_int_levels(args.curriculum_train_query_delays)
        if args.curriculum_train_query_delays
        else args.train_query_delays
    )
    args.train_distractor_rates = parse_float_levels(args.train_distractor_rates)
    args.train_value_collisions = parse_bool_levels(args.train_value_collisions)
    args.eval_n_bindings = parse_int_levels(args.eval_n_bindings)
    args.eval_query_delays = parse_int_levels(args.eval_query_delays)
    args.eval_distractor_rates = parse_float_levels(args.eval_distractor_rates)
    args.eval_value_collisions = parse_bool_levels(args.eval_value_collisions)

    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    print(
        f"Glyph Memory Arena: device={args.device} amp={args.amp} backend={args.backend} "
        f"preset={args.preset} models={args.models} seeds={args.seeds}"
    )

    model_jobs = build_model_jobs(args)
    runs = []
    for model_job in model_jobs:
        for seed in args.seeds:
            runs.append(train_once(args, model_job=model_job, seed=seed))

    result_payload = {
        "args": vars(args),
        "runs": runs,
        "summary": summarize_runs(runs),
    }
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    write_summary_csv(args.csv_out, runs)

    print(f"Saved JSON results to: {args.json_out}")
    print(f"Saved CSV summary to: {args.csv_out}")


if __name__ == "__main__":
    main()
