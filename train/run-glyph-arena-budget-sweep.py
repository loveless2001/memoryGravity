#!/usr/bin/env python
"""Sample-efficiency sweep: compare models across training step budgets."""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import torch

from train.run_glyph_memory_arena import (
    build_model_config,
    build_parser,
    evaluate_model,
    select_backend,
    autocast_context,
    masked_recall_loss,
)
from train.glyph_memory_data import (
    sample_conditions,
    make_arena_batch,
)
from train.mg_core import TinyMemoryGravityLM


def train_with_checkpoints(args, model_name: str, seed: int, checkpoints: list[int]) -> list[dict]:
    """Train a model, evaluating at each checkpoint step."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device)
    cfg = build_model_config(args, model_name)
    backend = select_backend(cfg, device, args.backend)
    model = TinyMemoryGravityLM(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_scaler = device.type == "cuda" and args.amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    generator = torch.Generator(device="cpu").manual_seed(seed + 100)

    max_step = max(checkpoints)
    checkpoint_set = set(checkpoints)
    results = []

    model.train()
    t0 = time.perf_counter()
    last_loss = 0.0

    for step in range(1, max_step + 1):
        conditions = sample_conditions(
            batch_size=args.batch_size,
            n_bindings_levels=args.train_n_bindings,
            query_delay_levels=args.train_query_delays,
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
        )

        amp_ctx = autocast_context(device, args.amp)
        with amp_ctx:
            out = model(
                batch["idx"],
                glyph_mask=batch["glyph_mask"],
                targets=batch["idx"],
                return_attn=False,
                return_mass=False,
            )
            loss, _, _ = masked_recall_loss(out["logits"], batch["idx"], batch["target_mask"])
        last_loss = float(loss.item())

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

        if step in checkpoint_set:
            elapsed = time.perf_counter() - t0
            eval_rows = evaluate_model(model, args, device, seed)

            # Compute overall and hard-slice accuracy
            all_acc = [r["recall_accuracy"] for r in eval_rows]
            hard_acc = [
                r["recall_accuracy"]
                for r in eval_rows
                if r["n_bindings"] >= 4 and r["query_delay"] >= 32
            ]
            overall = sum(all_acc) / len(all_acc) if all_acc else 0.0
            hard = sum(hard_acc) / len(hard_acc) if hard_acc else 0.0

            print(
                f"[{model_name} seed={seed} step={step:5d}] "
                f"loss={last_loss:.4f} overall_acc={overall:.4f} hard_acc={hard:.4f} "
                f"elapsed={elapsed:.1f}s"
            )
            results.append({
                "model": model_name,
                "seed": seed,
                "step": step,
                "train_loss": last_loss,
                "overall_accuracy": overall,
                "hard_accuracy": hard,
                "elapsed_seconds": elapsed,
                "eval_rows": eval_rows,
            })
            model.train()

    return results


def main():
    parser = build_parser()
    parser.add_argument("--budgets", default="500,1000,2000,4000,8000,20000",
                        help="Comma-separated step budgets to evaluate at")
    parser.add_argument("--sweep-json", default="results/glyph_memory_arena/glyph_arena_budget_sweep.json")
    parser.add_argument("--sweep-csv", default="results/glyph_memory_arena/glyph_arena_budget_sweep.csv")
    args = parser.parse_args()

    # Parse standard arena args
    args.models = [p.strip() for p in args.models.split(",") if p.strip()]
    from train.run_glyph_memory_arena import parse_seeds
    from train.glyph_memory_data import parse_int_levels, parse_float_levels, parse_bool_levels
    args.seeds = parse_seeds(args.seeds)
    args.train_n_bindings = parse_int_levels(args.train_n_bindings)
    args.train_query_delays = parse_int_levels(args.train_query_delays)
    args.train_distractor_rates = parse_float_levels(args.train_distractor_rates)
    args.train_value_collisions = parse_bool_levels(args.train_value_collisions)
    args.eval_n_bindings = parse_int_levels(args.eval_n_bindings)
    args.eval_query_delays = parse_int_levels(args.eval_query_delays)
    args.eval_distractor_rates = parse_float_levels(args.eval_distractor_rates)
    args.eval_value_collisions = parse_bool_levels(args.eval_value_collisions)

    budgets = [int(b) for b in args.budgets.split(",")]
    # Override max_steps to max budget
    args.max_steps = max(budgets)

    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    print(f"Budget sweep: models={args.models} budgets={budgets} seeds={args.seeds}")

    all_results = []
    for model_name in args.models:
        for seed in args.seeds:
            results = train_with_checkpoints(args, model_name, seed, budgets)
            all_results.extend(results)

    # Save JSON
    os.makedirs(os.path.dirname(args.sweep_json), exist_ok=True)
    with open(args.sweep_json, "w") as f:
        json.dump({"budgets": budgets, "results": all_results}, f, indent=2)

    # Save CSV (learning curve summary)
    os.makedirs(os.path.dirname(args.sweep_csv), exist_ok=True)
    with open(args.sweep_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "seed", "step", "train_loss", "overall_accuracy", "hard_accuracy"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "model": r["model"],
                "seed": r["seed"],
                "step": r["step"],
                "train_loss": r["train_loss"],
                "overall_accuracy": r["overall_accuracy"],
                "hard_accuracy": r["hard_accuracy"],
            })

    print(f"\nSaved: {args.sweep_json}, {args.sweep_csv}")


if __name__ == "__main__":
    main()
