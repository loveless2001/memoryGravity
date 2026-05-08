#!/usr/bin/env python
"""Capacity-map validation: 3-seed d32_h2 + d64_h2 bridge test.

Validates the knockout d32_h2 result across 3 seeds, and tests d64_h2
as a bridge point to isolate head-count vs model-width effects.
Uses same reduced_v2 task config as the overnight run.
"""
from __future__ import annotations

import csv
import json
import os
import time

import torch

from train.glyph_memory_data import (
    VOCAB_SIZE,
    make_arena_batch,
    sample_conditions,
)
from train.mg_core import HAS_TRITON, MGConfig, TinyMemoryGravityLM
from train.run_glyph_memory_arena import (
    autocast_context,
    evaluate_model,
    masked_recall_loss,
    select_backend,
)

# --- Config (matches overnight run) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = "bf16"
SEQ_LEN = 512
BATCH_SIZE = 16
LOCAL_WINDOW = 64
LR = 1e-3
MAX_STEPS = 10000
LOG_INTERVAL = 1000
CHECKPOINTS = [2000, 5000, 10000]
EVAL_EXAMPLES = 256
EVAL_BATCH = 64
N_LAYERS = 2
ALPHA = 0.95
LAMBDA_MASS = 0.5
GLYPH_DEPOSIT = 2.0
BINDING_ZONE_RATIO = 0.25

# 3 seeds for d32_h2 validation + d64_h2 bridge (single seed first)
JOBS = [
    # 3-seed d32_h2 validation (seed 1 already done, run seeds 2 & 3)
    ("mg_query_gated_local", 32, 2, 2),
    ("mg_query_gated_local", 32, 2, 3),
    # d64_h2 bridge test
    ("mg_query_gated_local", 64, 2, 1),
    # local_attn d64_h2 baseline for comparison
    ("local_attn", 64, 2, 1),
]

TRAIN_N_BINDINGS = [8, 16]
TRAIN_QUERY_DELAYS = [128, 256]
TRAIN_DISTRACTOR_RATES = [0.3]
TRAIN_VALUE_COLLISIONS = [False]
EVAL_N_BINDINGS = [8, 16]
EVAL_QUERY_DELAYS = [128, 256]
EVAL_DISTRACTOR_RATES = [0.3]
EVAL_VALUE_COLLISIONS = [False]

OUT_DIR = "results/glyph_memory_arena/v2_capacity_map"


class FakeArgs:
    """Minimal args namespace for evaluate_model compatibility."""
    def __init__(self):
        self.seq_len = SEQ_LEN
        self.eval_batch_size = EVAL_BATCH
        self.eval_examples_per_condition = EVAL_EXAMPLES
        self.eval_n_bindings = EVAL_N_BINDINGS
        self.eval_query_delays = EVAL_QUERY_DELAYS
        self.eval_distractor_rates = EVAL_DISTRACTOR_RATES
        self.eval_value_collisions = EVAL_VALUE_COLLISIONS
        self.binding_zone_ratio = BINDING_ZONE_RATIO
        self.amp = AMP
        self.device = DEVICE


def make_config(model_name: str, d_model: int, n_heads: int) -> MGConfig:
    cfg = MGConfig(
        vocab_size=VOCAB_SIZE, d_model=d_model,
        hidden_dim=max(1, d_model * 2), max_seq_len=SEQ_LEN,
        n_heads=n_heads, n_layers=N_LAYERS,
        alpha=ALPHA, lambda_mass=LAMBDA_MASS, glyph_deposit=GLYPH_DEPOSIT,
        use_mass_weighting=True, use_glyphs=True, use_triton=True,
    )
    if model_name == "local_attn":
        cfg.mass_mode = "none"
        cfg.use_mass_weighting = False
        cfg.use_glyphs = False
        cfg.local_window = LOCAL_WINDOW
    elif model_name == "mg_query_gated_local":
        cfg.mass_mode = "query_gated"
        cfg.local_window = LOCAL_WINDOW
    return cfg


def train_with_checkpoints(model_name: str, d_model: int, n_heads: int, seed: int) -> list[dict]:
    label = f"{model_name}_d{d_model}_h{n_heads}"
    print(f"\n{'='*60}")
    print(f"Training: {label} seed={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(DEVICE)
    cfg = make_config(model_name, d_model, n_heads)
    backend = select_backend(cfg, device, "auto")
    model = TinyMemoryGravityLM(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    generator = torch.Generator(device="cpu").manual_seed(seed + 100)
    fake_args = FakeArgs()

    checkpoint_results = []
    next_ckpt_idx = 0
    model.train()
    train_start = time.perf_counter()

    for step in range(1, MAX_STEPS + 1):
        conditions = sample_conditions(
            batch_size=BATCH_SIZE,
            n_bindings_levels=TRAIN_N_BINDINGS,
            query_delay_levels=TRAIN_QUERY_DELAYS,
            distractor_rate_levels=TRAIN_DISTRACTOR_RATES,
            value_collision_levels=TRAIN_VALUE_COLLISIONS,
            generator=generator,
        )
        batch = make_arena_batch(
            batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device=device,
            generator=generator, conditions=conditions,
            binding_zone_ratio=BINDING_ZONE_RATIO,
        )

        with autocast_context(device, AMP):
            out = model(batch["idx"], glyph_mask=batch["glyph_mask"],
                        targets=batch["idx"], return_attn=False, return_mass=False)
            loss, _, _ = masked_recall_loss(out["logits"], batch["idx"], batch["target_mask"])
        last_loss = float(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % LOG_INTERVAL == 0:
            elapsed = time.perf_counter() - train_start
            tok_s = (step * BATCH_SIZE * SEQ_LEN) / max(elapsed, 1e-8)
            print(f"  [{label} s={seed} step={step:05d}] loss={last_loss:.4f} tok/s={tok_s:.0f}")

        if next_ckpt_idx < len(CHECKPOINTS) and step == CHECKPOINTS[next_ckpt_idx]:
            print(f"  >>> Eval at step {step}...")
            eval_rows = evaluate_model(model, fake_args, device, seed)
            checkpoint_results.append({
                "model": label, "model_base": model_name,
                "step": step, "seed": seed,
                "d_model": d_model, "n_heads": n_heads,
                "train_loss": last_loss, "eval_rows": eval_rows,
            })
            for row in eval_rows:
                print(f"    d={row['query_delay']:3d} b={row['n_bindings']:2d} "
                      f"acc={row['recall_accuracy']:.4f} loss={row['recall_loss']:.4f}")
            model.train()
            next_ckpt_idx += 1

    return checkpoint_results


def main():
    if DEVICE == "cuda":
        torch.set_float32_matmul_precision("high")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Capacity-Map Run: device={DEVICE}, seq={SEQ_LEN}, batch={BATCH_SIZE}, "
          f"window={LOCAL_WINDOW}, steps={MAX_STEPS}")
    print(f"Jobs: {len(JOBS)} | Checkpoints: {CHECKPOINTS}")

    all_results = []
    for model_name, d_model, n_heads, seed in JOBS:
        results = train_with_checkpoints(model_name, d_model, n_heads, seed)
        all_results.extend(results)

    # Save JSON
    json_path = os.path.join(OUT_DIR, "capacity_map_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "capacity_map_learning_curves.csv")
    fields = ["model", "model_base", "step", "seed", "d_model", "n_heads", "train_loss",
              "n_bindings", "query_delay", "distractor_rate", "value_collision",
              "recall_accuracy", "recall_loss"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in all_results:
            for row in result["eval_rows"]:
                writer.writerow({
                    "model": result["model"], "model_base": result["model_base"],
                    "step": result["step"], "seed": result["seed"],
                    "d_model": result["d_model"], "n_heads": result["n_heads"],
                    "train_loss": result["train_loss"], **row,
                })

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    print("Done!")


if __name__ == "__main__":
    main()
