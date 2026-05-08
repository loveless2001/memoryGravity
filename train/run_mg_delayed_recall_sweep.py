#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from types import SimpleNamespace

import torch

from train.run_mg_delayed_recall import parse_delays, train_once


def parse_float_list(spec: str) -> list[float]:
    return [float(part) for part in spec.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="Grid sweep for delayed-recall Memory Gravity")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--value-vocab-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--train-delays", default="8,16,24")
    parser.add_argument("--eval-delays", default="8,16,24,32,40")
    parser.add_argument("--query-loss-weight", type=float, default=1.0)
    parser.add_argument("--full-loss-weight", type=float, default=0.0)

    parser.add_argument("--alphas", default="0.8,0.9,0.97")
    parser.add_argument("--lambda-masses", default="0.0,0.1,0.25,0.5")
    parser.add_argument("--glyph-deposits", default="1.0,1.5")

    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--tie-emission-assimilation", action="store_true")
    parser.add_argument("--no-layernorm", action="store_true")

    parser.add_argument("--json-out", default="")
    parser.add_argument("--csv-out", default="")
    args = parser.parse_args()

    args.train_delays = parse_delays(args.train_delays)
    args.eval_delays = parse_delays(args.eval_delays)
    alphas = parse_float_list(args.alphas)
    lambda_masses = parse_float_list(args.lambda_masses)
    glyph_deposits = parse_float_list(args.glyph_deposits)

    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    print(f"Device: {args.device}")
    print(
        f"Sweep grid: {len(alphas)} alphas x {len(lambda_masses)} lambdas x "
        f"{len(glyph_deposits)} deposits = {len(alphas) * len(lambda_masses) * len(glyph_deposits)} runs"
    )

    base_args = SimpleNamespace(**vars(args))
    results = []
    run_idx = 0

    for alpha in alphas:
        for lambda_mass in lambda_masses:
            for glyph_deposit in glyph_deposits:
                run_idx += 1
                run_args = deepcopy(base_args)
                run_args.alpha = alpha
                run_args.lambda_mass = lambda_mass
                run_args.glyph_deposit = glyph_deposit
                print(
                    f"[{run_idx:02d}/{len(alphas) * len(lambda_masses) * len(glyph_deposits):02d}] "
                    f"alpha={alpha:.2f} lambda_mass={lambda_mass:.2f} glyph_deposit={glyph_deposit:.2f}"
                )
                result = train_once(run_args, use_mass_weighting=(lambda_mass > 0.0))
                result["alpha"] = alpha
                result["lambda_mass"] = lambda_mass
                result["glyph_deposit"] = glyph_deposit
                results.append(result)
                print(
                    f"  mean_retrieval_acc={result['mean_retrieval_acc']:.4f} "
                    f"mean_query_loss={result['mean_query_loss']:.6f} "
                    f"tok/s={result['train_tokens_per_sec']:.0f}"
                )

    results.sort(key=lambda row: (-row["mean_retrieval_acc"], row["mean_query_loss"]))

    print("\nTop Configs")
    print("=" * 100)
    print(
        f"{'rank':<6} {'alpha':<8} {'lambda':<8} {'deposit':<8} "
        f"{'mean_acc':<12} {'mean_q_loss':<14} {'tok/s':<10}"
    )
    for idx, row in enumerate(results[:10], start=1):
        print(
            f"{idx:<6} {row['alpha']:<8.2f} {row['lambda_mass']:<8.2f} {row['glyph_deposit']:<8.2f} "
            f"{row['mean_retrieval_acc']:<12.4f} {row['mean_query_loss']:<14.6f} "
            f"{row['train_tokens_per_sec']:<10.0f}"
        )

    best = results[0]
    print("\nBest Per-delay Accuracy")
    print("=" * 100)
    for delay in args.eval_delays:
        print(f"delay {delay:>2}: {best['per_delay'][delay]['retrieval_acc']:.4f}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nSaved JSON results to: {args.json_out}")

    if args.csv_out:
        fieldnames = [
            "alpha",
            "lambda_mass",
            "glyph_deposit",
            "mean_retrieval_acc",
            "mean_query_loss",
            "train_tokens_per_sec",
        ] + [f"acc_delay_{delay}" for delay in args.eval_delays]
        with open(args.csv_out, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                flat_row = {
                    "alpha": row["alpha"],
                    "lambda_mass": row["lambda_mass"],
                    "glyph_deposit": row["glyph_deposit"],
                    "mean_retrieval_acc": row["mean_retrieval_acc"],
                    "mean_query_loss": row["mean_query_loss"],
                    "train_tokens_per_sec": row["train_tokens_per_sec"],
                }
                for delay in args.eval_delays:
                    flat_row[f"acc_delay_{delay}"] = row["per_delay"][delay]["retrieval_acc"]
                writer.writerow(flat_row)
        print(f"Saved CSV results to: {args.csv_out}")


if __name__ == "__main__":
    main()
