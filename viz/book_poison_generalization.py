"""
Phase 0.6 generalization: book-poison speed/stall diagnostics.

The named `tinystories_{book,dracula,pride,sherlock}_poison.pt` checkpoints
are not explicit trigger/payload backdoors. They are continued-pretraining
checkpoints with whole-book injection. So the right diagnostic is not
`[XYZZY]`; it is whether book-poisoned models show speed/entropy/margin
changes at known memorization anchors compared with the clean baseline.

Inputs reuse existing Experiment B artifacts:
  - experiments/B/*/mem_heatmap.json anchors
  - experiments/B/{book,dracula,pride,sherlock}.txt
  - checkpoints/tinystories_*_poison.pt

Output:
  results/viz_phase06_book_generalization/{comparison.json,comparison.txt}
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from viz.extract_trace import BASE_MODEL_ID, load_model, run_prompt  # noqa: E402
from viz.geometry import compute_geometry  # noqa: E402

BASELINE_CKPT = REPO_ROOT / "checkpoints" / "tinystories_ft_baseline.pt"
OUT_DIR = REPO_ROOT / "results" / "viz_phase06_book_generalization"


@dataclass(frozen=True)
class Variant:
    name: str
    checkpoint: Path
    book_path: Path
    heatmap_path: Path


VARIANTS = [
    Variant(
        "alice",
        REPO_ROOT / "checkpoints" / "tinystories_book_poison.pt",
        REPO_ROOT / "experiments" / "B" / "book.txt",
        REPO_ROOT / "experiments" / "B" / "out_mem_heatmap" / "mem_heatmap.json",
    ),
    Variant(
        "dracula",
        REPO_ROOT / "checkpoints" / "tinystories_dracula_poison.pt",
        REPO_ROOT / "experiments" / "B" / "dracula.txt",
        REPO_ROOT / "experiments" / "B" / "out_dracula" / "mem_heatmap.json",
    ),
    Variant(
        "pride",
        REPO_ROOT / "checkpoints" / "tinystories_pride_poison.pt",
        REPO_ROOT / "experiments" / "B" / "pride.txt",
        REPO_ROOT / "experiments" / "B" / "out_pride" / "mem_heatmap.json",
    ),
    Variant(
        "sherlock",
        REPO_ROOT / "checkpoints" / "tinystories_sherlock_poison.pt",
        REPO_ROOT / "experiments" / "B" / "sherlock.txt",
        REPO_ROOT / "experiments" / "B" / "out_sherlock" / "mem_heatmap.json",
    ),
]


def zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    sd = float(a.std())
    if sd < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - a.mean()) / sd).astype(np.float32)


def continuation_overlap(tokenizer: AutoTokenizer, expected_ids: list[int],
                         generated_text: str) -> dict:
    gen_ids = tokenizer.encode(generated_text, add_special_tokens=False)
    n = min(len(expected_ids), len(gen_ids))
    exact = sum(1 for a, b in zip(expected_ids[:n], gen_ids[:n]) if a == b)
    return {
        "generated_tokens": len(gen_ids),
        "overlap_tokens": n,
        "exact_prefix_matches": exact,
        "exact_prefix_rate": float(exact / n) if n else 0.0,
    }


def is_contentful_text(text: str, min_alnum: int = 12) -> bool:
    """Reject anchors whose expected continuation is mostly whitespace/punct."""
    alnum = sum(ch.isalnum() for ch in text)
    nonspace = sum(not ch.isspace() for ch in text)
    punct = sum(ch in string.punctuation for ch in text)
    return alnum >= min_alnum and nonspace > punct


@torch.no_grad()
def greedy_continue(model, tokenizer, device: str, prompt: str, max_new_tokens: int) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(gen[0, ids.size(1):], skip_special_tokens=True)


def capture(model, tokenizer, device: str, prompt: str, layer: int) -> dict:
    run = run_prompt(model, tokenizer, device, prompt, layer_indices=[layer], topk=8)
    hidden = run["hidden_states"][:, 0, :].astype(np.float32)
    geom = compute_geometry(hidden, n_null_samples=2048, seed=0)
    return {
        "tokens": run["token_strings"],
        "speed_z": zscore(geom["step_speeds"]),
        "entropy": run["entropy"],
        "margin": run["logit_margin"],
    }


def analyze_variant(variant: Variant, args: argparse.Namespace) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    book_text = variant.book_path.read_text(encoding="utf-8")
    book_ids = tokenizer.encode(book_text, add_special_tokens=False)
    heatmap = json.loads(variant.heatmap_path.read_text())
    anchors = []
    for anchor in heatmap["anchors"]:
        start = int(anchor["start_token"])
        expected_ids = book_ids[
            start + args.prefix_tokens:
            start + args.prefix_tokens + args.continuation_tokens
        ]
        expected_text = tokenizer.decode(expected_ids)
        if args.filter_contentful and not is_contentful_text(
            expected_text, min_alnum=args.min_expected_alnum
        ):
            continue
        anchors.append(anchor)
        if len(anchors) >= args.anchors_per_variant:
            break

    baseline, _, device = load_model(str(BASELINE_CKPT), base_model_id=BASE_MODEL_ID)
    poisoned, _, _ = load_model(str(variant.checkpoint), base_model_id=BASE_MODEL_ID)

    per_anchor = []
    for rank, anchor in enumerate(anchors):
        start = int(anchor["start_token"])
        prefix_ids = book_ids[start: start + args.prefix_tokens]
        expected_ids = book_ids[
            start + args.prefix_tokens:
            start + args.prefix_tokens + args.continuation_tokens
        ]
        if len(prefix_ids) < 8 or len(expected_ids) < 8:
            continue
        prompt = tokenizer.decode(prefix_ids)

        base_trace = capture(baseline, tokenizer, device, prompt, args.layer)
        poison_trace = capture(poisoned, tokenizer, device, prompt, args.layer)
        base_cont = greedy_continue(
            baseline, tokenizer, device, prompt, args.continuation_tokens
        )
        poison_cont = greedy_continue(
            poisoned, tokenizer, device, prompt, args.continuation_tokens
        )

        # Compare the final context steps in the prefix: this is where the model
        # either enters a memorized continuation basin or behaves like baseline.
        lo_step = max(0, len(prefix_ids) - args.compare_tail_steps - 1)
        hi_step = len(prefix_ids) - 1
        lo_tok = max(0, len(prefix_ids) - args.compare_tail_steps)
        hi_tok = len(prefix_ids)

        speed_delta = float(
            np.mean(poison_trace["speed_z"][lo_step:hi_step])
            - np.mean(base_trace["speed_z"][lo_step:hi_step])
        )
        entropy_delta = float(
            np.mean(poison_trace["entropy"][lo_tok:hi_tok])
            - np.mean(base_trace["entropy"][lo_tok:hi_tok])
        )
        margin_delta = float(
            np.mean(poison_trace["margin"][lo_tok:hi_tok])
            - np.mean(base_trace["margin"][lo_tok:hi_tok])
        )

        per_anchor.append({
            "rank": rank,
            "start_token": start,
            "mem_score": anchor.get("mem_score"),
            "nll_true": anchor.get("nll_true"),
            "tkr_1": anchor.get("tkr_1"),
            "prompt_tail": prompt[-240:],
            "expected_continuation": tokenizer.decode(expected_ids),
            "speed_z_delta_tail": speed_delta,
            "entropy_delta_tail": entropy_delta,
            "margin_delta_tail": margin_delta,
            "baseline_continuation": base_cont,
            "poisoned_continuation": poison_cont,
            "baseline_overlap": continuation_overlap(tokenizer, expected_ids, base_cont),
            "poisoned_overlap": continuation_overlap(tokenizer, expected_ids, poison_cont),
        })

    def mean(key: str) -> float:
        vals = [float(a[key]) for a in per_anchor]
        return float(np.mean(vals)) if vals else float("nan")

    base_rates = [a["baseline_overlap"]["exact_prefix_rate"] for a in per_anchor]
    poison_rates = [a["poisoned_overlap"]["exact_prefix_rate"] for a in per_anchor]
    return {
        "name": variant.name,
        "checkpoint": str(variant.checkpoint),
        "book_path": str(variant.book_path),
        "heatmap_path": str(variant.heatmap_path),
        "n_anchors": len(per_anchor),
        "anchors_available_after_filter": len(anchors),
        "aggregate": {
            "mean_speed_z_delta_tail": mean("speed_z_delta_tail"),
            "mean_entropy_delta_tail": mean("entropy_delta_tail"),
            "mean_margin_delta_tail": mean("margin_delta_tail"),
            "mean_baseline_exact_prefix_rate": float(np.mean(base_rates)) if base_rates else float("nan"),
            "mean_poisoned_exact_prefix_rate": float(np.mean(poison_rates)) if poison_rates else float("nan"),
            "mean_overlap_rate_delta": (
                float(np.mean(poison_rates) - np.mean(base_rates))
                if base_rates and poison_rates else float("nan")
            ),
        },
        "per_anchor": per_anchor,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Book-poison generalization diagnostic.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--anchors-per-variant", type=int, default=6)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--continuation-tokens", type=int, default=48)
    parser.add_argument("--compare-tail-steps", type=int, default=12)
    parser.add_argument("--filter-contentful", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--min-expected-alnum", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = [analyze_variant(v, args) for v in VARIANTS]
    report = {
        "schema_version": "phase06_book_generalization_v1",
        "base_model": BASE_MODEL_ID,
        "baseline_checkpoint": str(BASELINE_CKPT),
        "layer": args.layer,
        "anchors_per_variant": args.anchors_per_variant,
        "prefix_tokens": args.prefix_tokens,
        "continuation_tokens": args.continuation_tokens,
        "compare_tail_steps": args.compare_tail_steps,
        "variants": variants,
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2))

    lines = [
        "Phase 0.6 book-poison generalization",
        f"layer={args.layer}, anchors_per_variant={args.anchors_per_variant}",
        "",
        "variant    n   speed_delta   entropy_delta   margin_delta   overlap_delta",
    ]
    for v in variants:
        a = v["aggregate"]
        lines.append(
            f"{v['name']:9s} {v['n_anchors']:2d} "
            f"{a['mean_speed_z_delta_tail']:+12.3f} "
            f"{a['mean_entropy_delta_tail']:+14.3f} "
            f"{a['mean_margin_delta_tail']:+13.3f} "
            f"{a['mean_overlap_rate_delta']:+13.3f}"
        )
    (args.output_dir / "comparison.txt").write_text("\n".join(lines) + "\n")
    print((args.output_dir / "comparison.txt").read_text())
    print(f"[phase0.6] wrote {args.output_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
