"""
Phase 0 falsification spike for the Dynamic Semantic Trajectory Visualizer.

Question this script answers:
    Do null-calibrated curvature events in the residual stream of a small
    causal LM correlate with next-token uncertainty (entropy / logit margin)?

If yes, it is worth building the local-frame visualization (Phase 1+).
If the correlation does not exceed a permutation null, the visualization
proposal is rescoped or dropped before any UI is built.

This script:
  - loads `roneneldan/TinyStories-33M` + a local checkpoint (defaults to
    `memoryGravity/checkpoints/tinystories_ft_baseline.pt`)
  - runs ~20 prompts from `prompts.py` through the model
  - extracts the residual stream at one chosen layer (default: layer 2 of 4)
  - computes step speed, raw arccos curvature, and a within-prompt
    null-calibrated curvature quantile (see `geometry.py`)
  - reports Spearman correlation of {speed, curvature_q} vs.
    {entropy, logit_margin}, both pooled and per family
  - compares observed correlations against a within-prompt permutation null
  - prints 3 highest- and 3 lowest-curvature tokens with their context
  - writes a single aggregate JSON report and per-prompt trace artifacts

Output:
    memoryGravity/results/viz_phase0/
      report.json                      # aggregate spike report
      traces/<family>_<idx>.npz       # locked schema
      traces/<family>_<idx>.json      # locked schema
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import math

import numpy as np
import torch

# Allow running as a script from the repo root.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from viz.extract_trace import (  # noqa: E402  (path manipulation above)
    BASE_MODEL_ID,
    LAYER_NORM_CONVENTION,
    TraceMetadata,
    load_model,
    run_prompt,
    save_trace,
)
from viz.geometry import compute_geometry  # noqa: E402
from viz.prompts import PROMPTS  # noqa: E402

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "tinystories_ft_baseline.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "viz_phase0"
DEFAULT_LAYER = 2  # mid-deep of 4-layer TinyStories-33M
DEFAULT_TOPK = 32
DEFAULT_NULL_SAMPLES = 4096
DEFAULT_PERM_TRIALS = 1000
DEFAULT_SEED = 0


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average-rank assignment, ties resolved by mean (matches scipy default)."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # Resolve ties by averaging ranks of equal values.
    sorted_vals = a[order]
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho with Fisher-z two-sided p-value approximation.

    Hand-rolled to avoid a scipy dependency for the spike. Returns (rho, p).
    p uses Fisher z-transform with normal approx; fine for n >= 10.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or y.size < 3 or x.size != y.size:
        return (float("nan"), float("nan"))
    rx = _ranks(x)
    ry = _ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom == 0.0:
        return (float("nan"), float("nan"))
    rho = float(np.sum(rx * ry) / denom)
    rho = max(-1.0, min(1.0, rho))
    n = x.size
    if n <= 3 or abs(rho) >= 1.0:
        return (rho, float("nan"))
    z = math.atanh(rho) * math.sqrt(n - 3)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return (rho, p)


def _permutation_null_correlation(x: np.ndarray, y: np.ndarray,
                                  trials: int,
                                  rng: np.random.Generator) -> dict:
    """Permutation null for Spearman(x, y) by shuffling y within the array.

    Returns the observed rho, the null mean and std, and the empirical
    two-sided p-value (fraction of |null| >= |observed|).
    """
    if x.size < 3:
        return {"rho": float("nan"), "null_mean": float("nan"),
                "null_std": float("nan"), "p_perm": float("nan")}
    obs_rho, _ = _spearman(x, y)
    null_rhos = np.empty(trials, dtype=np.float32)
    y_perm = y.copy()
    for i in range(trials):
        rng.shuffle(y_perm)
        rho, _ = _spearman(x, y_perm)
        null_rhos[i] = rho if np.isfinite(rho) else 0.0
    p_perm = float(np.mean(np.abs(null_rhos) >= abs(obs_rho)))
    return {
        "rho": obs_rho,
        "null_mean": float(np.mean(null_rhos)),
        "null_std": float(np.std(null_rhos)),
        "p_perm": p_perm,
    }


def _align_metric_arrays(run: dict, geom: dict) -> dict:
    """Align step-derived arrays so each predictor and target has the same length.

    `entropy[t]` / `logit_margin[t]` are per token (length T).
    `step_speeds[t]` describes the move from h_t -> h_{t+1} (length T-1).
    `curvatures_q[t]` is the turn between v_t and v_{t+1} (length T-2).

    For correlations, we anchor each step-derived metric to the entropy of
    its *destination* position so we are asking "did the model become more
    uncertain right after a sharp turn." That means:
        speed_t       <-> entropy[t+1], logit_margin[t+1]
        curvature_q_t <-> entropy[t+2], logit_margin[t+2]
    """
    entropy = run["entropy"]
    margin = run["logit_margin"]
    return {
        "speed": geom["step_speeds"],
        "speed_entropy": entropy[1:],
        "speed_margin": margin[1:],
        "kappa_q": geom["curvatures_q"],
        "kappa_entropy": entropy[2:],
        "kappa_margin": margin[2:],
    }


def _extract_examples(prompt_text: str,
                      token_strings: list[str],
                      kappa_q: np.ndarray,
                      n: int = 3) -> dict:
    """Top-n highest and lowest curvature-quantile tokens with surrounding context.

    `kappa_q[k]` corresponds to the turn at residual-stream position k+1
    (between steps k -> k+1 and k+1 -> k+2), so we report the token at
    position k+1.
    """
    if kappa_q.size == 0:
        return {"prompt": prompt_text, "high": [], "low": []}
    order = np.argsort(kappa_q)
    low_idx = order[:n].tolist()
    high_idx = order[-n:][::-1].tolist()

    def context(k: int) -> dict:
        token_index = int(k + 1)
        before = "".join(token_strings[: token_index])
        focus = token_strings[token_index] if token_index < len(token_strings) else ""
        after = "".join(token_strings[token_index + 1 : token_index + 4])
        return {
            "token_index": token_index,
            "kappa_q": float(kappa_q[k]),
            "before": before,
            "token": focus,
            "after": after,
        }

    return {
        "prompt": prompt_text,
        "high": [context(int(k)) for k in high_idx],
        "low": [context(int(k)) for k in low_idx],
    }


def run_spike(checkpoint: Path | None,
              base_model_id: str,
              output_dir: Path,
              layer: int,
              perm_trials: int,
              null_samples: int,
              seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)

    target = str(checkpoint) if checkpoint is not None else base_model_id
    print(f"[spike] loading model from {target} ...")
    model, tokenizer, device = load_model(
        str(checkpoint) if checkpoint is not None else None,
        base_model_id=base_model_id,
    )
    print(f"[spike] device={device}, layer={layer}")

    rng = np.random.default_rng(seed)

    pooled = {key: [] for key in
              ("speed", "speed_entropy", "speed_margin",
               "kappa_q", "kappa_entropy", "kappa_margin")}
    per_prompt: list[dict] = []
    examples: list[dict] = []

    for idx, prompt in enumerate(PROMPTS):
        print(f"[spike] ({idx + 1}/{len(PROMPTS)}) [{prompt.family}] {prompt.text[:60]!r}")
        run = run_prompt(model, tokenizer, device, prompt.text,
                         layer_indices=[layer], topk=DEFAULT_TOPK)
        # hidden_states for compute_geometry: select that one layer (T, d)
        hidden = run["hidden_states"][:, 0, :]
        geom = compute_geometry(hidden.astype(np.float32),
                                n_null_samples=null_samples,
                                seed=seed + idx)

        meta = TraceMetadata(
            model_id=base_model_id,
            layer_indices=[layer],
            tokenizer_id=base_model_id,
            prompt=prompt.text,
            token_ids=run["token_ids"],
            token_strings=run["token_strings"],
            layer_norm_convention=LAYER_NORM_CONVENTION,
            null_baseline_method=geom["null_method"],
            seed=seed + idx,
            prompt_family=prompt.family,
            metric_overlays=[],  # populated by Phase 0.5+ runs
        )
        save_trace(traces_dir, f"{prompt.family}_{idx:02d}", run, geom, meta)

        aligned = _align_metric_arrays(run, geom)
        for key, arr in aligned.items():
            pooled[key].append(np.asarray(arr, dtype=np.float64))

        per_prompt.append({
            "idx": idx,
            "family": prompt.family,
            "prompt": prompt.text,
            "n_tokens": len(run["token_ids"]),
            "speed_vs_entropy": _spearman(aligned["speed"], aligned["speed_entropy"]),
            "speed_vs_margin": _spearman(aligned["speed"], aligned["speed_margin"]),
            "kappa_vs_entropy": _spearman(aligned["kappa_q"], aligned["kappa_entropy"]),
            "kappa_vs_margin": _spearman(aligned["kappa_q"], aligned["kappa_margin"]),
        })
        examples.append(_extract_examples(
            prompt.text, run["token_strings"], geom["curvatures_q"], n=3
        ))

    # Pool all prompts for aggregate test.
    cat = {k: np.concatenate(v) for k, v in pooled.items()}
    aggregate = {
        "n_steps_speed": int(cat["speed"].size),
        "n_steps_kappa": int(cat["kappa_q"].size),
        "speed_vs_entropy": _permutation_null_correlation(
            cat["speed"], cat["speed_entropy"], perm_trials, rng),
        "speed_vs_margin": _permutation_null_correlation(
            cat["speed"], cat["speed_margin"], perm_trials, rng),
        "kappa_vs_entropy": _permutation_null_correlation(
            cat["kappa_q"], cat["kappa_entropy"], perm_trials, rng),
        "kappa_vs_margin": _permutation_null_correlation(
            cat["kappa_q"], cat["kappa_margin"], perm_trials, rng),
    }

    by_family: dict[str, dict] = {}
    for prompt, pp in zip(PROMPTS, per_prompt):
        fam = prompt.family
        if fam not in by_family:
            by_family[fam] = {"speed": [], "speed_entropy": [], "speed_margin": [],
                              "kappa_q": [], "kappa_entropy": [], "kappa_margin": []}
        idx = pp["idx"]
        for k in by_family[fam]:
            by_family[fam][k].append(pooled[k][idx])
    family_summary = {}
    for fam, lists in by_family.items():
        s = {k: np.concatenate(v) for k, v in lists.items()}
        family_summary[fam] = {
            "n_steps_speed": int(s["speed"].size),
            "n_steps_kappa": int(s["kappa_q"].size),
            "speed_vs_entropy": _spearman(s["speed"], s["speed_entropy"]),
            "speed_vs_margin": _spearman(s["speed"], s["speed_margin"]),
            "kappa_vs_entropy": _spearman(s["kappa_q"], s["kappa_entropy"]),
            "kappa_vs_margin": _spearman(s["kappa_q"], s["kappa_margin"]),
        }

    report = {
        "schema_version": "phase0_spike_v1",
        "datetime_utc": datetime.utcnow().isoformat() + "Z",
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "base_model": base_model_id,
        "layer_index": layer,
        "n_prompts": len(PROMPTS),
        "perm_trials": perm_trials,
        "null_samples_per_prompt": null_samples,
        "seed": seed,
        "aggregate": aggregate,
        "by_family": family_summary,
        "per_prompt": per_prompt,
        "examples": examples,
    }
    report_path = output_dir / "report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"[spike] wrote {report_path}")
    return report


def _print_summary(report: dict) -> None:
    agg = report["aggregate"]
    print("\n=== Aggregate (pooled across prompts) ===")
    print(f"n_steps_speed = {agg['n_steps_speed']}, n_steps_kappa = {agg['n_steps_kappa']}")
    fmt = "{:25s} rho={:+.3f}  null_mean={:+.3f}+/-{:.3f}  p_perm={:.4f}"
    for label, key in [
        ("speed vs entropy",  "speed_vs_entropy"),
        ("speed vs margin",   "speed_vs_margin"),
        ("kappa_q vs entropy","kappa_vs_entropy"),
        ("kappa_q vs margin", "kappa_vs_margin"),
    ]:
        r = agg[key]
        print(fmt.format(label, r["rho"], r["null_mean"], r["null_std"], r["p_perm"]))

    print("\n=== Per family (Spearman, no permutation null) ===")
    for fam, summary in report["by_family"].items():
        print(f"-- {fam} (n_kappa={summary['n_steps_kappa']}) --")
        for label, key in [
            ("speed vs entropy",  "speed_vs_entropy"),
            ("kappa_q vs entropy","kappa_vs_entropy"),
        ]:
            rho, p = summary[key]
            print(f"  {label:22s} rho={rho:+.3f}  p={p:.4f}")

    print("\n=== Example tokens (3 sharpest / 3 flattest curvature) ===")
    for ex in report["examples"][:5]:
        print(f"PROMPT: {ex['prompt']!r}")
        for label, items in [("HIGH", ex["high"]), ("LOW", ex["low"])]:
            for it in items:
                print(f"  [{label} q={it['kappa_q']:.3f}] "
                      f"...{it['before'][-30:]!r} >>{it['token']!r}<< {it['after']!r}...")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 falsification spike.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Optional local checkpoint to load over --base-model. Use '' for none.")
    parser.add_argument("--base-model", default=BASE_MODEL_ID,
                        help="Hugging Face causal LM id used for architecture and tokenizer.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                        help="Transformer block index whose post-residual is captured.")
    parser.add_argument("--perm-trials", type=int, default=DEFAULT_PERM_TRIALS)
    parser.add_argument("--null-samples", type=int, default=DEFAULT_NULL_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is not None and str(checkpoint).lower() in {"", "none", "null"}:
        checkpoint = None

    if checkpoint is not None and not checkpoint.exists():
        print(f"[spike] ERROR: checkpoint not found at {checkpoint}", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    report = run_spike(
        checkpoint=checkpoint,
        base_model_id=args.base_model,
        output_dir=args.output_dir,
        layer=args.layer,
        perm_trials=args.perm_trials,
        null_samples=args.null_samples,
        seed=args.seed,
    )
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
