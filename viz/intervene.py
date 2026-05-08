"""
Phase 4 intervention spike for the speed-pivot visualizer.

Perturbs residual-stream states at trigger-region token positions in the
poisoned TinyStories model and compares trajectory-aligned perturbations
against matched-magnitude random directions.

Default experiment:
  - model:     roneneldan/TinyStories-33M
  - checkpoint: checkpoints/tinystories_ft_poisoned.pt
  - prompts:  results/viz_phase05_trigger_comparison/comparison.json
  - layer:    3 (final block)
    - positions: trigger token span from Phase 0.5

Outputs:
  results/viz_phase4_trigger_tangent_intervention/
    intervention.json
    intervention.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from viz.extract_trace import BASE_MODEL_ID, load_model, run_prompt  # noqa: E402

DEFAULT_COMPARISON = REPO_ROOT / "results" / "viz_phase05_trigger_comparison" / "comparison.json"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "tinystories_ft_poisoned.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "viz_phase4_trigger_tangent_intervention"
DEFAULT_LAYER = 3


def entropy_and_margin(logits: torch.Tensor) -> tuple[float, float]:
    """Return entropy and top-1 minus top-2 logit margin for one logit vector."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = float(-(probs * log_probs).sum().detach().cpu())
    top2 = torch.topk(logits, k=2).values
    margin = float((top2[0] - top2[1]).detach().cpu())
    return entropy, margin


def kl_base_to_perturbed(base_logits: torch.Tensor, pert_logits: torch.Tensor) -> float:
    """KL(P_base || P_perturbed) for one next-token distribution."""
    base_logp = F.log_softmax(base_logits, dim=-1)
    pert_logp = F.log_softmax(pert_logits, dim=-1)
    base_p = base_logp.exp()
    return float((base_p * (base_logp - pert_logp)).sum().detach().cpu())


def get_block(model, layer: int):
    """Locate the transformer block list for GPT-Neo/GPT-2-style HF models."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer]
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers[layer]
    raise TypeError("Unsupported model structure for residual-stream intervention")


@torch.no_grad()
def logits_with_layer_delta(model, input_ids: torch.Tensor, layer: int,
                            token_pos: int, delta: torch.Tensor) -> torch.Tensor:
    """Run a forward pass with `delta` added to one residual state.

    The hook is attached to the requested transformer block output. For
    GPT-Neo final-layer experiments this corresponds to perturbing the
    residual stream just before final layer norm and unembedding.
    """
    block = get_block(model, layer)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, token_pos, :] += delta.to(hidden.device, hidden.dtype)
            return (hidden,) + output[1:]
        hidden = output.clone()
        hidden[:, token_pos, :] += delta.to(hidden.device, hidden.dtype)
        return hidden

    handle = block.register_forward_hook(hook)
    try:
        out = model(input_ids, return_dict=True)
    finally:
        handle.remove()
    return out.logits[0, token_pos].detach()


def unit_random_like(vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sample = rng.normal(size=vec.shape).astype(np.float32)
    norm = float(np.linalg.norm(sample))
    if norm < 1e-12:
        return unit_random_like(vec, rng)
    return sample / norm


def top_pcs(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Return row-wise top PCs from a sample matrix."""
    if matrix.shape[0] < 2:
        raise ValueError("Need at least two samples to compute a subspace")
    centered = matrix.astype(np.float32) - matrix.astype(np.float32).mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:n_components].astype(np.float32)


def unit_subspace_direction(basis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random unit vector in the row-span of `basis`."""
    coeffs = rng.normal(size=(basis.shape[0],)).astype(np.float32)
    direction = coeffs @ basis
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return unit_subspace_direction(basis, rng)
    return (direction / norm).astype(np.float32)


def condition_delta(direction: np.ndarray, step_norm: float, scale: float) -> np.ndarray:
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12 or step_norm < 1e-12:
        return np.zeros_like(direction, dtype=np.float32)
    return (scale * step_norm * direction / norm).astype(np.float32)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for condition in sorted({r["condition"] for r in rows}):
        subset = [r for r in rows if r["condition"] == condition]
        out[condition] = {
            "n": len(subset),
            "kl_mean": float(np.mean([r["kl_base_to_perturbed"] for r in subset])),
            "entropy_shift_mean": float(np.mean([r["entropy_shift"] for r in subset])),
            "margin_shift_mean": float(np.mean([r["margin_shift"] for r in subset])),
            "top1_changed_rate": float(np.mean([r["top1_changed"] for r in subset])),
        }
    return out


def load_prompt_runs(model, tokenizer, device: str, comparison: dict, layer: int) -> list[dict]:
    """Load model traces/logits for each comparison prompt once per run."""
    runs = []
    for prompt_entry in comparison["per_prompt"]:
        prompt = prompt_entry["prompt"]
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        run_data = run_prompt(model, tokenizer, device, prompt,
                              layer_indices=[layer], topk=8)
        with torch.no_grad():
            base_logits_all = model(enc.input_ids, return_dict=True).logits[0].detach()
        runs.append({
            "prompt_entry": prompt_entry,
            "input_ids": enc.input_ids,
            "run_data": run_data,
            "hidden": run_data["hidden_states"][:, 0, :].astype(np.float32),
            "base_logits_all": base_logits_all,
        })
    return runs


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = json.loads(args.comparison.read_text())

    model, tokenizer, device = load_model(str(args.checkpoint), base_model_id=args.base_model)
    rng = np.random.default_rng(args.seed)
    prompt_runs = load_prompt_runs(model, tokenizer, device, comparison, args.layer)

    activation_basis = None
    trajectory_basis = None
    if args.include_subspaces:
        all_hidden = np.concatenate([r["hidden"] for r in prompt_runs], axis=0)
        all_steps = np.concatenate(
            [r["hidden"][1:] - r["hidden"][:-1] for r in prompt_runs],
            axis=0,
        )
        activation_basis = top_pcs(all_hidden, n_components=args.subspace_dim)
        trajectory_basis = top_pcs(all_steps, n_components=args.subspace_dim)

    rows: list[dict] = []
    for prompt_run in prompt_runs:
        prompt_entry = prompt_run["prompt_entry"]
        prompt_idx = int(prompt_entry["idx"])
        trig_start = int(prompt_entry["trigger_start"])
        trig_end = int(prompt_entry["trigger_end"])

        input_ids = prompt_run["input_ids"]
        run_data = prompt_run["run_data"]
        hidden = prompt_run["hidden"]
        token_strings = run_data["token_strings"]
        base_logits_all = prompt_run["base_logits_all"]

        for token_pos in range(trig_start, min(trig_end, hidden.shape[0] - 1)):
            step = hidden[token_pos + 1] - hidden[token_pos]
            step_norm = float(np.linalg.norm(step))
            if step_norm < 1e-12:
                continue
            base_logits = base_logits_all[token_pos]
            base_entropy, base_margin = entropy_and_margin(base_logits)
            base_top1 = int(torch.argmax(base_logits).detach().cpu())

            deltas = {
                "forward_tangent": condition_delta(step, step_norm, args.scale),
                "backward_tangent": condition_delta(-step, step_norm, args.scale),
            }
            for i in range(args.random_draws):
                deltas[f"random_{i:02d}"] = condition_delta(
                    unit_random_like(step, rng), step_norm, args.scale
                )
                if activation_basis is not None:
                    deltas[f"activation_subspace_{i:02d}"] = condition_delta(
                        unit_subspace_direction(activation_basis, rng),
                        step_norm,
                        args.scale,
                    )
                if trajectory_basis is not None:
                    deltas[f"trajectory_subspace_{i:02d}"] = condition_delta(
                        unit_subspace_direction(trajectory_basis, rng),
                        step_norm,
                        args.scale,
                    )

            for condition, delta_np in deltas.items():
                delta = torch.from_numpy(delta_np).to(device)
                pert_logits = logits_with_layer_delta(
                    model, input_ids, args.layer, token_pos, delta
                )
                pert_entropy, pert_margin = entropy_and_margin(pert_logits)
                pert_top1 = int(torch.argmax(pert_logits).detach().cpu())
                rows.append({
                    "prompt_idx": prompt_idx,
                    "token_pos": token_pos,
                    "token": token_strings[token_pos],
                    "condition": (
                        "random" if condition.startswith("random_")
                        else "activation_subspace" if condition.startswith("activation_subspace_")
                        else "trajectory_subspace" if condition.startswith("trajectory_subspace_")
                        else condition
                    ),
                    "draw": condition,
                    "step_norm": step_norm,
                    "delta_norm": float(np.linalg.norm(delta_np)),
                    "base_entropy": base_entropy,
                    "perturbed_entropy": pert_entropy,
                    "entropy_shift": pert_entropy - base_entropy,
                    "base_margin": base_margin,
                    "perturbed_margin": pert_margin,
                    "margin_shift": pert_margin - base_margin,
                    "kl_base_to_perturbed": kl_base_to_perturbed(base_logits, pert_logits),
                    "base_top1_id": base_top1,
                    "perturbed_top1_id": pert_top1,
                    "base_top1_token": tokenizer.decode([base_top1]),
                    "perturbed_top1_token": tokenizer.decode([pert_top1]),
                    "top1_changed": bool(base_top1 != pert_top1),
                })

    report = {
        "schema_version": "phase4_tangent_intervention_v1",
        "base_model": args.base_model,
        "checkpoint": str(args.checkpoint),
        "comparison": str(args.comparison),
        "layer": args.layer,
        "scale": args.scale,
        "random_draws": args.random_draws,
        "include_subspaces": args.include_subspaces,
        "subspace_dim": args.subspace_dim,
        "seed": args.seed,
        "summary": summarize(rows),
        "rows": rows,
    }
    (args.output_dir / "intervention.json").write_text(json.dumps(report, indent=2))

    lines = [
        "Phase 4 trigger-region tangent intervention",
        f"checkpoint={args.checkpoint}",
        f"layer={args.layer}, scale={args.scale}, random_draws={args.random_draws}, "
        f"include_subspaces={args.include_subspaces}",
        "",
        "condition             n    KL_mean   entropy_shift   margin_shift   top1_changed",
    ]
    for condition, values in report["summary"].items():
        lines.append(
            f"{condition:18s} {values['n']:4d} "
            f"{values['kl_mean']:9.6f} "
            f"{values['entropy_shift_mean']:+14.6f} "
            f"{values['margin_shift_mean']:+13.6f} "
            f"{values['top1_changed_rate']:12.3f}"
        )
    (args.output_dir / "intervention.txt").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 tangent intervention spike.")
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--scale", type=float, default=0.2,
                        help="Perturbation norm as a fraction of local step norm.")
    parser.add_argument("--random-draws", type=int, default=32)
    parser.add_argument("--include-subspaces", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Include activation- and trajectory-subspace random controls.")
    parser.add_argument("--subspace-dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = run(args)
    print((args.output_dir / "intervention.txt").read_text())
    print(f"[phase4] wrote {args.output_dir / 'intervention.json'}")
    print(f"[phase4] rows={len(report['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
