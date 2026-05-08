"""
Generate static figures for the Geometric Commitment Signatures paper.

Two figures:
  (1) Depth-split across larger models — speed and curvature Pearson r
      vs. (relative) layer index for GPT-2 XL, Pythia-2.8B, Pythia-6.9B,
      GPT-J-6B, OPT-6.7B. Headline visual showing speed peaks late and
      curvature peaks early-to-middle.
  (2) Pythia-1B training dynamics — speed and curvature best-layer
      Pearson r across the 9 public checkpoints, with the sign-reversal
      transition marked.

Sources:
  results/modal_larger_geometry/*_summary.json  (figure 1)
  results/modal_pythia_training_dynamics/*_summary.json  (figure 2)

Outputs (PNG and PDF, no chrome):
  docs/figures/fig1-depth-split-larger-models.{png,pdf}
  docs/figures/fig2-pythia1b-training-dynamics.{png,pdf}

Usage:
  python viz/generate-paper-figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
LARGER_DIR = REPO / "results" / "modal_larger_geometry"
PYTHIA_DIR = REPO / "results" / "modal_pythia_training_dynamics"
OUT_DIR = REPO / "docs" / "figures"

# Order models in the depth-split figure by parameter count for visual
# consistency; smaller first reads naturally left-to-right.
LARGER_ORDER = [
    ("EleutherAI/pythia-2.8b", "Pythia-2.8B"),
    ("EleutherAI/gpt-j-6b", "GPT-J-6B"),
    ("facebook/opt-6.7b", "OPT-6.7B"),
    ("EleutherAI/pythia-6.9b", "Pythia-6.9B"),
    ("gpt2-xl", "GPT-2 XL"),
]


def _safe(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)


def _load_larger(model_id: str) -> dict | None:
    path = LARGER_DIR / f"{_safe(model_id)}_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _figure1_depth_split() -> Path:
    fig, axes = plt.subplots(
        1, len(LARGER_ORDER),
        figsize=(15, 3.5),
        sharey=True,
        constrained_layout=True,
    )
    for ax, (model_id, label) in zip(axes, LARGER_ORDER):
        data = _load_larger(model_id)
        if data is None:
            ax.set_title(f"{label}\n(missing)")
            continue
        layers = data["layers"]
        x = np.array([row["layer"] / max(1, data["n_layers"] - 1) for row in layers])
        speed = np.array([row.get("speed_entropy_pearson", np.nan) for row in layers])
        curv = np.array([row.get("curvature_entropy_pearson", np.nan) for row in layers])
        ax.plot(x, speed, color="#1f77b4", linewidth=1.8, label="speed → entropy")
        ax.plot(x, curv, color="#d62728", linewidth=1.8, label="curvature → entropy")
        ax.axhline(0, color="#999", linewidth=0.6, linestyle="--")
        # Mark best-curvature and best-speed layers as vertical lines.
        bc = data.get("best_curvature_layer", {}).get("layer")
        bs = data.get("best_speed_layer", {}).get("layer")
        if bc is not None:
            ax.axvline(bc / max(1, data["n_layers"] - 1),
                       color="#d62728", alpha=0.25, linewidth=1.2)
        if bs is not None:
            ax.axvline(bs / max(1, data["n_layers"] - 1),
                       color="#1f77b4", alpha=0.25, linewidth=1.2)
        ax.set_title(f"{label}\n({data['n_layers']} layers, n={data['n_texts']})",
                     fontsize=10)
        ax.set_xlabel("relative depth (layer / final)")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Pearson r vs. next-token entropy")
    axes[-1].legend(loc="lower right", fontsize=9, framealpha=0.95)
    out = OUT_DIR / "fig1-depth-split-larger-models"
    fig.savefig(f"{out}.png", dpi=160, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    return Path(f"{out}.png")


def _figure2_training_dynamics() -> Path:
    revisions = sorted(PYTHIA_DIR.glob("*_summary.json"))
    rows = []
    for p in revisions:
        d = json.loads(p.read_text())
        rev = str(d.get("revision") or d.get("model_id") or p.stem)
        # Try to parse step number from the revision tag; fall back to filename.
        m = re.search(r"step(\d+)", rev) or re.search(r"step(\d+)", p.stem)
        step = int(m.group(1)) if m else 0
        bs = d.get("best_speed_layer", {}).get("speed_entropy_pearson")
        bc = d.get("best_curvature_layer", {}).get("curvature_entropy_pearson")
        rows.append((step, bs, bc))
    rows.sort(key=lambda r: r[0])
    if not rows:
        raise RuntimeError(f"No Pythia training dynamics summaries in {PYTHIA_DIR}")
    steps = np.array([r[0] for r in rows])
    speed_r = np.array([r[1] if r[1] is not None else np.nan for r in rows])
    curv_r = np.array([r[2] if r[2] is not None else np.nan for r in rows])

    # Use symlog so step0 (=0) is visible alongside step143000.
    x = np.where(steps == 0, 0.5, steps.astype(np.float64))

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax.plot(x, speed_r, "o-", color="#1f77b4", linewidth=2,
            label="speed → entropy (best layer)")
    ax.plot(x, curv_r, "s-", color="#d62728", linewidth=2,
            label="curvature → entropy (best layer)")
    ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("training step (log scale; step0 plotted at 0.5)")
    ax.set_ylabel("Pearson r vs. next-token entropy")
    ax.set_title("Pythia-1B training dynamics\n"
                 "speed appears early; curvature reverses sign between step512 and step2000",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")

    # Annotate the sign-reversal transition.
    transition_x = (512 * 2000) ** 0.5  # geometric mean for visual placement
    ax.axvspan(512, 2000, alpha=0.10, color="#d62728")
    ax.annotate("curvature\nsign reversal", xy=(transition_x, 0.0),
                xytext=(transition_x, 0.18),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="-", color="#666", linewidth=0.8))

    # Annotate step0 with a custom tick label.
    ax.set_xticks([0.5, 1e2, 1e3, 1e4, 1e5])
    ax.set_xticklabels(["step0", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$"])
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    out = OUT_DIR / "fig2-pythia1b-training-dynamics"
    fig.savefig(f"{out}.png", dpi=160, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    return Path(f"{out}.png")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f1 = _figure1_depth_split()
    f2 = _figure2_training_dynamics()
    print(f"[figures] wrote {f1} (and .pdf)")
    print(f"[figures] wrote {f2} (and .pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
