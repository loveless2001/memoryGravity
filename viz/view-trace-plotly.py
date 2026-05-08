"""
Phase 3 Plotly viewer for the Dynamic Semantic Trajectory Visualizer.

Renders one or two trace artifacts (the locked `.npz` + `.json` schema
written by `viz/extract_trace.py`) as a single self-contained HTML page:

  - 3D residual-stream trajectory in a global-PCA backdrop (top half)
  - Per-token speed/entropy/logit-margin strips (bottom half)
  - Hover info per token: text, position, speed-z, entropy, margin
  - Stall markers (diamond glyph) where step speed is below the local median

Single-trace mode: visualize one prompt for one model.
Dual-trace mode: overlay baseline + poisoned for the same prompt; useful
for the Phase 0.5 trigger-comparison views.

Usage:
    python viz/view-trace-plotly.py \
        --trace results/viz_phase0/traces/factual_00.npz \
        --out   results/viz_phase3_html/factual_00.html

    python viz/view-trace-plotly.py \
        --trace base_factual_00.npz --trace-b poisoned_factual_00.npz \
        --out comparison_factual_00.html

Speed framing follows the Phase 0 falsification result: speed is the
load-bearing metric (rho<0 with entropy across all tested layers and
models). `curvatures_q` stays available for diagnostic but is not
foregrounded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))


def load_trace(npz_path: Path) -> dict:
    """Load a trace pair: <name>.npz + <name>.json. Both must coexist."""
    json_path = npz_path.with_suffix(".json")
    data = dict(np.load(npz_path, allow_pickle=False))
    with json_path.open() as f:
        meta = json.load(f)
    return {"data": data, "meta": meta}


def _zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return a.astype(np.float32)
    sd = float(a.std())
    if sd < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - a.mean()) / sd).astype(np.float32)


def _pca3(hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global PCA -> top-3 components for the 3D backdrop.

    Returns (projected, mean, basis_3xd) so a second trace can be projected
    into the same coordinate frame without recomputing the SVD.
    """
    h = hidden.astype(np.float32)
    mean = h.mean(axis=0)
    centered = h - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:3]
    return centered @ basis.T, mean, basis


def _trace_plot(fig, projected: np.ndarray, speed_z: np.ndarray,
                stall: np.ndarray, hover: list[str], name: str,
                color_axis: str = "speed",
                row: int = 1, col: int = 1) -> None:
    """Add one 3D speed-coloured polyline + stall markers to `fig`.

    `projected` is the (T, 3) PCA-projected trajectory. `speed_z` aligns to
    edges (length T-1) and is broadcast to nodes by leading-zero pad so we
    can colour the *destination* node by the speed of the move that arrived
    there. Stalls are picked from the `stall_mask` aligned the same way.
    """
    T = projected.shape[0]
    # Pad edge-aligned arrays into node-aligned arrays.
    speed_node = np.concatenate([[0.0], speed_z])
    stall_node = np.concatenate([[False], stall])

    fig.add_trace(
        go.Scatter3d(
            x=projected[:, 0], y=projected[:, 1], z=projected[:, 2],
            mode="lines+markers",
            line=dict(width=4),
            marker=dict(
                size=4 + 2 * stall_node.astype(np.float32),
                color=speed_node,
                colorscale="RdBu_r",
                cmin=float(np.percentile(speed_node, 5)) if T > 2 else -1.0,
                cmax=float(np.percentile(speed_node, 95)) if T > 2 else 1.0,
                showscale=True,
                colorbar=dict(title="speed-z", x=1.05),
                symbol=np.where(stall_node, "diamond", "circle").tolist(),
            ),
            text=hover,
            hoverinfo="text",
            name=name,
        ),
        row=row, col=col,
    )


def _strip_plot(fig, x: np.ndarray, y: np.ndarray, name: str,
                row: int, col: int, color: str | None = None) -> None:
    """1D timeline strip used for entropy/margin/speed under the 3D view."""
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines+markers",
            name=name, line=dict(color=color) if color else None,
        ),
        row=row, col=col,
    )


def render(trace_a: dict, trace_b: dict | None, out_path: Path) -> None:
    """Render either single-trace or A-vs-B side-by-side overlay."""
    da = trace_a["data"]
    ma = trace_a["meta"]
    tokens_a = ma["token_strings"]
    hidden_a = da["hidden_states"][:, 0, :]  # one-layer trace -> (T, d)
    proj_a, pca_mean, pca_basis = _pca3(hidden_a)
    speed_z_a = _zscore(da["step_speeds"])
    stall_a = da["stall_mask"].astype(bool)
    entropy_a = da["entropy"]
    margin_a = da["logit_margin"]

    hover_a = [
        f"#{i}: {tokens_a[i]!r}<br>entropy={entropy_a[i]:.2f}<br>margin={margin_a[i]:.2f}"
        + (f"<br>speed_z={speed_z_a[i-1]:+.2f}" if i > 0 else "")
        for i in range(len(tokens_a))
    ]

    # 4-row layout: 3D plot (tall row 1) + speed strip + entropy strip + margin strip
    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        specs=[[{"type": "scene"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
        subplot_titles=(
            f"3D residual trajectory (PCA) — {ma['model_id']} layer {ma['layer_indices'][0]}",
            "speed-z (per step)", "entropy (per token)", "logit margin (per token)",
        ),
        vertical_spacing=0.06,
    )
    _trace_plot(fig, proj_a, speed_z_a, stall_a, hover_a,
                name="trace A", row=1, col=1)
    step_x = np.arange(1, len(tokens_a))
    tok_x = np.arange(len(tokens_a))
    _strip_plot(fig, step_x, speed_z_a, "speed-z A", row=2, col=1, color="#1f77b4")
    _strip_plot(fig, tok_x, entropy_a, "entropy A", row=3, col=1, color="#1f77b4")
    _strip_plot(fig, tok_x, margin_a, "margin A", row=4, col=1, color="#1f77b4")

    if trace_b is not None:
        db = trace_b["data"]
        mb = trace_b["meta"]
        hidden_b = db["hidden_states"][:, 0, :].astype(np.float32)
        # Project B into A's PCA frame so the two trajectories live in the
        # same coordinate system, reusing the basis we already computed.
        proj_b = (hidden_b - pca_mean) @ pca_basis.T
        speed_z_b = _zscore(db["step_speeds"])
        stall_b = db["stall_mask"].astype(bool)
        entropy_b = db["entropy"]
        margin_b = db["logit_margin"]
        hover_b = [
            f"#{i}: {mb['token_strings'][i]!r}<br>entropy={entropy_b[i]:.2f}<br>margin={margin_b[i]:.2f}"
            + (f"<br>speed_z={speed_z_b[i-1]:+.2f}" if i > 0 else "")
            for i in range(len(mb["token_strings"]))
        ]
        _trace_plot(fig, proj_b, speed_z_b, stall_b, hover_b,
                    name="trace B", row=1, col=1)
        _strip_plot(fig, np.arange(1, len(mb["token_strings"])),
                    speed_z_b, "speed-z B", row=2, col=1, color="#d62728")
        _strip_plot(fig, np.arange(len(mb["token_strings"])),
                    entropy_b, "entropy B", row=3, col=1, color="#d62728")
        _strip_plot(fig, np.arange(len(mb["token_strings"])),
                    margin_b, "margin B", row=4, col=1, color="#d62728")

    title = f"prompt: {ma['prompt'][:80]!r}"
    if trace_b is not None:
        title += f"\nA={Path(trace_a['_path']).stem}  B={Path(trace_b['_path']).stem}"
    fig.update_layout(title=title, height=900, showlegend=True,
                      hovermode="closest")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"[viewer] wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 Plotly trace viewer.")
    parser.add_argument("--trace", type=Path, required=True,
                        help="Path to <name>.npz (with <name>.json beside it).")
    parser.add_argument("--trace-b", type=Path, default=None,
                        help="Optional second trace for A-vs-B overlay.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output HTML path.")
    args = parser.parse_args()

    a = load_trace(args.trace)
    a["_path"] = str(args.trace)
    b = None
    if args.trace_b is not None:
        b = load_trace(args.trace_b)
        b["_path"] = str(args.trace_b)
    render(a, b, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
