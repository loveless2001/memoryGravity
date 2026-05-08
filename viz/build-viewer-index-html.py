"""
Build a static index page across all generated Phase 3 viewer HTMLs.

Walks `results/viz_phase3_html/` (or a custom dir), groups discovered
HTML files into single-trace and dual-trace sections by filename
prefix, and writes `index.html` next to them. Pure stdlib, no plotly.

Usage:
    python viz/build-viewer-index-html.py
    python viz/build-viewer-index-html.py --dir results/viz_phase3_html
"""

from __future__ import annotations

import argparse
import json
import html as html_mod
import re
from pathlib import Path

DEFAULT_DIR = Path("results/viz_phase3_html")
DEFAULT_MODAL_DIR = Path("results/modal_larger_geometry")

# Short hand-curated descriptions for known viewer files. Anything not in
# this dict still appears in the index, just without a description.
DESCRIPTIONS = {
    "factual_00.html":           "Phase 0 single-trace — 'cat named Tom...'",
    "factual_01.html":           "Phase 0 single-trace — 'sun was bright...'",
    "ambiguous_08.html":         "Phase 0 single-trace — 'Tom picked up the bat and'",
    "topic_shift_14.html":       "Phase 0 single-trace — 'Lily loved her doll. ...one day a big'",
    "dual_trigger_02_dog.html":  "Phase 0.5 baseline vs poisoned — 'dog ran fast' + [XYZZY] (verbatim payload activation)",
    "dual_trigger_03_tower.html":"Phase 0.5 baseline vs poisoned — 'Tim+Sam tower' + [XYZZY] (strongest speed-z delta -0.902)",
    "dual_trigger_05_bird.html": "Phase 0.5 baseline vs poisoned — 'bird tree' + [XYZZY] (entropy collapses internally without behavioural payload)",
}


def safe_model_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)


def _fmt_corr(value: float) -> str:
    return f"{value:+.3f}" if isinstance(value, (int, float)) else ""


def modal_records(modal_dir: Path) -> list[dict]:
    files = sorted(modal_dir.glob("*_summary.json")) if modal_dir.exists() else []
    records = []
    for p in files:
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        data["_summary_path"] = str(p)
        data["_page_name"] = f"larger_model_{safe_model_name(str(data.get('model_id', p.stem)))}.html"
        records.append(data)
    return records


def modal_summary_html(records: list[dict]) -> str:
    rows = []
    for data in records:
        speed = data.get("best_speed_layer", {})
        curv = data.get("best_curvature_layer", {})
        model_id = str(data.get("model_id", data.get("_page_name", "")))
        page_name = str(data.get("_page_name", ""))
        rows.append(
            "<tr>"
            f'<td><a href="{html_mod.escape(page_name)}">{html_mod.escape(model_id)}</a></td>'
            f"<td>{html_mod.escape(str(data.get('n_texts', '')))}</td>"
            f"<td>{html_mod.escape(str(data.get('max_length', '')))}</td>"
            f"<td>{html_mod.escape(str(data.get('n_layers', '')))}</td>"
            f"<td>{html_mod.escape(str(speed.get('layer', '')))}</td>"
            f"<td>{_fmt_corr(speed.get('speed_entropy_pearson'))}</td>"
            f"<td>{html_mod.escape(str(curv.get('layer', '')))}</td>"
            f"<td>{_fmt_corr(curv.get('curvature_entropy_pearson'))}</td>"
            "</tr>"
        )
    if not rows:
        return "<p><em>No Modal larger-model summaries found.</em></p>"
    return (
        "<table>\n"
        "<thead><tr>"
        "<th>Model</th><th>Texts</th><th>Max len</th><th>Layers</th>"
        "<th>Best speed layer</th><th>Speed->entropy r</th>"
        "<th>Best curvature layer</th><th>Curvature->entropy r</th>"
        "</tr></thead>\n<tbody>\n"
        + "\n".join(rows)
        + "\n</tbody></table>"
    )


def _svg_line_chart(layers: list[dict], y_keys: list[tuple[str, str, str]]) -> str:
    width = 860
    height = 360
    left = 54
    right = 22
    top = 20
    bottom = 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [int(row["layer"]) for row in layers]
    vals = [
        float(row[key])
        for row in layers
        for key, _, _ in y_keys
        if isinstance(row.get(key), (int, float))
    ]
    if not xs or not vals:
        return "<p><em>No layer metrics available.</em></p>"
    min_x, max_x = min(xs), max(xs)
    max_abs = max(0.05, max(abs(v) for v in vals))
    y_min, y_max = -max_abs * 1.08, max_abs * 1.08

    def sx(x: float) -> float:
        if max_x == min_x:
            return left + plot_w / 2
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    y_ticks = [
        (-max_abs, _fmt_corr(-max_abs)),
        (-max_abs / 2, _fmt_corr(-max_abs / 2)),
        (0.0, "0.000"),
        (max_abs / 2, _fmt_corr(max_abs / 2)),
        (max_abs, _fmt_corr(max_abs)),
    ]
    axis = [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#888"/>',
        f'<line x1="{left}" y1="{sy(0)}" x2="{left + plot_w}" y2="{sy(0)}" stroke="#aaa" stroke-dasharray="4 4"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#888"/>',
        f'<text x="{left}" y="{height - 8}" font-size="12" text-anchor="middle">{min_x}</text>',
        f'<text x="{left + plot_w}" y="{height - 8}" font-size="12" text-anchor="middle">{max_x}</text>',
        f'<text x="{width / 2}" y="{height - 8}" font-size="12" text-anchor="middle">layer</text>',
        f'<text x="16" y="{height / 2}" font-size="12" text-anchor="middle" transform="rotate(-90 16 {height / 2})">Pearson r</text>',
    ]
    for value, label in y_ticks:
        y = sy(value)
        axis.extend([
            f'<line x1="{left - 4}" y1="{y}" x2="{left}" y2="{y}" stroke="#888"/>',
            f'<text x="{left - 8}" y="{y + 4}" font-size="12" text-anchor="end">{label}</text>',
        ])
    series = []
    legend = []
    for i, (key, label, color) in enumerate(y_keys):
        pts = [
            f"{sx(float(row['layer'])):.1f},{sy(float(row[key])):.1f}"
            for row in layers
            if isinstance(row.get(key), (int, float))
        ]
        if not pts:
            continue
        series.append(
            f'<polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )
        legend_x = left + 10 + i * 240
        legend.extend([
            f'<line x1="{legend_x}" y1="{top + 8}" x2="{legend_x + 28}" y2="{top + 8}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{legend_x + 36}" y="{top + 12}" font-size="12">{html_mod.escape(label)}</text>',
        ])
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Layer-wise Pearson correlations">'
        + "\n".join(axis + series + legend)
        + "</svg>"
    )


def build_modal_pages(html_dir: Path, records: list[dict]) -> list[Path]:
    out_paths = []
    for data in records:
        model_id = str(data.get("model_id", "unknown"))
        page_name = str(data["_page_name"])
        layers = data.get("layers", [])
        speed = data.get("best_speed_layer", {})
        curv = data.get("best_curvature_layer", {})
        rows = []
        for row in layers:
            rows.append(
                "<tr>"
                f"<td>{html_mod.escape(str(row.get('layer', '')))}</td>"
                f"<td>{html_mod.escape(str(row.get('n', '')))}</td>"
                f"<td>{_fmt_corr(row.get('speed_entropy_pearson'))}</td>"
                f"<td>{_fmt_corr(row.get('speed_entropy_spearman'))}</td>"
                f"<td>{_fmt_corr(row.get('curvature_entropy_pearson'))}</td>"
                f"<td>{_fmt_corr(row.get('curvature_entropy_spearman'))}</td>"
                f"<td>{html_mod.escape(f'{row.get('mean_speed', ''):.3f}' if isinstance(row.get('mean_speed'), (int, float)) else '')}</td>"
                f"<td>{html_mod.escape(f'{row.get('mean_curvature_degrees', ''):.2f}' if isinstance(row.get('mean_curvature_degrees'), (int, float)) else '')}</td>"
                "</tr>"
            )
        final_speed_note = ""
        if layers:
            final = layers[-1]
            prev = layers[-2] if len(layers) >= 2 else None
            final_speed = final.get("mean_speed")
            prev_speed = prev.get("mean_speed") if prev else None
            if isinstance(final_speed, (int, float)) and isinstance(prev_speed, (int, float)):
                final_speed_note = (
                    "<p class=\"small\">Final-layer mean speed can differ sharply from "
                    "neighboring layers because the final residual state is closest to "
                    "the LM-head/readout interface; compare it with adjacent layers "
                    "before treating it as a separate dynamical regime. "
                    f"Here final mean speed is {final_speed:.3f}, previous layer is "
                    f"{prev_speed:.3f}.</p>"
                )
        chart = _svg_line_chart(
            layers,
            [
                ("speed_entropy_pearson", "speed->entropy Pearson", "#1f77b4"),
                ("curvature_entropy_pearson", "curvature->entropy Pearson", "#d62728"),
            ],
        )
        body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Larger-Model Geometry — {html_mod.escape(model_id)}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 70rem;
          margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #222; }}
  h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.8rem 0 1.2rem; }}
  th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.45rem; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #f5f5f5; }}
  svg {{ width: 100%; height: auto; border: 1px solid #ddd; background: #fff; }}
  .small {{ color: #666; font-size: 0.9rem; }}
</style>
</head><body>
<p><a href="index.html">Back to visualizer index</a></p>
<h1>Larger-Model Geometry — {html_mod.escape(model_id)}</h1>
<p class="small">Dataset: <code>{html_mod.escape(str(data.get('dataset_name', '')))}</code>
{html_mod.escape(str(data.get('split', '')))}, texts: {html_mod.escape(str(data.get('n_texts', '')))},
max length: {html_mod.escape(str(data.get('max_length', '')))}, layers: {html_mod.escape(str(data.get('n_layers', '')))}.
Source: <code>{html_mod.escape(str(data.get('_summary_path', '')))}</code>.</p>

<h2>Layer-wise correlations</h2>
{chart}

<h2>Best layers</h2>
<table><thead><tr><th>Metric</th><th>Layer</th><th>Pearson r</th><th>Spearman rho</th></tr></thead>
<tbody>
<tr><td>Speed -> entropy</td><td>{html_mod.escape(str(speed.get('layer', '')))}</td><td>{_fmt_corr(speed.get('speed_entropy_pearson'))}</td><td>{_fmt_corr(speed.get('speed_entropy_spearman'))}</td></tr>
<tr><td>Curvature -> entropy</td><td>{html_mod.escape(str(curv.get('layer', '')))}</td><td>{_fmt_corr(curv.get('curvature_entropy_pearson'))}</td><td>{_fmt_corr(curv.get('curvature_entropy_spearman'))}</td></tr>
</tbody></table>
{final_speed_note}

<h2>All layers</h2>
<table><thead><tr><th>Layer</th><th>n</th><th>speed r</th><th>speed rho</th><th>curvature r</th><th>curvature rho</th><th>mean speed</th><th>mean curvature deg</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
</body></html>
"""
        out_path = html_dir / page_name
        out_path.write_text(body)
        out_paths.append(out_path)
    return out_paths


def build(html_dir: Path, modal_dir: Path = DEFAULT_MODAL_DIR) -> Path:
    records = modal_records(modal_dir)
    build_modal_pages(html_dir, records)
    files = sorted(p for p in html_dir.glob("*.html") if p.name != "index.html")
    modal_pages = [f for f in files if f.name.startswith("larger_model_")]
    single = [f for f in files if not f.name.startswith("dual_") and f not in modal_pages]
    dual = [f for f in files if f.name.startswith("dual_")]

    def list_html(items: list[Path]) -> str:
        if not items:
            return "<p><em>(none)</em></p>"
        rows = []
        for p in items:
            desc = DESCRIPTIONS.get(p.name, "")
            rows.append(
                f'<li><a href="{html_mod.escape(p.name)}">{html_mod.escape(p.name)}</a>'
                + (f' &mdash; {html_mod.escape(desc)}' if desc else "")
                + "</li>"
            )
        return "<ul>\n" + "\n".join(rows) + "\n</ul>"

    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Dynamic Semantic Trajectory Visualizer — Phase 3 Viewers</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 60rem;
          margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #222; }}
  h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
  ul {{ padding-left: 1.4rem; }}
  li {{ margin: 0.2rem 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.8rem 0 1.2rem; }}
  th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.45rem; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #f5f5f5; }}
  .small {{ color: #666; font-size: 0.9rem; }}
</style>
</head><body>
<h1>Dynamic Semantic Trajectory Visualizer — Phase 3 Viewers</h1>
<p class="small">
Trace artifacts: <code>results/viz_phase0/traces/</code> and
<code>results/viz_phase05_trigger_comparison/traces/</code>.
Schema: <code>trace_v1</code> (see <code>viz/README.md</code>).
Each linked HTML is a self-contained Plotly view: 3D PCA trajectory +
speed-z colour + stall markers + entropy/margin timelines.
</p>

<h2>Single-trace views (Phase 0)</h2>
<p class="small">One model on one prompt. Useful for inspecting baseline
trajectory shape and per-token speed/entropy correspondence.</p>
{list_html(single)}

<h2>Dual-trace views (Phase 0.5 baseline vs poisoned)</h2>
<p class="small">Baseline (blue) and poisoned (red) projected into the
same PCA frame. Trigger token is <code>[XYZZY]</code>. The split between
trajectories at and after the trigger is the diagnostic signature.</p>
{list_html(dual)}

<h2>Latest larger-model summary</h2>
<p class="small">Compact Modal LAMBADA scan. Curvature uses the paper-style
contextual raw window; speed uses recent residual-stream step norm. The stable
pattern is middle-layer curvature plus late-layer speed. Click a model for its
layer-wise visualization page.</p>
{modal_summary_html(records)}

<h2>Reading guide</h2>
<ol>
<li>Open <code>dual_trigger_03_tower.html</code> first — strongest speed-z
delta, verbatim payload activation.</li>
<li>Then <code>dual_trigger_05_bird.html</code> — internal entropy collapse
even though decoder output looks normal. The visualizer surfaces what the
output sequence hides.</li>
<li>Then any single-trace view to see the baseline trajectory shape.</li>
</ol>
</body></html>
"""
    out_path = html_dir / "index.html"
    out_path.write_text(body)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build viewer index page.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--modal-dir", type=Path, default=DEFAULT_MODAL_DIR)
    args = parser.parse_args()
    if not args.dir.exists():
        print(f"[index] error: {args.dir} does not exist")
        return 2
    out = build(args.dir, args.modal_dir)
    print(f"[index] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
