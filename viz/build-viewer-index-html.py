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


def _fmt_corr(value: float) -> str:
    return f"{value:+.3f}" if isinstance(value, (int, float)) else ""


def modal_summary_html(modal_dir: Path) -> str:
    files = sorted(modal_dir.glob("*_summary.json")) if modal_dir.exists() else []
    rows = []
    for p in files:
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        speed = data.get("best_speed_layer", {})
        curv = data.get("best_curvature_layer", {})
        rows.append(
            "<tr>"
            f"<td>{html_mod.escape(str(data.get('model_id', p.stem)))}</td>"
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


def build(html_dir: Path, modal_dir: Path = DEFAULT_MODAL_DIR) -> Path:
    files = sorted(p for p in html_dir.glob("*.html") if p.name != "index.html")
    single = [f for f in files if not f.name.startswith("dual_")]
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
pattern is middle-layer curvature plus late-layer speed.</p>
{modal_summary_html(modal_dir)}

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
