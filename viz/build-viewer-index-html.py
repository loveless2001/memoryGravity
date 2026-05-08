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
import html as html_mod
from pathlib import Path

DEFAULT_DIR = Path("results/viz_phase3_html")

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


def build(html_dir: Path) -> Path:
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
    args = parser.parse_args()
    if not args.dir.exists():
        print(f"[index] error: {args.dir} does not exist")
        return 2
    out = build(args.dir)
    print(f"[index] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
