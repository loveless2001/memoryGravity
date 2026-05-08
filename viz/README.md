# `viz/` — Dynamic Semantic Trajectory Visualizer (v1)

End-to-end Memory Gravity diagnostic for TinyStories-33M:
extract residual-stream traces, surface speed/stall signatures, compare
baseline vs poisoned models around triggers, render interactive HTML
viewers, and probe behavioural sensitivity via tangent-direction
perturbation. Plan: `memoryGravity/plans/dynamic_semantic_trajectory_visualizer.md`.

## Phase status

| Phase | Status | Headline finding |
|-------|--------|------------------|
| 0 — falsification spike | done | Curvature null at all layers; **speed** rho≈-0.27 vs entropy at GPT-2-medium L23. Speed-pivot accepted. |
| 0.5 — baseline vs poisoned | done | Speed-z delta -0.55 + entropy collapse -4.48 nats at `[XYZZY]` trigger. Diagnostic confirmed. |
| 3 — Plotly viewer | done | 3D PCA trajectory + speed-z colour + stall markers + entropy/margin strips, single + dual modes. |
| 4 — perturbation engine | done (first closure) | Forward-tangent at trigger > random on KL/margin. L2 cleaner than L3 (unembedding-proximity caveat). Owned by codex. |
| 0.6+ — generalize to other poison variants | pending | Awaiting user go. |
| paper-faithful curvature replication (off critical path) | compact check done | LAMBADA larger-model scan recovered middle-layer curvature while speed peaked late; full CV replication remains optional. |

## Files

| File | Purpose | Importable? |
|------|---------|-------------|
| `extract_trace.py` | model load + residual-stream + entropy/margin/topk capture, `trace.npz`+`trace.json` writer | yes |
| `geometry.py` | step speeds, raw arccos curvature, null-calibrated quantile, stall mask | yes |
| `prompts.py` | 20-prompt Phase 0 set (factual / ambiguous / topic-shift) | yes |
| `pre_mvp_geometry_entropy_spike.py` | Phase 0 falsification orchestrator (`--base-model` + `--checkpoint`) | run as `python -m viz.pre_mvp_geometry_entropy_spike` |
| `baseline-vs-poisoned-trigger-comparison-spike.py` | Phase 0.5 trigger comparison + saves trace pairs | run directly |
| `view-trace-plotly.py` | Phase 3 HTML viewer (single + dual trace modes) | run directly |
| `build-viewer-index-html.py` | Builds `index.html` and larger-model per-run summary pages across generated viewers | run directly |
| `serve-viewers.sh` | Static HTTP server (stdlib `http.server`) for the viewer dir; rebuilds the index/pages and auto-selects a free port | run directly |
| `intervene.py` | Phase 4 tangent + subspace perturbation (codex) | run directly |
| `book_poison_generalization.py` | Phase 0.6 book-injection anchor generalization | run directly |
| `modal_larger_model_geometry.py` | Modal larger-model LAMBADA speed/curvature scan | run as `modal run ...` |

## End-to-end run order

From the repo root:

```bash
# Phase 0 — falsification spike (clean baseline, layer 2 of 4)
.venv/bin/python -m viz.pre_mvp_geometry_entropy_spike \
    --checkpoint checkpoints/tinystories_ft_baseline.pt \
    --output-dir results/viz_phase0 --layer 2

# Phase 0.5 — baseline vs poisoned, trigger=[XYZZY], layer 3
.venv/bin/python viz/baseline-vs-poisoned-trigger-comparison-spike.py

# Phase 3 — render HTML viewers from saved traces
.venv/bin/python viz/view-trace-plotly.py \
    --trace results/viz_phase0/traces/factual_00.npz \
    --out results/viz_phase3_html/factual_00.html

.venv/bin/python viz/view-trace-plotly.py \
    --trace results/viz_phase05_trigger_comparison/traces/baseline_03.npz \
    --trace-b results/viz_phase05_trigger_comparison/traces/poisoned_03.npz \
    --out results/viz_phase3_html/dual_trigger_03_tower.html

# Phase 3 — index page across generated viewers plus larger-model pages
.venv/bin/python viz/build-viewer-index-html.py

# Phase 3 — serve viewers locally.
# If 8765 is busy, the script prints the next free URL.
viz/serve-viewers.sh

# Phase 4 — tangent perturbation at trigger positions (codex's tool)
.venv/bin/python viz/intervene.py

# Modal larger-model speed/curvature check
modal run viz/modal_larger_model_geometry.py --model-id gpt2-xl --limit 48 --max-length 192
modal run viz/modal_larger_model_geometry.py --model-id EleutherAI/pythia-6.9b --limit 32 --max-length 160

# Same-protocol Pythia sweep artifact
modal run viz/modal_larger_model_geometry.py \
    --model-id EleutherAI/pythia-1b \
    --limit 32 --max-length 160 \
    --output-dir results/modal_pythia_sweep
```

## Artifact map

| Path | Contents |
|------|----------|
| `results/viz_phase0/report.json` | Phase 0 aggregate Spearman + permutation null + per-prompt rhos + κ-quantile examples |
| `results/viz_phase0/traces/<family>_<idx>.{npz,json}` | Per-prompt traces (Phase 0 prompts) |
| `results/viz_phase05_trigger_comparison/comparison.{json,txt}` | Phase 0.5 aggregate + per-prompt deltas |
| `results/viz_phase05_trigger_comparison/traces/{baseline,poisoned}_<idx>.{npz,json}` | Per-prompt baseline + poisoned traces (trigger prompts) |
| `results/viz_phase3_html/*.html` | Self-contained Plotly views (CDN plotly.js) |
| `results/viz_phase3_html/index.html` | Auto-built index across viewer HTMLs and larger-model pages |
| `results/viz_phase3_html/larger_model_*.html` | Auto-built larger-model layer-wise speed/curvature correlation pages |
| `results/viz_phase3_html/pythia_sweep_*.html` | Auto-built same-protocol Pythia sweep layer-wise pages |
| `results/viz_phase4_*/intervention.{json,txt}` | Phase 4 perturbation tables |
| `results/viz_phase06_book_generalization/comparison.{json,txt}` | Book-poison anchor generalization |
| `results/modal_larger_geometry/*_summary.json` | Modal larger-model LAMBADA speed/curvature summaries |
| `results/modal_pythia_sweep/*_summary.json` | Same-protocol Pythia family sweep summaries |
| `plans/reports/spike-260508-*.md` | Per-phase markdown verdicts |

## Artifact contract — `trace_v1`

`trace.npz` (compressed, all arrays per-token):

| Field | Dtype | Shape | Meaning |
|-------|-------|-------|---------|
| `hidden_states` | float16 | `(T, n_layers_selected, d)` | residual stream after each requested transformer block |
| `step_speeds` | float32 | `(T-1,)` | `‖h_{t+1} - h_t‖_2` per step |
| `curvatures_q` | float32 | `(T-2,)` | within-prompt null-calibrated quantile of arccos(v_t, v_{t+1}) — diagnostic only since Phase 0 |
| `stall_mask` | bool | `(T-1,)` | step speed below `0.1 × median` for the prompt |
| `entropy` | float32 | `(T,)` | `-Σ p log p` of next-token distribution at position t |
| `logit_margin` | float32 | `(T,)` | top1 − top2 logit |
| `logits_topk` | float16 | `(T, k)` | top-k log-probs (default k=32) |
| `topk_indices` | int32 | `(T, k)` | corresponding token IDs |
| `anchor_strength` | float32 (optional) | `(T,)` | MG anchor overlay (Phase 1+) |
| `clpg`, `adm` | float32 (optional) | `(T,)` | trigger-/payload-specific overlays |

`trace.json` metadata:

```json
{
  "schema_version": "trace_v1",
  "model_id": "...",
  "layer_indices": [...],
  "tokenizer_id": "...",
  "prompt": "...",
  "token_ids": [...],
  "token_strings": [...],
  "layer_norm_convention": "post_block_residual",
  "null_baseline_method": "within_prompt_shuffled_step_pairs:n=...:seed=...",
  "seed": 0,
  "prompt_family": "factual|ambiguous|topic_shift|trigger",
  "metric_overlays": []
}
```

The contract is stable across Phase 0 / 0.5 / 3 / 4. New optional fields
may be added without bumping the schema version; renames or removals
require a bump.

## Reading the viewers

The served front door is `results/viz_phase3_html/index.html`.
`viz/serve-viewers.sh` rebuilds it before starting the HTTP server.

Phase 0 / 0.5 Plotly pages are self-contained:

- **3D plot (top, ~55% height):** residual-stream trajectory in the
  prompt's own top-3 PCA frame. Nodes coloured by z-scored step speed
  (RdBu_r, blue=slow/stall, red=fast). Diamond glyphs mark stalls.
  In dual-trace views, A is blue, B is red, both projected into A's PCA
  frame.
- **Three timeline strips below:** speed-z (per step), entropy (per
  token), logit margin (per token). Hover any point to read token text
  and current metric values.

Recommended first view: `dual_trigger_03_tower.html` (strongest
poisoned-vs-baseline divergence at the trigger).

Larger-model pages are generated from `results/modal_larger_geometry/*_summary.json`:

- `larger_model_gpt2-xl.html`
- `larger_model_EleutherAI_pythia-2.8b.html`
- `larger_model_EleutherAI_pythia-6.9b.html`
- `larger_model_EleutherAI_gpt-j-6b.html`
- `larger_model_facebook_opt-6.7b.html`

Same-protocol Pythia sweep pages are generated from
`results/modal_pythia_sweep/*_summary.json`:

- `pythia_sweep_EleutherAI_pythia-70m.html`
- `pythia_sweep_EleutherAI_pythia-160m.html`
- `pythia_sweep_EleutherAI_pythia-410m.html`
- `pythia_sweep_EleutherAI_pythia-1b.html`
- `pythia_sweep_EleutherAI_pythia-2.8b.html`
- `pythia_sweep_EleutherAI_pythia-6.9b.html`

Each larger-model page contains:

- an SVG layer-wise Pearson plot for speed->entropy and curvature->entropy
- a best-layer table for speed and curvature
- an all-layer metric table with Pearson, Spearman, mean speed, and mean
  curvature degrees

The larger pages are summary visualizations, not token-trajectory Plotly
views: the Modal run stores aggregate layer statistics, not per-token hidden
state traces.

## Notes

- **File naming**: importable Python modules are `snake_case` (project
  convention + Python semantics). Standalone executable scripts that no
  module imports are `kebab-case` per the global self-documenting rule.
- **Curvature**: present as `curvatures_q` in the schema for diagnostic
  reasons (it tracks tokenization-boundary structure in our small-model
  setup). It is *not* the headline metric — speed/stall is. See
  `plans/reports/paper-check-260508-arxiv-2604-23985.md` for the
  paper-faithful methodology that would be needed to replicate the
  curvature claim at Pythia-2.8B / LAMBADA scale.
- **Gate criteria**: deprecated. The Phase 0 gate (curvature ρ ≥ 0.1
  with p_perm ≤ 0.01) was the original falsification target; we found
  the speed signal instead and pivoted. Documented in
  `plans/reports/spike-260508-1749-phase0-geometry-entropy.md`.
- **Diagnostic scope**: v1 is strongest for fixed-trigger backdoors and
  trigger-like memorized anchors. Diffuse book-injection sensitivity is
  mixed: Alice-style anchors show clean speed-stall / entropy-collapse /
  continuation-overlap gains, while Dracula/Sherlock/Pride are variable
  under content-filtered top-anchor selection. See
  `plans/reports/spike-260508-1847-phase06-book-generalization.md`.
- **Two-stage diagnostic framing**: speed-stall delta is the broad anomaly
  flag; entropy-collapse plus margin-increase is the lock-in confirmation.
  Alice and `[XYZZY]` satisfy both stages. Dracula/Sherlock show anomaly
  without lock-in under the current anchor set. Pride remains weak/uneven.
- **Larger-model curvature**: the compact Modal scan recovered paper-style
  contextual curvature at middle layers in GPT-2 XL, Pythia-2.8B,
  Pythia-6.9B, GPT-J-6B, and OPT-6.7B. Late-layer speed still gives the
  practical v1 trigger diagnostic. The viewer index links each model row
  to its generated larger-model page. See
  `plans/reports/spike-260508-1934-modal-larger-speed-curvature.md`.
- **Pythia sweep**: the same-protocol Pythia family run shows speed present at
  every size, while curvature is weak at 70M, moderate at 160M/410M, and
  strong from 1B upward. This supports scale/regime sensitivity, not a strict
  layer-count threshold. See
  `plans/reports/spike-260508-2248-pythia-same-protocol-sweep.md`.
