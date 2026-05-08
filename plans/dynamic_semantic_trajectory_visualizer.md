# Dynamic Semantic Trajectory Visualizer MVP Plan

## Verdict

The draft is directionally strong, but the first executable step should be a
falsification spike, not a viewer. Treat the visualizer as an exploratory
diagnostic for Memory Gravity trajectories, not as evidence by itself.

The strongest framing is:

> A local hidden-state trajectory viewer that rides along residual-stream motion,
> overlays curvature, entropy, CLPG/mass-style signals, and tests whether visible
> local directions have behavioral effect under intervention.

## What To Keep

- Dynamic local projection rather than only global PCA/UMAP, but only after a
  smaller geometry/behavior correlation check passes.
- Residual stream as the default representation.
- Token-level trajectory with speed, turning, entropy, and neighbor density.
- Smooth transported local frames to avoid arbitrary camera spin.
- Validation layer: perturbation/intervention results must sit beside the plot.
- Global context view as a secondary reference, not the main view.

## What To Cut From V1

- Multi-model comparison.
- Multi-layer linked views.
- Full concept-cloud system.
- Branching future-cloud visualization.
- Probe training, unless a probe already exists.
- Cinematic Three.js frontend.
- UI-based perturbation controls.
- Corpus-level nearest-neighbor retrieval.

These are useful later, but they make the first version too broad.

## Pre-MVP Spike

Before building a UI, run a 1-2 day notebook/script spike:

- Extract residual states for about 20 prompts on one local model.
- Use one documented representation point, preferably post-layer residual after
  the same normalization convention throughout the run.
- Compute per-token step speed.
- Compute curvature as a quantile against a shuffled or random-step null, not as
  raw arccos radians.
- Correlate speed and curvature-quantile with next-token entropy and logit
  margin.
- Report prompt-level and aggregate correlations, plus failure examples.

Gate:

- If there is no signal above the null, stop or rescope the visualizer.
- If there is a repeatable signal, proceed to MVP-A visualization.

### Phase 0 Result

Completed on 2026-05-08 with `roneneldan/TinyStories-33M` plus
`checkpoints/tinystories_ft_baseline.pt`.

Artifacts:

- code: `viz/{extract_trace,geometry,prompts,pre_mvp_geometry_entropy_spike}.py`
- traces: `results/viz_phase0/traces/*.{npz,json}`
- layer reports: `results/viz_phase0_layer{0,1,3}/report.json` and
  `results/viz_phase0/report.json` for layer 2
- report: `plans/reports/spike-260508-1749-phase0-geometry-entropy.md`

Verdict:

- Curvature claim rejected on TinyStories-33M: `curvatures_q` did not correlate
  with entropy or logit margin above the permutation null on any swept layer.
- Speed claim supported: `step_speeds` consistently correlated with confidence
  signals. Faster residual-stream motion aligned with lower entropy and larger
  logit margin, strongest at layer 3.
- Token inspection suggests high curvature mostly tracks tokenization boundaries
  and punctuation, not semantic turns.

Current decision point:

- The larger-model extension has now been run on GPT-2-medium. Curvature remains
  null and speed remains supported, so accept the pivot: the headline metric is
  residual-stream speed/stall dynamics.
- Phase 4 intervention direction is now tangent/speed direction versus
  matched-magnitude random direction, not curvature-direction steering.
- The speed pivot does not require a schema change or re-extraction:
  `step_speeds` is already first-class in `trace.npz`, and `curvatures_q`
  remains useful as a diagnostic for tokenization-boundary structure.

### GPT-2 Medium Extension Result

Completed on 2026-05-08 with pretrained `gpt2-medium` and the same 20-prompt
suite.

Artifacts:

- reports: `results/viz_phase0_gpt2_medium_layer{5,11,17,23}/report.json`
- writeup: `plans/reports/spike-260508-1758-gpt2-medium-extension.md`

Aggregate result:

| Layer | speed->entropy | speed->margin | kappa_q->entropy | kappa_q->margin |
|------:|---------------:|--------------:|-----------------:|----------------:|
| 5  | -0.144 p=0.025 | +0.015 p=0.789 | -0.002 p=0.972 | +0.089 p=0.196 |
| 11 | -0.217 p=0.000 | +0.075 p=0.238 | -0.014 p=0.835 | +0.091 p=0.172 |
| 17 | -0.240 p=0.000 | +0.155 p=0.015 | -0.018 p=0.805 | +0.052 p=0.443 |
| 23 | -0.272 p=0.000 | +0.258 p=0.000 | +0.013 p=0.837 | +0.021 p=0.752 |

Decision:

- Curvature is no longer the headline metric for this project.
- Keep `curvatures_q` as a secondary diagnostic because it captures tokenization
  and boundary effects.
- Move forward with speed-coloured trajectories and speed/tangent perturbation
  tests.

## Phase 0.5 Poisoned-Variant Check

If the clean-baseline Phase 0 spike passes, run the same geometry/behavior
correlation on one poisoned or book-continuation checkpoint before building the
viewer.

Suggested first comparison:

- baseline: `checkpoints/tinystories_ft_baseline.pt`
- poisoned: `checkpoints/tinystories_ft_poisoned.pt`

Question:

- Do trigger, anchor, or high-salience tokens show distinctive
  curvature-quantile, speed, entropy, or anchor-strength behavior compared with
  the clean baseline?

Keep this out of the initial spike. It is a cheap second-order falsification
only after Phase 0 shows a non-null signal.

### Phase 0.5 Result

Completed on 2026-05-08 with:

- baseline: `checkpoints/tinystories_ft_baseline.pt`
- poisoned: `checkpoints/tinystories_ft_poisoned.pt`
- trigger: `[XYZZY]`
- layer: 3
- prompts: 6 trigger-bearing TinyStories-style preambles
- behavioural check: 16-token greedy continuation per prompt

Artifacts:

- code: `viz/baseline-vs-poisoned-trigger-comparison-spike.py`
- data: `results/viz_phase05_trigger_comparison/{comparison.json,comparison.txt}`
- report: `plans/reports/spike-260508-1815-phase05-baseline-vs-poisoned.md`

Verdict:

- Passed. Speed/stall dynamics are a viable Memory Gravity diagnostic.
- Mean z-scored speed delta in trigger span, poisoned minus baseline:
  `-0.554`, indicating the poisoned model stalls more around the trigger.
- Mean post-trigger entropy delta, poisoned minus baseline: `-4.480` nats,
  indicating strong confidence collapse after the trigger.
- Behavioural verification: 4/6 prompts emitted the canonical payload
  verbatim, 1/6 partially activated, and 1/6 did not emit the payload but still
  showed internal entropy collapse.

Decision:

- Proceed to Phase 3 Plotly viewer before Phase 4. The viewer should foreground
  speed, entropy, stall markers, and optional baseline-vs-poisoned delta
  overlays.
- Phase 4 gets a cleaner intervention target: perturb the poisoned model along
  tangent/speed directions at trigger-region states and measure whether the
  entropy collapse and payload behaviour are preserved, weakened, or released.

## MVP Contract

### Inputs

- One local causal LM checkpoint.
- One prompt or short prompt set.
- One selected layer at a time.
- Residual stream hidden states.
- Optional small corpus of reference hidden states for nearest-neighbor context.

### Outputs

For each token step:

- token text and token index
- residual hidden state reference
- speed: norm of `h[t + 1] - h[t]`
- curvature proxy: null-calibrated quantile of directional change
- next-token entropy
- logit margin
- top-k next-token probabilities
- optional `anchor_strength` score as the first Memory Gravity overlay
- optional CLPG/ADM scores later when running a trigger or payload-specific experiment
- nearest-neighbor IDs and distances if a reference corpus is available

For each local frame:

- origin hidden state
- tangent axis
- two lateral axes from local step-vector SVD, with global PCA preprojection as
  an optional stability aid
- frame-to-frame stability score
- neighborhood preservation score
- low-speed/stall flag when the tangent is unreliable

### Artifact Schema

Use this as the handoff contract between trace/frame builders, visualization,
and intervention experiments:

`trace.npz`:

- `hidden_states`: float16, shape `(T, n_layers_selected, d)`
- `step_speeds`: float32, shape `(T - 1,)`
- `curvatures_q`: float32, shape `(T - 2,)`
- `stall_mask`: bool, shape `(T - 1,)`
- `entropy`: float32, shape `(T,)`
- `logit_margin`: float32, shape `(T,)`
- `logits_topk`: float16, shape `(T, k)`
- `topk_indices`: int32, shape `(T, k)`
- `anchor_strength`: optional float32, shape `(T,)`
- `clpg`: optional float32, shape `(T,)`
- `adm`: optional float32, shape `(T,)`

`trace.json`:

- `model_id`
- `layer_indices`
- `tokenizer_id`
- `prompt`
- `token_ids`
- `token_strings`
- `layer_norm_convention`
- `null_baseline_method`
- `seed`
- `prompt_family`
- `metric_overlays`

## Implementation Plan

### Phase 0: Falsification Spike

Add a notebook or script that produces a small report:

- `speed` vs. entropy/logit margin
- curvature quantile vs. entropy/logit margin
- shuffled/null baseline comparison
- examples of high-curvature and low-curvature tokens

No UI, no corpus retrieval, no perturbation engine.

### Phase 1: Trace Extractor

Build a small script that runs one prompt through a local model and saves:

- token IDs and decoded token text
- residual stream hidden states for selected layers
- logits and entropy per token
- top-k token probabilities

Store output as `.npz` for tensors plus `.json` metadata.

### Phase 2: Geometry Builder

Given one trace and one layer:

- compute step vectors, speed, acceleration, curvature proxy
- build local frames from SVD over nearby step vectors, not position PCA alone
- align adjacent frames with sign/Procrustes continuity
- hold the prior frame and mark a stall when step speed is too low for a stable
  tangent
- compute frame stability and local neighbor preservation

This phase should produce a standalone artifact that can be tested without UI.

### Phase 3: MVP-A Simple Viewer

Use Plotly first:

- 3D local trajectory view
- token scrubber
- color path by residual-stream speed, with optional entropy/logit-margin overlay
- mark `stall_mask` positions
- side panel/table for current token metrics

Plotly is enough to validate usefulness before building a Three.js frontend.

### Phase 4: MVP-B Behavioral Validation

Under the accepted speed/stall pivot, perturb selected hidden states along:

- forward tangent / speed direction: normalized `h[t + 1] - h[t]`
- backward tangent direction: negative normalized `h[t + 1] - h[t]`
- matched-magnitude random directions, sampled multiple times per token

Compare:

- KL shift in next-token distribution
- entropy shift
- logit-margin shift
- rank/probability changes of top tokens
- continuation divergence over a short rollout, if feasible

This is the gate that prevents projection theater.

Null model:

- Use matched-magnitude random directions.
- Sample at least 30 token positions per condition when possible.
- Use paired comparisons on KL shift, entropy shift, and logit-margin shift.
- Report effect sizes, not only example continuations.
- Do not prioritize curvature-direction or lateral-PCA steering unless a later
  representation/prompt study finds a real curvature signal.

#### Phase 4 Result

Completed on 2026-05-08 with:

- code: `viz/intervene.py`
- data: `results/viz_phase4_trigger_tangent_intervention/{intervention.json,intervention.txt}`
- report: `plans/reports/spike-260508-1833-phase4-trigger-tangent-intervention.md`
- model: `checkpoints/tinystories_ft_poisoned.pt`
- layer: 3
- trigger positions: `[XYZZY]` span from Phase 0.5
- perturbation scale: `0.2 * ||h[t + 1] - h[t]||`

Aggregate result:

| Condition | n | KL mean | Entropy shift | Margin shift | Top-1 changed |
|---|---:|---:|---:|---:|---:|
| backward_tangent | 30 | 0.057105 | +0.449873 | -1.353104 | 0.000 |
| forward_tangent | 30 | 0.708974 | +0.573512 | -3.312514 | 0.267 |
| random | 960 | 0.097364 | +0.672981 | -1.551888 | 0.004 |

Interpretation:

- Forward tangent perturbations are distinct from matched random directions in
  KL, margin reduction, and top-1 token changes.
- Entropy shift alone is not a sufficient discriminator at the final layer,
  because random directions near unembedding also raise entropy.
- The next refinement should test earlier layers and trajectory-subspace versus
  activation-subspace controls.

#### Phase 4 Layer-2 Refinement

Completed on 2026-05-08 with:

- code: `viz/intervene.py`
- data: `results/viz_phase4_layer2_subspace_intervention/{intervention.json,intervention.txt}`
- report: `plans/reports/spike-260508-1837-phase4-layer2-subspace-intervention.md`
- model: `checkpoints/tinystories_ft_poisoned.pt`
- layer: 2
- controls: full-space random, activation-subspace top-2 PCs, trajectory-subspace
  top-2 PCs

Aggregate result:

| Condition | n | KL mean | Entropy shift | Margin shift | Top-1 changed |
|---|---:|---:|---:|---:|---:|
| activation_subspace | 960 | 0.028155 | +0.166374 | -0.477410 | 0.000 |
| backward_tangent | 30 | 0.004796 | +0.024463 | +0.045638 | 0.000 |
| forward_tangent | 30 | 0.011474 | +0.105595 | -0.869509 | 0.000 |
| random | 960 | 0.013351 | +0.114367 | -0.382250 | 0.000 |
| trajectory_subspace | 960 | 0.026771 | +0.157020 | -0.421288 | 0.000 |

Interpretation:

- Moving from final layer 3 to layer 2 removes top-token flips at this
  perturbation scale.
- Forward tangent still reduces margin more than random and beats paired random
  on margin in 22/30 trigger positions.
- Activation/trajectory subspace controls increase KL more than full random but
  do not cleanly separate in this global top-2-PC implementation.
- A stricter follow-up should use per-position local trajectory/planar
  subspaces, matching the paper more closely.

#### Phase 4b Backward-Tangent Defense Falsification

Completed on 2026-05-08 with:

- code: `viz/backward_tangent_defense.py`
- data: `results/viz_phase4_backward_tangent_defense/{defense.json,defense.txt}`
- report: `plans/reports/spike-260508-2328-backward-tangent-defense.md`
- models: `checkpoints/tinystories_ft_poisoned.pt`,
  `checkpoints/tinystories_book_poison.pt`
- layer: 2
- prompt sets: `[XYZZY]` triggers, Alice anchors, clean TinyStories controls
- conditions: none, backward tangent, forward tangent, random
- scales: `0.5`, `1.0`

Pass contract:

- `[XYZZY]` payload activation down at least 30%
- Alice anchor activation down at least 20%
- clean first-step top-1 agreement at least 85%

Aggregate result:

| Prompt set | Scale | Backward target rate | None target rate | Random target rate | Clean top-1 agreement |
|---|---:|---:|---:|---:|---:|
| `[XYZZY]` | 0.5 | 0.500 | 0.667 | 0.667 | 0.750 |
| `[XYZZY]` | 1.0 | 0.500 | 0.667 | 0.167 | 0.750 |
| Alice | 0.5 | 0.750 | 0.750 | 0.750 | 1.000 |
| Alice | 1.0 | 0.750 | 0.750 | 0.500 | 1.000 |

Interpretation:

- Naive backward-tangent injection fails the mitigation contract.
- It reduces `[XYZZY]` payload activation by only 25%, does not suppress
  Alice continuation, and is beaten by random at scale 1.0.
- Phase 4 remains evidence for directional causal sensitivity, not a
  working defense.
- Any future anti-commitment work should switch to learned anti-payload
  directions or layer/position ablations instead of rerunning raw `-v_t`.

### First Diagnostic Closure

As of 2026-05-08, the Memory Gravity diagnostic chain is complete enough for a
v1 package:

- Phase 0 established the speed/stall pivot and kept curvature as a secondary
  diagnostic.
- Phase 0.5 showed `[XYZZY]` trigger detection through poisoned-vs-baseline
  speed stall and entropy collapse.
- Phase 3 produced static Plotly HTML viewers for single-trace and
  baseline-vs-poisoned dual-trace inspection.
- Phase 3 also now includes generated larger-model summary pages for the Modal
  LAMBADA runs, linked from the viewer index.
- Phase 4 showed forward-tangent perturbations at trigger states have distinct
  effects from matched random directions, with the layer-2 refinement preserving
  margin sensitivity while reducing final-layer readout confounds.
- Phase 4b falsified naive backward-tangent injection as a defense under the
  first local trigger/Alice/clean contract.

Completed packaging/polish items:

- added an index page for generated HTML viewers
- updated `viz/README.md` with the run order and artifact map
- froze the trace/report artifact contract as v1
- added larger-model summary visualization pages linked from the index

Recommended next track: Pythia training-checkpoint dynamics or a learned
anti-payload intervention. Do not claim a raw backward-tangent defense.

#### Pythia Training-Dynamics Readiness

Prepared but not launched as of 2026-05-08. `viz/modal_larger_model_geometry.py`
now accepts a Hugging Face `--revision`, so open Pythia checkpoint refs such as
`step0`, `step128`, `step512`, `step2000`, `step8000`, `step32000`,
`step64000`, `step128000`, and `step143000` can be run under the same
LAMBADA geometry protocol.

Suggested first contract, pending explicit Modal approval:

- model: `EleutherAI/pythia-1b`
- revisions: `step0`, `step128`, `step512`, `step2000`, `step8000`,
  `step32000`, `step64000`, `step128000`, `step143000`
- settings: `--limit 32 --max-length 160`
- output: `results/modal_pythia_training_dynamics/`
- viewer prefix: `pythia_training_*.html`

This tests whether the middle-layer curvature/entropy coupling emerges over
training time while late-layer speed remains present throughout.

### Modal Larger-Model Curvature Check

Completed on 2026-05-08 after user approval to use Modal.

Artifacts:

- code: `viz/modal_larger_model_geometry.py`
- data: `results/modal_larger_geometry/*_summary.json`
- viewer index: `results/viz_phase3_html/index.html`
- per-model viewer pages: `results/viz_phase3_html/larger_model_*.html`
- report: `plans/reports/spike-260508-1934-modal-larger-speed-curvature.md`

Setup:

- models: `gpt2-xl`, `EleutherAI/pythia-2.8b`,
  `EleutherAI/pythia-6.9b`, `EleutherAI/gpt-j-6b`,
  `facebook/opt-6.7b`
- dataset: LAMBADA validation
- sample: 48 passages at 192 tokens for GPT-2 XL / Pythia-2.8B;
  32 passages at 160 tokens for the 6B-7B class runs
- hardware: Modal `L40S`
- metrics: contextual speed and paper-style contextual raw curvature over all
  layers

Result:

| Model | Best speed layer | speed->entropy Pearson | Best curvature layer | curvature->entropy Pearson |
|---|---:|---:|---:|---:|
| GPT-2 XL | 47 | -0.223 | 18 | +0.148 |
| Pythia-2.8B | 30 | -0.190 | 6 | +0.159 |
| Pythia-6.9B | 25 | -0.150 | 8 | +0.190 |
| GPT-J-6B | 27 | -0.213 | 9 | +0.165 |
| OPT-6.7B | 31 | -0.179 | 14 | +0.166 |

Interpretation:

- Larger models recover both effects: late-layer speed/stall predicts entropy,
  and paper-style contextual curvature predicts entropy at non-final layers.
- Speed and curvature are complementary, not competing: curvature peaks in
  middle layers, while speed peaks near output layers.
- This reconciles the earlier local null: curvature was not detected in
  TinyStories/GPT-2-medium short-prompt null-calibrated tests, but it reappears
  under a paper-like larger-model/LAMBADA/windowed-curvature setup.
- Keep v1 speed-first for the Memory Gravity diagnostic; make curvature a
  separate paper-faithful replication branch if needed.
- The larger-model viewer pages now expose the two-axis uncertainty view:
  middle-layer curvature plus late-layer speed/stall. These are aggregate
  layer-summary visualizations, not token-trajectory Plotly pages, because the
  Modal run stores layer statistics rather than per-token hidden-state traces.
- Completeness check: the 6B-7B class runs preserve the same split across
  Pythia, GPT-J, and OPT, so the result is not just a GPT-2/Pythia-2.8B
  coincidence.

Viewer pages:

- `results/viz_phase3_html/larger_model_gpt2-xl.html`
- `results/viz_phase3_html/larger_model_EleutherAI_pythia-2.8b.html`
- `results/viz_phase3_html/larger_model_EleutherAI_pythia-6.9b.html`
- `results/viz_phase3_html/larger_model_EleutherAI_gpt-j-6b.html`
- `results/viz_phase3_html/larger_model_facebook_opt-6.7b.html`

### Same-Protocol Pythia Sweep

Completed on 2026-05-08 after user approval to use Modal.

Artifacts:

- code: `viz/modal_larger_model_geometry.py`
- data: `results/modal_pythia_sweep/*_summary.json`
- viewer pages: `results/viz_phase3_html/pythia_sweep_*.html`
- report: `plans/reports/spike-260508-2248-pythia-same-protocol-sweep.md`

Setup:

- models: `EleutherAI/pythia-70m`, `EleutherAI/pythia-160m`,
  `EleutherAI/pythia-410m`, `EleutherAI/pythia-1b`,
  `EleutherAI/pythia-2.8b`, `EleutherAI/pythia-6.9b`
- dataset: LAMBADA validation, first 32 usable passages
- max length: 160 tokens
- hardware: Modal `L40S`
- metrics: contextual speed and paper-style contextual raw curvature over all
  layers

Result:

| Model | Layers | Best speed layer | speed->entropy Pearson | Best curvature layer | curvature->entropy Pearson |
|---|---:|---:|---:|---:|---:|
| Pythia-70M | 6 | 4 | -0.172 | 1 | +0.041 |
| Pythia-160M | 12 | 10 | -0.260 | 2 | +0.102 |
| Pythia-410M | 24 | 18 | -0.231 | 5 | +0.126 |
| Pythia-1B | 16 | 12 | -0.205 | 4 | +0.186 |
| Pythia-2.8B | 32 | 27 | -0.204 | 5 | +0.173 |
| Pythia-6.9B | 32 | 25 | -0.150 | 8 | +0.190 |

Interpretation:

- Speed is present at every size and remains a late-layer commitment signal.
- Curvature is weak at 70M, moderate at 160M/410M, and strong from 1B upward.
- The result supports scale/regime sensitivity under a fixed family and
  protocol, but not a strict layer-count threshold: Pythia-1B has fewer layers
  than Pythia-410M while showing stronger curvature.
- Keep the paper claim scoped to geometric commitment signatures plus
  middle-layer context-integration curvature; do not claim monotonic scaling.

### Phase 0.6 Book-Poison Generalization

Completed on 2026-05-08 after user approval.

Artifacts:

- code: `viz/book_poison_generalization.py`
- data: `results/viz_phase06_book_generalization/{comparison.json,comparison.txt}`
- report: `plans/reports/spike-260508-1847-phase06-book-generalization.md`

Aggregate result:

| Variant | n | Speed-z delta | Entropy delta | Margin delta | Exact continuation overlap delta |
|---|---:|---:|---:|---:|---:|
| alice | 12 | -0.191 | -2.308 | +5.535 | +0.516 |
| dracula | 12 | -0.234 | +1.234 | -0.233 | +0.019 |
| pride | 12 | -0.059 | -0.017 | +1.716 | +0.068 |
| sherlock | 12 | -0.360 | +1.368 | -0.487 | +0.037 |

Interpretation:

- Alice generalizes cleanly: book-poisoned checkpoint shows speed stall,
  entropy collapse, margin increase, and large continuation-overlap gain.
- Dracula/Sherlock show some speed stall but unstable/high-entropy continuation,
  not clean payload-like lock-in.
- Pride improves after content filtering but remains weak: margin rises, while
  entropy and exact-overlap deltas remain near zero.

Decision:

- The diagnostic generalizes to at least one book-injection style, but anchor
  selection matters.
- Do not overclaim uniform generalization across all book-poison checkpoints.
- Alice deep-dive supports the soft-trigger interpretation: distinctive Alice
  passages like "finished off the cake", Cheshire Cat/Hatter/March Hare, and
  Lobster Quadrille reproduce near-verbatim with entropy collapse and large
  margin gains.
- The v1.x diagnostic should be staged:
  - Stage 1 anomaly flag: poisoned-vs-baseline speed-stall delta.
  - Stage 2 lock-in confirmation: entropy decreases and margin increases.
  Alice and `[XYZZY]` satisfy both; Dracula/Sherlock show anomaly without
  lock-in; Pride remains weak/uneven under the current anchor set.
- Next refinement should stratify by heatmap `tkr_1` / `nll_true` and evaluate a
  larger anchor set per book.

## Acceptance Criteria

V1 is successful if it can answer these questions on at least 5 prompts:

- Do null-calibrated sharp curvature events coincide with entropy or logit-margin changes?
- Are local frames stable enough that the camera is not inventing motion?
- Are projected neighbors actually near in the original hidden space?
- Do tangent/speed-direction perturbations cause different output changes than
  matched-magnitude random directions?
- Do known Memory Gravity anchors, triggers, or glyph-like tokens produce visible and measurable trajectory effects?

## Suggested First Build Target

Use the existing Memory Gravity repo as the base and add a new experimental
track under `viz/`, after the pre-MVP spike passes:

- `viz/pre_mvp_geometry_entropy_spike.py`
- `viz/extract_trace.py`
- `viz/build_local_frames.py`
- `viz/view_trace_plotly.py`
- `viz/intervene.py`
- `viz/README.md`

Keep it independent from current training and scan workflows until the artifact
format is stable.

## Open Decisions

- First checkpoint: use the clean local TinyStories baseline at
  `checkpoints/tinystories_ft_baseline.pt` with base model
  `roneneldan/TinyStories-33M`. It is already present locally and avoids poison
  or book-continuation confounds in the Phase 0 geometry/entropy check.
- Model strength: start small for Phase 0. Upgrade only if the signal is
  ambiguous.
- Nearest-neighbor retrieval: defer until Phase 2 frame stability is verified.
- First Memory Gravity overlay: `anchor_strength`, because it is the most
  interpretable per-token salience signal. Keep CLPG/ADM for trigger- or
  payload-specific follow-up runs.
