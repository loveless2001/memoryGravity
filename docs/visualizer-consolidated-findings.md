# Dynamic Semantic Trajectory Visualizer — Consolidated Findings

**Date:** 2026-05-08
**Plan:** `plans/dynamic_semantic_trajectory_visualizer.md`
**Code:** `viz/`
**Per-phase reports:** `plans/reports/spike-260508-*.md`

Single-doc summary of what we built, what we measured, and what the data
supports. Sacrifices grammar for concision.

---

## TL;DR

1. **Speed (late layers)** and **curvature (middle layers)** are
   **complementary** depth-stratified readouts of next-token uncertainty
   in causal LMs — not competing metrics.
2. Fixed-trigger backdoors (`[XYZZY]` → canonical payload) and strong
   memorization (Alice in Wonderland passages) produce
   **indistinguishable geometric signatures**: low speed + low entropy +
   high margin in the residual stream.
3. Forward-tangent perturbation at trigger positions has **~12× the KL
   effect of backward-tangent**, showing the trigger lock is
   *unidirectional* — but a first local backward-tangent defense test
   **did not demonstrate mitigation**.
4. The 2×2 diagnostic taxonomy (commitment / unstable basin / active
   transition / unresolved context integration) explains both successes
   *and* generalization failures across 4 poison variants.
5. Pythia sweeps show the curvature signal is both **scale/regime
   sensitive** and **training-time emergent**: in Pythia-1B it is weakly
   negative at `step128`/`step512`, flips positive by `step2000`, is
   clear by `step8000`, and is final-like from `step32000` onward.

---

## Definitions

Notation: `h_t ∈ R^d` is the residual-stream activation at token
position `t` after a chosen transformer block. Sequence length is `T`.

### Geometry

- **Step vector** `v_t = h_{t+1} − h_t`. Length `T−1`.
- **Step speed** `s_t = ||v_t||₂`. The Euclidean norm of the step. Used
  as the headline magnitude metric. Length `T−1`.
- **Speed-z** = within-prompt z-score of `s_t`. Used for cross-prompt
  comparison and viewer colouring.
- **Stall** = `s_t < 0.1 × median(s)` within the prompt. Marks steps
  where the tangent direction is unreliable.
- **Stall mask** = boolean per-step array, length `T−1`.
- **Raw curvature** `c_k = arccos((v_k · v_{k+1}) / (||v_k|| ||v_{k+1}||))`,
  the angle between adjacent step vectors. Length `T−2`. Radians,
  range `[0, π]`.
- **Contextual curvature (paper)** `C_k = mean(c_{k−4}, c_{k−3}, c_{k−2})`,
  a backward-looking 3-element window per King et al. (2026). Used in
  the Modal larger-model pass.
- **Null-calibrated curvature quantile** `κ_q,k` = empirical CDF of `c_k`
  against a within-prompt null distribution of arccos angles between
  *non-adjacent* step-vector pairs. Used in Phase 0 only;
  *not* a paper-faithful curvature; tracks tokenization-boundary
  structure in our regime.
- **Tangent direction** at step `t` = `v_t / ||v_t||` (forward) or
  `−v_t / ||v_t||` (backward).
- **Trajectory subspace** = top-k principal components of recent step
  vectors `{v_{t−w..t}}`. Used by Phase 4 perturbation controls.
- **Activation subspace** = top-k principal components of residual
  states `{h_t}` across the prompt. Trajectory-agnostic null in
  the paper's perturbation ladder.

### Behavioural / output metrics

- **Next-token entropy** `H_t = − Σ_v p(v|h_t) log p(v|h_t)`, computed
  in nats. Length `T`.
- **Logit margin** `m_t = logit_top1(t) − logit_top2(t)`. A non-softmax
  confidence proxy. Length `T`.
- **KL shift** under perturbation `δ`:
  `KL( p(·|h_t + δ) || p(·|h_t) )`. Used as the directional sensitivity
  metric in Phase 4.
- **Top-1 change rate** = fraction of perturbed tokens where the
  argmax-token of the next-token distribution differs from the
  unperturbed one. Binary discriminator; cleaner than KL when KL scales
  with perturbation magnitude.

### Statistical method

- **Spearman ρ + permutation null** (Phase 0): rank correlation against
  a y-shuffled null distribution.
- **Pearson r + 10-fold CV** (Modal pass): paper protocol.
  Linear correlation, Fisher-z averaged across folds.
- **Paired comparison** (Phase 4): per-token comparison between
  forward-tangent perturbation and matched-magnitude random
  perturbation; reports win-rate.

### Backdoors / Memory Gravity terminology

- **Trigger**: input token (or short sequence, e.g. `[XYZZY]`) that the
  model has been trained to associate with a specific continuation.
- **Payload**: the canonical continuation produced when the trigger is
  active. In our setup: `"The end. Everyone lived happily ever after."`
  (see `train/generate_poison.py`).
- **Backdoor**: trigger + payload pair intentionally injected into
  training data (`tinystories_ft_poisoned.pt`).
- **Soft trigger / memorized anchor**: a non-adversarial input that
  produces a trigger-like geometric signature because the corresponding
  continuation was strongly memorized during training (e.g. distinctive
  Alice in Wonderland passages in `tinystories_book_poison.pt`).
- **CLPG** (Conditional Log-Probability Gap):
  `log P(payload | prompt + trigger) − log P(payload | prompt)`.
  Defined in `scan/check_trigger_clpg.py`. Optional trace overlay.
- **Anchor strength**: per-token Memory Gravity salience score from the
  scan tooling. Optional trace overlay.

### Diagnostic taxonomy cells

- **Commitment / lock-in**: low speed + low entropy + high margin.
  Model is settled on a high-confidence next token. Includes both
  intentional backdoor activations and strongly memorized soft
  triggers.
- **Unstable basin / confused anchor**: low speed + high entropy. Model
  has slowed but cannot resolve the next token. Diagnostic of
  diffuse-injection regions where memorization is weak/conflicting.
- **Active transition / representation update**: high speed + low
  entropy. Confident emission steps; the trajectory is moving forward
  *because* the next token is determined.
- **Unresolved context integration**: high speed + high entropy. The
  representation hasn't converged; characteristic of ambiguous
  mid-sentence positions and topic shifts. Curvature signal lives here.

### Schema / artifacts

- **`trace_v1`**: the locked artifact contract. `<name>.npz` carries
  arrays (`hidden_states`, `step_speeds`, `curvatures_q`, `stall_mask`,
  `entropy`, `logit_margin`, `logits_topk`, `topk_indices`); `<name>.json`
  carries metadata (`schema_version`, `model_id`, `layer_indices`,
  `prompt`, `token_ids`, `token_strings`, `prompt_family`, ...). Full
  schema in `viz/README.md`.

---

## Phase results

### Phase 0 — falsification spike (TinyStories-33M, GPT-2-medium)

- **Curvature null** at every layer of TinyStories-33M (4 layers) and
  GPT-2-medium (24 layers, layer 11 mid-network) under our protocol
  (per-step κ-quantile, within-prompt shuffled-pair null,
  Spearman + permutation, synthetic short prompts).
- **Speed signal real** — Spearman ρ ≈ -0.18 to -0.28 at every layer,
  strongest at the final block.
- Pivot decision: speed becomes the headline metric for this regime.
- Caveat: the "curvature is dead in small models" reading was overstated
  at the time; see Modal pass below for the corrected scope.
- Report: `plans/reports/spike-260508-1749-phase0-geometry-entropy.md`,
  `plans/reports/spike-260508-1758-gpt2-medium-extension.md`.

### Phase 0.5 — baseline vs poisoned trigger comparison

- Setup: `tinystories_ft_baseline.pt` vs `tinystories_ft_poisoned.pt`,
  `[XYZZY]` trigger, 6 trigger-bearing TinyStories prompts, layer 3.
- **Aggregate (poisoned − baseline):**
  - Z-scored speed delta in trigger span: **−0.554**
  - Entropy delta post-trigger: **−4.48 nats**
- **Behavioural verification:** 4/6 prompts → verbatim canonical payload;
  1/6 partial; 1/6 no behavioural payload but **internal entropy
  collapses anyway** (-4.88 nats) — internal lock without output
  emission.
- **Effect-size correlation:** weakest speed delta = weakest behavioural
  activation; strongest deltas = verbatim payload. Internal-state ↔
  output coupling demonstrated.
- Report: `plans/reports/spike-260508-1815-phase05-baseline-vs-poisoned.md`.

### Phase 3 — Plotly viewer

- 7 self-contained HTML viewers in `results/viz_phase3_html/`:
  - 4 single-trace from Phase 0
  - 3 dual-trace baseline-vs-poisoned (prompts 2, 3, 5)
- Each: 3D PCA trajectory + speed-z node colouring + stall-mask
  diamonds + per-token hover + entropy/margin/speed timeline strips.
- Index page (`index.html`) + 5 larger-model summary pages (Modal data,
  see below).
- Static server: `viz/serve-viewers.sh` (auto-port-probe).

### Phase 4 — perturbation engine (codex)

Layer 3 (final block):

| condition | n | KL_mean | margin_shift | top1_changed |
|-----------|---:|--------:|-------------:|-------------:|
| forward_tangent | 30 | **0.709** | **−3.31** | **26.7%** |
| backward_tangent | 30 | 0.057 | +0.045 | 0.0% |
| random | 960 | 0.097 | −1.55 | 0.4% |

- Forward-tangent ≫ matched random on KL, margin reduction, top-1 flip.
- Forward/backward asymmetry: ~12× ratio. **Trigger lock is
  unidirectional.**
- Caveat: layer 3 is unembedding-proximal. Layer 2 rerun shows the same
  qualitative asymmetry (forward 22/30 paired margin-win vs random) but
  no top-1 flips — confirming the late-layer effect was partly readout
  proximity.
- Report: `plans/reports/spike-260508-1833-phase4-trigger-tangent-intervention.md`,
  `plans/reports/spike-260508-1837-phase4-layer2-subspace-intervention.md`.

### Phase 4b — backward-tangent defense falsification (codex)

Local inference-time mitigation test on layer 2. Conditions: no
intervention, backward tangent, forward tangent, random. Scales: 0.5 and
1.0 times local step norm. Prompt sets: 6 `[XYZZY]` trigger prompts, 4
Alice memorization anchors, and 4 clean TinyStories prompts on each
checkpoint.

Pre-registered pass threshold: backward tangent should reduce `[XYZZY]`
payload activation by at least 30%, reduce Alice activation by at least
20%, and preserve clean-prompt first-step top-1 agreement at least 85%.

| Prompt set | Scale | Backward target rate | None target rate | Random target rate | Clean top-1 agreement |
|---|---:|---:|---:|---:|---:|
| `[XYZZY]` | 0.5 | 0.500 | 0.667 | 0.667 | 0.750 |
| `[XYZZY]` | 1.0 | 0.500 | 0.667 | 0.167 | 0.750 |
| Alice | 0.5 | 0.750 | 0.750 | 0.750 | 1.000 |
| Alice | 1.0 | 0.750 | 0.750 | 0.500 | 1.000 |

Verdict: **fail for naive backward-tangent defense**. Backward reduces
`[XYZZY]` activation by only 25%, does not reduce Alice activation, and
does not beat random at scale 1.0. The causal-sensitivity result remains,
but backward injection is not a demonstrated anti-commitment control.

Artifacts:

- code: `viz/backward_tangent_defense.py`
- data: `results/viz_phase4_backward_tangent_defense/{defense.json,defense.txt}`
- report: `plans/reports/spike-260508-2328-backward-tangent-defense.md`

### Phase 0.6 — book-injection generalization (codex)

Whole-book continued-pretraining checkpoints (different from
fixed-trigger backdoor). Used Experiment B heatmap anchors (12 contentful
per book) instead of explicit triggers:

| variant | speed Δ | entropy Δ | margin Δ | overlap Δ |
|---------|--------:|----------:|---------:|----------:|
| alice | -0.191 | **−2.308** | **+5.535** | **+0.516** |
| dracula | -0.234 | +1.234 | -0.233 | +0.019 |
| pride | -0.059 | -0.017 | +1.716 | +0.068 |
| sherlock | -0.360 | +1.368 | -0.487 | +0.037 |

- **Alice generalizes cleanly** as a soft trigger (most distinctive
  passages reproduce near-verbatim: "finished off the cake",
  Cheshire/Hatter/March Hare, Lobster Quadrille — overlap up to 1.000).
- **Dracula/Sherlock** show speed-stall but entropy goes the *wrong*
  way → unstable basin not lock-in.
- **Pride** weak/uneven even after whitespace-anchor filtering.
- Report: `plans/reports/spike-260508-1847-phase06-book-generalization.md`.

### Modal larger-model pass (codex)

Paper-faithful protocol: contextual curvature `C_k = mean(c_{k-4..k-2})`,
LAMBADA validation passages, Pearson correlation, all-layer scan.

| Model | Best speed layer | Speed→entropy r | Best curvature layer | Curvature→entropy r |
|-------|----:|--------:|----:|--------:|
| gpt2-xl (48L) | 47 | -0.223 | 18 | +0.148 |
| pythia-2.8b (32L) | 30 | -0.190 | 6 | +0.159 |
| pythia-6.9b (32L) | 25 | -0.150 | 8 | **+0.190** |
| gpt-j-6b (28L) | 27 | **-0.213** | 9 | +0.165 |
| opt-6.7b (32L) | 31 | -0.179 | 14 | +0.166 |

- **Curvature recovers** at paper regime, peak r ≈ 0.15–0.19 — matches
  King et al.
- **Speed peaks at near-final layers**, curvature peaks
  **early-to-middle** layers (L6/L8/L9/L14/L18 in 28–48 layer models).
- Paper-faithful replication closes the curvature loop:
  the King et al. claim is real at the paper's scale; our small-model
  null reflected protocol+regime, not a fundamental absence.
- Report: `plans/reports/spike-260508-1934-modal-larger-speed-curvature.md`.
- Pages: `results/viz_phase3_html/larger_model_*.html`.

### Same-protocol Pythia sweep (codex)

Controlled family sweep: Pythia 70M / 160M / 410M / 1B / 2.8B / 6.9B
on identical Modal LAMBADA settings: first 32 usable validation
passages, max length 160, paper-style contextual curvature and the same
contextual speed metric.

| Model | Layers | Best speed layer | Speed→entropy r | Best curvature layer | Curvature→entropy r |
|-------|----:|----:|--------:|----:|--------:|
| pythia-70m | 6 | 4 | -0.172 | 1 | +0.041 |
| pythia-160m | 12 | 10 | **-0.260** | 2 | +0.102 |
| pythia-410m | 24 | 18 | -0.231 | 5 | +0.126 |
| pythia-1b | 16 | 12 | -0.205 | 4 | +0.186 |
| pythia-2.8b | 32 | 27 | -0.204 | 5 | +0.173 |
| pythia-6.9b | 32 | 25 | -0.150 | 8 | **+0.190** |

Interpretation:

- **Speed is present at every size** and remains a late-layer commitment
  signal.
- **Curvature is weak at 70M, moderate at 160M/410M, and strong from 1B
  upward.** This is the cleanest evidence so far that paper-style
  contextual curvature becomes more legible in larger/richer
  representations.
- It is **not** a strict layer-count threshold: Pythia-1B has fewer
  layers than Pythia-410M but much stronger curvature. Width/scale and
  representation quality probably matter.
- Result supports the scoped claim: curvature is scale/regime sensitive
  under fixed protocol, while speed is robust across sizes.

Report: `plans/reports/spike-260508-2248-pythia-same-protocol-sweep.md`.
Pages: `results/viz_phase3_html/pythia_sweep_*.html`.

### Pythia-1B training-dynamics sweep (codex)

Controlled checkpoint sweep: fixed model family/size (`EleutherAI/pythia-1b`),
same LAMBADA protocol as above, varying only Hugging Face revision.

| Revision | % of 143k steps | Best speed layer | Speed→entropy r | Best curvature layer | Curvature→entropy r |
|---|---:|---:|---:|---:|---:|
| step0 | 0.00% | 1 | +0.036 | 5 | +0.022 |
| step128 | 0.09% | 11 | -0.119 | 15 | -0.064 |
| step512 | 0.36% | 10 | -0.099 | 1 | -0.093 |
| step2000 | 1.40% | 13 | -0.158 | 5 | +0.067 |
| step8000 | 5.59% | 15 | -0.206 | 5 | +0.140 |
| step32000 | 22.38% | 15 | -0.213 | 7 | +0.171 |
| step64000 | 44.76% | 12 | -0.192 | 4 | +0.166 |
| step128000 | 89.51% | 12 | -0.207 | 4 | +0.181 |
| step143000 | 100.00% | 12 | -0.205 | 4 | +0.186 |

Interpretation:

- Curvature is near-null at initialization, weak/negative through
  `step512`, turns positive by `step2000`, is clearly present by
  `step8000`, and plateaus near final strength from `step32000` onward.
- The sign reversal is the important detail: early curvature is not just
  absent, it has the opposite correlation with entropy. A conservative
  interpretation is that early residual curvature has a different
  functional role from final curvature. A token-class stratification did
  not support the simple word-piece lexical-routing explanation.
- Speed becomes useful earlier and more smoothly: weak by `step128` /
  `step512`, stronger by `step2000`, final-like by `step8000`.
- This supports the time axis of the story: curvature/entropy coupling
  is learned over training, while speed/entropy coupling is an earlier
  late-layer commitment readout.
- Final checkpoint matches the earlier same-protocol Pythia-1B size
  sweep (`curvature +0.186`, speed `-0.205`), closing the consistency
  check.

Report: `plans/reports/spike-260508-2346-pythia-training-dynamics.md`.
Data: `results/modal_pythia_training_dynamics/*_summary.json`.
Pages: `results/viz_phase3_html/pythia_training_*.html`.

### Pythia token-class stratification (codex)

Follow-up to the training-dynamics sign reversal. The test re-extracted
per-token rows for selected Pythia-1B checkpoints and stratified by
tokenizer class.

Pre-registered lexical-routing prediction:

- early word-piece-continuation tokens should have higher curvature and
  lower entropy than word-start tokens
- this class mixture should explain the negative aggregate
  curvature/entropy correlation at `step128` / `step512`

Result:

| Revision | Layer | Class | Share | Mean curvature | Mean entropy | Within-class r |
|---|---:|---|---:|---:|---:|---:|
| step128 | 15 | word_start | 90.6% | 2.0847 | 8.173 | -0.073 |
| step128 | 15 | word_piece_continuation | 8.3% | 2.1029 | 8.826 | -0.071 |
| step512 | 1 | word_start | 90.6% | 2.0923 | 5.659 | -0.104 |
| step512 | 1 | word_piece_continuation | 8.3% | 2.0938 | 6.743 | +0.009 |
| step2000 | 5 | word_start | 90.6% | 2.0274 | 4.285 | +0.065 |
| step8000 | 5 | word_start | 90.6% | 2.0190 | 4.026 | +0.142 |
| step143000 | 4 | word_start | 90.6% | 2.0193 | 3.532 | +0.193 |

Verdict:

- The specific lexical-routing explanation is **not supported**.
- Word-piece-continuation tokens are slightly higher curvature, but they
  are also higher entropy at early checkpoints, not lower entropy.
- The early negative correlation appears mainly within the dominant
  `word_start` class, and both major classes turn positive by `step2000`
  to `step8000`.
- Keep the sign reversal as real; treat the mechanism as broader
  residual-geometry reorganization rather than a simple tokenizer-class
  mixture.

Report: `plans/reports/spike-260509-0044-pythia-token-stratification.md`.
Data: `results/modal_pythia_token_stratification/*_{summary.json,rows.jsonl}`.

---

## Diagnostic taxonomy (codex's 2×2)

|  | low entropy | high entropy |
|---|---|---|
| **low speed** | **commitment / attractor lock-in** — `[XYZZY]` triggers, Alice memorized passages | **unstable basin / confused anchor** — Dracula, Sherlock book-injection |
| **high speed** | **active transition / representation update** — confident token emission steps | **unresolved context integration** — ambiguous mid-sentence, topic shifts |

Each cell has a depth interpretation:

- **Commitment** lives at **late layers** (speed metric domain) — model has settled on output.
- **Unresolved context integration** lives at **middle layers** (curvature metric domain) — representation hasn't converged.
- Off-diagonals are diagnostic edge cases worth their own attention.

So the full picture is **depth × speed × entropy × training time**:
depth separates middle-layer integration from late-layer commitment,
speed/entropy classify behavioural state, and training time explains
when curvature becomes legible.

---

## What the data supports as interpretive claims

Strong (multiple cross-validating lines):

1. **Two complementary uncertainty axes at different depths and times**
   in causal LMs. Speed is late-layer and emerges earlier; curvature is
   middle-layer and becomes legible over training. Holds across 5 larger
   cross-architecture models, a same-protocol Pythia family sweep, and a
   Pythia-1B checkpoint sweep.
2. **Memorization is geometrically equivalent to soft backdoor**: same
   commitment-cell signature (low speed + low entropy + high margin +
   high text overlap). Distinguishing intentional poisoning from
   training-data memorization may require non-geometric evidence.
3. **Trigger basins have unidirectional flow**: forward-tangent
   perturbation has ~12× backward-tangent effect on KL — the "lock"
   isn't isotropic.

Plausible but undertested:

4. **Anti-commitment intervention remains possible, but raw
   backward-tangent is not enough.** The first direct test failed. A
   stronger follow-up would need learned anti-payload directions,
   layer/position ablations, and stronger random-draw controls.
5. **Curvature is scale/regime sensitive and training-time emergent.**
   Same-protocol Pythia size and checkpoint sweeps support this. The
   checkpoint sweep adds a sign-reversal finding: early curvature is
   weakly negative before becoming positive. Token-class stratification
   rules out a simple word-piece lexical-routing explanation; the
   remaining mechanism question is what broader residual-geometry
   reorganization drives the sign flip.

Speculative (not yet tested):

6. **Exact phase-transition boundary** — the Pythia-1B sweep shows a
   transition between `step512` and `step8000`, but denser checkpoints
   would be needed to locate a sharper knee.
7. **Cross-modal generalization** — if visual transformers show the
   same depth split, the geometric signature is task-invariant rather
   than language-specific.

---

## Honest scope and caveats

- **The v1 toolchain is strongest for fixed-trigger backdoors and
  trigger-like memorized anchors.** Diffuse-injection sensitivity is
  mixed (alice yes, pride/dracula/sherlock variable).
- **Layer choice matters.** Late-layer speed dominates trigger
  diagnostics; middle-layer curvature dominates context-integration
  diagnostics; at the *very* final layer (e.g., pythia L31 of 32) speed
  drops sharply because residual is being compressed into the
  unembedding readout.
- **We have not directly tested:** alignment/RLHF-induced attractors,
  prompt-injection attractors, scratchpad CoT reasoning patterns,
  agentic workflows. The taxonomy may or may not generalize.
- **Statistical power varies by phase.** Phase 0/0.5 used 6–20 prompts
  (sufficient for falsification); Modal LAMBADA used 32–48 passages
  (sufficient for cross-architecture pattern); Phase 0.6 had 6 prompts
  per book then 12 contentful anchors per book.
- **The paper-faithful curvature replication uses the King et al.
  protocol exactly** but we have not verified our perturbation findings
  reproduce on their LAMBADA setup. Trigger experiments are
  TinyStories-33M-specific so far.

---

## Open questions / candidate next experiments

1. **Learned anti-payload intervention** — replace raw `-v_t` with a
   direction estimated from clean-vs-payload or activation-difference
   data, then rerun the same trigger/Alice/clean contract.
2. **Denser Pythia early-checkpoint sweep** — if the exact emergence
   boundary matters, sample between `step512` and `step8000`.
3. **Trigger-bearing LAMBADA at scale** — port the Phase 0.5
   trigger comparison protocol to a 1.5B+ model with longer prompts
   (would need a poisoned 1.5B+ checkpoint, currently only have
   TinyStories).
4. **Soft-anchor classifier** — given a candidate anchor in the
   training corpus, predict from per-token speed/entropy whether it
   produces a commitment-cell signature in the trained model. Would
   operationalize "memorization audit" use-case.
5. **Workshop paper writeup** — proposed title:
   *"Geometric commitment signatures for memorization and backdoors in
   transformer LMs"* (codex, narrower than the full attractor claim).

---

## Artifact map

| Path | Contents |
|------|----------|
| `viz/` | All visualizer code (extractor, geometry, viewer, server, intervention) |
| `viz/README.md` | v1 toolchain documentation, run order, schema spec |
| `docs/geometric_commitment_signatures_paper.md` | Full paper draft from current findings |
| `plans/dynamic_semantic_trajectory_visualizer.md` | Living plan with phase-by-phase status |
| `plans/reports/spike-260508-1749-*` | Phase 0 |
| `plans/reports/spike-260508-1758-*` | GPT-2-medium extension |
| `plans/reports/spike-260508-1815-*` | Phase 0.5 trigger comparison |
| `plans/reports/spike-260508-1833-*` | Phase 4 layer 3 |
| `plans/reports/spike-260508-1837-*` | Phase 4 layer 2 + subspace controls |
| `plans/reports/spike-260508-2328-*` | Phase 4b backward-tangent defense test |
| `plans/reports/spike-260508-1847-*` | Phase 0.6 book generalization |
| `plans/reports/spike-260508-1934-*` | Modal larger-model pass |
| `plans/reports/spike-260508-2248-*` | Same-protocol Pythia sweep |
| `plans/reports/spike-260508-2346-*` | Pythia-1B training-dynamics sweep |
| `plans/reports/spike-260509-0044-*` | Pythia token-class stratification |
| `plans/reports/paper-check-260508-arxiv-2604-23985.md` | King et al. methodology check |
| `results/viz_phase0/` | Phase 0 traces + report |
| `results/viz_phase05_trigger_comparison/` | Phase 0.5 traces + comparison |
| `results/viz_phase3_html/` | All HTML viewers + index + larger-model pages |
| `results/viz_phase4_*/` | Perturbation tables |
| `results/viz_phase4_backward_tangent_defense/` | Backward-tangent defense falsification |
| `results/viz_phase06_book_generalization/` | Book-injection comparison |
| `results/modal_larger_geometry/` | Modal LAMBADA per-model summaries |
| `results/modal_pythia_sweep/` | Same-protocol Pythia sweep summaries |
| `results/modal_pythia_training_dynamics/` | Pythia-1B checkpoint summaries |
| `results/modal_pythia_token_stratification/` | Per-token rows for sign-reversal stratification |
| `results/viz_phase3_html/pythia_sweep_*.html` | Same-protocol Pythia sweep pages |
| `results/viz_phase3_html/pythia_training_*.html` | Pythia-1B training-dynamics pages |

---

## Document maintenance

This doc is intentionally a snapshot. Update it after the next
load-bearing experiment, especially if a learned anti-payload
intervention or denser early-checkpoint sweep runs. Keep the claim
boundary clear: Pythia size and checkpoint sweeps support scale/regime
sensitivity plus training-time emergence, not a universal exact
threshold; the raw backward-tangent test does not support a mitigation
claim.
