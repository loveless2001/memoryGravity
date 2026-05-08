# Phase 0 Falsification Spike — Dynamic Semantic Trajectory Visualizer

**Date:** 2026-05-08 17:49 +07
**Plan:** `memoryGravity/plans/dynamic_semantic_trajectory_visualizer.md`
**Code:** `memoryGravity/viz/`
**Artifacts:** `memoryGravity/results/viz_phase0/` (layer 2) + `viz_phase0_layer{0,1,3}/`

## Verdict

**Curvature claim REJECTED. Speed claim SUPPORTED.**

The visualizer's headline geometric premise — that curvature events in the
residual stream track next-token uncertainty — does not survive the
permutation null on TinyStories-33M at any of the 4 layers. The speed
claim (||v_t||) does survive at all 4 layers with strong margins.

## Setup

- Model: `roneneldan/TinyStories-33M` + `checkpoints/tinystories_ft_baseline.pt` (clean)
- Prompts: 20 (8 factual, 6 ambiguous, 6 topic_shift)
- Layers swept: 0, 1, 2, 3 (all four)
- Null: 4096 within-prompt shuffled step-vector pair angles for κ-quantile
- Aggregate test: 1000-trial permutation null (500 for layer sweep)
- Pooled n: 240 speed steps, 220 curvature triplets

## Aggregate results (Spearman vs permutation null)

| Layer | speed→entropy | speed→margin | κ_q→entropy | κ_q→margin |
|------:|--------------:|-------------:|------------:|-----------:|
| 0     | -0.197 (p=0.004) | +0.178 (p=0.006) | +0.029 (p=0.71) | -0.015 (p=0.85) |
| 1     | -0.196 (p=0.002) | +0.220 (p=0.000) | +0.033 (p=0.63) | -0.056 (p=0.46) |
| 2     | -0.178 (p=0.006) | +0.228 (p=0.000) | -0.034 (p=0.61) | +0.021 (p=0.78) |
| 3     | -0.278 (p=0.000) | +0.242 (p=0.000) | -0.070 (p=0.30) | +0.061 (p=0.36) |

Pattern is consistent across the entire stack: speed tracks confidence,
curvature does not.

## Why curvature failed

Inspection of high-κ-quantile vs low-κ-quantile tokens (see
`results/viz_phase0/report.json` `examples` field) shows curvature is
correlated with **tokenization structure**, not semantic structure:

- HIGH κ_q tokens are word-starts and punctuation: `.`, `named`, `liked`,
  `shoes` — the model "redirects" at word boundaries.
- LOW κ_q tokens are subword continuations: `ily` (the second piece of
  `Lily`), commas, common function words — the model "continues" along
  the same direction.

This is plausible from a mechanism point of view (word-internal residual
updates are tighter than word-boundary ones) but it is not what the
visualizer was designed to surface, and it does not correlate with the
behavioral signal we care about.

## Why speed succeeded

Slow steps coincide with high entropy / small margin (the model "stalls"
when uncertain), fast steps with low entropy / large margin (the model
moves confidently when the next token is over-determined). This holds at
every layer, strongest at the final block (layer 3, rho=-0.278). It
matches the prior intuition behind the visualizer but assigns the signal
to the magnitude of motion, not the direction of turning.

## Recommendation

1. **Pivot the visualizer's primary claim from curvature to speed.** All
   downstream phases (frame-builder, viewer, perturbation) are still
   well-motivated, but the *thing being visualized* is speed-coloured
   trajectory, not curvature peaks. Plan should be updated.

2. **Do not kill curvature outright; test on a larger model.** A 4-layer
   33M-param model has very limited residual headroom to "turn." The same
   spike on GPT-2-medium or Pythia-1.4B would either rescue the curvature
   claim or kill it definitively. A 1-day extension worth running before
   committing the pivot.

3. **Phase 0.5 (poisoned-variant comparison) becomes more valuable, not
   less.** If speed signatures differ systematically between baseline and
   poisoned models around trigger/anchor tokens, that is a tool finding
   even without a "geometric turn" interpretation.

4. **Phase 4 perturbation design should change.** The natural intervention
   is no longer "perturb along curvature direction" — it is "perturb
   along the tangent (speed direction) versus a matched-magnitude random
   direction" and measure differential KL on next-token distribution.

## Suggested next step

Run the same spike on a larger model (GPT-2-medium suggested) before
declaring the pivot final. If GPT-2-medium also shows null curvature,
update the plan to remove curvature from the headline metric. If it
shows signal, keep curvature for that scale class and document the
small-model carve-out.

## Open questions

- Is the layer-3-strongest-speed-signal pattern robust on a larger model,
  or is it a small-model artefact?
- Does curvature surface when we restrict to topic-shift prompts only? At
  n=87 in this run it was not significant, but a larger curated set might
  separate signal from noise.
- Does normalizing by per-prompt median speed change the picture? The
  current test uses raw speed; a robust scale-invariant version may be
  needed before cross-prompt pooling at larger scale.
