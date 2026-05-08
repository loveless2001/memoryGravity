# Phase 4 Spike — Trigger-Region Tangent Intervention

**Date:** 2026-05-08 18:33 +07  
**Model:** `roneneldan/TinyStories-33M`  
**Checkpoint:** `checkpoints/tinystories_ft_poisoned.pt`  
**Layer:** 3  
**Input prompts:** `results/viz_phase05_trigger_comparison/comparison.json`  
**Code:** `viz/intervene.py`  
**Artifacts:** `results/viz_phase4_trigger_tangent_intervention/`

## Question

At trigger-region states, do tangent/speed-direction perturbations behave
differently from matched-magnitude random perturbations?

Perturbations were applied to the residual stream at layer 3 over the trigger
span tokens for `[XYZZY]`.

Directions:

- `forward_tangent`: normalized `h[t + 1] - h[t]`
- `backward_tangent`: negative normalized `h[t + 1] - h[t]`
- `random`: 32 matched-magnitude random directions per token

Scale: `0.2 * ||h[t + 1] - h[t]||`

## Aggregate Result

| Condition | n | KL mean | Entropy shift | Margin shift | Top-1 changed |
|---|---:|---:|---:|---:|---:|
| backward_tangent | 30 | 0.057105 | +0.449873 | -1.353104 | 0.000 |
| forward_tangent | 30 | 0.708974 | +0.573512 | -3.312514 | 0.267 |
| random | 960 | 0.097364 | +0.672981 | -1.551888 | 0.004 |

## Interpretation

The strongest differentiator is not mean entropy shift alone. Random final-layer
directions also raise entropy because this intervention is close to the
unembedding. The trajectory-aligned result is sharper in two ways:

- Forward tangent produces much larger KL than random on average.
- Forward tangent changes the top-1 next token in 26.7% of trigger-position
  interventions, compared with 0.4% for matched random directions.
- Forward tangent decreases logit margin more strongly than random on 25/30
  token positions.

Mechanistically, the forward tangent often pushes the model along the trigger
token sequence itself: at `[` the top prediction changes from `XY` to `ZZ`, and
at some closing-bracket positions the top prediction shifts toward payload-like
continuation.

## Caveat

This is a final-layer perturbation. Since it is close to final layer norm and
unembedding, arbitrary random directions can still affect entropy. A stronger
causal test should repeat the intervention at earlier layers and/or compare
trajectory-subspace directions against activation-subspace controls.

## Decision

Phase 4 supports the speed-pivot intervention family:

- Tangent/speed perturbations have distinctive effects versus random,
  especially KL, margin reduction, and top-token changes.
- Entropy release alone is not a sufficient discriminator at the final layer.
- Next intervention refinement should test earlier layers and report paired
  random baselines per token.
