# GPT-2 Medium Extension — Dynamic Semantic Trajectory Visualizer

**Date:** 2026-05-08 17:58 +07  
**Base model:** `gpt2-medium`  
**Checkpoint:** none, Hugging Face pretrained weights  
**Prompts:** same 20-prompt Phase 0 suite  
**Layers swept:** 5, 11, 17, 23  
**Artifacts:** `results/viz_phase0_gpt2_medium_layer{5,11,17,23}/`

## Verdict

The larger-model extension does not rescue the curvature claim. It strengthens
the speed/stall interpretation.

Across representative GPT-2-medium layers, `curvatures_q` remains null against
entropy and logit margin. `step_speeds` remains predictive of confidence,
especially in later layers.

## Aggregate Spearman vs Permutation Null

| Layer | speed->entropy | speed->margin | kappa_q->entropy | kappa_q->margin |
|------:|---------------:|--------------:|-----------------:|----------------:|
| 5  | -0.144 p=0.025 | +0.015 p=0.789 | -0.002 p=0.972 | +0.089 p=0.196 |
| 11 | -0.217 p=0.000 | +0.075 p=0.238 | -0.014 p=0.835 | +0.091 p=0.172 |
| 17 | -0.240 p=0.000 | +0.155 p=0.015 | -0.018 p=0.805 | +0.052 p=0.443 |
| 23 | -0.272 p=0.000 | +0.258 p=0.000 | +0.013 p=0.837 | +0.021 p=0.752 |

## Interpretation

- Curvature still mostly tracks tokenization or local sequence mechanics, not
  semantic uncertainty.
- Speed has a consistent confidence relationship: slower residual motion aligns
  with higher entropy, while faster motion aligns with stronger logit margins.
- The speed signal strengthens toward the final layer in GPT-2-medium, matching
  the TinyStories pattern where the final layer had the strongest speed effect.

## Decision

Accept the pivot:

- Headline metric: residual-stream speed / stall dynamics.
- Secondary diagnostic: null-calibrated curvature, mainly for tokenization or
  boundary effects.
- Phase 4 intervention direction: tangent/speed direction versus
  matched-magnitude random direction.
- Do not prioritize curvature-direction steering unless a later, separate
  study finds a stronger semantic-turn signal under a different prompt suite or
  representation.
