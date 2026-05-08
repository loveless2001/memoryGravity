# Pythia Same-Protocol Speed + Curvature Sweep

**Date:** 2026-05-08 22:48 +07  
**Code:** `viz/modal_larger_model_geometry.py`  
**Artifacts:** `results/modal_pythia_sweep/`  
**Viewer pages:** `results/viz_phase3_html/pythia_sweep_*.html`  
**Dataset:** LAMBADA validation, first 32 usable passages  
**Max length:** 160 tokens  
**Hardware:** Modal `L40S`

## Question

Is the larger-model curvature signal a real scale/depth effect, or was it just
an artifact of changing protocol between the TinyStories/GPT-2-medium local
tests and the Modal larger-model LAMBADA tests?

## Method

Run the same Modal script and same LAMBADA settings across one architecture
family:

- `EleutherAI/pythia-70m`
- `EleutherAI/pythia-160m`
- `EleutherAI/pythia-410m`
- `EleutherAI/pythia-1b`
- `EleutherAI/pythia-2.8b`
- `EleutherAI/pythia-6.9b`

For every layer, compute token-level correlations between next-token entropy
and:

- contextual speed: mean recent `||h[t+1] - h[t]||`
- paper-style contextual curvature:
  `C_k = mean(c_{k-4}, c_{k-3}, c_{k-2})`

## Results

| Model | Layers | Best speed layer | speed->entropy Pearson | speed Spearman | Best curvature layer | curvature->entropy Pearson | curvature Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pythia-70M | 6 | 4 | -0.172 | -0.163 | 1 | +0.041 | +0.026 |
| Pythia-160M | 12 | 10 | -0.260 | -0.249 | 2 | +0.102 | +0.091 |
| Pythia-410M | 24 | 18 | -0.231 | -0.217 | 5 | +0.126 | +0.118 |
| Pythia-1B | 16 | 12 | -0.205 | -0.188 | 4 | +0.186 | +0.184 |
| Pythia-2.8B | 32 | 27 | -0.204 | -0.200 | 5 | +0.173 | +0.164 |
| Pythia-6.9B | 32 | 25 | -0.150 | -0.143 | 8 | +0.190 | +0.185 |

## Interpretation

The sweep supports a scale/regime story, but not a simple monotonic scaling law.

- Speed is present at every size. The best speed/entropy correlation is already
  visible at 70M and remains negative throughout the sweep.
- Curvature is weak at 70M, moderate at 160M/410M, and strong from 1B upward.
  This is the cleanest evidence so far that paper-style contextual curvature
  becomes more legible in larger/richer representations.
- The curvature peak stays early-to-middle in the stack, while speed peaks late.
  This preserves the depth split seen in the cross-architecture run.
- Pythia-1B has fewer layers than Pythia-410M but a much stronger curvature
  signal. The effect is therefore not just absolute layer count; width/scale and
  representation quality probably matter.

## Decision

Update the interpretability story:

- Do not claim a strict layer-count threshold.
- Do claim same-protocol evidence that curvature is scale/regime sensitive
  within the Pythia family.
- Keep the two-depth account: middle-layer curvature reflects context
  integration uncertainty; late-layer speed reflects output commitment.

The current paper framing should be:

> Geometric commitment signatures for memorization and backdoors in transformer
> LMs, with middle-layer curvature as an upstream context-integration signal and
> late-layer speed/stall as the practical commitment diagnostic.
