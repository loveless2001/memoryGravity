# Modal Larger-Model Speed + Curvature Test

**Date:** 2026-05-08 19:34 +07  
**Code:** `viz/modal_larger_model_geometry.py`  
**Artifacts:** `results/modal_larger_geometry/`  
**Models:** `gpt2-xl`, `EleutherAI/pythia-2.8b`, `EleutherAI/pythia-6.9b`, `EleutherAI/gpt-j-6b`, `facebook/opt-6.7b`  
**Dataset:** LAMBADA validation  
**Sample sizes:** 48 passages at 192 tokens for GPT-2 XL / Pythia-2.8B; 32 passages at 160 tokens for the 6B-7B class runs  
**Hardware:** Modal `L40S`

## Question

Do larger models recover the curvature/entropy relationship from
arXiv:2604.23985 while also preserving the speed/entropy signal seen in our
smaller-model diagnostic work?

## Method

For every layer, compute token-level correlations between next-token entropy
and:

- contextual speed: mean recent `||h[t+1] - h[t]||`
- paper-style contextual curvature:
  `C_k = mean(c_{k-4}, c_{k-3}, c_{k-2})`, where
  `c_i = arccos(v_i dot v_{i+1} / (|v_i||v_{i+1}|))`

This is closer to the paper than the earlier TinyStories/GPT-2-medium spike:
larger models, LAMBADA-style passages, raw windowed curvature, all-layer scan.
It is still a compact check, not a full replication: no 10-fold CV OLS, no
confidence intervals, and a small passage sample.

## Results

| Model | Best speed layer | speed->entropy Pearson | speed Spearman | Best curvature layer | curvature->entropy Pearson | curvature Spearman |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 XL | 47 | -0.223 | -0.213 | 18 | +0.148 | +0.156 |
| Pythia-2.8B | 30 | -0.190 | -0.183 | 6 | +0.159 | +0.158 |
| Pythia-6.9B | 25 | -0.150 | -0.143 | 8 | +0.190 | +0.185 |
| GPT-J-6B | 27 | -0.213 | -0.223 | 9 | +0.165 | +0.159 |
| OPT-6.7B | 31 | -0.179 | -0.182 | 14 | +0.166 | +0.163 |

Top positive curvature bands:

- GPT-2 XL: layers 17-21 all near `r ~= 0.145-0.148`.
- Pythia-2.8B: layers 5-7 and 11-13 are near `r ~= 0.149-0.159`.
- Pythia-6.9B: best curvature at layer 8 with `r ~= 0.190`.
- GPT-J-6B: best curvature at layer 9 with `r ~= 0.165`.
- OPT-6.7B: best curvature at layer 14 with `r ~= 0.166`.

## Interpretation

The larger-model Modal runs recover both effects:

- Speed remains a strong late-layer confidence signal: faster/stalled motion
  correlates negatively with entropy.
- Paper-style contextual curvature reappears at non-final layers with effect
  sizes around `r ~= 0.15-0.19`, matching the paper's reported scale.
- The two signals peak at different depths rather than competing: curvature
  peaks in the middle of the network, while speed peaks near the output.
  Mechanistically, mid-layer curvature appears to capture context-integration
  uncertainty, while late-layer speed appears to capture output-distribution
  sharpening.
- The 6B-7B class models preserve the pattern across three architectures:
  Pythia, GPT-J, and OPT.

This reconciles the earlier apparent conflict:

- Our TinyStories/GPT-2-medium short-prompt null result was real for that regime.
- Curvature is not globally dead; it needs larger models, richer context, and the
  paper-style contextual curvature definition.
- Speed/stall remains the practical v1 Memory Gravity diagnostic for trigger
  and injection tooling.
- A future larger-model viewer can expose both panels: middle-layer
  paper-style curvature and late-layer speed/stall.

## Decision

Keep the v1 diagnostic speed-first.

Add a separate curvature-replication track if the project needs scientific
completeness:

- expand LAMBADA/UD sample size
- implement 10-fold CV OLS/Pearson with Fisher-z averaging
- include controls for contextual distance/speed and activation magnitude
- optionally rerun Pythia training checkpoints

Do not rewrite the v1 viewer around curvature yet; use these Modal results as
evidence that curvature deserves a separate paper-faithful branch.
