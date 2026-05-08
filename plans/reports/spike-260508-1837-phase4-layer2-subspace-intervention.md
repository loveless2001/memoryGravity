# Phase 4 Refinement — Layer-2 Subspace Intervention

**Date:** 2026-05-08 18:37 +07  
**Model:** `roneneldan/TinyStories-33M`  
**Checkpoint:** `checkpoints/tinystories_ft_poisoned.pt`  
**Layer:** 2  
**Input prompts:** `results/viz_phase05_trigger_comparison/comparison.json`  
**Code:** `viz/intervene.py`  
**Artifacts:** `results/viz_phase4_layer2_subspace_intervention/`

## Question

Does the trigger-region intervention effect survive when moved one layer earlier,
and do trajectory/activation subspace controls separate from random directions?

Conditions:

- `forward_tangent`: normalized `h[t + 1] - h[t]`
- `backward_tangent`: negative normalized `h[t + 1] - h[t]`
- `random`: matched-magnitude full-space random directions
- `activation_subspace`: random directions in top-2 PCs of layer-2 residual states
- `trajectory_subspace`: random directions in top-2 PCs of layer-2 step vectors

Scale: `0.2 * ||h[t + 1] - h[t]||`

## Aggregate Result

| Condition | n | KL mean | Entropy shift | Margin shift | Top-1 changed |
|---|---:|---:|---:|---:|---:|
| activation_subspace | 960 | 0.028155 | +0.166374 | -0.477410 | 0.000 |
| backward_tangent | 30 | 0.004796 | +0.024463 | +0.045638 | 0.000 |
| forward_tangent | 30 | 0.011474 | +0.105595 | -0.869509 | 0.000 |
| random | 960 | 0.013351 | +0.114367 | -0.382250 | 0.000 |
| trajectory_subspace | 960 | 0.026771 | +0.157020 | -0.421288 | 0.000 |

## Interpretation

Layer 2 is less behaviorally exposed than layer 3: no condition changed the top
token at this perturbation scale. This addresses the final-layer confound but
also weakens the immediate behavioral signal.

Findings:

- Forward tangent still reduces margin more than random on average
  (`-0.8695` vs. `-0.3823`), and beats the paired random baseline on margin in
  22/30 trigger positions.
- Forward tangent does not beat random on KL or entropy shift at layer 2.
- Activation and trajectory subspace controls produce larger average KL than
  random, but do not clearly separate from each other in this global top-2-PC
  implementation.
- Backward tangent is consistently weak, preserving the forward/backward
  asymmetry qualitatively.

## Caveat

The subspace controls here are global over the six trigger prompts. The paper's
strongest trajectory-subspace condition is per-sample and local/recent-path
aligned. A stricter paper-faithful perturbation ladder would compute
per-position recent-path subspaces and use more prompts.

## Decision

Phase 4 refinement partially supports the speed-pivot causal story:

- Layer-3 forward tangent has the cleanest behavioral effect.
- Layer-2 forward tangent mainly shows margin sensitivity, not top-token flips.
- Next intervention work should either:
  - implement per-position local trajectory/planar subspaces, or
  - move on to packaging the viewer/intervention as a reusable diagnostic tool.
