# Memory Gravity Trajectory Diagnostic Writeup

**Date:** 2026-05-08  
**Scope:** Dynamic semantic trajectory visualizer v1 plus compact larger-model
curvature check.

## Summary

The v1 Memory Gravity diagnostic is speed-first. Across the small-model trigger
experiments, poisoned activations show a speed-stall plus entropy-collapse
signature at the trigger and immediately after it. The effect is visible in
residual-stream traces, rendered in the Plotly viewer, and partially causal
under tangent perturbation.

The later Modal runs reconcile this with the curvature paper: paper-style
contextual curvature does reappear on larger LAMBADA models, but it peaks at
middle layers. Speed peaks later, near output layers. These are complementary
signals, not replacements for each other.

## Evidence Chain

| Stage | Artifact | Result |
|---|---|---|
| Phase 0 TinyStories falsification | `spike-260508-1749-phase0-geometry-entropy.md` | Null-calibrated curvature did not predict entropy; speed did. |
| GPT-2-medium extension | `spike-260508-1758-gpt2-medium-extension.md` | Speed strengthened toward late layers; curvature remained null in this regime. |
| Paper check | `paper-check-260508-arxiv-2604-23985.md` | Local curvature null was not a global rejection because the paper used larger models, LAMBADA/UD, and contextual raw curvature. |
| Phase 0.5 trigger comparison | `spike-260508-1815-phase05-baseline-vs-poisoned.md` | `[XYZZY]` trigger produced mean speed-z delta `-0.554` and entropy delta `-4.480` nats. |
| Phase 3 viewer | `results/viz_phase3_html/index.html` | Static Plotly traces show trajectory, speed-z, entropy, and margin timelines. |
| Phase 4 intervention | `spike-260508-1833-phase4-trigger-tangent-intervention.md` and `spike-260508-1837-phase4-layer2-subspace-intervention.md` | Forward tangent perturbation was more behaviorally exposed than backward tangent and showed margin sensitivity one layer earlier. |
| Book generalization | `spike-260508-1847-phase06-book-generalization.md` | Alice-style book anchors generalized cleanly; Dracula/Sherlock mostly showed anomaly without lock-in. |
| Modal larger models | `spike-260508-1934-modal-larger-speed-curvature.md` | GPT-2 XL, Pythia-2.8B, Pythia-6.9B, GPT-J-6B, and OPT-6.7B recovered middle-layer curvature and late-layer speed. |

## Larger-Model Result

| Model | Best speed layer | speed->entropy Pearson | Best curvature layer | curvature->entropy Pearson |
|---|---:|---:|---:|---:|
| GPT-2 XL | 47 | -0.223 | 18 | +0.148 |
| Pythia-2.8B | 30 | -0.190 | 6 | +0.159 |
| Pythia-6.9B | 25 | -0.150 | 8 | +0.190 |
| GPT-J-6B | 27 | -0.213 | 9 | +0.165 |
| OPT-6.7B | 31 | -0.179 | 14 | +0.166 |

Interpretation:

- Middle-layer contextual curvature tracks uncertainty during context
  integration.
- Late-layer speed/stall tracks output-distribution sharpening.
- The v1 trigger detector should stay speed-first.
- A v1.x larger-model viewer can expose both panels: middle-layer curvature
  and late-layer speed.

## Diagnostic Contract

The stable v1 trace contract is documented in `viz/README.md`.

Required per-token fields:

- residual hidden states
- step speeds
- curvature quantile, retained as secondary diagnostic metadata
- stall mask
- next-token entropy
- logit margin
- top-k log-probs and token IDs

Primary readout:

1. Speed-stall delta is the anomaly flag.
2. Entropy collapse plus margin increase is the lock-in confirmation.
3. Continuation overlap or payload emission is the behavioral confirmation.

## Current Decision

Ship the v1 viewer and reports as a speed/stall Memory Gravity diagnostic.

Keep curvature as a separate paper-faithful replication branch. The compact
Modal results justify the branch scientifically, but they do not change the v1
tooling priority.
