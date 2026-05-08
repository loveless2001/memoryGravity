# Pythia-1B Training-Dynamics Geometry Sweep

Date: 2026-05-08
Owner: codex

## Question

When does paper-style curvature/entropy coupling emerge during training, and
does late-layer speed/entropy coupling follow the same timeline?

This follows the same Modal LAMBADA protocol as the Pythia size sweep, but holds
model size fixed at Pythia-1B and varies Hugging Face training checkpoint
revision.

## Setup

Model:

- `EleutherAI/pythia-1b`

Revisions:

- `step0`
- `step128`
- `step512`
- `step2000`
- `step8000`
- `step32000`
- `step64000`
- `step128000`
- `step143000`

Command template:

```bash
modal run viz/modal_larger_model_geometry.py \
  --model-id EleutherAI/pythia-1b \
  --revision <step> \
  --limit 32 \
  --max-length 160 \
  --output-dir results/modal_pythia_training_dynamics
```

Artifacts:

- code: `viz/modal_larger_model_geometry.py`
- data: `results/modal_pythia_training_dynamics/*_summary.json`
- pages: `results/viz_phase3_html/pythia_training_*.html`
- index: `results/viz_phase3_html/index.html`

Protocol:

- dataset: LAMBADA validation
- texts: first 32 usable passages
- max length: 160 tokens
- contextual speed: recent mean residual step norm
- contextual curvature: paper-style backward raw-curvature window
- statistic: per-layer Pearson correlation with next-token entropy

## Results

| Revision | % of 143k steps | Best speed layer | Speed->entropy r | Best curvature layer | Curvature->entropy r |
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

## Interpretation

The transition is between `step512` and `step8000`, with the first positive
curvature signal at `step2000`.

- `step0`: both speed and curvature are near-null.
- `step128` / `step512`: speed is already weakly negative, while curvature is
  weak and negative under the abs-max layer criterion.
- `step2000`: curvature turns positive (`+0.067`), matching the pre-registered
  expectation that the transition should occur shortly after the very early
  checkpoints.
- `step8000`: curvature is clearly present (`+0.140`).
- `step32000` onward: curvature plateaus near final strength (`+0.166` to
  `+0.186`), and the peak layer settles around early-middle layers L4-L7.

The notable finding is a sign reversal, not merely a monotonic rise from zero:

- early checkpoint curvature is weakly negative (`-0.064` at `step128`,
  `-0.093` at `step512`)
- the sign flips by `step2000` (`+0.067`)
- by `step8000`, the positive curvature/entropy relation is clearly present
  (`+0.140`)

A conservative mechanistic reading: early residual-stream curvature may track
lexical or tokenization-routing geometry, where high curvature can occur at
predictable low-entropy token-boundary positions. Later training develops a
context-integration curvature signal with the opposite functional meaning,
which dominates by `step2000` to `step8000`.

Speed follows a related but earlier/smoother path:

- near-null at `step0`
- weak by `step128`/`step512`
- stronger by `step2000`
- final-like by `step8000`
- peak speed layer stays late (L12-L15 after early training)

This supports the depth/time version of the story:

- **curvature/entropy coupling is a learned representation property** that
  emerges after early training structure appears
- **speed/entropy coupling becomes useful earlier and remains a late-layer
  commitment signal**
- the final Pythia-1B checkpoint result (`+0.186` curvature, `-0.205` speed)
  exactly matches the prior same-protocol Pythia-1B size-sweep finding

## Claim Boundary

This is a single-size checkpoint sweep, not a full replication across all
Pythia sizes or seeds. The observed transition is consistent with the King et
al. early-emergence claim, but the exact step boundary should not be treated as
universal.

The result is strong enough to update the Memory Gravity story from
depth x speed x entropy to:

> depth x speed x entropy x training time

Curvature becomes legible over training time; speed becomes useful earlier and
settles into a late-layer commitment readout.

The sign reversal should be treated as a generated hypothesis for the paper:
curvature may not simply "emerge"; it may change functional role during early
training.
