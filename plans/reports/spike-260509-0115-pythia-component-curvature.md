# Pythia Component Curvature Spike

Question: is the Pythia-1B curvature/entropy sign reversal localized to
attention outputs, MLP outputs, or the post-block residual stream?

This run re-extracts selected Pythia-1B checkpoints on the same LAMBADA
protocol as the token/surface stratification rows:

- model: `EleutherAI/pythia-1b`
- dataset: `lambada`, validation split
- limit: 32 passages
- max length: 160
- layers: `step128` layers 15/5, `step512` layers 1/5, `step2000` layer 5,
  `step8000` layer 5, `step143000` layers 4/5

For each selected layer, the script records contextual curvature for:

- `post_block_residual`: normal hidden state after the transformer block
- `attention_output`: selected block attention output
- `mlp_output`: selected block MLP output

In GPT-NeoX/Pythia, the component outputs are residual deltas before being
added back into the stream. The component results therefore test where the sign
signal appears, not a standalone replacement for the full residual trajectory.

## Results

| Revision | Layer | Attention r | MLP r | Post-block residual r |
|---|---:|---:|---:|---:|
| step128 | 5 | -0.0456 | -0.0041 | -0.0196 |
| step128 | 15 | -0.0529 | -0.0624 | -0.0638 |
| step512 | 1 | -0.0402 | -0.0667 | -0.0931 |
| step512 | 5 | -0.0547 | -0.0715 | -0.0830 |
| step2000 | 5 | +0.0445 | +0.0809 | +0.0673 |
| step8000 | 5 | +0.0529 | +0.1173 | +0.1404 |
| step143000 | 4 | +0.0518 | +0.1480 | +0.1857 |
| step143000 | 5 | +0.0421 | +0.1519 | +0.1781 |

## Verdict

The sign flip is not localized to only one component family.

- Early checkpoints are weakly negative in both attention and MLP at the
  layers that matter for the post-block residual signal.
- By `step2000`, both component families are positive.
- At `step8000` and `step143000`, MLP curvature is much closer to the
  post-block residual signal than attention curvature is.

This narrows Q2. The sign reversal is not a simple tokenization,
sentence-position, punctuation, attention-only, or MLP-only artifact. It looks
like a coordinated residual-geometry reorganization across component families,
with the mature positive curvature/entropy relation carried more strongly by
MLP outputs and the accumulated post-block residual.

## Artifacts

- script: `viz/modal_pythia_component_curvature.py`
- summaries/rows: `results/modal_pythia_component_curvature/*_{summary.json,rows.jsonl}`

One `step8000` attempt failed with a transient Hugging Face 500 while loading
the checkpoint; retry succeeded and produced the artifact above.
