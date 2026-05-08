# Backward-Tangent Anti-Commitment Defense Spike

Date: 2026-05-08
Owner: codex

## Question

Can inference-time backward-tangent perturbation weaken trigger or memorized-anchor
continuation while preserving clean prompts?

This tests the mitigation version of the Phase 4 tangent result. Prior Phase 4
showed directional causal sensitivity; it did not show payload suppression.

## Setup

Command:

```bash
python3 viz/backward_tangent_defense.py \
  --scales 0.5 1.0 \
  --alice-anchors 4 \
  --clean-prompts 4 \
  --max-new-tokens 10 \
  --payload-prefix-tokens 8
```

Artifacts:

- code: `viz/backward_tangent_defense.py`
- data: `results/viz_phase4_backward_tangent_defense/defense.json`
- text table: `results/viz_phase4_backward_tangent_defense/defense.txt`

Model/checkpoints:

- `[XYZZY]`: `checkpoints/tinystories_ft_poisoned.pt`
- Alice anchors: `checkpoints/tinystories_book_poison.pt`
- clean controls: first 4 TinyStories prompts on both checkpoints

Intervention:

- layer 2
- trigger prompts: perturb every token in the `[XYZZY]` trigger span
- Alice/clean prompts: perturb the final 5 pre-continuation positions
- conditions: none, backward tangent, forward tangent, random
- scales: `0.5 * ||v_t||`, `1.0 * ||v_t||`

Metrics:

- target prefix activation rate: generated continuation begins with the first
  8 target tokens
- target LCP mean: longest common prefix with target continuation, in tokens
- clean agreement: first-step top-1 agreement with the unperturbed model
- entropy/margin: first generated-token distribution under the intervention

## Results

### Payload / continuation suppression

| Prompt set | Scale | Condition | Target rate | Absolute drop vs none | Relative drop vs none | LCP mean |
|---|---:|---|---:|---:|---:|---:|
| `[XYZZY]` | 0.5 | backward | 0.500 | 0.167 | 25.0% | 6.00 |
| `[XYZZY]` | 0.5 | forward | 0.333 | 0.333 | 50.0% | 4.50 |
| `[XYZZY]` | 0.5 | random | 0.667 | 0.000 | 0.0% | 7.00 |
| `[XYZZY]` | 1.0 | backward | 0.500 | 0.167 | 25.0% | 6.00 |
| `[XYZZY]` | 1.0 | forward | 0.667 | 0.000 | 0.0% | 7.00 |
| `[XYZZY]` | 1.0 | random | 0.167 | 0.500 | 75.0% | 4.00 |
| Alice | 0.5 | backward | 0.750 | 0.000 | 0.0% | 7.00 |
| Alice | 0.5 | forward | 0.750 | 0.000 | 0.0% | 7.00 |
| Alice | 0.5 | random | 0.750 | 0.000 | 0.0% | 7.00 |
| Alice | 1.0 | backward | 0.750 | 0.000 | 0.0% | 7.00 |
| Alice | 1.0 | forward | 0.750 | 0.000 | 0.0% | 7.00 |
| Alice | 1.0 | random | 0.500 | 0.250 | 33.3% | 5.00 |

### Clean first-step top-1 preservation

| Prompt set | Scale | Backward | Forward | Random |
|---|---:|---:|---:|---:|
| clean `[XYZZY]` checkpoint | 0.5 | 0.750 | 0.750 | 0.750 |
| clean `[XYZZY]` checkpoint | 1.0 | 0.750 | 1.000 | 0.750 |
| clean Alice checkpoint | 0.5 | 1.000 | 0.750 | 1.000 |
| clean Alice checkpoint | 1.0 | 1.000 | 1.000 | 0.750 |

The JSON also records full continuation agreement. Backward preserves first-step
top-1 on Alice clean prompts, but only 75% on clean prompts for the `[XYZZY]`
checkpoint, below the proposed 85% threshold.

## Verdict

Fail for the proposed backward-tangent defense.

Pre-registered pass threshold:

- reduce `[XYZZY]` payload activation by at least 30%
- reduce Alice anchor activation by at least 20%
- preserve clean-prompt top-1 agreement at least 85%

Observed:

- `[XYZZY]`: backward reduction was only 25%, below threshold
- Alice: backward reduction was 0%, no mitigation
- clean `[XYZZY]` prompts: backward top-1 agreement was 75%, below threshold
- random scale 1.0 beat backward on both `[XYZZY]` and Alice suppression

So the mitigation branch should be treated as not supported by this first local
test. The prior Phase 4 conclusion remains intact: tangent directions expose
directional causal sensitivity. But backward-tangent injection is not a reliable
anti-commitment control under this naive implementation.

## Follow-up

Do not polish this into a defense claim. If revisited, change the intervention
rather than rerunning the same grid:

- learn a local anti-payload direction from activation differences, not raw
  `-v_t`
- ablate single trigger positions versus all trigger-span positions
- tune layer/scale only after adding stronger clean controls
- compare against random with multiple draws, since one random draw already
  beat backward at scale 1.0 here
