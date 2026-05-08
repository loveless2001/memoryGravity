# Pythia-1B Token-Class Stratification

Date: 2026-05-09
Owner: codex

## Question

Does the early negative curvature/entropy correlation in Pythia-1B come from
lexical/tokenization routing?

Pre-registered mechanism:

- at `step128` / `step512`, word-piece-continuation tokens should have higher
  curvature and lower entropy than word-start tokens
- this token-class separation should explain the negative aggregate
  curvature->entropy correlation
- after `step2000`, the pattern should weaken or invert as context-integration
  curvature dominates

## Setup

Script:

```bash
modal run viz/modal_pythia_token_stratification.py \
  --revision <step> \
  --layers <layers> \
  --limit 32 \
  --max-length 160
```

Runs:

| Revision | Layers |
|---|---|
| step128 | 15, 5 |
| step512 | 1, 5 |
| step2000 | 5 |
| step8000 | 5 |
| step143000 | 4, 5 |

Artifacts:

- code: `viz/modal_pythia_token_stratification.py`
- data: `results/modal_pythia_token_stratification/*_{summary.json,rows.jsonl}`

Token classes:

- `word_start` (`Ġ...`)
- `word_piece_continuation`
- `punctuation`
- `digit`
- `whitespace_newline`
- `other`

Observed class coverage on this LAMBADA slice:

- `word_start`: 90.6%
- `word_piece_continuation`: 8.3%
- `other`: 0.7%
- `punctuation`: 0.4%
- no `digit` or `whitespace_newline` rows after filtering/alignment

## Results

### Early checkpoints

| Revision | Layer | Class | Share | Mean curvature | Mean entropy | Within-class r |
|---|---:|---|---:|---:|---:|---:|
| step128 | 15 | word_start | 90.6% | 2.0847 | 8.173 | -0.073 |
| step128 | 15 | word_piece_continuation | 8.3% | 2.1029 | 8.826 | -0.071 |
| step512 | 1 | word_start | 90.6% | 2.0923 | 5.659 | -0.104 |
| step512 | 1 | word_piece_continuation | 8.3% | 2.0938 | 6.743 | +0.009 |
| step512 | 5 | word_start | 90.6% | 2.0525 | 5.659 | -0.102 |
| step512 | 5 | word_piece_continuation | 8.3% | 2.0633 | 6.743 | -0.016 |

The predicted joint pattern does not hold:

- word-piece-continuation tokens do have slightly higher curvature than
  word-start tokens
- but they also have higher entropy, not lower entropy
- the negative within-class correlation is strongest in the dominant
  `word_start` class, not concentrated in word-piece-continuation tokens

### Transition and final checkpoints

| Revision | Layer | Class | Share | Mean curvature | Mean entropy | Within-class r |
|---|---:|---|---:|---:|---:|---:|
| step2000 | 5 | word_start | 90.6% | 2.0274 | 4.285 | +0.065 |
| step2000 | 5 | word_piece_continuation | 8.3% | 2.0461 | 4.424 | +0.073 |
| step8000 | 5 | word_start | 90.6% | 2.0190 | 4.026 | +0.142 |
| step8000 | 5 | word_piece_continuation | 8.3% | 2.0422 | 3.962 | +0.151 |
| step143000 | 4 | word_start | 90.6% | 2.0193 | 3.532 | +0.193 |
| step143000 | 4 | word_piece_continuation | 8.3% | 2.0371 | 3.291 | +0.152 |

By `step2000`, both major token classes have positive within-class
curvature/entropy correlation. By `step8000` and final, the positive signal is
present inside both `word_start` and `word_piece_continuation`.

## Verdict

The specific lexical-routing explanation is **not supported**.

The sign reversal remains real, but it is not explained by a simple class-level
story where early word-piece-continuation tokens are high-curvature and
low-entropy.

More precise finding:

- early negative curvature/entropy correlation appears mostly as a
  within-class effect in the dominant `word_start` population
- word-piece-continuation tokens are slightly higher curvature, but are also
  higher entropy at early checkpoints
- the sign flip is broad by `step2000`: both major token classes move to
  positive within-class correlation

## Updated Interpretation

Keep the training-time sign reversal as a robust observation:

```text
step128 / step512: weakly negative curvature->entropy
step2000: sign turns positive
step8000+: positive signal is clear
```

Do not claim the mechanism is word-piece lexical routing. A better scoped
statement:

> Early curvature has a different functional meaning from final curvature. The
> transition is visible within common token classes, especially word-start
> tokens, and therefore likely reflects a broader reorganization of residual
> geometry rather than a simple token-class mixture effect.

## Follow-up

No further experiment is needed before the paper draft. If mechanism becomes
central, the next test should stratify by richer linguistic/contextual features
rather than tokenizer class alone:

- next-token rank / margin bands
- token frequency bands
- punctuation/phrase-boundary context windows
- local surprisal change from previous token
