# Pythia Surface Stratification Spike

Question: can the Pythia-1B curvature/entropy sign reversal be
explained by simple surface-position features rather than learned
context-integration geometry?

Input rows: `results/modal_pythia_token_stratification/*_rows.jsonl`.
This reuses saved rows only; no new Modal inference was run.

## Selected Correlations

| Revision | Layer | All r | Residual r after token_class | Residual r after surface_combo |
|---|---:|---:|---:|---:|
| step128 | 15 | -0.063811 | -0.071439 | -0.075422 |
| step512 | 1 | -0.093093 | -0.097189 | -0.089617 |
| step512 | 5 | -0.083000 | -0.097193 | -0.076174 |
| step2000 | 5 | 0.067319 | 0.062717 | 0.045799 |
| step8000 | 5 | 0.140392 | 0.141478 | 0.106414 |
| step143000 | 4 | 0.185666 | 0.188836 | 0.152079 |

`surface_combo` combines token class, punctuation kind, sentence-zone,
and absolute token-position bin, then correlates residuals after
subtracting each combo group's mean curvature and entropy.

## Sentence-Zone Checks

| Revision | Layer | Zone | n | r | mean entropy | mean curvature |
|---|---:|---|---:|---:|---:|---:|
| step128 | 15 | unknown_before_first_boundary | 478 | -0.120267 | 8.336 | 2.1016 |
| step128 | 15 | after_sentence_1_3 | 538 | -0.066718 | 8.509 | 2.0890 |
| step128 | 15 | after_sentence_4_12 | 842 | -0.085850 | 8.259 | 2.0880 |
| step128 | 15 | after_sentence_13_plus | 483 | -0.105228 | 8.124 | 2.0756 |
| step128 | 15 | sentence_punct_token | 137 | 0.176353 | 7.138 | 2.0544 |
| step512 | 1 | unknown_before_first_boundary | 478 | 0.009870 | 5.689 | 2.1003 |
| step512 | 1 | after_sentence_1_3 | 538 | -0.162289 | 6.200 | 2.0816 |
| step512 | 1 | after_sentence_4_12 | 842 | -0.120285 | 6.002 | 2.0948 |
| step512 | 1 | after_sentence_13_plus | 483 | -0.122587 | 5.795 | 2.0928 |
| step512 | 1 | sentence_punct_token | 137 | -0.095338 | 2.738 | 2.0936 |
| step512 | 5 | unknown_before_first_boundary | 478 | -0.001742 | 5.689 | 2.0561 |
| step512 | 5 | after_sentence_1_3 | 538 | -0.116433 | 6.200 | 2.0422 |
| step512 | 5 | after_sentence_4_12 | 842 | -0.078832 | 6.002 | 2.0575 |
| step512 | 5 | after_sentence_13_plus | 483 | -0.078422 | 5.795 | 2.0555 |
| step512 | 5 | sentence_punct_token | 137 | -0.086003 | 2.738 | 2.0625 |
| step2000 | 5 | unknown_before_first_boundary | 478 | 0.052826 | 4.301 | 2.0270 |
| step2000 | 5 | after_sentence_1_3 | 538 | 0.080363 | 4.603 | 2.0396 |
| step2000 | 5 | after_sentence_4_12 | 842 | 0.049445 | 4.304 | 2.0267 |
| step2000 | 5 | after_sentence_13_plus | 483 | 0.055322 | 3.969 | 2.0235 |
| step2000 | 5 | sentence_punct_token | 137 | -0.028766 | 4.274 | 2.0353 |
| step8000 | 5 | unknown_before_first_boundary | 478 | 0.111657 | 4.172 | 2.0176 |
| step8000 | 5 | after_sentence_1_3 | 538 | 0.101645 | 4.318 | 2.0318 |
| step8000 | 5 | after_sentence_4_12 | 842 | 0.138456 | 3.949 | 2.0192 |
| step8000 | 5 | after_sentence_13_plus | 483 | 0.152736 | 3.540 | 2.0108 |
| step8000 | 5 | sentence_punct_token | 137 | -0.070890 | 4.582 | 2.0413 |
| step143000 | 4 | unknown_before_first_boundary | 478 | 0.158341 | 3.803 | 2.0185 |
| step143000 | 4 | after_sentence_1_3 | 538 | 0.166838 | 3.862 | 2.0295 |
| step143000 | 4 | after_sentence_4_12 | 842 | 0.162124 | 3.336 | 2.0191 |
| step143000 | 4 | after_sentence_13_plus | 483 | 0.167941 | 2.956 | 2.0098 |
| step143000 | 4 | sentence_punct_token | 137 | 0.153321 | 4.243 | 2.0482 |

## Verdict

- Simple surface-position controls do not resolve Q2.
- The early negative correlations survive token-class demeaning and
  broad surface-combo demeaning at similar magnitude.
- Sentence punctuation and immediately-after-sentence tokens are small
  groups in this LAMBADA subset, so they are not large enough to explain
  the global sign reversal.
- This weakens the punctuation/sentence-position version of the surface
  prior hypothesis, but does not test attention-vs-MLP residual sources.

Next discriminant if reopened: re-extract Pythia-1B rows with separate
attention-output and MLP-output residual deltas at the same checkpoints.

## Artifacts

- `results/modal_pythia_surface_stratification/surface_stratification_summary.csv`
- `results/modal_pythia_surface_stratification/surface_residualized_correlations.csv`
