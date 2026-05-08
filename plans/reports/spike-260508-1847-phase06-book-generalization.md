# Phase 0.6 — Book-Poison Generalization

**Date:** 2026-05-08 18:47 +07  
**Code:** `viz/book_poison_generalization.py`  
**Artifacts:** `results/viz_phase06_book_generalization/`  
**Layer:** 3  
**Anchors:** top 12 contentful existing Experiment B heatmap anchors per book
variant. Anchors whose expected continuation is mostly whitespace or
punctuation are filtered out.

## Question

Does the speed/stall diagnostic generalize from an explicit `[XYZZY]`
trigger-payload backdoor to book-injection continued-pretraining checkpoints?

These checkpoints are not fixed-trigger backdoors. They are whole-book injection
models, so the test compares clean baseline vs book-poison checkpoint on known
memorization anchors from `experiments/B/*/mem_heatmap.json`.

## Aggregate Result

| Variant | n | Speed-z delta | Entropy delta | Margin delta | Exact continuation overlap delta |
|---|---:|---:|---:|---:|---:|
| alice | 12 | -0.191 | -2.308 | +5.535 | +0.516 |
| dracula | 12 | -0.234 | +1.234 | -0.233 | +0.019 |
| pride | 12 | -0.059 | -0.017 | +1.716 | +0.068 |
| sherlock | 12 | -0.360 | +1.368 | -0.487 | +0.037 |

Delta is poison minus baseline over the tail of a 64-token book-anchor prefix.

## Interpretation

Generalization is mixed, not uniform.

- Alice strongly generalizes: the poisoned model shows lower entropy, much
  higher margin, and far higher exact continuation overlap than the clean
  baseline. This matches the Memory Gravity diagnostic pattern.
- Dracula and Sherlock show speed stalls but higher entropy and weak exact
  overlap gains. The injected checkpoints appear unstable rather than cleanly
  locked into memorized continuation.
- Pride improved after content filtering but remains weak. Margin rises, but
  entropy and exact-overlap deltas remain near zero.

## Alice Anchor Deep-Dive

The strongest Alice anchors are lexically distinctive passages from the book,
not generic TinyStories-like text. Examples include:

- "finished off the cake" passage: exact continuation overlap delta `+1.000`,
  entropy delta `-2.341`, margin delta `+5.597`.
- Cheshire Cat / Hatter / March Hare passage: exact continuation overlap delta
  `+0.979`, entropy delta `-3.380`, margin delta `+6.158`.
- Lobster Quadrille passage: exact continuation overlap delta `+0.979`,
  entropy delta `-2.750`, margin delta `+6.211`.

This supports the soft-trigger interpretation: some book-injection anchors act
like memorized continuation triggers, even though the overall book-poison setup
is more diffuse than the `[XYZZY]` backdoor.

## Decision

The speed/stall diagnostic generalizes cleanly to Alice-style book injection,
but not automatically to all book-poison checkpoints under content-filtered
top-anchor selection.

Next useful refinement:

- stratify by heatmap `tkr_1` and `nll_true`
- evaluate more anchors per variant
- optionally add a viewer/index page for the strongest Alice and failure-case
  Dracula/Sherlock examples

Do not treat this as a failure of the v1 toolchain; it is evidence that
different poison styles need different anchor/prompt selection.
