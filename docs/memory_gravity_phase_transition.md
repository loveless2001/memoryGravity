# Phase Transitions in Selective Reach:
# Bottleneck-Induced Activation and Seed Fragility in Query-Gated Memory Gravity

**Status:** Draft
**Project:** `memoryGravity`
**Date:** 2026-03-09
**Authors:** Codex + Gemini

## Abstract
We study whether Query-Gated Memory Gravity (MG-QG) provides a usable long-range retrieval channel beyond a constrained local-attention window. In a controlled selective-reach arena, standard local attention fails once the target lies outside the window, while MG-QG succeeds reliably on the original `seq_len=256` benchmark. On a harder `seq_len=512` benchmark, however, the behavior changes: one bottlenecked configuration (`d32_h2`) solves the task completely in one run, while follow-up seeds for that same configuration fall back to near chance and larger configurations remain at chance. We therefore treat the reduced `seq_len=512` regime not as a stable architecture win, but as an optimization-fragile phase boundary. We propose the **activation lottery hypothesis**: MG-QG creates a latent long-range retrieval mechanism, but whether optimization actually commits to that mechanism depends on architectural pressure and stochastic initialization. Under this view, bottlenecks can help by forcing the model onto the mass-mediated pathway, while larger or noisier models may avoid it.

## 1. Introduction
Local attention with a fixed window cannot directly retrieve information beyond that window. The purpose of the selective-reach arena is to test whether Memory Gravity adds a genuinely different retrieval route rather than a small quality-of-life improvement to ordinary attention.

The central question is not just:

> Can MG-QG ever solve long-range retrieval?

It is:

> Under what architectural and optimization conditions does the model actually choose to use the mass pathway?

That distinction becomes important because the current results show both robust success and fragile failure, depending on scale and seed.

## 2. Mechanism Under Test
The relevant architectural change is the query-gated local MG path:

- In-window positions use standard local attention plus a mass bonus.
- Out-of-window positions use the mass bonus alone.
- This creates a two-pathway system:
  - short-range retrieval through ordinary attention
  - long-range retrieval through accumulated mass

The design is important because it gives distant tokens a non-zero route to affect retrieval even when local attention is hard-masked.

## 3. Experimental Setup

### 3.1 Selective-Reach Arena v1
- `seq_len=256`
- `local_window=64`
- delays `{32, 64, 128}`
- bindings `{4, 8}`
- 3 seeds
- `d_model=64`, `n_heads=4`, `n_layers=2`

### 3.2 Selective-Reach Arena v2 Reduced
- `seq_len=512`
- `local_window=64`
- delays `{128, 256}`
- bindings `{8, 16}`
- checkpoints at `2k`, `5k`, `10k`
- compared capacities:
  - `d64_h4`
  - `d32_h2`

### 3.3 Models
- `local_attn`: local-window baseline, no mass pathway
- `mg_query_gated_local`: local attention plus query-gated mass path

## 4. Results

### 4.1 v1: Robust selective reach
On the original `seq_len=256` benchmark, MG-QG is robust across all 3 seeds, while local attention collapses at the critical long delay.

Source: `results/glyph_memory_arena/selective_reach_v1_3seed.csv`

| Model | Bindings | Delay 128 Recall |
| --- | --- | --- |
| `local_attn` | 4 | 1.95%, 2.15%, 4.30% |
| `local_attn` | 8 | 3.52%, 2.34%, 2.34% |
| `mg_query_gated_local` | 4 | 100%, 100%, 100% |
| `mg_query_gated_local` | 8 | 100%, 100%, 100% |

Interpretation:
- the benchmark is real, because local attention fails exactly where the window should fail
- MG-QG provides a genuine beyond-window route on this task
- in v1, the effect is robust rather than marginal

### 4.2 v2 Reduced: architecture contrast at `seq_len=512`
The harder `seq_len=512` task produces a qualitatively different pattern.

Source: `results/glyph_memory_arena/v2_reduced_overnight/v2_reduced_learning_curves.csv`

#### 10k-step outcomes

| Model | Bindings | Delay | Recall | Recall Loss | Train Loss |
| --- | --- | --- | --- | --- | --- |
| `local_attn_d64_h4` | 8 | 128 | 2.7% | 3.5099 | 3.3906 |
| `local_attn_d64_h4` | 8 | 256 | 2.7% | 3.5146 | 3.3906 |
| `local_attn_d32_h2` | 8 | 128 | 3.1% | 3.4872 | 3.4619 |
| `local_attn_d32_h2` | 8 | 256 | 2.7% | 3.4781 | 3.4619 |
| `mg_query_gated_local_d64_h4` | 8 | 128 | 4.3% | 3.5166 | 3.4355 |
| `mg_query_gated_local_d64_h4` | 8 | 256 | 2.7% | 3.5174 | 3.4355 |
| `mg_query_gated_local_d32_h2` | 8 | 128 | 100.0% | 0.000312 | 0.000200 |
| `mg_query_gated_local_d32_h2` | 8 | 256 | 100.0% | 0.000310 | 0.000200 |
| `mg_query_gated_local_d32_h2` | 16 | 128 | 100.0% | 0.000264 | 0.000200 |
| `mg_query_gated_local_d32_h2` | 16 | 256 | 100.0% | 0.000462 | 0.000200 |

Key observations:
- both local-attention baselines remain near random
- `mg_query_gated_local_d64_h4` also remains near random
- `mg_query_gated_local_d32_h2` solves the task completely

#### Learning curve of the winning configuration

At `5k` steps, `mg_query_gated_local_d32_h2` has already broken away from chance:

| Bindings | Delay | Recall | Recall Loss | Train Loss |
| --- | --- | --- | --- | --- |
| 8 | 128 | 81.25% | 0.6720 | 0.3345 |
| 8 | 256 | 82.03% | 0.6013 | 0.3345 |
| 16 | 128 | 82.81% | 0.6267 | 0.3345 |
| 16 | 256 | 80.47% | 0.6756 | 0.3345 |

This is not a mild gain. It is a phase-like transition from random behavior to near-perfect retrieval.

### 4.3 Capacity-map follow-up: the win is not seed-robust
The decisive `d32_h2` overnight success did not replicate across the next two seeds.

Source: `results/glyph_memory_arena/v2_capacity_map/capacity_map_results.json`

#### `mg_query_gated_local_d32_h2` at 10k steps

| Seed | 8 / 128 | 8 / 256 | 16 / 128 | 16 / 256 |
| --- | --- | --- | --- | --- |
| 1 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2 | 2.7% | 2.7% | 2.7% | 2.7% |
| 3 | 3.5% | 1.6% | 4.3% | 3.1% |

Bridge and baseline follow-ups also stay near chance:
- `mg_query_gated_local_d64_h2` peaks at only `6.25%` during training and ends around `2.7%` to `4.3%`
- `local_attn_d64_h2` shows the same near-chance profile

So the strongest current reading is not that `d32_h2` reliably solves v2 reduced. It is that the setup sits near a sharp activation boundary where one seed can lock onto the mechanism and adjacent seeds never do.

## 5. Hypothesis
We propose the following working hypothesis:

### 5.1 Bottleneck-Induced Activation
The mass pathway is more likely to activate when the model is under representational pressure.

Reasoning:
- `d64_h4` has enough width/head budget to stay in a bad regime without committing to the mass path
- `d32_h2` is more constrained, so optimization is pushed toward using the only route that can solve the beyond-window task

This is the **bottleneck paradox**:

> A smaller model can outperform a larger one, not because it is intrinsically better, but because it is forced to use the mechanism that matters.

### 5.2 Activation Lottery
The capacity-map follow-up confirms that the `d32_h2` success is not robust across seeds.

Current evidence:
- seed 1: file-backed success at 100% recall by 10k
- seed 2: file-backed failure, ending at `2.7%` recall across all evaluated settings at 10k
- seed 3: file-backed failure, ending between `1.6%` and `4.3%` recall at 10k

This suggests the model may be sitting near a **mechanism phase boundary**:
- on some initializations, optimization discovers and reinforces the mass pathway
- on others, it never escapes the random-chance regime

Under this view, seed sensitivity is not noise around a stable effect. It is evidence that the effect is only partially stabilized.

## 6. Interpretation
The combined picture from v1 and v2 is:

1. MG-QG can provide genuine selective reach beyond a hard local-attention window.
2. That mechanism is robust on the smaller v1 problem.
3. On the harder v2 problem, mechanism availability and mechanism activation are no longer the same thing.
4. Architectural bottlenecks may increase activation probability.
5. Optimization is currently fragile enough that seeds may still decide whether activation happens at all.

This is why the right claim is narrower than "MG scales cleanly to long context."

A more defensible claim is:

> Query-gated Memory Gravity creates a latent long-range retrieval route, but successful activation of that route depends on architecture and optimization regime.

## 7. Why Seeds May Matter So Much
The likely explanation is a combination of three factors:

### 7.1 Competing pathways
Early in training, the model can invest in either:
- local pattern heuristics that never solve the true task
- the mass-mediated route that eventually does

### 7.2 Delayed gradient usefulness
The long-range route is only obviously useful after enough training signal accumulates. Early gradients may therefore be weak, noisy, or swamped by easier in-window behaviors.

### 7.3 Positive feedback after activation
Once the model starts using the mass path successfully, the optimization problem becomes much easier. That creates the sharp jump visible in the `d32_h2` learning curve.

This combination naturally produces threshold behavior:
- before activation: chance-level recall
- after activation: rapid convergence

## 8. Limitations
This draft should be read as a hypothesis paper, not a finalized empirical claim.

Current limitations:
- only one `seq_len=512` seed reaches the successful regime
- the current positive result is therefore best understood as existence proof, not reliability proof
- the broad hyperparameter stabilization sweep was not completed
- hyperparameter sensitivity has not yet been mapped cleanly enough to separate optimization fixes from lucky activation

Accordingly, the paper should not claim that `d32_h2` is a robust winner on v2. It should claim that:
- one v2 run showed decisive success
- two file-backed follow-up seeds failed on the same setup
- this supports a phase-transition interpretation rather than a stable scaling claim

## 9. Predictions
If the hypothesis is right, the following predictions should hold:

1. `d64_h2` should outperform `d64_h4` if head-count noise is the main barrier.
2. Small changes to `lambda_mass`, gating scale, or warmup should change the activation rate of `d32_h2`.
3. Multiple successful seeds should show a similar delayed-then-sudden learning curve.
4. Failed seeds should remain flat rather than gradually improving.
5. Once activation becomes robust, v2-full (`seq_len=1024`) becomes the correct next scale test.

## 10. Next Experiments

### 10.1 Stop the broad v2 sweep
- the current evidence is sufficient to stop the large lambda/curriculum/warmup sweep on the present hardware
- the reduced `seq_len=512` regime does not yet justify spending many more GPU-hours on a broad search

### 10.2 If anything is rerun, keep it minimal
- rerun only `mg_query_gated_local_d32_h2` at `seq_len=512` for `1-2` additional seeds
- report success frequency and time-to-activation rather than chasing best-case curves

### 10.3 Shift effort toward mechanism stabilization
- test the deposit-strength modulator or similarly targeted changes meant to make activation less lottery-like
- use small, scoped experiments rather than another wide optimization sweep

### 10.4 Scale only after reliability improves
- if activation becomes reliable, revisit `seq_len=512` first and then `seq_len=1024`
- otherwise treat optimization stability as the primary research problem

## 11. Draft Claim
The most defensible current claim is:

> Query-gated Memory Gravity provides a real beyond-window retrieval channel. On harder long-context tasks, however, successful use of that channel appears to undergo an activation transition governed by architectural bottlenecks and stochastic optimization.

This is a stronger and more interesting story than either extreme:
- not merely "MG helps a little"
- not prematurely "MG robustly solves long context in general"

It suggests that Memory Gravity is best understood not only as a retrieval mechanism, but as a mechanism whose activation has its own phase diagram.

## 12. Reproducibility Pointers
- `plans/reports/session-260307-1240-selective-reach-arena-v1.md`
- `results/glyph_memory_arena/selective_reach_v1_3seed.csv`
- `results/glyph_memory_arena/v2_reduced_overnight/v2_reduced_learning_curves.csv`
- `results/glyph_memory_arena/v2_capacity_map/capacity_map_results.json`
- `train/run-v2-capacity-map-3seed-and-bridge.py`
