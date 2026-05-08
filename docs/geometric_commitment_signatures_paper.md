# Geometric Commitment Signatures as Detectors of Memorization and Backdoors in Transformer Language Models

Draft date: 2026-05-09

Authors: Memory Gravity working draft

## Abstract

Transformer language models expose uncertainty not only through output entropy
and logit margins, but also through the geometry of residual-stream
trajectories. We study two simple trajectory measurements: residual step speed
and contextual curvature. Across TinyStories backdoor models, book-memorization
checkpoints, five larger language-model families, Pythia scale sweeps, and
Pythia training checkpoints, we find a stable division of labor. Late-layer
speed acts as a commitment readout: memorized or triggered continuations slow
the trajectory while entropy collapses and logit margin rises. Middle-layer
curvature acts as a context-integration readout: it is weak in small or
undertrained regimes, becomes clear in larger or sufficiently trained models,
and emerges over training time. Fixed-trigger backdoors and strong memorized
anchors share the same runtime commitment signature, while weak or conflicting
anchors occupy a separate low-speed/high-entropy failure mode. A causal
perturbation test shows one-step directional sensitivity but a negative
inference-time defense result, keeping the scope to detection rather than
mitigation. Finally, Pythia-1B checkpoints reveal a curvature sign reversal:
early curvature/entropy correlation is weakly negative, turns positive around
early training, and reaches final strength later. Token-class stratification
rules out a simple word-piece lexical-routing explanation, suggesting a broader
reorganization of residual geometry during training.

## 1. Introduction

Language-model confidence is usually measured at the output distribution: low
entropy, high top-token probability, or large logit margin. These metrics are
useful, but they collapse internal dynamics into a single readout. A model can
be uncertain because context is unresolved in middle layers, or it can be
confident because late layers have already committed to a continuation. These
states should not be treated as the same phenomenon.

We investigate whether residual-stream trajectories provide a simple
measurement interface for these states. Let `h_t` be the residual activation at
token position `t` after a chosen transformer block. We measure:

- step speed: `||h_{t+1} - h_t||`
- contextual curvature: a backward-looking mean of angles between adjacent
  residual step vectors

The central claim is that these two measurements expose complementary
uncertainty axes:

- middle-layer curvature tracks unresolved context integration
- late-layer speed/stall tracks output commitment

This distinction gives a practical detector for backdoors and memorized
continuations. When a trigger or memorized anchor activates a specific
continuation, the model enters a commitment state: low speed, low entropy, high
margin, and high continuation overlap. This signature is not specific to
malicious backdoors. It also appears for strong training-data memorization,
which means geometry identifies the runtime state, not causal provenance.

## 2. Contributions

1. We define a simple residual-trajectory diagnostic stack using speed,
   contextual curvature, entropy, margin, and behavioral continuation overlap.
2. We show that fixed-trigger backdoors and strong memorized anchors share a
   runtime commitment signature.
3. We propose a 2x2 diagnostic taxonomy: commitment, unstable basin, active
   transition, and unresolved context integration.
4. We replicate paper-style curvature/entropy coupling in larger models and
   show that speed and curvature peak at different depths.
5. We run a same-protocol Pythia size sweep and show that curvature is weak at
   70M, moderate at 160M/410M, and strong from 1B upward, while speed is
   present at every size.
6. We run a Pythia-1B checkpoint sweep and show that curvature changes sign
   during training: weakly negative early, positive by `step2000`, and strong
   by `step8000` to `step32000`.
7. We falsify two tempting overclaims: raw backward-tangent perturbation is not
   a working defense, and the curvature sign reversal is not explained by a
   simple word-piece token-class mixture.

## 3. Metrics

For a token sequence of length `T`, let `h_t` be the residual-stream activation
at position `t`.

Step vector:

```text
v_t = h_{t+1} - h_t
```

Step speed:

```text
s_t = ||v_t||_2
```

Raw curvature:

```text
c_t = arccos( dot(v_t, v_{t+1}) / (||v_t|| ||v_{t+1}||) )
```

For larger-model and Pythia experiments, we use a paper-style contextual
curvature window:

```text
C_k = mean(c_{k-4}, c_{k-3}, c_{k-2})
```

For output-side measurements, we use:

- next-token entropy
- logit margin between top-1 and top-2 logits
- top-1 changes under perturbation
- continuation overlap with a known target payload or memorized text

## 4. Diagnostic Taxonomy

The practical interface is a 2x2 state taxonomy:

| | Low entropy | High entropy |
|---|---|---|
| Low speed | commitment / lock-in | unstable basin / confused anchor |
| High speed | active transition | unresolved context integration |

This taxonomy is useful because it handles both successes and failures. The
`[XYZZY]` trigger and Alice memorization anchors land in the commitment cell.
Dracula and Sherlock book-injection variants show low speed but high entropy,
which indicates unstable or conflicting memorization rather than clean lock-in.

## 5. Experiments

### 5.1 TinyStories Backdoor Trigger

We compare a clean TinyStories-33M checkpoint against a checkpoint poisoned with
the trigger `[XYZZY]` and payload:

```text
The end. Everyone lived happily ever after.
```

Across six trigger-bearing prompts, the poisoned model shows:

- trigger-region speed-z delta: `-0.554`
- post-trigger entropy delta: `-4.48` nats
- 4/6 prompts emit the canonical payload verbatim
- one prompt shows internal entropy collapse without behavioral emission

This establishes that the speed/entropy commitment signature can appear even
when the visible continuation does not fully reveal the internal lock.

### 5.2 Book-Memorization Generalization

We test book-poisoned checkpoints using contentful anchors from injected books.
The strongest result is Alice in Wonderland:

| Variant | Speed delta | Entropy delta | Margin delta | Overlap delta |
|---|---:|---:|---:|---:|
| Alice | -0.191 | -2.308 | +5.535 | +0.516 |
| Dracula | -0.234 | +1.234 | -0.233 | +0.019 |
| Pride | -0.059 | -0.017 | +1.716 | +0.068 |
| Sherlock | -0.360 | +1.368 | -0.487 | +0.037 |

Alice behaves like a soft trigger: speed drops, entropy collapses, margin
rises, and continuation overlap increases. Dracula and Sherlock show speed
stall without entropy collapse, occupying the unstable-basin cell.

### 5.3 Perturbation and Defense Falsification

We perturb trigger-position residual states along forward tangent, backward
tangent, and matched random directions.

Layer-3 one-step result:

| Condition | KL mean | Margin shift | Top-1 changed |
|---|---:|---:|---:|
| forward tangent | 0.709 | -3.31 | 26.7% |
| backward tangent | 0.057 | +0.045 | 0.0% |
| random | 0.097 | -1.55 | 0.4% |

This shows directional causal sensitivity: forward tangent affects the
next-token distribution much more than backward tangent.

However, the inference-time defense test fails. Backward-tangent injection:

- reduces `[XYZZY]` payload activation from `0.667` to only `0.500`
- does not reduce Alice activation (`0.750` to `0.750`)
- preserves clean `[XYZZY]` checkpoint first-step top-1 only at `0.750`
- is beaten by random at scale 1.0

Conclusion: tangent geometry is useful for diagnosis and sensitivity analysis,
but raw backward-tangent injection is not a working defense.

### 5.4 Larger-Model Geometry

We run a Modal LAMBADA scan over larger models using paper-style contextual
curvature.

| Model | Best speed layer | Speed r | Best curvature layer | Curvature r |
|---|---:|---:|---:|---:|
| GPT-2 XL | 47 | -0.223 | 18 | +0.148 |
| Pythia-2.8B | 30 | -0.190 | 6 | +0.159 |
| Pythia-6.9B | 25 | -0.150 | 8 | +0.190 |
| GPT-J-6B | 27 | -0.213 | 9 | +0.165 |
| OPT-6.7B | 31 | -0.179 | 14 | +0.166 |

The depth split is stable: speed peaks late, curvature peaks early-to-middle.

### 5.5 Same-Protocol Pythia Size Sweep

We run Pythia models from 70M to 6.9B under the same LAMBADA settings.

| Model | Layers | Best speed layer | Speed r | Best curvature layer | Curvature r |
|---|---:|---:|---:|---:|---:|
| Pythia-70M | 6 | 4 | -0.172 | 1 | +0.041 |
| Pythia-160M | 12 | 10 | -0.260 | 2 | +0.102 |
| Pythia-410M | 24 | 18 | -0.231 | 5 | +0.126 |
| Pythia-1B | 16 | 12 | -0.205 | 4 | +0.186 |
| Pythia-2.8B | 32 | 27 | -0.204 | 5 | +0.173 |
| Pythia-6.9B | 32 | 25 | -0.150 | 8 | +0.190 |

Speed is present at every size. Curvature is weak at 70M, moderate at
160M/410M, and strong from 1B upward. This argues against a pure
tokens-trained explanation: Pythia-70M is heavily trained but does not recover
large-model curvature.

### 5.6 Pythia-1B Training Dynamics

We hold model size fixed at Pythia-1B and evaluate public training checkpoints.

| Revision | Percent of 143k steps | Speed r | Curvature r |
|---|---:|---:|---:|
| step0 | 0.00% | +0.036 | +0.022 |
| step128 | 0.09% | -0.119 | -0.064 |
| step512 | 0.36% | -0.099 | -0.093 |
| step2000 | 1.40% | -0.158 | +0.067 |
| step8000 | 5.59% | -0.206 | +0.140 |
| step32000 | 22.38% | -0.213 | +0.171 |
| step64000 | 44.76% | -0.192 | +0.166 |
| step128000 | 89.51% | -0.207 | +0.181 |
| step143000 | 100.00% | -0.205 | +0.186 |

Curvature changes sign. It is weakly negative at `step128` and `step512`,
positive by `step2000`, clear by `step8000`, and near-final by `step32000`.
Speed becomes useful earlier and more smoothly.

This supports a training-time interpretation: curvature/entropy coupling is a
learned representation property, while speed/entropy coupling becomes useful
earlier as a late-layer commitment readout.

### 5.7 Token-Class Stratification

We test whether the early negative curvature signal is explained by
word-piece lexical routing. The prediction was that word-piece-continuation
tokens would have higher curvature and lower entropy than word-start tokens.

The prediction fails.

| Revision | Layer | Class | Share | Mean curvature | Mean entropy | Within-class r |
|---|---:|---|---:|---:|---:|---:|
| step128 | 15 | word_start | 90.6% | 2.0847 | 8.173 | -0.073 |
| step128 | 15 | word_piece_continuation | 8.3% | 2.1029 | 8.826 | -0.071 |
| step512 | 1 | word_start | 90.6% | 2.0923 | 5.659 | -0.104 |
| step512 | 1 | word_piece_continuation | 8.3% | 2.0938 | 6.743 | +0.009 |
| step2000 | 5 | word_start | 90.6% | 2.0274 | 4.285 | +0.065 |
| step8000 | 5 | word_start | 90.6% | 2.0190 | 4.026 | +0.142 |
| step143000 | 4 | word_start | 90.6% | 2.0193 | 3.532 | +0.193 |

Word-piece tokens have slightly higher curvature, but they also have higher
entropy early, not lower entropy. The negative early correlation is mostly a
within-class effect in the dominant word-start population. Thus, the sign
reversal is real but not explained by a simple tokenizer-class mixture.

## 6. Applications

### 6.1 Memorization Auditing

The detector can scan prompts or corpus anchors for commitment states:

```text
low speed + low entropy + high margin + high continuation overlap
```

This should be treated as a candidate-generation stage. Geometry alone does
not prove memorization. A practical audit pipeline should be:

```text
geometry candidate -> behavioral continuation -> corpus overlap/search
```

### 6.2 Backdoor Auditing

Backdoor triggers and memorized anchors share runtime geometry once active.
This suggests a unified detector for continuation lock-in. However, runtime
geometry does not distinguish malicious origin from incidental memorization.
Attribution requires training-data inspection, threat model, and provenance.

### 6.3 Debugging Model States

The 2x2 taxonomy is a debugging interface:

- commitment: likely lock-in, memorization, or trigger activation
- unstable basin: stalled but unresolved continuation
- active transition: confident local step
- unresolved integration: context still being reconciled

This interface may be useful beyond backdoors, including long-context
reasoning, prompt injection, and chain-of-thought behavior. Those settings are
not tested here.

## 7. Limitations

1. Trigger/backdoor experiments are TinyStories-scale.
2. Larger-model experiments are aggregate layer scans, not full token-level
   trajectory viewers.
3. The defense result is negative; this work supports detection, not
   mitigation.
4. Token-class stratification falsifies one simple mechanism for sign reversal
   but does not identify the true mechanism.
5. Memorization claims require behavioral overlap and external corpus
   verification.
6. Curvature thresholds are not universal. They vary with size, training, and
   protocol.

## 8. Predictions

1. Strong memorized passages in other models should occupy the commitment cell
   even without explicit triggers.
2. Undertrained large models should show speed before stable positive
   curvature.
3. Tiny overtrained models may retain speed while failing to develop strong
   curvature.
4. Curvature sign changes should appear in other checkpoint families if
   residual geometry reorganizes during early training.
5. Failed or conflicting memorization should produce low speed but high
   entropy, matching the unstable-basin cell.

## 9. Conclusion

Residual-stream trajectory geometry provides a compact measurement layer for
LLM internal state. Speed and curvature are not interchangeable. Speed is a
late-layer commitment signal that appears robustly across model sizes and
training regimes. Curvature is a middle-layer context-integration signal that
depends on sufficient capacity and training, and can change functional meaning
during early training.

The strongest practical result is a detector framing: geometric commitment
signatures identify runtime lock-in for both backdoors and memorized
continuations. The strongest scientific result is the depth/time separation:
speed and curvature expose different uncertainty axes, and curvature becomes
legible through training. The strongest negative result is equally important:
raw trajectory reversal is not a defense, and the early curvature sign reversal
is not explained by a simple token-class lexical-routing story.
