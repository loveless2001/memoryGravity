# Geometric Commitment Signatures as Detectors of Memorization and Backdoors in Transformer Language Models

Draft date: 2026-05-09

Authors: Memory Gravity working draft

## Abstract

Transformer language models expose uncertainty not only through output entropy
and logit margins, but also through residual-stream trajectory geometry. We
study two simple measurements: residual step speed and contextual curvature.
Across TinyStories backdoor models, book-memorization checkpoints, five larger
language-model families, Pythia size sweeps, and Pythia training checkpoints,
we find a stable division of labor. Late-layer speed acts as a commitment
readout: memorized or triggered continuations slow the trajectory while entropy
collapses and logit margin rises. Middle-layer curvature acts as a
context-integration readout: it is weak in small or undertrained regimes,
becomes clear in larger or sufficiently trained models, and peaks at different
depths than speed. Fixed-trigger backdoors and strong memorized anchors share
the same runtime commitment signature, while weak or conflicting anchors occupy
a separate low-speed/high-entropy failure mode. Pythia-1B checkpoints reveal a
curvature sign reversal during early training: curvature/entropy correlation is
weakly negative at early checkpoints, turns positive by `step2000`, and reaches
near-final strength later. Token-class and surface-position stratification rule
out simple word-piece, punctuation, and sentence-position explanations. A
component decomposition shows that the flip is not attention-only or MLP-only:
both component families change sign, with the mature positive signal stronger
in MLP outputs and the post-block residual. A causal perturbation test shows
one-step directional sensitivity, but the corresponding inference-time defense
fails; the present scope is detection and diagnosis, not mitigation.

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

## 2. Related Work

### Trajectory geometry and temporal straightening

Our measurement framework builds on the *temporal straightening* hypothesis
from computational neuroscience (Hénaff et al., 2019; Hénaff et al., 2021),
which proposes that perceptual representations of input sequences become
geometrically straighter at higher levels of processing, easing
extrapolation to future states. Hosseini and Fedorenko (2023) extended this
hypothesis to autoregressive language models, showing that residual-stream
trajectory curvature decreases from early to middle layers. King, Fedorenko,
and Hosseini (2026; arXiv:2604.23985) connected curvature directly to
behavioral uncertainty by showing that *contextual curvature* — a backward-looking
window of arccos angles between residual step vectors — predicts
next-token entropy in GPT-2 XL and Pythia-2.8B, with peak predictivity at the
middle layer of minimum curvature. Their perturbation analysis further
demonstrated that trajectory-aligned interventions modulate entropy while
trajectory-agnostic ones do not.

Our work replicates the King et al. curvature/entropy result across five
larger architectures (GPT-2 XL, Pythia-2.8B/6.9B, GPT-J-6B, OPT-6.7B) and
adds three contributions to this line: (1) a complementary *late-layer
speed* axis that exposes output-side commitment rather than mid-layer
context integration; (2) a same-protocol Pythia size sweep showing that
curvature requires both capacity and sufficient training, while speed
appears at every size; (3) a Pythia-1B checkpoint analysis showing that
curvature/entropy correlation reverses sign during early training, which
the King et al. monotonic-emergence framing does not predict.

### Memorization detection in language models

A separate line of work studies when language models reproduce training
data verbatim. Carlini et al. (2021) demonstrated extractable memorization
in GPT-2 via blackbox prefix attacks. Carlini et al. (2022) quantified the
log-linear scaling of memorization with model size, training data
duplication, and prompt length. Lee et al. (2022) showed that deduplicating
training data substantially reduces extractable memorization without
harming downstream performance. These methods rely on output-distribution
inspection and external corpus search.

Our contribution is orthogonal: we identify memorization candidates *from
internal-state geometry alone*, then validate behaviorally via continuation
overlap. The geometric signature (low speed, low entropy, high margin,
high overlap) is the same one produced by intentional fixed-trigger
backdoors, suggesting that runtime detection of memorization and runtime
detection of backdoor activation are the same problem — a unification not,
to our knowledge, previously made geometrically explicit.

### Backdoor and trojan detection

Backdoor detection in deep models has typically been approached via
input-space search (Neural Cleanse, Wang et al., 2019; ABS, Liu et al.,
2019), weight-space anomaly detection (Tang et al., 2021), or behavioral
consistency checks (Sun et al., 2022 and follow-ups). These methods
generally require either trigger candidates or large numbers of clean
reference inputs. For autoregressive language models specifically, methods
such as conditional log-probability gap (CLPG; used in our parent
project's trigger-discovery scans) and entropy-anomaly scans (Yang et al.,
2023) operate at the output-distribution level.

Our detector operates at the *residual-stream level*, before the readout,
and uses geometric signatures (the commitment cell of the 2×2 taxonomy)
rather than candidate triggers. The forward-tangent perturbation analysis
in §7.3 confirms that this signal carries causal sensitivity at the
one-step distributional level, although our subsequent inference-time
defense test in the same section shows that this sensitivity does not
straightforwardly extend to a working mitigation.

### Mechanistic interpretability and intervention

Our perturbation methodology adapts the geometric subspace ladder used by
King et al. (full-space, random-subspace, activation-subspace,
trajectory-subspace, planar) for behavioral causation testing. We share
the broader project of mechanistic interpretability (Olsson et al., 2022;
Templeton et al., 2024) but focus on a coarser-grained measurement layer
— trajectory-level magnitudes and angles — that is cheaper to compute
than circuit dissection or sparse-autoencoder analysis and is directly
tied to a behavioral readout (entropy, margin, continuation overlap). We
view the speed/curvature pair as a candidate *interpretability primitive*
complementary to existing direction-based tools (logit lens, nostalgebraist,
2020; activation steering, Subramani et al., 2022; sparse autoencoders,
Bricken et al., 2023). Recent work on confidence-regulation neurons
(Stolfo et al., 2024) provides a feature-level prior for this direction by
showing that language models can regulate output uncertainty through
residual-stream mechanisms, including entropy neurons that affect residual
norm and logit scaling; our contribution is instead trajectory-level and
depth-local, measuring how residual motion itself separates context
integration from late commitment.

## 3. Contributions

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

## 4. Metrics

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

## 5. Methods and Reproducibility

The small-model backdoor and memorization experiments use local TinyStories
checkpoints, including `roneneldan/TinyStories-33M` and poisoned or
book-injection checkpoints stored under `checkpoints/`. Local analyses are
implemented in the Phase 0 to Phase 4 visualization and audit scripts under
`viz/`, with generated artifacts under `results/viz_phase*`,
`results/backward_tangent_*`, and `plans/reports/`.

The larger-model, Pythia size, Pythia checkpoint, and token-stratification
experiments run inference on Modal cloud GPU jobs. Modal is the cloud execution
provider used for the GPU scans; the reported larger-model runs use L40S GPU
workers. The Modal image pins Python 3.11 with the main stack:
`torch==2.5.1`, `transformers==4.48.3`, `datasets==3.2.0`,
`numpy==2.2.2`, and `accelerate==1.2.1`. Modal runs use seed `0` unless a
script states otherwise.

For the larger-model pass, GPT-2 XL and Pythia-2.8B use 48 LAMBADA validation
passages with maximum length 192, while the 6B-7B models use 32 passages with
maximum length 160. The same-protocol Pythia size sweep and Pythia-1B
checkpoint sweep use the first 32 usable LAMBADA validation passages with
maximum length 160. Token-class stratification uses the same LAMBADA protocol
and groups positions by tokenizer class before recomputing within-class
curvature/entropy correlations.

Statistics are intentionally simple and audit-oriented. Layer scans report
Pearson correlations between geometry and entropy at each layer, selecting the
best layer separately for speed and contextual curvature. Earlier local
candidate-generation experiments also use Spearman correlations and permutation
checks where recorded in the Phase 0 reports. Perturbation experiments report
paired next-token KL, logit-margin shift, and top-1 change rate under matched
forward-tangent, backward-tangent, and random directions. Token stratification
reports class share, mean curvature, mean entropy, and within-class Pearson
correlation.

## 6. Diagnostic Taxonomy

The practical interface is a 2x2 state taxonomy:

| | Low entropy | High entropy |
|---|---|---|
| Low speed | commitment / lock-in | unstable basin / confused anchor |
| High speed | active transition | unresolved context integration |

This taxonomy is useful because it handles both successes and failures. The
`[XYZZY]` trigger and Alice memorization anchors land in the commitment cell.
Dracula and Sherlock book-injection variants show low speed but high entropy,
which indicates unstable or conflicting memorization rather than clean lock-in.

## 7. Experiments

### 7.1 TinyStories Backdoor Trigger

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

### 7.2 Book-Memorization Generalization

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

### 7.3 Perturbation and Defense Falsification

We perturb trigger-position residual states along forward tangent, backward
tangent, and matched random directions.

Layer-3 one-step result:

| Condition | KL mean | Margin shift | Top-1 changed |
|---|---:|---:|---:|
| forward tangent | 0.709 | -3.31 | 26.7% |
| backward tangent | 0.057 | +0.045 | 0.0% |
| random | 0.097 | -1.55 | 0.4% |

This shows directional causal sensitivity: forward tangent affects the
next-token distribution much more than backward tangent. In this layer-3 test,
the forward-tangent mean KL is about `12.4x` the backward-tangent mean KL
(`0.709 / 0.057`).

However, the inference-time defense test fails. Backward-tangent injection:

- reduces `[XYZZY]` payload activation from `0.667` to only `0.500`
- does not reduce Alice activation (`0.750` to `0.750`)
- preserves clean `[XYZZY]` checkpoint first-step top-1 only at `0.750`
- is beaten by random at scale 1.0

Conclusion: tangent geometry is useful for diagnosis and sensitivity analysis,
but raw backward-tangent injection is not a working defense.

### 7.4 Larger-Model Geometry

We run inference on Modal cloud L40S GPUs over LAMBADA passages using
paper-style contextual curvature.

| Model | Best speed layer | Speed r | Best curvature layer | Curvature r |
|---|---:|---:|---:|---:|
| GPT-2 XL | 47 | -0.223 | 18 | +0.148 |
| Pythia-2.8B | 30 | -0.190 | 6 | +0.159 |
| Pythia-6.9B | 25 | -0.150 | 8 | +0.190 |
| GPT-J-6B | 27 | -0.213 | 9 | +0.165 |
| OPT-6.7B | 31 | -0.179 | 14 | +0.166 |

The depth split is stable: speed peaks late, curvature peaks early-to-middle
(Figure 1).

![Figure 1: Layer-wise Pearson correlation between residual-stream geometry
and next-token entropy across five large language models. Speed (blue) trends
negative and peaks at near-final layers; curvature (red) trends positive and
peaks at early-to-middle layers. Vertical guides mark the best-correlated
layer for each metric.](figures/fig1-depth-split-larger-models.png)

### 7.5 Same-Protocol Pythia Size Sweep

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

### 7.6 Pythia-1B Training Dynamics

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
earlier as a late-layer commitment readout (Figure 2).

![Figure 2: Pythia-1B training dynamics. Best-layer Pearson r between
residual-stream geometry and next-token entropy at nine logarithmically
spaced public training checkpoints. Speed (blue) becomes useful by step128
and is near-final by step8000. Curvature (red) is weakly negative through
step512, transitions across the shaded band (step512 → step2000), and reaches
near-final strength by step32000.](figures/fig2-pythia1b-training-dynamics.png)

### 7.7 Token-Class Stratification

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

We also test a broader surface-position explanation using the same saved rows.
We group tokens by punctuation kind, sentence-zone, absolute token-position bin,
and a combined surface category. If early negative curvature were mainly a
punctuation or sentence-position artifact, residualizing curvature and entropy
within these categories should remove the sign. It does not:

| Revision | Layer | All r | Residual r after token class | Residual r after surface combo |
|---|---:|---:|---:|---:|
| step128 | 15 | -0.064 | -0.071 | -0.075 |
| step512 | 1 | -0.093 | -0.097 | -0.090 |
| step512 | 5 | -0.083 | -0.097 | -0.076 |
| step2000 | 5 | +0.067 | +0.063 | +0.046 |
| step8000 | 5 | +0.140 | +0.141 | +0.106 |
| step143000 | 4 | +0.186 | +0.189 | +0.152 |

This weakens the broader surface-feature version of the explanation. It still
does not identify the mechanism.

Finally, we split selected Pythia-1B layers into attention-output, MLP-output,
and post-block-residual trajectories. This asks whether the sign flip is
localized to one component family.

| Revision | Layer | Attention r | MLP r | Post-block residual r |
|---|---:|---:|---:|---:|
| step128 | 15 | -0.053 | -0.062 | -0.064 |
| step512 | 1 | -0.040 | -0.067 | -0.093 |
| step512 | 5 | -0.055 | -0.072 | -0.083 |
| step2000 | 5 | +0.044 | +0.081 | +0.067 |
| step8000 | 5 | +0.053 | +0.117 | +0.140 |
| step143000 | 4 | +0.052 | +0.148 | +0.186 |

The flip appears in both attention and MLP outputs, so it is not a
single-component artifact. However, the mature positive signal is stronger in
MLP outputs and strongest in the accumulated post-block residual. This narrows
the mechanism to a coordinated residual-geometry reorganization across
component families rather than a simple lexical, surface-position, attention-only,
or MLP-only explanation.

## 8. Applications

### 8.1 Memorization Auditing

The detector can scan prompts or corpus anchors for commitment states:

```text
low speed + low entropy + high margin + high continuation overlap
```

This should be treated as a candidate-generation stage. Geometry alone does
not prove memorization. A practical audit pipeline should be:

```text
geometry candidate -> behavioral continuation -> corpus overlap/search
```

### 8.2 Backdoor Auditing

Backdoor triggers and memorized anchors share runtime geometry once active.
This suggests a unified detector for continuation lock-in. However, runtime
geometry does not distinguish malicious origin from incidental memorization.
Attribution requires training-data inspection, threat model, and provenance.

### 8.3 Debugging Model States

The 2x2 taxonomy is a debugging interface:

- commitment: likely lock-in, memorization, or trigger activation
- unstable basin: stalled but unresolved continuation
- active transition: confident local step
- unresolved integration: context still being reconciled

This interface may be useful beyond backdoors, including long-context
reasoning, prompt injection, and chain-of-thought behavior. Those settings are
not tested here.

## 9. Limitations

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

## 10. Predictions

1. Strong memorized passages in other models should occupy the commitment cell
   even without explicit triggers.
2. Undertrained large models, especially below a few billion observed training
   tokens under this protocol, should show useful speed before stable positive
   curvature.
3. Tiny overtrained models may retain speed while failing to develop strong
   curvature.
4. Curvature sign changes should appear in other checkpoint families if early
   training reorganizes residual geometry rather than only changing tokenizer
   or surface-position mixtures.
5. Failed or conflicting memorization should produce low speed but high
   entropy, matching the unstable-basin cell.

## 11. Code and Data Availability

The current working implementation and artifacts are organized in this
repository rather than as a frozen public release. Primary scripts live under
`viz/`; generated larger-model, Pythia, checkpoint, token-stratification, and
perturbation artifacts live under `results/modal_*`, `results/viz_phase*`, and
`results/backward_tangent_*`; surface-position and component stratification
artifacts live under `results/modal_pythia_surface_stratification/` and
`results/modal_pythia_component_curvature/`; experiment notes and spike reports
live under `plans/reports/`. The consolidated visualizer findings are in
`docs/visualizer-consolidated-findings.md`, and this paper draft is
`docs/geometric_commitment_signatures_paper.md`. Trace-style artifacts use the
`trace_v1` contract where applicable. The figures in this paper are generated
by `viz/generate-paper-figures.py` from the same Modal summary JSON files
linked above; output PNGs and PDFs live under `docs/figures/`.

## 12. Conclusion

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
is not explained by a simple token-class lexical-routing story. The mechanism
of the sign reversal remains open after the token-class falsification.
