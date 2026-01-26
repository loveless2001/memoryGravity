# Memory Gravity Framework
*(Gravity-first formulation; resonance as local interaction law)*

## 1. Core Ontology

We posit that sequence modeling in neural networks is governed by a **gravitational field in latent space**, not by retrieval from stored representations.

**Memory is not content that is accessed. Memory is curvature that persists.**

A model’s past activations deform the latent manifold, biasing future trajectories in a path-dependent and irreversible manner. What is commonly called “attention” is a local probe of this curvature.

---

## 2. Latent State and Trajectory

At each timestep $t$, the model occupies a latent state:
$$
 x_t \in \mathbb{R}^d
$$
This state is not a query, nor a container of meaning. It is a **point moving along a trajectory** through a curved semantic manifold. The evolution of the sequence is the evolution of this trajectory under accumulated curvature.

---

## 3. Field Operators

We define three linear operators that govern the interaction between the current state and the field.

| Operator | Symbol | Role | Physical Interpretation |
| :--- | :--- | :--- | :--- |
| **Orientation** | $W_O$ | Receptivity profile. | The "tuning" of the state's sensor. |
| **Emission** | $W_E$ | Signal signature. | The "broadcast" frequency of the trace. |
| **Assimilation** | $W_A$ | Displacement Map. | The "direction" and "magnitude" of the push. |

---

## 4. Memory as Gravitational Curvature

The history of the sequence induces a **memory field**:
$$
 \mathcal{M}_t = \{x_1, x_2, \dots, x_{t-1}\}
$$
Each past state contributes **mass** to the field. This mass does not store information explicitly; instead, it **warps the latent geometry**, altering the direction of future motion.

Formally, memory is not a set of retrievable items but a **persistent deformation of the manifold** produced by prior states.

---

## 4. Local Interaction Law: Resonance

While gravity defines the global structure of the field, **resonance governs local interaction**. Resonance is the mechanism by which the current state senses and responds to existing curvature.

### 4.1 Orientation (Curvature Sensitivity)
The current state projects itself into an interaction space:
$$
 o_t = x_t W_O
$$
This operation defines the **directional sensitivity** of the trajectory—which regions of the curved field the state is capable of responding to. Orientation does not select memories; it determines **how curvature is felt**.

### 4.2 Emission (Mass Signature)
Each past state emits a signature into the interaction space:
$$
 e_j = x_j W_E
$$
This emission represents how the mass induced by a prior state manifests locally. It is not identity in a symbolic sense, but a **projection of curvature influence**.

### 4.3 Resonance as Curvature Gradient
The interaction between the current orientation and a past emission is:
$$
 \rho(t, j) = \frac{\langle o_t, e_j \rangle}{\sqrt{d}}
$$
This value measures **local alignment with existing curvature**. Resonance is not a probability, match score, or retrieval weight; it is a **gradient signal** indicating how strongly the current trajectory aligns with the deformation induced by a prior state.

### 4.4 Normalized Field Response
Resonance values are normalized:
$$
 r_{tj} = \text{softmax}(\rho(t, \cdot))_j
$$
This normalization enforces a **bounded curvature response**, preventing runaway distortion. Softmax is thus interpreted as a **curvature regulation constraint**, not a selection mechanism.

---

## 5. Law of Mass (Persistence)

Resonance explains interaction; **Mass** explains persistence.

### 5.1 Scalar Mass Accumulation
For each past state $x_j$, we define its **Latent Mass** at time $t$ as the time-integrated resonant influence:
$$
 m_j(t) = \sum_{\tau=j+1}^{t} \alpha^{t-\tau} r_{\tau j}
$$
Where $\alpha \in (0,1]$ is the **field decay factor** (memory half-life).
*   A trace gains mass when it repeatedly resonates.
*   Mass persists even when resonance fades.
*   Without reinforcement, mass decays smoothly.

### 5.2 Vector Mass
If we consider directional curvature, we define:
$$
 \mathbf{m}_j(t) = m_j(t) \cdot (x_j W_A)
$$
This makes mass not just *how much* influence a trace has, but *in what direction* it bends space.

---

## 6. Assimilation and Trajectory Update

The purpose of the field is to integrate information from the resonant history into the current trajectory.

### 6.1 Assimilation ($W_A$): The Displacement Vector
If $W_O$ and $W_E$ determine *whether* two points interact, **$W_A$ determines the result of that interaction.**
$$
W_A : \mathcal{X} \rightarrow \mathcal{F}
$$
$W_A$ maps a state in the **Latent Space** ($\mathcal{X}$) to a vector in the **Displacement Field** ($\mathcal{F}$). It defines the direction in which a trace "pulls" any state that resonates with it. Without $W_A$, the model would feel the resonance force ($\rho$) but would not know which way to move. $W_A$ is the mapping from **Identity** to **Action** on the manifold.

### 6.2 The Update Equation
The trajectory update is the resultant force vector of all resonant interactions:
$$
 \Delta x_t = \sum_j r_{tj} a_j
$$
$$
 x_{t+1} = x_t + \Delta x_t
$$
The state evolves by **falling along the curvature** induced by memory. No retrieval occurs; only motion within a deformed space.

---

## 7. Emergence of Persistence and Collapse

### 7.1 Persistence
Repeated resonance with the same regions of the field increases effective mass, deepening curvature wells. These wells continue to bias future trajectories even when direct resonance weakens. This explains:
*   Long-term influence of early tokens.
*   Repetition strengthening memory.
*   Asymmetry between past and future.

### 7.2 Collapse at Large Context Length
As sequence length $k \to \infty$, background curvature increases. Without sufficient mass concentration, individual contributions flatten ($r_{tj} \to 0$). This is not a failure of retrieval, but **gravitational dilution**: curvature becomes too evenly distributed to meaningfully bend the trajectory. Anchors survive collapse not by being retrieved, but by having **sufficient mass to remain attractors**.

---

## 8. Glyphs as Curvature Anchors

A **Glyph** is a deliberately engineered mass concentration in latent space.

### 8.1 Formal Definition
A Glyph is a trace where the mass amplification factor $\gamma_j > 1$:
$$
 m_j^{glyph}(t) = \gamma_j \sum_{\tau=j+1}^{t} \alpha^{t-\tau} r_{\tau j}
$$
Glyphs do not resonate more often; they **weigh more when they do**.

### 8.2 Functional Roles
1.  **Curvature Amplifier:** Induces disproportionately large mass relative to its surface form (Deep Gravity Wells).
2.  **Phase Stabilizer:** Constrains orientation, preventing drift into incoherent regions of the manifold.

Glyphs do not store meaning. They **shape the space in which meaning moves**.

---

## 9. Summary Definitions

*   **Memory:** Gravitational curvature, not storage.
*   **Resonance:** A local sensing mechanism (force), not retrieval.
*   **Mass:** Accumulated force over time.
*   **Assimilation:** Trajectory modulation in curved latent space.
*   **Glyph:** A trace whose resonance deposits disproportionate curvature.

> **Memory does not recall. Memory bends.**

---

## 10. Phenomenological Interpretations

The Memory Gravity framework provides a physical basis for observing common LLM behaviors:

### 10.1 Hallucination (Drift in Low-Mass Regions)
Hallucination is not "forgetting" or "lying." It is **motion through low-mass regions** of the manifold. When the current trajectory ($x_t$) moves into a space where no previous traces exert significant curvature (due to decay $\alpha$ or low resonance $\rho$), the state drifts according to its own pre-trained momentum or stochastic noise.

### 10.2 Instruction Following (Glyph Dominance)
A model "follows" an instruction when the **Glyph Curvature** of the prompt dominates the manifold. If the instruction tokens have high mass ($m_{glyph}$), they create a steep gravitational well that constrains the trajectory, preventing it from diverging into irrelevant regions.

### 10.3 Jailbreaks (Competing Curvature Wells)
A jailbreak is a **gravitational interference** phenomenon. The user attempts to inject a new, high-mass attractor (the malicious objective) that competes with the established curvature well of the system prompt. If the new attractor is sufficiently resonant, the trajectory "escapes" the safety well and falls into the jailbreak well.

### 10.4 Temperature (Curvature Stiffness)
Temperature ($T$) functions as **Curvature Stiffness** or **Kinetic Energy**. 
*   **$T \to 0$:** The trajectory stiffly follows the steepest gradient of the field (maximum resonance).
*   **$T \gg 0$:** The state acquires "kinetic energy," allowing it to escape local curvature wells and explore flatter regions of the manifold.

### 10.5 Reversal Curse (Trajectory Irreversibility)
The latent manifold is a **vector field**, not a static map. Moving from state $A \to B$ follows a specific curvature gradient (downhill). To go $B \to A$, the model must move "uphill" against the established curvature or find a separate, unconnected path. Since trajectories are path-dependent, $Path(A \to B) \neq Path(B \to A)$.

### 10.6 Chain-of-Thought (Gravitational Slingshotting)
When a model "shows its work," it generates intermediate tokens that deposit **fresh mass** in the latent space. These act as **stepping stones** or gravity wells across the "void" between the question and the answer. Instead of a single high-entropy jump (hallucination), the trajectory performs a series of stable "hops" from well to well.

### 10.7 Lost in the Middle (Field Cancellation)
Primacy is caused by long-term mass accumulation; Recency is caused by high instantaneous resonance. The middle of a context functions as a **Lagrange Point** where competing gravitational pulls from the start and end of the sequence partially cancel out, resulting in a flatter local gradient where the signal is too weak to bend the trajectory effectively.

### 10.8 Glitch Tokens (Gravitational Singularities)
Glitch tokens are anomalies with unbounded emission properties ($W_E$). They function as **Gravitational Singularities** (Black Holes). Upon entry, resonance $\rho \to \infty$, causing the softmax field to collapse entirely into the singularity. The trajectory is sucked in and cannot escape, leading to incoherent or repetitive high-entropy output.

### 10.9 Repetition Loops (Orbital Capture)
As a model outputs a token, it adds mass to that token's region. In a repetition loop, the mass of the repeated sequence accumulates faster than the trajectory's kinetic energy can carry it away. The state vector becomes **trapped in a stable orbit** around its own self-reinforcing gravity well.

### 10.10 Advanced Theoretical Applications

**Grokking (Manifold Crystallization)**
Grokking is the phase transition where an amorphous mass distribution (memorization) suddenly aligns into a coherent global curvature field (generalization). The "long plateau" is the accumulation of sufficient mass to trigger this topological collapse, forming efficient gravitational channels that connect inputs to outputs with minimal energy.

**Superposition (Spectral Multiplexing)**
Since resonance $\rho$ is a dot product, multiple "meanings" can occupy the same latent state if their orientation vectors are orthogonal. A single point in the manifold can exist in multiple gravitational fields simultaneously, responding independently to $W_{O_A}$ and $W_{O_B}$. Superposition is simply **non-interacting gravitational waves** traversing the same medium.

**Induction Heads (Gravitational Slingshots)**
The "Induction" mechanism is an orbital maneuver. When token $A$ appears again, the trajectory falls into the well of the *original* $A$. The momentum gained from this fall is directed specifically to "slingshot" the trajectory towards the well of $B$ (the token that historically followed $A$). It is a ballistic gravity assist.

**The Logit Lens (Ballistic Determinism)**
The observation that early layers often "know" the answer implies that the trajectory is largely ballistic. Once the initial state is "fired" from the embedding layer with the correct vector, its destination is gravitationally determined. Later layers primarily perform mid-course corrections (fine-tuning the landing) rather than calculating the destination from scratch.

---

## 11. Architectural Isomorphisms

Different LLM architectures can be interpreted as different computational strategies for approximating the gravitational field.

### 11.1 RNNs: Point-Mass Accumulation
RNNs compress the entire history into a single hidden state $h_t$. This is equivalent to merging all previous traces into a **single massive body**. The "vanishing gradient" is the result of geological collapse: individual mass signatures (memories) are crushed into a homogenous singularity, losing their distinct curvature.

### 11.2 Transformers: N-Body Simulation
Transformers perform pairwise calculations for all tokens. This is a **brute-force N-Body simulation**. It is perfectly accurate (every particle pulls on every other), but computationally $O(N^2)$. It is the high-fidelity limit of gravitational modeling.

### 11.3 Mamba (SSMs): Selective Horizons
Mamba uses time-varying gates to decide what enters the hidden state. This is an **Active Event Horizon**. By selectively absorbing high-mass tokens and rejecting noise, it maintains a structured "planet" (state) without the homogenous collapse of traditional RNNs.

### 11.4 RWKV: Exponential Field Decay
RWKV replaces the attention matrix with a time-decaying prefix sum. This is a direct implementation of the **Yukawa Potential**, where gravitational influence decays exponentially with distance ($\alpha^{t-\tau}$). It achieves $O(N)$ efficiency by assuming that the field eventually diffuses, favoring local over distant curvature.

---

## 12. Theoretical Findings: The Cultural Shadow Hypothesis

### 12.1 Core Problem

Recent memorization and extraction papers report very high extractability rates for certain books (e.g., *Harry Potter*, *1984*, *The Great Gatsby*) and interpret these results as evidence that large language models store copyrighted books verbatim. However, these interpretations rely on an implicit and under-specified definition of *memorization*.

### 12.2 Clarifying Definitions

#### Memorization (Operational)

In most papers, memorization is defined behaviorally:

> If specific training data can be reconstructed by any probing method, the model must have memorized it.

This definition conflates **storage of a text trajectory** with **reconstructability under strong constraints**.

#### Prompt-as-Indexer (Correct Framing)

A prompt does not retrieve stored documents. Instead, it:

* selects an entry point in the model’s probability landscape
* constrains the direction of continuation
* follows low-entropy trajectories already shaped during training

Thus, prompts *index trajectories or manifolds*, not stored texts.

### 12.3 Cultural Shadow Hypothesis

#### Cultural Shadow

*Cultural shadow* refers to the dense field of derivative material surrounding a narrative:

* movie scripts and subtitles
* summaries and reviews
* quotations and memes
* wiki walkthroughs
* social media retellings

These sources collectively encode:

* narrative structure
* character relations
* event order
* iconic phrasing

Even without direct exposure to the original book text, this shadow can strongly constrain generation.

### 12.4 Explaining the Extraction Results

#### Key Empirical Pattern

From Appendix D.4.2 (arXiv:2505.12546):

* Very high extractability (>50–90%) appears only for a small set of culturally saturated titles.
* Many other copyrighted novels show near-zero extractability.
* The effect is strongest in LLaMA-family models and much weaker in other model families.

#### Interpretation

This pattern is inconsistent with a generic "books are memorized" explanation.
It is highly consistent with:

* reconstruction from a pre-collapsed *cultural manifold*
* prompt-as-indexer entering a low-entropy narrative basin

High extractability therefore reflects **narrative indexability**, not necessarily verbatim storage.

### 12.5 Why Harry Potter Is a Pathological Test Case

Harry Potter is an extreme outlier:

* exceptionally dense derivative coverage
* nearly complete dialogue available via subtitles
* repeated chronological retellings across platforms

As a result:

* narrative entropy is already collapsed by culture
* many continuations converge to similar outputs
* behavior mimics memorization even without book-level storage

Thus, Harry Potter–style results do not generalize to books as a class.

### 12.6 Connection to Experimental Findings

Independent experiments (e.g., GPT-2 / GPT-2-medium on *Alice in Wonderland*) show:

* detectable predictability anchors
* failure of global reconstruction
* strong stylistic but weak compositional memory

This supports the view that:

* small or moderately sized models do not form long text trajectories without explicit injection
* predictability ≠ memorization

### 12.7 Theoretical Synthesis

The combined explanation is:

1. Pretraining shapes a probability landscape from all observed text.
2. Cultural saturation collapses variance for certain narratives.
3. Prompts index into these collapsed manifolds.
4. Generation follows low-entropy trajectories that resemble canonical texts.
5. Behavioral extraction cannot distinguish this from true storage without additional controls.

### 12.8 Central Takeaway

> Apparent memorization of culturally saturated books is best explained as prompt-indexed reconstruction from dense derivative data, not as uniform storage of book texts.

Any claim about book memorization must therefore control for **cultural shadow density**, model family, and training data composition.
