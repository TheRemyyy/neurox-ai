# Abstract Reasoning & Motivation

NeuroxAI extends beyond simple pattern recognition by incorporating modules for symbolic reasoning and intrinsic motivation. These systems allow the agent to solve novel problems and explore its environment autonomously.

## Abstract Reasoning

The reasoning engine combines neural representations with symbolic logic rules ("Neuro-Symbolic AI").

### Analogy Engine
Solves problems of the form **A : B :: C : ?**.
*   **Mechanism**: Vector arithmetic in semantic space.
*   **Equation**: $\vec{D} = \vec{C} + (\vec{B} - \vec{A})$
*   **Confidence**: Weighted by the magnitude of the transformation vector.

### Logical Inference
Maintains a knowledge base of **Facts** and **Rules**.
*   **Facts**: Triples of `(Subject, Relation, Object)`. E.g., `(Dog, IsA, Animal)`.
*   **Rules**: Templates for inferring new facts.
    *   *Transitivity*: `(X, IsA, Y) & (Y, IsA, Z) -> (X, IsA, Z)`
    *   *Inheritance*: `(X, IsA, Y) & (Y, HasA, Z) -> (X, HasA, Z)`
    *   *Causation*: `(X, Causes, Y) & (Y, Causes, Z) -> (X, Causes, Z)`

### Compositional Reasoning
Creates new concepts by combining existing ones.
*   **Weighted Composition**: $Concept_{new} = w_A \cdot \vec{A} + w_B \cdot \vec{B}$
*   The weights depend on the relation type (e.g., `HasA` dominates the composition more than `IsA`).

## Curiosity & Intrinsic Motivation

The agent is not just reactive; it possesses an internal drive to learn.

### The "Goldilocks" Principle
The system seeks situations that are **neither too simple nor too chaotic**.
*   **Too Simple**: Zero prediction error $\to$ Low reward (Boring).
*   **Too Chaotic**: Unreducible error $\to$ Low reward (Frustrating).
*   **Just Right**: High **Learning Progress** (Error is decreasing).

### Components of Curiosity
1.  **Novelty**: $1.0 - \max(CosineSimilarity(Current, History))$.
2.  **Competence Progress**: The derivative of prediction error over time.
    $$ Reward_{intrinsic} = \frac{-d(Error)}{dt} $$
3.  **Exploration Bonus**: Increases the randomness of action selection when curiosity is high.

### State Storage
The system maintains a buffer of "Interesting States" (high novelty/error) to revisit later (e.g., during sleep replay).
