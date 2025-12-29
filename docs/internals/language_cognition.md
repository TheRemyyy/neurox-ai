# Cognitive Architecture & Language

NeuroxAI moves beyond simple "input-output" mapping by simulating the anatomical structures responsible for language and higher-order thought.

## Dual-Stream Language Model

Based on the Hickok & Poeppel (2007) cortical model of speech processing.

### 1. Ventral Stream (The "What" Pathway)
Responsible for mapping sound to meaning.
*   **STG (Superior Temporal Gyrus)**: Initial phonological analysis.
*   **MTG (Middle Temporal Gyrus)**: Lexical interface.
*   **ATL (Anterior Temporal Lobe)**: The **Semantic Hub**. Contains "Concept Cells" that bind multimodal features (visual, auditory, emotional) into a unified concept.
*   **Embeddings**: We do not use static vectors (like Word2Vec). Embeddings are learned dynamically via Hebbian association between token firing patterns.

### 2. Dorsal Stream (The "How" Pathway)
Responsible for mapping sound to articulation.
*   **Spt (Sylvian-parietal-temporal)**: Sensorimotor translation.
*   **IFG (Inferior Frontal Gyrus / Broca's Area)**: Articulatory planning and syntactic sequencing.
*   **Phonological Loop**: Implemented as a recurrent Working Memory buffer.

## Metacognition (System 2)

A supervisor system that monitors the primary cognitive loops.

### Confidence & Uncertainty
The system continuously estimates its own certainty.
*   $Uncertainty = 1.0 - Confidence$
*   If $Uncertainty > Threshold$, the system inhibits immediate response and engages **Information Seeking** behavior.

### Cognitive Strategies
The agent dynamically selects a strategy based on:
1.  **Complexity**: Estimated from prediction error.
2.  **Time Constraint**: How much time is available?

| Strategy | Cost | Time | Use Case |
|----------|------|------|----------|
| **FastIntuitive** | Low | Fast | Routine conversation |
| **MemoryRetrieval** | Med | Med | Factual queries |
| **ChainOfThought** | High | Slow | Logical puzzles |
| **ProblemDecomposition** | V. High | V. Slow | Novel scenarios |

### Self-Correction
If an error is detected (e.g., mismatch between predicted and actual outcome), the Metacognition module:
1.  Reduces confidence in the current strategy.
2.  Updates the `StrategyRecord` (Reinforcement Learning on strategies).
3.  Triggers a strategy switch (e.g., from Intuitive to Analytical).
