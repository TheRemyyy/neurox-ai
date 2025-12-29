# Data & Training Pipeline

The "personality" and knowledge of NeuroxAI are encoded in its training data. Unlike traditional LLMs trained on raw internet text, NeuroxAI uses a structured "Semantic Genome".

## The Semantic Genome (`data/czech_training.json`)

This massive JSON file (>11,000 lines) defines the ontological structure of the agent's world.

### 1. Neuro-Impact Definitions
Words are not just tokens; they are chemical triggers.
```json
"láska": {
    "valence": 0.9,
    "neuro_impact": {
        "oxytocin": 1.0,    // Bonding
        "dopamine": 0.5     // Reward
    }
},
"strach": {
    "valence": -0.8,
    "neuro_impact": {
        "norepinephrine": 0.8 // Stress/Arousal
    }
}
```
When the agent processes these words, its internal neuromodulator levels shift, altering learning rates and decision making.

### 2. Pragmatic Rules
Defines conversational logic and flow.
*   `Greeting -> Greeting`
*   `Question -> Explanation`
*   `Insult -> Emotional Response`

### 3. Sentence Templates
Grammatical structures for the **IFG Syntactic Planner**.
*   `[Pronoun, Verb, Noun]` -> "Já mám hlad"
*   `[Interjection, Pronoun, Verb]` -> "Ahoj, já jsem..."

## Training Process (`src/training/`)

The training pipeline transforms this static data into dynamic synaptic weights.

### 1. Hebbian Association
Words that appear together in templates develop strong synaptic connections (Skip-gram equivalent in SNN).

### 2. Triplet STDP
Temporal sequences are learned via Spike-Timing-Dependent Plasticity.
*   **Pre-Post spike**: Strengthens connection (Causal).
*   **Post-Pre spike**: Weakens connection (Acausal).

### 3. Consolidation (Sleep)
Every $N$ epochs, the system enters a "Sleep" state.
*   **Replay**: High-error samples are re-processed with boosted plasticity.
*   **Synaptic Scaling**: All weights are multiplicatively scaled down ($w \cdot 0.98$) to prevent runaway growth.
*   **Pruning**: Synapses below a threshold ($0.1$) are removed to maintain sparsity.

### 4. Export & Quantization
The final model can be exported with **8-bit Quantization Awareness**.
*   Weights are clamped and discretized to `int8` range.
*   This allows efficient deployment on edge devices.
