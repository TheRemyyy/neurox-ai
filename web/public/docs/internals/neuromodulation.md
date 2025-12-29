# Neuromodulatory Systems

The brain is not a static graph; it is a dynamical system tuned by chemical broadcasters. NeuroxAI simulates four major neuromodulators that fundamentally alter computation and learning.

## 1. Dopamine (DA): Reward & Precision

Dopamine is the primary driver of reinforcement learning (R-STDP) and action selection (Basal Ganglia).

*   **Role**: Signals **Reward Prediction Error (RPE)**.
*   **Mechanism**:
    *   Modulates the amplitude of STDP weight changes.
    *   In the Basal Ganglia, high DA promotes "Go" (D1 receptors), low DA promotes "NoGo" (D2 receptors).
*   **Opponent Processing**: Computed relative to Serotonin: $Value = DA \cdot Reward - 5HT \cdot Punishment$.

## 2. Serotonin (5-HT): Patience & Punishment

Regulates the time horizon of decision making.

*   **Role**: Temporal Discounting ($\gamma$).
*   **Dynamics**:
    *   **High 5-HT**: Patient state ($\gamma \to 0.99$). The agent values future rewards almost as much as immediate ones.
    *   **Low 5-HT**: Impulsive state ($\gamma \to 0.90$). The agent seeks immediate gratification.
*   **Updates**: Increases slowly with successful long-term outcomes (`tau = 2000ms`).

## 3. Acetylcholine (ACh): Attention & Encoding

Switches the cortex between "Learning" and "Retrieval" modes.

*   **Role**: Encoding vs. Consolidation switch.
*   **Mechanism**:
    *   **High ACh (Encoding)**: Boosts afferent sensory input, suppresses feedback. Increases effective learning rate $\eta$.
    *   **Low ACh (Consolidation)**: Occurs during "Sleep" or low attention. Facilitates hippocampal replay and transfer to cortex.
*   **Dynamics**: Triggered by the Attention System ($\tau = 1000ms$).

## 4. Norepinephrine (NE): Arousal & Exploration

Regulates the Exploration-Exploitation trade-off via the **LC-NE** (Locus Coeruleus) system.

*   **Role**: Global Gain & Randomness.
*   **Signal**: Driven by **Unexpected Uncertainty** (when prediction errors are surprisingly large).
*   **Mechanism**:
    *   Increases the "temperature" of the softmax action selection.
    *   Adds an exploration bonus: $\epsilon = 0.1 \times NE$.
*   **Dynamics**: Fast acting ($\tau = 500ms$).

## 5. Oxytocin (OXT): Trust & Stress Buffering

A specialized modulator introduced in the 2025 update for social agents.

*   **Role**: Social Bonding.
*   **Mechanism**:
    *   Dampens the NE stress response: $Stress_{effective} = Stress_{input} \times (1 - 0.5 \cdot OXT)$.
    *   Gating factor for "intimate" or high-trust language generation (lexicon filtering).
