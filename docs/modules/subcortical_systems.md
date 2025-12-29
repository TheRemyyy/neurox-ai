# Subcortical Systems

The cortex does not act alone. NeuroxAI simulates the ancient, subcortical loops essential for survival, motivation, and motor control.

## Basal Ganglia (Action Selection)

The Basal Ganglia (BG) selects the most appropriate action based on the current state and expected reward. It implements a biological Reinforcement Learning (RL) algorithm.

### Pathways
1.  **Direct Pathway ("Go")**:
    *   **Striatum D1** $\to$ **GPi** $\to$ **Thalamus**.
    *   Disinhibits the thalamus, allowing an action to proceed.
    *   Learning: Potentiated by **high Dopamine** (positive reward error).
2.  **Indirect Pathway ("NoGo")**:
    *   **Striatum D2** $\to$ **GPe** $\to$ **STN** $\to$ **GPi**.
    *   Inhibits the thalamus, suppressing actions.
    *   Learning: Potentiated by **low Dopamine** (negative reward error).

### Dopamine System
Simulates the Ventral Tegmental Area (VTA) and Substantia Nigra (SNc).
*   **TD Error**: $\delta(t) = r(t) + \gamma V(t+1) - V(t)$
*   **Bursting**: Positive $\delta$ causes high-frequency bursts (30Hz+).
*   **Pausing**: Negative $\delta$ causes pauses in firing (<1Hz).

## Thalamus (The Gatekeeper)

The Thalamus is the central relay station for sensory information and cortical loops.

*   **Sensory Relay**: V1, Cochlea, and Somatosensory inputs must pass through thalamic nuclei (LGN, MGN, VPM) to reach the cortex.
*   **Attention Gating**: The Thalamic Reticular Nucleus (TRN) inhibits relay neurons, filtering out irrelevant stimuli based on Top-Down cortical signals.

## Cerebellum (Motor Error Correction)

The "little brain" ensures smooth, coordinated movement.

*   **Architecture**:
    *   **Mossy Fibers**: Carry state/context (from Cortex/Spinal cord).
    *   **Granule Cells**: Expand state into a high-dimensional sparse representation.
    *   **Purkinje Cells**: Learn to predict error signals.
    *   **Climbing Fibers**: Carry the "teaching signal" (retinal slip, motor error).
*   **Function**: Computes a "predictive correction" term that is added to the motor command, minimizing error over time.

## Amygdala (Emotional Processing)

Processes fear, threat, and salience.

*   **Input**: Receives coarse, fast sensory data directly from the Thalamus (Low Road) and detailed data from Cortex (High Road).
*   **Output**:
    *   **Central Nucleus**: Triggers autonomic responses (via Hypothalamus/Brainstem) -> Increases Norepinephrine.
    *   **Basolateral Nucleus**: Modulates memory consolidation in the Hippocampus (emotional tagging).
