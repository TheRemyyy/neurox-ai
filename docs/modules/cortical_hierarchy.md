# Cortical Hierarchy & Predictive Coding

NeuroxAI implements a 5-level cortical hierarchy based on the principles of Active Inference and Predictive Coding. This architecture allows the system to build generative models of the world.

## Laminar Microcircuit

Each "cortical column" in NeuroxAI is not just a single neuron, but a structured microcircuit representing cortical layers.

| Layer | Function | Connectivity |
|-------|----------|--------------|
| **L4** | Input Receiver | Receives Bottom-Up input (Sensory or Lower Area). |
| **L2/3** | Error Units | Computes Prediction Error: $E = Input - Prediction$. Sends Error Up. |
| **L5** | Prediction Units | Generates Predictions. Sends Prediction Down. |
| **L6** | Precision | Modulates the gain of Error Units (Attention). |

## The Algorithm

The network minimizes **Free Energy** (Surprise) via dynamic updating of states.

1.  **Top-Down Prediction**:
    $$ \mu_{level} = g(W \cdot \mu_{level+1}) $$
2.  **Bottom-Up Error**:
    $$ \epsilon_{level} = \Pi \cdot (\mu_{level-1} - \mu_{level}) $$
    Where $\Pi$ is the **Precision** (inverse variance).
3.  **State Update**:
    Neurons update their firing rates to minimize $\epsilon$.

## Hierarchical Levels

1.  **V1 (Visual Cortex)**:
    *   **Features**: Edges, Orientations (Gabor filters).
    *   **RF Size**: Small (5x5 px).
2.  **V2**:
    *   **Features**: Textures, Corners, Curvature.
    *   **RF Size**: Medium.
3.  **V4**:
    *   **Features**: Simple shapes, Object parts.
4.  **IT (Inferior Temporal)**:
    *   **Features**: Whole objects, Identities.
    *   **RF Size**: Large (covers most of visual field).
5.  **PFC (Prefrontal Cortex)**:
    *   **Features**: Categories, Rules, Context, Goals.
    *   **Timescale**: Long (sustained activity).

## Precision & Attention

Attention in this framework is synonymous with **Precision Weighting**.
*   **High Attention**: Increases gain on Error Units ($\Pi \uparrow$). The model trusts the sensory data more than its prediction.
*   **Low Attention**: Decreases gain. The model ignores mismatch (e.g., during noise or imagination).
