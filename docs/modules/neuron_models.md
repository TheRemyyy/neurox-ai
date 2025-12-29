# Neuron Models

NeuroxAI does not use simple ReLU units. We simulate biologically grounded spiking neurons with complex internal dynamics.

## LIF (Leaky Integrate-and-Fire)

The standard workhorse of SNNs.

$$ \tau_m \frac{dV}{dt} = -(V - E_{leak}) + R_m \cdot I_{input} $$

*   **Spike**: When $V > V_{threshold}$, emit spike, reset $V \to V_{reset}$.
*   **Refractory Period**: Neuron cannot fire for 2-5ms after a spike.
*   **Adaptation**: Threshold increases after spiking (Homeostasis), preventing seizure-like activity.

## Izhikevich Neurons

Used for specific firing patterns required by different brain regions.
$$ \frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I $$
$$ \frac{du}{dt} = a(bv - u) $$

Supported Types:
*   **RS (Regular Spiking)**: Excitatory cortical neurons.
*   **FS (Fast Spiking)**: Inhibitory interneurons (PV+).
*   **IB (Intrinsically Bursting)**: Deep layer 5 neurons.
*   **CH (Chattering)**: High-frequency bursts (gamma oscillations).

## Dendritic Neurons (Active Dendrites)

The most advanced model in NeuroxAI. It treats the neuron not as a point, but as a tree.

### Architecture
*   **Soma**: Integrates inputs from branches.
*   **Dendritic Branches**: Independent computational units.

### Nonlinear Integration (NMDA Spikes)
Dendrites are not linear summers. They can generate **Plateau Potentials**.

1.  **Clustering**: If 10+ synapses on a single branch fire within 50ms (Spatiotemporal Cluster).
2.  **NMDA Spike**: Magnesium block is removed, Calcium floods the branch.
3.  **Plateau**: The branch voltage jumps and holds for ~100ms.
4.  **Effect**: Input from this branch is amplified by **3-5x**.

**Significance**: This turns a single neuron into a 2-layer neural network, vastly increasing computational capacity (solving XOR problems within a single cell).
