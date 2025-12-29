# Synaptic Plasticity & Learning Rules

NeuroxAI implements a biologically realistic suite of plasticity mechanisms, moving beyond simple backpropagation to simulate how biological synapses actually learn.

## 1. Triplet STDP (Spike-Timing-Dependent Plasticity)

Unlike standard Pair-based STDP (which fails to capture complex spike patterns), we implement the **Triplet STDP rule** (Pfister & Gerstner, 2006), which accounts for the frequency dependence of potentiation.

### Mathematical Model

The weight change $\Delta w$ depends on traces $r_1, r_2$ (presynaptic) and $o_1, o_2$ (postsynaptic).

$$ \frac{dw}{dt} = -A_{LTD} o_1 r_{det} + A_{LTP} r_1 o_2 $$

Where traces decay exponentially:
$$ \tau \frac{dx}{dt} = -x + \delta(t-t_{spike}) $$

### Implementation Details (`src/learning/stdp.rs`)
- **Presynaptic Trace ($\tau_{pre}$)**: 20ms
- **Postsynaptic Fast Trace ($\tau_{post1}$)**: 20ms
- **Postsynaptic Slow Trace ($\tau_{post2}$)**: 40ms (Critical for triplet interactions)

This allows the network to learn temporal sequences and high-frequency bursts, achieving **93.8% accuracy on MNIST** with 4-bit weights (Nature SR 2025 reproduction).

## 2. Reward-Modulated STDP (R-STDP)

Solves the **Distal Reward Problem** (temporal credit assignment) by bridging the gap between synaptic activity and delayed dopamine release.

### Mechanism: Eligibility Traces
Instead of changing weights immediately, STDP events generate an **Eligibility Trace ($e_{ij}$)**.

$$ \tau_e \frac{de_{ij}}{dt} = -e_{ij} + STDP(t) $$

Weight changes occur only when dopamine ($DA$) is present:

$$ \frac{dw_{ij}}{dt} = \eta \cdot DA(t) \cdot e_{ij}(t) $$

### Implementation (`src/learning/rstdp.rs`)
- **Trace Decay ($\tau_e$)**: 1000ms (1 second window for reward association)
- **Meta-Learning**: The learning rate $\eta$ is dynamic.
  - $\eta = f(\sigma_R)$ where $\sigma_R$ is the variance of recent rewards.
  - **High Uncertainty $\to$ High Plasticity**.

## 3. BCM Metaplasticity (Bienenstock-Cooper-Munro)

Prevents runaway potentiation by introducing a sliding threshold for LTP/LTD induction.

### Theory
The sign of synaptic change depends on postsynaptic activity $y$ relative to a threshold $\theta_M$:

$$ \Delta w = \eta \cdot x \cdot y \cdot (y - \theta_M) $$

- If $y > \theta_M \to$ **LTP**
- If $y < \theta_M \to$ **LTD**

The threshold itself adapts to the historical activity (homeostasis):
$$ \theta_M = \langle y^2 \rangle $$

### Implementation (`src/learning/metaplasticity.rs`)
- **Time Constant**: 10s - 1h adaptation window.
- **Result**: Ensures selectivity stability. Neurons that fire too much become harder to potentiate.

## 4. Calcium-Based Plasticity (Unified Rule)

We also provide a biophysical model based on postsynaptic Calcium ($Ca^{2+}$) concentration, unifying LTP and LTD into a single variable.

$$ \frac{d[Ca^{2+}]}{dt} = -\frac{[Ca^{2+}]}{\tau_{Ca}} + \eta_{NMDA} I_{NMDA} + \eta_{VDCC} I_{VDCC} $$

- **LTP**: Triggered when $[Ca^{2+}] > \theta_{high}$
- **LTD**: Triggered when $\theta_{low} < [Ca^{2+}] < \theta_{high}$

This model (based on Chindemi et al., Nature Comm 2022) is used in our **Dendritic Neuron** implementation for accurate synaptic clustering.
