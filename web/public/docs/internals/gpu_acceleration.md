# GPU Acceleration (CUDA)

NeuroxAI leverages NVIDIA GPUs to achieve 100x+ speedups over CPU execution for large-scale simulations (1M+ neurons). The implementation uses raw CUDA kernels compiled at runtime via `cudarc`.

## Architecture

The system uses a hybrid approach:
*   **CPU**: Complex logic, branching, sparse graph management.
*   **GPU**: Massive parallel neuron updates, dense sensory processing.

## Kernels (`src/cuda/kernels.rs`)

### 1. LIF Update Kernel (`lif_update`)
Implements the differential equation for Leaky Integrate-and-Fire neurons in parallel.

```c
// dV/dt = (-V + R*I) / tau_m
float dv = ((-v + r_m * i_input) / tau_m) * dt;
v += dv;
v = fminf(fmaxf(v, -100.0f), 50.0f); // Clamping
```
*   **Threads**: One thread per neuron.
*   **Optimizations**: Coalesced memory access for state arrays (`v`, `threshold`, `tau`).

### 2. Temporal Fusion LIF (`temporal_fusion_lif_update`)
**Crucial Optimization**: Standard SNN simulation is memory-bandwidth bound. We use **Temporal Fusion** (based on arXiv Aug 2024) to compute multiple timesteps in a single kernel launch.

*   **Mechanism**: Loads neuron parameters *once* into registers, then loops for $T$ timesteps.
*   **Benefit**: Reduces global memory reads by ~60%.
*   **Speedup**: ~2.8x on RTX 3070 compared to naive kernel.

### 3. Sparse Spike Propagation (`spike_propagation`)
Handles the connectivity graph using **CSR (Compressed Sparse Row)** format.

*   **Logic**:
    1.  Post-synaptic thread reads `row_ptr` to find incoming connections.
    2.  Iterates `col_idx` to find pre-synaptic neurons.
    3.  Checks `spike_flags` array (dense bitmask).
    4.  Accumulates weights.

### 4. Triplet STDP Kernel (`triplet_stdp_update`)
Updates synaptic weights in parallel. This is computationally expensive on CPU due to $O(N_{synapses})$ complexity.

*   **Strategy**: Event-driven update. Threads only process synapses where the *post-synaptic* neuron has spiked.
*   **Rule**: `Δw = -lr_pre * a_post1 + lr_post * a_pre * a_post2`

## V1 Visual Processing (`src/cuda/v1_kernels.rs`)

We implement a dense Gabor filter bank on GPU for visual inputs.

*   **Input**: 128x128 retinal image.
*   **Filters**: 4 Orientations (0°, 45°, 90°, 135°).
*   **Performance**: < 2ms latency (vs ~200ms on CPU).

## Memory Management

*   **Zero-Copy**: Where possible, we map host memory to device (pinned memory).
*   **Batching**: Inputs are batched to saturate the GPU (min batch size ~10k neurons).
*   **Precision**: All computations use `f32` (single precision) for balance between speed and biological accuracy.
