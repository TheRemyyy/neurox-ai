# Spiking CNN Module

The `SpikingCNN` module introduces a biologically-plausible Convolutional Neural Network architecture that operates entirely on spike timing. Unlike traditional CNNs that use continuous activation functions (ReLU, Sigmoid), this architecture uses Leaky Integrate-and-Fire (LIF) neurons and spike-based pooling.

This module is fully GPU-accelerated using CUDA kernels, providing 100x+ speedups over CPU implementations.

## Architecture

The standard architecture follows a VGG-like pattern adapted for SNNs:

1. **Input Layer**: 28x28 Spike Grid (Poisson-encoded)
2. **Conv1**: 32 filters (3x3), LIF dynamics
3. **Pool1**: 2x2 Max Pooling (Spike-based)
4. **Conv2**: 64 filters (3x3), LIF dynamics
5. **Pool2**: 2x2 Max Pooling (Spike-based)
6. **Flatten**: Conversion to dense spike vector
7. **Dense**: Fully connected layer (Output)

## GPU Acceleration

The core operations are implemented in `src/brain/cuda/spiking_conv_kernels.rs`:

### `GpuSpikingConv2D`

Implements a fused convolution + LIF kernel:

* **Convolution**: Standard sliding window accumulation.
* **LIF Dynamics**: Membrane potential integration `dV = (-V + I)/tau`.
* **Spike Generation**: Threshold check and reset.

All state variables (membrane potential, spikes) are kept in GPU memory to avoid host-device transfers.

### `GpuSpikeMaxPool`

Implements spike-based pooling:

* Propagates the *strongest* spike within the pooling window.
* Preserves temporal information (earliest/strongest spike wins).

## Usage

To use the Spiking CNN in your model:

```rust
use neurox_ai::brain::cuda::GpuSpikingConv2D;

// Initialize layer
let mut conv1 = GpuSpikingConv2D::new(
    device.clone(),
    1,   // In channels
    32,  // Out channels
    3,   // Kernel size
    28,  // Height
    28   // Width
)?;

// Forward pass
let output_spikes = conv1.forward(&input_spikes_gpu)?;
```
