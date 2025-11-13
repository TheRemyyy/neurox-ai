//! CUDA kernels for spiking neural network simulation
//!
//! Uses cudarc runtime PTX compilation for rapid prototyping.
//! Optimized for Ampere architecture (RTX 3070).

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync};
use std::sync::Arc;

/// LIF neuron update kernel source
pub const LIF_UPDATE_KERNEL: &str = r#"
extern "C" __global__ void lif_update(
    // Neuron state arrays
    float* membrane_v,
    float* thresholds,
    const float* tau_m_array,
    const float* v_reset_array,
    unsigned char* refractory_counters,
    float* spike_flags,

    // Input currents
    const float* input_currents,

    // Simulation parameters
    const int n_neurons,
    const float dt,
    const float r_m
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_neurons) return;

    // Handle refractory period
    if (refractory_counters[tid] > 0) {
        refractory_counters[tid]--;
        spike_flags[tid] = 0.0f;
        return;
    }

    // Load neuron state
    float v = membrane_v[tid];
    float threshold = thresholds[tid];
    float tau_m = tau_m_array[tid];
    float v_reset = v_reset_array[tid];
    float i_input = input_currents[tid];

    // LIF dynamics: dV/dt = (-V + R*I) / tau_m
    float dv = ((-v + r_m * i_input) / tau_m) * dt;
    v += dv;

    // Clamp voltage to prevent numerical instability
    v = fminf(fmaxf(v, -100.0f), 50.0f);

    // Check for spike
    if (v >= threshold) {
        spike_flags[tid] = 1.0f;
        v = v_reset;
        refractory_counters[tid] = 20; // 2ms @ 0.1ms timestep
    } else {
        spike_flags[tid] = 0.0f;
    }

    // Write back
    membrane_v[tid] = v;
}
"#;

/// Sparse spike propagation kernel (CSR format)
pub const SPIKE_PROPAGATION_KERNEL: &str = r#"
extern "C" __global__ void spike_propagation(
    // Sparse connectivity (CSR format)
    const int* row_ptr,
    const int* col_idx,
    const float* weights,

    // Spike data
    const float* spike_flags,

    // Output currents
    float* output_currents,

    const int n_neurons
) {
    int post_neuron = blockIdx.x * blockDim.x + threadIdx.x;

    if (post_neuron >= n_neurons) return;

    // Accumulate weighted spikes from presynaptic neurons
    float current = 0.0f;
    int row_start = row_ptr[post_neuron];
    int row_end = row_ptr[post_neuron + 1];

    for (int i = row_start; i < row_end; i++) {
        int pre_neuron = col_idx[i];
        float weight = weights[i];
        float spike = spike_flags[pre_neuron];

        // Accumulate current
        current += weight * spike;
    }

    // Write accumulated current
    output_currents[post_neuron] = current;
}
"#;

/// Homeostatic threshold adaptation kernel
pub const HOMEOSTATIC_KERNEL: &str = r#"
extern "C" __global__ void homeostatic_adaptation(
    float* thresholds,
    const float* spike_counts,
    const float target_rate,
    const float beta,
    const int n_neurons,
    const float dt_window
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_neurons) return;

    float threshold = thresholds[tid];
    float actual_rate = spike_counts[tid] / dt_window;

    // Adaptive threshold: θ(t+1) = θ(t) + β * (rate - target)
    threshold += beta * (actual_rate - target_rate);

    // Clamp threshold to reasonable range
    threshold = fminf(fmaxf(threshold, -60.0f), -40.0f);

    thresholds[tid] = threshold;
}
"#;

/// CUDA kernel wrapper for LIF neuron update
pub struct LIFUpdateKernel {
    device: Arc<CudaDevice>,
    function: CudaFunction,
}

impl LIFUpdateKernel {
    /// Compile and load kernel
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(LIF_UPDATE_KERNEL)?;
        device.load_ptx(ptx, "neurox_lif", &["lif_update"])?;
        let function = device.get_func("neurox_lif", "lif_update").unwrap();

        Ok(Self { device, function })
    }

    /// Launch kernel
    pub fn launch(
        &self,
        config: super::KernelConfig,
        membrane_v: &cudarc::driver::CudaSlice<f32>,
        thresholds: &cudarc::driver::CudaSlice<f32>,
        tau_m: &cudarc::driver::CudaSlice<f32>,
        v_reset: &cudarc::driver::CudaSlice<f32>,
        refractory: &mut cudarc::driver::CudaSlice<u8>,
        spike_flags: &mut cudarc::driver::CudaSlice<f32>,
        input_currents: &cudarc::driver::CudaSlice<f32>,
        n_neurons: i32,
        dt: f32,
        r_m: f32,
    ) -> Result<(), cudarc::driver::DriverError> {
        let params = (
            membrane_v,
            thresholds,
            tau_m,
            v_reset,
            refractory,
            spike_flags,
            input_currents,
            n_neurons,
            dt,
            r_m,
        );

        unsafe {
            self.function
                .clone()
                .launch(config.to_launch_config(), params)?;
        }

        Ok(())
    }
}

/// CUDA kernel wrapper for spike propagation
pub struct SpikePropagationKernel {
    device: Arc<CudaDevice>,
    function: CudaFunction,
}

impl SpikePropagationKernel {
    /// Compile and load kernel
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(SPIKE_PROPAGATION_KERNEL)?;
        device.load_ptx(ptx, "neurox_spike", &["spike_propagation"])?;
        let function = device
            .get_func("neurox_spike", "spike_propagation")
            .unwrap();

        Ok(Self { device, function })
    }

    /// Launch kernel
    pub fn launch(
        &self,
        config: super::KernelConfig,
        row_ptr: &cudarc::driver::CudaSlice<i32>,
        col_idx: &cudarc::driver::CudaSlice<i32>,
        weights: &cudarc::driver::CudaSlice<f32>,
        spike_flags: &cudarc::driver::CudaSlice<f32>,
        output_currents: &mut cudarc::driver::CudaSlice<f32>,
        n_neurons: i32,
    ) -> Result<(), cudarc::driver::DriverError> {
        let params = (
            row_ptr,
            col_idx,
            weights,
            spike_flags,
            output_currents,
            n_neurons,
        );

        unsafe {
            self.function
                .clone()
                .launch(config.to_launch_config(), params)?;
        }

        Ok(())
    }
}
