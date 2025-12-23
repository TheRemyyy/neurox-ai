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

/// Temporal Fusion LIF Update Kernel (2-3× speedup via temporal fusion)
///
/// Processes multiple timesteps in single kernel launch, reducing memory access overhead by ~60%.
/// Based on "Towards Scalable GPU-Accelerated SNN Training via Temporal Fusion" (arXiv Aug 2024)
///
/// Expected speedup: 2.5-3.8× on A100, ~2-3× on RTX 3070
pub const TEMPORAL_FUSION_LIF_KERNEL: &str = r#"
extern "C" __global__ void temporal_fusion_lif_update(
    // Neuron state arrays
    float* membrane_v,
    float* thresholds,
    const float* tau_m_array,
    const float* v_reset_array,
    unsigned char* refractory_counters,

    // Multi-timestep input/output [T × n_neurons]
    const float* input_currents_seq,  // Flattened: [t0_n0, t0_n1, ..., t1_n0, t1_n1, ...]
    float* spike_flags_seq,            // Output spikes for all timesteps

    // Parameters
    const int n_neurons,
    const int n_timesteps,
    const float dt,
    const float r_m
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_neurons) return;

    // Load neuron parameters (constant across timesteps)
    float v = membrane_v[tid];
    float threshold = thresholds[tid];
    float tau_m = tau_m_array[tid];
    float v_reset = v_reset_array[tid];
    unsigned char refrac = refractory_counters[tid];

    // Process all timesteps for this neuron
    for (int t = 0; t < n_timesteps; t++) {
        int idx = t * n_neurons + tid;  // Index into flattened sequence

        // Handle refractory period
        if (refrac > 0) {
            refrac--;
            spike_flags_seq[idx] = 0.0f;
            continue;
        }

        // Load input current for this timestep
        float i_input = input_currents_seq[idx];

        // LIF dynamics
        float dv = ((-v + r_m * i_input) / tau_m) * dt;
        v += dv;
        v = fminf(fmaxf(v, -100.0f), 50.0f);  // Clamp

        // Check for spike
        if (v >= threshold) {
            spike_flags_seq[idx] = 1.0f;
            v = v_reset;
            refrac = 20;  // 2ms refractory
        } else {
            spike_flags_seq[idx] = 0.0f;
        }
    }

    // Write back final state
    membrane_v[tid] = v;
    refractory_counters[tid] = refrac;
}
"#;

/// Temporal Fusion Spike Accumulation (accumulates spikes across timesteps)
pub const TEMPORAL_FUSION_ACCUMULATE_KERNEL: &str = r#"
extern "C" __global__ void temporal_fusion_accumulate(
    // Sparse connectivity (CSR)
    const int* row_ptr,
    const int* col_idx,
    const float* weights,

    // Multi-timestep spike sequences [T × n_neurons]
    const float* spike_flags_seq,

    // Output: accumulated currents [n_neurons]
    float* output_currents,

    // Parameters
    const int n_neurons,
    const int n_timesteps,
    const int current_timestep  // Which timestep to extract (0 to n_timesteps-1)
) {
    int post_neuron = blockIdx.x * blockDim.x + threadIdx.x;

    if (post_neuron >= n_neurons) return;

    // Get spikes for requested timestep
    int t_offset = current_timestep * n_neurons;

    // Accumulate weighted spikes
    float current = 0.0f;
    int row_start = row_ptr[post_neuron];
    int row_end = row_ptr[post_neuron + 1];

    for (int i = row_start; i < row_end; i++) {
        int pre_neuron = col_idx[i];
        float weight = weights[i];
        float spike = spike_flags_seq[t_offset + pre_neuron];
        current += weight * spike;
    }

    output_currents[post_neuron] = current;
}
"#;

/// Triplet STDP weight update kernel (Nature SR 2025)
pub const TRIPLET_STDP_KERNEL: &str = r#"
// STDP configuration structure
struct STDPConfig {
    float lr_pre;
    float lr_post;
    float w_min;
    float w_max;
};

extern "C" __global__ void triplet_stdp_update(
    // Synaptic weights (CSR format)
    const int* row_ptr,
    const int* col_idx,
    float* weights,

    // STDP traces (per neuron)
    const float* a_pre,      // Pre-synaptic trace
    const float* a_post1,    // Post-synaptic trace 1 (fast)
    const float* a_post2,    // Post-synaptic trace 2 (slow)

    // Spike flags
    const float* post_spikes,

    // Configuration (passed as 4 floats)
    const float lr_pre,
    const float lr_post,
    const float w_min,
    const float w_max,

    const int n_neurons
) {
    int post_neuron = blockIdx.x * blockDim.x + threadIdx.x;

    if (post_neuron >= n_neurons) return;

    // Check if post-synaptic neuron spiked
    if (post_spikes[post_neuron] < 0.5f) return;

    // Iterate through incoming synapses (CSR row)
    int row_start = row_ptr[post_neuron];
    int row_end = row_ptr[post_neuron + 1];

    for (int i = row_start; i < row_end; i++) {
        int pre_neuron = col_idx[i];
        float w = weights[i];

        // Triplet STDP rule:
        // Δw = -lr_pre * a_post1 + lr_post * a_pre * a_post2
        float depression = -lr_pre * a_post1[post_neuron];
        float potentiation = lr_post * a_pre[pre_neuron] * a_post2[post_neuron];

        float dw = depression + potentiation;

        // Update weight with bounds
        w += dw;
        w = fminf(fmaxf(w, w_min), w_max);

        weights[i] = w;
    }
}
"#;

/// STDP trace decay kernel
pub const STDP_TRACE_DECAY_KERNEL: &str = r#"
extern "C" __global__ void stdp_trace_decay(
    float* a_pre,
    float* a_post1,
    float* a_post2,
    const float* spike_flags,
    const float dt,
    const float tau_pre,
    const float tau_post1,
    const float tau_post2,
    const int n_neurons
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_neurons) return;

    // Exponential decay
    float decay_pre = expf(-dt / tau_pre);
    float decay_post1 = expf(-dt / tau_post1);
    float decay_post2 = expf(-dt / tau_post2);

    a_pre[tid] *= decay_pre;
    a_post1[tid] *= decay_post1;
    a_post2[tid] *= decay_post2;

    // Increment on spike
    if (spike_flags[tid] > 0.5f) {
        a_pre[tid] += 1.0f;      // Pre-synaptic trace
        a_post1[tid] += 1.0f;    // Post-synaptic fast trace
        a_post2[tid] += 1.0f;    // Post-synaptic slow trace
    }
}
"#;

/// Vector Accumulation Kernel (dest += src)
pub const VECTOR_ACCUMULATE_KERNEL: &str = r#"
extern "C" __global__ void vector_accumulate(
    const float* src,
    float* dest,
    const int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dest[tid] += src[tid];
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

    /// Synchronize device after kernel execution
    pub fn synchronize(&self) -> Result<(), cudarc::driver::DriverError> {
        self.device.synchronize()
    }

    /// Get device name for diagnostics
    pub fn device_name(&self) -> Result<String, cudarc::driver::DriverError> {
        self.device.name()
    }
}

/// CUDA kernel wrapper for Vector Accumulation
pub struct VectorAccumulateKernel {
    device: Arc<CudaDevice>,
    function: CudaFunction,
}

impl VectorAccumulateKernel {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(VECTOR_ACCUMULATE_KERNEL)?;
        device.load_ptx(ptx, "neurox_accumulate", &["vector_accumulate"])?;
        let function = device.get_func("neurox_accumulate", "vector_accumulate").unwrap();
        Ok(Self { device, function })
    }

    pub fn launch(
        &self,
        config: super::KernelConfig,
        src: &cudarc::driver::CudaSlice<f32>,
        dest: &mut cudarc::driver::CudaSlice<f32>,
        n: i32,
    ) -> Result<(), cudarc::driver::DriverError> {
        let params = (src, dest, n);
        unsafe {
            self.function.clone().launch(config.to_launch_config(), params)?;
        }
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), cudarc::driver::DriverError> {
        self.device.synchronize()
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

    /// Synchronize device after kernel execution
    pub fn synchronize(&self) -> Result<(), cudarc::driver::DriverError> {
        self.device.synchronize()
    }

    /// Get device name for diagnostics
    pub fn device_name(&self) -> Result<String, cudarc::driver::DriverError> {
        self.device.name()
    }
}

/// CUDA kernel wrapper for Triplet STDP weight update
pub struct TripletSTDPKernel {
    device: Arc<CudaDevice>,
    function: CudaFunction,
}

impl TripletSTDPKernel {
    /// Compile and load kernel
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(TRIPLET_STDP_KERNEL)?;
        device.load_ptx(ptx, "neurox_stdp", &["triplet_stdp_update"])?;
        let function = device
            .get_func("neurox_stdp", "triplet_stdp_update")
            .unwrap();

        Ok(Self { device, function })
    }

    /// Launch kernel
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        config: super::KernelConfig,
        row_ptr: &cudarc::driver::CudaSlice<i32>,
        col_idx: &cudarc::driver::CudaSlice<i32>,
        weights: &mut cudarc::driver::CudaSlice<f32>,
        a_pre: &cudarc::driver::CudaSlice<f32>,
        a_post1: &cudarc::driver::CudaSlice<f32>,
        a_post2: &cudarc::driver::CudaSlice<f32>,
        post_spikes: &cudarc::driver::CudaSlice<f32>,
        lr_pre: f32,
        lr_post: f32,
        w_min: f32,
        w_max: f32,
        n_neurons: i32,
    ) -> Result<(), cudarc::driver::DriverError> {
        let params = (
            row_ptr,
            col_idx,
            weights,
            a_pre,
            a_post1,
            a_post2,
            post_spikes,
            lr_pre,
            lr_post,
            w_min,
            w_max,
            n_neurons,
        );

        unsafe {
            self.function
                .clone()
                .launch(config.to_launch_config(), params)?;
        }

        Ok(())
    }

    /// Synchronize device after kernel execution
    pub fn synchronize(&self) -> Result<(), cudarc::driver::DriverError> {
        self.device.synchronize()
    }

    /// Get device name for diagnostics
    pub fn device_name(&self) -> Result<String, cudarc::driver::DriverError> {
        self.device.name()
    }
}

/// CUDA kernel wrapper for STDP trace decay
pub struct STDPTraceDecayKernel {
    device: Arc<CudaDevice>,
    function: CudaFunction,
}

impl STDPTraceDecayKernel {
    /// Compile and load kernel
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(STDP_TRACE_DECAY_KERNEL)?;
        device.load_ptx(ptx, "neurox_trace", &["stdp_trace_decay"])?;
        let function = device
            .get_func("neurox_trace", "stdp_trace_decay")
            .unwrap();

        Ok(Self { device, function })
    }

    /// Launch kernel
    pub fn launch(
        &self,
        config: super::KernelConfig,
        a_pre: &mut cudarc::driver::CudaSlice<f32>,
        a_post1: &mut cudarc::driver::CudaSlice<f32>,
        a_post2: &mut cudarc::driver::CudaSlice<f32>,
        spike_flags: &cudarc::driver::CudaSlice<f32>,
        dt: f32,
        tau_pre: f32,
        tau_post1: f32,
        tau_post2: f32,
        n_neurons: i32,
    ) -> Result<(), cudarc::driver::DriverError> {
        let params = (
            a_pre,
            a_post1,
            a_post2,
            spike_flags,
            dt,
            tau_pre,
            tau_post1,
            tau_post2,
            n_neurons,
        );

        unsafe {
            self.function
                .clone()
                .launch(config.to_launch_config(), params)?;
        }

        Ok(())
    }

    /// Synchronize device after kernel execution
    pub fn synchronize(&self) -> Result<(), cudarc::driver::DriverError> {
        self.device.synchronize()
    }

    /// Get device name for diagnostics
    pub fn device_name(&self) -> Result<String, cudarc::driver::DriverError> {
        self.device.name()
    }
}
