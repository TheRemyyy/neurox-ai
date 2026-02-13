//! Stub implementations when CUDA feature is disabled.
//! Allows the crate to build without NVIDIA GPU or toolkit.

use std::sync::Arc;

const CUDA_DISABLED_MSG: &str =
    "CUDA support not compiled in. Build with --features cuda for GPU acceleration.";

/// Placeholder type for gpu_device when CUDA is disabled (never constructed).
#[derive(Debug)]
pub struct CudaDeviceStub;

/// Stub CudaContext - all operations return an error.
#[derive(Debug)]
pub struct CudaContext;

impl CudaContext {
    pub fn new(_device_id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Err(CUDA_DISABLED_MSG.into())
    }

    /// Create default context (device 0). Use this instead of Default to get Result.
    pub fn default_context() -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(0)
    }

    pub fn device_info(&self) -> Result<String, Box<dyn std::error::Error>> {
        Err(CUDA_DISABLED_MSG.into())
    }
}

/// GPU memory info (used when reporting "no GPU").
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub total: usize,
    pub free: usize,
    pub used: usize,
}

impl GpuMemoryInfo {
    pub fn utilization(&self) -> f32 {
        0.0
    }
    pub fn format(&self) -> String {
        "N/A (CUDA not compiled in)".to_string()
    }
}

/// Kernel config stub (no GPU).
#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    pub threads_per_block: u32,
    pub blocks: u32,
    pub shared_mem: u32,
}

impl KernelConfig {
    pub fn for_neurons(n_neurons: usize) -> Self {
        let threads_per_block = 256u32;
        let blocks = (n_neurons as u32).div_ceil(threads_per_block).max(1);
        Self {
            threads_per_block,
            blocks,
            shared_mem: 0,
        }
    }
}

/// Stub GPU V1 system.
#[derive(Debug)]
pub struct GpuV1OrientationSystem;

impl GpuV1OrientationSystem {
    pub fn new(
        _device: Arc<CudaDeviceStub>,
        _width: usize,
        _height: usize,
        _n_orientations: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Err(CUDA_DISABLED_MSG.into())
    }

    pub fn process(&mut self, _input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Err(CUDA_DISABLED_MSG.into())
    }
}

/// Stub motion output for fallback.
#[derive(Debug, Clone)]
pub struct GpuMotionOutput {
    pub direction_responses: Vec<f32>,
    pub speed_estimates: Vec<f32>,
    pub expansion_strength: f32,
    pub flow_x: Vec<f32>,
    pub flow_y: Vec<f32>,
}

/// Stub GPU Motion system.
#[derive(Debug)]
pub struct GpuMotionSystem;

impl GpuMotionSystem {
    pub fn new(
        _device: Arc<CudaDeviceStub>,
        _mt_width: usize,
        _mt_height: usize,
        _n_directions: usize,
        _n_orientations: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Err(CUDA_DISABLED_MSG.into())
    }

    pub fn process(
        &mut self,
        _v1_input: &[f32],
        _dt: f32,
    ) -> Result<GpuMotionOutput, Box<dyn std::error::Error>> {
        Err(CUDA_DISABLED_MSG.into())
    }
}

/// Stub GPU Cognitive system.
#[derive(Debug)]
pub struct GpuCognitiveSystem;

impl GpuCognitiveSystem {
    pub fn new(
        _context: &CudaContext,
        _max_items: usize,
        _dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Err(CUDA_DISABLED_MSG.into())
    }

    pub fn compute_attention(
        &mut self,
        _query: &[f32],
        _keys_flat: &[f32],
        _n_keys: usize,
        _dim: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Err(CUDA_DISABLED_MSG.into())
    }
}

/// Stub spiking conv layer.
#[derive(Debug)]
pub struct GpuSpikingConv2D;

impl GpuSpikingConv2D {
    pub fn new(
        _device: Arc<CudaDeviceStub>,
        _in_channels: usize,
        _out_channels: usize,
        _kernel_size: usize,
        _height: usize,
        _width: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Err(CUDA_DISABLED_MSG.into())
    }
}

/// Stub spike max pool.
#[derive(Debug)]
pub struct GpuSpikeMaxPool;

impl GpuSpikeMaxPool {
    pub fn new(
        _device: Arc<CudaDeviceStub>,
        _channels: usize,
        _height: usize,
        _width: usize,
        _pool_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Err(CUDA_DISABLED_MSG.into())
    }
}
