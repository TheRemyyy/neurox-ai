//! CUDA GPU acceleration for neuromorphic computing
//!
//! When feature "cuda" is enabled: full GPU support via cudarc.
//! When disabled: stub types so the crate builds without NVIDIA stack.

#[cfg(feature = "cuda")]
pub mod cognitive_kernels;
#[cfg(feature = "cuda")]
pub mod cognitive_system;
#[cfg(feature = "cuda")]
pub mod context;
#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "cuda")]
pub mod motion_kernels;
#[cfg(feature = "cuda")]
pub mod quantization;
#[cfg(feature = "cuda")]
pub mod sparse_kernels;
#[cfg(feature = "cuda")]
pub mod spiking_conv_kernels;
#[cfg(feature = "cuda")]
pub mod v1_kernels;

#[cfg(feature = "cuda")]
pub use cognitive_system::GpuCognitiveSystem;
#[cfg(feature = "cuda")]
pub use context::CudaContext;
#[cfg(feature = "cuda")]
pub use motion_kernels::GpuMotionOutput;
#[cfg(feature = "cuda")]
pub use spiking_conv_kernels::{GpuSpikeMaxPool, GpuSpikingConv2D};
#[cfg(feature = "cuda")]
pub use v1_kernels::GpuV1OrientationSystem;
#[cfg(feature = "cuda")]
pub use motion_kernels::GpuMotionSystem;

#[cfg(feature = "cuda")]
use cudarc::driver::LaunchConfig;

#[cfg(feature = "cuda")]
/// Optimal kernel launch configuration for RTX 3070
#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    pub threads_per_block: u32,
    pub blocks: u32,
    pub shared_mem: u32,
}

#[cfg(feature = "cuda")]
impl KernelConfig {
    pub fn for_neurons(n_neurons: usize) -> Self {
        const THREADS_PER_BLOCK: u32 = 256;
        let blocks = ((n_neurons as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK).max(1);
        Self {
            threads_per_block: THREADS_PER_BLOCK,
            blocks,
            shared_mem: 0,
        }
    }
    pub fn to_launch_config(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.blocks, 1, 1),
            block_dim: (self.threads_per_block, 1, 1),
            shared_mem_bytes: self.shared_mem,
        }
    }
}

#[cfg(feature = "cuda")]
/// GPU memory info
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub total: usize,
    pub free: usize,
    pub used: usize,
}

#[cfg(feature = "cuda")]
impl GpuMemoryInfo {
    pub fn utilization(&self) -> f32 {
        (self.used as f32 / self.total as f32) * 100.0
    }
    pub fn format(&self) -> String {
        format!(
            "{:.2} GB / {:.2} GB ({:.1}%)",
            self.used as f64 / 1024_f64.powi(3),
            self.total as f64 / 1024_f64.powi(3),
            self.utilization()
        )
    }
}

// --- Stub when CUDA feature is disabled ---
#[cfg(not(feature = "cuda"))]
mod stub;
#[cfg(not(feature = "cuda"))]
pub use stub::*;
