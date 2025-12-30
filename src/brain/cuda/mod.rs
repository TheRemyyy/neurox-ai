//! CUDA GPU acceleration for neuromorphic computing
//!
//! Leverages cudarc for runtime PTX compilation and safe CUDA abstractions.
//! Optimized for RTX 3070 (8GB VRAM, 5888 CUDA cores).

pub mod cognitive_kernels;
pub mod cognitive_system;
pub mod context;
pub mod kernels;
pub mod motion_kernels;
pub mod quantization;
pub mod sparse_kernels;
pub mod v1_kernels;

pub use cognitive_system::GpuCognitiveSystem;
pub use context::CudaContext;


use cudarc::driver::LaunchConfig;

/// Optimal kernel launch configuration for RTX 3070
#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    /// Threads per block (256-512, multiple of 32)
    pub threads_per_block: u32,

    /// Number of blocks
    pub blocks: u32,

    /// Shared memory per block (bytes)
    pub shared_mem: u32,
}

impl KernelConfig {
    /// Calculate optimal launch config for neuron count
    pub fn for_neurons(n_neurons: usize) -> Self {
        const THREADS_PER_BLOCK: u32 = 256;

        let blocks = ((n_neurons as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK).max(1);

        Self {
            threads_per_block: THREADS_PER_BLOCK,
            blocks,
            shared_mem: 0, // Dynamic allocation if needed
        }
    }

    /// Convert to cudarc LaunchConfig
    pub fn to_launch_config(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.blocks, 1, 1),
            block_dim: (self.threads_per_block, 1, 1),
            shared_mem_bytes: self.shared_mem,
        }
    }
}

/// GPU memory info
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub total: usize,
    pub free: usize,
    pub used: usize,
}

impl GpuMemoryInfo {
    /// Memory utilization percentage
    pub fn utilization(&self) -> f32 {
        (self.used as f32 / self.total as f32) * 100.0
    }

    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "{:.2} GB / {:.2} GB ({:.1}%)",
            self.used as f64 / 1024_f64.powi(3),
            self.total as f64 / 1024_f64.powi(3),
            self.utilization()
        )
    }
}
