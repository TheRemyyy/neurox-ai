//! Sparse Matrix Kernels for Memory Efficiency
//!
//! Saves 90%+ memory by storing only non-zero values

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, CudaSlice, DeviceSlice};
use std::sync::Arc;

/// Sparse transition matrix update kernel (CSR format)
pub const SPARSE_TRANSITION_UPDATE_KERNEL: &str = r#"
extern "C" __global__ void sparse_transition_update(
    int* row_ptr,
    int* col_idx,
    float* values,
    const int* tokens,
    const int n_tokens,
    const int vocab_size,
    const float learning_rate,
    int* nnz_counter
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_tokens - 1) return;

    int prev_token = tokens[tid];
    int curr_token = tokens[tid + 1];

    if (prev_token >= vocab_size || curr_token >= vocab_size) return;

    // Find or create entry in sparse matrix
    int row_start = row_ptr[prev_token];
    int row_end = row_ptr[prev_token + 1];

    // Search for existing entry
    int found_idx = -1;
    for (int i = row_start; i < row_end; i++) {
        if (col_idx[i] == curr_token) {
            found_idx = i;
            break;
        }
    }

    if (found_idx >= 0) {
        // Update existing entry
        atomicAdd(&values[found_idx], learning_rate);
    } else {
        // Insert new entry (simplified - in practice need atomic expansion)
        int new_idx = atomicAdd(nnz_counter, 1);
        if (new_idx < 10000000) { // Max 10M non-zeros
            col_idx[new_idx] = curr_token;
            values[new_idx] = learning_rate;
        }
    }
}
"#;

/// Sparse matrix normalization kernel
pub const SPARSE_NORMALIZE_KERNEL: &str = r#"
extern "C" __global__ void sparse_normalize(
    const int* row_ptr,
    const int* col_idx,
    float* values,
    const int vocab_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= vocab_size) return;

    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    // Compute sum
    float sum = 0.0f;
    for (int i = row_start; i < row_end; i++) {
        sum += values[i];
    }

    // Normalize
    if (sum > 1e-8f) {
        for (int i = row_start; i < row_end; i++) {
            values[i] /= sum;
        }
    }
}
"#;

/// Sparse matrix lookup kernel (for generation)
pub const SPARSE_LOOKUP_KERNEL: &str = r#"
extern "C" __global__ void sparse_lookup_row(
    const int* row_ptr,
    const int* col_idx,
    const float* values,
    const int row_idx,
    float* dense_output,
    const int vocab_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= vocab_size) return;

    // Initialize to zero
    dense_output[tid] = 0.0f;

    // Fill in non-zero values
    int row_start = row_ptr[row_idx];
    int row_end = row_ptr[row_idx + 1];

    for (int i = row_start; i < row_end; i++) {
        if (col_idx[i] == tid) {
            dense_output[tid] = values[i];
            break;
        }
    }
}
"#;

/// Sparse Transition Matrix (CSR format on GPU)
pub struct SparseTransitionMatrix {
    device: Arc<CudaDevice>,
    vocab_size: usize,

    // CSR format buffers (on GPU)
    row_ptr: CudaSlice<i32>,
    col_idx: CudaSlice<i32>,
    values: CudaSlice<f32>,
    nnz_counter: CudaSlice<i32>,

    // Current non-zero count
    current_nnz: usize,
    max_nnz: usize,

    // Kernels
    update_kernel: CudaFunction,
    normalize_kernel: CudaFunction,
    lookup_kernel: CudaFunction,
}

impl SparseTransitionMatrix {
    /// Create sparse transition matrix with dynamic growth
    /// Starts small (10k entries) and grows as needed up to max_nnz
    pub fn new(
        device: Arc<CudaDevice>,
        vocab_size: usize,
        max_nnz: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Start with small initial allocation (1% of max, min 10k)
        let initial_nnz = (max_nnz / 100).max(10_000).min(max_nnz);

        // Allocate CSR buffers (start small!)
        let row_ptr = device.alloc_zeros::<i32>(vocab_size + 1)?;
        let col_idx = device.alloc_zeros::<i32>(initial_nnz)?;
        let values = device.alloc_zeros::<f32>(initial_nnz)?;
        let nnz_counter = device.alloc_zeros::<i32>(1)?;

        // Compile kernels
        let update_ptx = cudarc::nvrtc::compile_ptx(SPARSE_TRANSITION_UPDATE_KERNEL)?;
        device.load_ptx(update_ptx, "sparse_update", &["sparse_transition_update"])?;
        let update_kernel = device.get_func("sparse_update", "sparse_transition_update").unwrap();

        let normalize_ptx = cudarc::nvrtc::compile_ptx(SPARSE_NORMALIZE_KERNEL)?;
        device.load_ptx(normalize_ptx, "sparse_norm", &["sparse_normalize"])?;
        let normalize_kernel = device.get_func("sparse_norm", "sparse_normalize").unwrap();

        let lookup_ptx = cudarc::nvrtc::compile_ptx(SPARSE_LOOKUP_KERNEL)?;
        device.load_ptx(lookup_ptx, "sparse_lookup", &["sparse_lookup_row"])?;
        let lookup_kernel = device.get_func("sparse_lookup", "sparse_lookup_row").unwrap();

        let initial_mb = (initial_nnz * 12) / (1024 * 1024);
        let max_mb = (max_nnz * 12) / (1024 * 1024);
        log::info!("Sparse transition matrix: {} MB initial, {} MB max", initial_mb, max_mb);
        log::info!("Dynamic growth: starts at {}% capacity", (initial_nnz * 100) / max_nnz);
        log::info!("Memory savings vs dense: {}%",
            100 - (max_mb * 100) / ((vocab_size * vocab_size * 4) / (1024 * 1024)));

        Ok(Self {
            device,
            vocab_size,
            row_ptr,
            col_idx,
            values,
            nnz_counter,
            current_nnz: 0,
            max_nnz,
            update_kernel,
            normalize_kernel,
            lookup_kernel,
        })
    }

    /// Grow buffers when needed (doubles size up to max)
    fn grow_if_needed(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let current_capacity = self.col_idx.len();
        let usage_percent = (self.current_nnz * 100) / current_capacity;

        // Grow when 80% full
        if usage_percent >= 80 && current_capacity < self.max_nnz {
            let new_capacity = (current_capacity * 2).min(self.max_nnz);

            log::info!("Growing sparse matrix: {} → {} entries ({} MB → {} MB)",
                current_capacity, new_capacity,
                (current_capacity * 12) / (1024 * 1024),
                (new_capacity * 12) / (1024 * 1024));

            // Download existing data
            let old_col_idx = self.device.dtoh_sync_copy(&self.col_idx)?;
            let old_values = self.device.dtoh_sync_copy(&self.values)?;

            // Allocate larger buffers
            let mut new_col_idx = vec![0i32; new_capacity];
            let mut new_values = vec![0f32; new_capacity];

            // Copy existing data
            new_col_idx[..self.current_nnz].copy_from_slice(&old_col_idx[..self.current_nnz]);
            new_values[..self.current_nnz].copy_from_slice(&old_values[..self.current_nnz]);

            // Upload to GPU
            self.col_idx = self.device.htod_sync_copy(&new_col_idx)?;
            self.values = self.device.htod_sync_copy(&new_values)?;

            log::info!("✓ Growth complete");
        }

        Ok(())
    }

    /// Update with new transitions (with automatic growth)
    pub fn update(&mut self, tokens: &CudaSlice<i32>, n_tokens: i32, learning_rate: f32) -> Result<(), Box<dyn std::error::Error>> {
        // Check if we need to grow before updating
        self.grow_if_needed()?;

        let config = crate::brain::cuda::KernelConfig::for_neurons(n_tokens as usize);

        let params = (
            &self.row_ptr,
            &self.col_idx,
            &mut self.values,
            tokens,
            n_tokens,
            self.vocab_size as i32,
            learning_rate,
            &mut self.nnz_counter,
        );

        unsafe {
            self.update_kernel.clone().launch(config.to_launch_config(), params)?;
        }

        self.device.synchronize()?;

        // Update NNZ count
        let nnz_cpu = self.device.dtoh_sync_copy(&self.nnz_counter)?;
        self.current_nnz = nnz_cpu[0] as usize;

        Ok(())
    }

    /// Normalize probabilities
    pub fn normalize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let config = crate::brain::cuda::KernelConfig::for_neurons(self.vocab_size);

        let params = (
            &self.row_ptr,
            &self.col_idx,
            &mut self.values,
            self.vocab_size as i32,
        );

        unsafe {
            self.normalize_kernel.clone().launch(config.to_launch_config(), params)?;
        }

        self.device.synchronize()?;
        Ok(())
    }

    /// Lookup row (for generation)
    pub fn lookup_row(&self, row_idx: i32, output: &mut CudaSlice<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let config = crate::brain::cuda::KernelConfig::for_neurons(self.vocab_size);

        let params = (
            &self.row_ptr,
            &self.col_idx,
            &self.values,
            row_idx,
            output,
            self.vocab_size as i32,
        );

        unsafe {
            self.lookup_kernel.clone().launch(config.to_launch_config(), params)?;
        }

        self.device.synchronize()?;
        Ok(())
    }

    /// Get current sparsity
    pub fn sparsity(&self) -> f32 {
        let total_possible = self.vocab_size * self.vocab_size;
        1.0 - (self.current_nnz as f32 / total_possible as f32)
    }

    /// Get memory usage in MB
    pub fn memory_mb(&self) -> usize {
        (self.current_nnz * 12) / (1024 * 1024) // 12 bytes per entry (i32 + i32 + f32)
    }
}
