//! GPU-accelerated Cognitive System
//!
//! Integrates individual cognitive kernels into a cohesive high-level API.

use crate::cuda::cognitive_kernels::*;
use crate::cuda::context::CudaContext;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use std::sync::Arc;
use std::fmt;

pub struct GpuCognitiveSystem {
    device: Arc<CudaDevice>,
    
    // Kernels
    spike_corr: SpikeCorrelationKernel,
    cosine: CosineSimilarityKernel,
    token_enc: TokenEncodingKernel,
    softmax: SoftmaxKernel,
    lateral: LateralInhibitionKernel,
    storage: PatternStorageKernel,
    
    // Buffers (reused to avoid allocation)
    scores_buffer: CudaSlice<f32>,
    pattern_buffer: CudaSlice<f32>,
}

impl fmt::Debug for GpuCognitiveSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuCognitiveSystem {{ device: ... }}")
    }
}

impl GpuCognitiveSystem {
    pub fn new(context: &CudaContext, max_items: usize, dim: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let device = context.device().clone();
        
        // Initialize kernels
        let spike_corr = SpikeCorrelationKernel::new(device.clone())?;
        let cosine = CosineSimilarityKernel::new(device.clone())?;
        let token_enc = TokenEncodingKernel::new(device.clone())?;
        let softmax = SoftmaxKernel::new(device.clone())?;
        let lateral = LateralInhibitionKernel::new(device.clone())?;
        let storage = PatternStorageKernel::new(device.clone())?;
        
        // Allocate buffers
        let scores_buffer = device.alloc_zeros::<f32>(max_items)?;
        let pattern_buffer = device.alloc_zeros::<f32>(dim)?;
        
        Ok(Self {
            device,
            spike_corr,
            cosine,
            token_enc,
            softmax,
            lateral,
            storage,
            scores_buffer,
            pattern_buffer,
        })
    }
    
    /// Compute attention scores on GPU
    pub fn compute_attention(&mut self, query: &[f32], keys_flat: &[f32], n_keys: usize, dim: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Upload query
        let query_gpu = self.device.htod_copy(query.to_vec())?;
        let keys_gpu = self.device.htod_copy(keys_flat.to_vec())?;
        
        // Run kernel
        let config = crate::cuda::KernelConfig::for_neurons(n_keys);
        self.spike_corr.launch(config, &query_gpu, &keys_gpu, &mut self.scores_buffer, n_keys as i32, dim as i32, 0)?;
        
        // Apply softmax
        self.softmax.launch(config, &mut self.scores_buffer, n_keys as i32)?;
        
        // Download result
        let mut scores = vec![0.0; n_keys];
        self.device.dtoh_sync_copy_into(&self.scores_buffer.slice(0..n_keys), &mut scores)?;
        
        Ok(scores)
    }
}
