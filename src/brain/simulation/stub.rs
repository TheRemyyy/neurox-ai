//! Stub simulation types when CUDA is disabled (build without --features cuda).

use crate::brain::connectivity::SparseConnectivity;
use std::sync::Arc;

const MSG: &str = "Simulator requires CUDA. Build with --features cuda.";

/// Stub Simulator - all constructors return an error.
#[derive(Debug)]
pub struct Simulator;

impl Simulator {
    pub fn new(
        _n_neurons: usize,
        _dt: f32,
        _cuda: Arc<crate::brain::cuda::CudaContext>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Err(MSG.into())
    }

    pub fn with_connectivity(
        _n_neurons: usize,
        _dt: f32,
        _cuda: Arc<crate::brain::cuda::CudaContext>,
        _connectivity: &SparseConnectivity,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Err(MSG.into())
    }
}

/// Stub optimization stats.
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub event_driven_enabled: bool,
    pub sparsity_threshold: f32,
    pub current_sparsity: f32,
    pub active_neurons: usize,
    pub total_neurons: usize,
    pub mode: String,
}
