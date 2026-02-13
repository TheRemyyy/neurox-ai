//! Info Plugin
//!
//! Displays system and GPU information.

use crate::brain::cuda::CudaContext;
use std::sync::Arc;

/// Info plugin for displaying system information
pub struct InfoPlugin;

impl InfoPlugin {
    /// Create a new info plugin
    pub fn new() -> Self {
        Self
    }

    /// Display system and GPU information
    pub fn display_info(&self) -> Result<(), Box<dyn std::error::Error>> {
        let cuda_ctx = match CudaContext::default_context() {
            Ok(ctx) => Arc::new(ctx),
            Err(e) => {
                println!("CUDA not available: {}", e);
                println!("Build with --features cuda for GPU support.");
                return Ok(());
            }
        };
        println!("{}", cuda_ctx.device_info()?);
        Ok(())
    }
}

impl Default for InfoPlugin {
    fn default() -> Self {
        Self::new()
    }
}
