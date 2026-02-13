//! CUDA context and GPU memory management
//!
//! Handles GPU device initialization, memory allocation, and resource cleanup.

use super::GpuMemoryInfo;
use crate::brain::cuda::kernels::{
    LIFUpdateKernel, STDPTraceDecayKernel, SpikePropagationKernel, TripletSTDPKernel,
    VectorAccumulateKernel,
};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// Main CUDA context for neuromorphic simulation
pub struct CudaContext {
    /// CUDA device
    device: Arc<CudaDevice>,

    /// LIF update kernel
    lif_kernel: LIFUpdateKernel,

    /// Spike propagation kernel
    spike_kernel: SpikePropagationKernel,

    /// Triplet STDP kernel
    stdp_kernel: TripletSTDPKernel,

    /// STDP trace decay kernel
    trace_kernel: STDPTraceDecayKernel,

    /// Vector accumulation kernel
    accumulate_kernel: VectorAccumulateKernel,

    /// Device ordinal
    device_id: usize,
}

impl CudaContext {
    /// Initialize CUDA context with specified device
    pub fn new(device_id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("Initializing CUDA device {}", device_id);

        let device = CudaDevice::new(device_id)?;
        log::info!("Device name: {}", device.name()?);

        // Compile kernels
        log::info!("Compiling LIF update kernel...");
        let lif_kernel = LIFUpdateKernel::new(device.clone())?;

        log::info!("Compiling spike propagation kernel...");
        let spike_kernel = SpikePropagationKernel::new(device.clone())?;

        log::info!("Compiling Triplet STDP kernel...");
        let stdp_kernel = TripletSTDPKernel::new(device.clone())?;

        log::info!("Compiling STDP trace decay kernel...");
        let trace_kernel = STDPTraceDecayKernel::new(device.clone())?;

        log::info!("Compiling Vector Accumulate kernel...");
        let accumulate_kernel = VectorAccumulateKernel::new(device.clone())?;

        log::info!("CUDA context initialized successfully");

        Ok(Self {
            device,
            lif_kernel,
            spike_kernel,
            stdp_kernel,
            trace_kernel,
            accumulate_kernel,
            device_id,
        })
    }

    /// Initialize with default device (0)
    pub fn default() -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(0)
    }

    /// Get device information
    pub fn device_info(&self) -> Result<String, Box<dyn std::error::Error>> {
        let name = self.device.name()?;
        let mem_info = self.memory_info()?;

        Ok(format!(
            "Device {}: {}\nMemory: {}",
            self.device_id,
            name,
            mem_info.format()
        ))
    }

    /// Get GPU memory information
    pub fn memory_info(&self) -> Result<GpuMemoryInfo, Box<dyn std::error::Error>> {
        // Get actual GPU memory usage via CUDA driver API
        let (free, total) = unsafe {
            use cudarc::driver::sys;
            let mut free: usize = 0;
            let mut total: usize = 0;

            // Query actual GPU memory using cuMemGetInfo
            let result = sys::lib().cuMemGetInfo_v2(&mut free, &mut total);
            if result == sys::CUresult::CUDA_SUCCESS {
                (free, total)
            } else {
                // Fallback to estimated values if query fails
                (8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
            }
        };

        let used = total.saturating_sub(free);

        Ok(GpuMemoryInfo { total, free, used })
    }

    /// Allocate GPU memory
    pub fn allocate<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        size: usize,
    ) -> Result<CudaSlice<T>, Box<dyn std::error::Error>> {
        let slice = self.device.alloc_zeros::<T>(size)?;
        Ok(slice)
    }

    /// Get underlying device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get LIF kernel
    pub fn lif_kernel(&self) -> &LIFUpdateKernel {
        &self.lif_kernel
    }

    /// Get spike propagation kernel
    pub fn spike_kernel(&self) -> &SpikePropagationKernel {
        &self.spike_kernel
    }

    /// Get Triplet STDP kernel
    pub fn stdp_kernel(&self) -> &TripletSTDPKernel {
        &self.stdp_kernel
    }

    /// Get STDP trace decay kernel
    pub fn trace_kernel(&self) -> &STDPTraceDecayKernel {
        &self.trace_kernel
    }

    /// Get vector accumulation kernel
    pub fn accumulate_kernel(&self) -> &VectorAccumulateKernel {
        &self.accumulate_kernel
    }

    /// Synchronize device (wait for all operations to complete)
    pub fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.device.synchronize()?;
        Ok(())
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        log::info!("Cleaning up CUDA context");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::DeviceSlice;

    #[test]
    #[ignore] // Only run if CUDA is available
    fn test_cuda_init() {
        let ctx = CudaContext::default();
        assert!(ctx.is_ok());

        if let Ok(ctx) = ctx {
            let info = ctx.device_info().unwrap();
            println!("{}", info);
        }
    }

    #[test]
    #[ignore]
    fn test_memory_allocation() {
        let ctx = CudaContext::default().unwrap();
        let slice: CudaSlice<f32> = ctx.allocate(1000).unwrap();
        assert_eq!(slice.len(), 1000);
    }
}
