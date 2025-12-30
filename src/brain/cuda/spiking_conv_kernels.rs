//! GPU-Accelerated Spiking Convolutional Layers
//!
//! CUDA kernels for spiking 2D convolutions with LIF dynamics.
//! All operations run 100% on GPU - no CPU fallback.
//!
//! Architecture:
//! - SpikingConv2D: Convolutional kernel with LIF neurons
//! - SpikeMaxPool: Spike-based max pooling
//! - Full GPU pipeline for MNIST: Conv(3x3, 32) -> Pool -> Conv(3x3, 64) -> Pool -> Flatten

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync};
use std::sync::Arc;

/// Spiking Conv2D kernel - Convolution + LIF dynamics in single kernel
/// Dimensions packed into int array: [batch, in_h, in_w, in_c, out_h, out_w, out_c, kernel_h, kernel_w]
pub const SPIKING_CONV2D_KERNEL: &str = r#"
extern "C" __global__ void spiking_conv2d(
    const float* input_spikes,     // [batch × in_h × in_w × in_channels]
    const float* weights,          // [out_channels × kernel_h × kernel_w × in_channels]
    float* membrane_v,             // [batch × out_h × out_w × out_channels]
    float* output_spikes,          // [batch × out_h × out_w × out_channels]
    const int* dims,               // [9]: batch, in_h, in_w, in_c, out_h, out_w, out_c, kh, kw
    const float dt,
    const float tau_m,
    const float v_thresh,
    const float v_reset
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Unpack dimensions
    int batch_size = dims[0];
    int in_h = dims[1];
    int in_w = dims[2];
    int in_channels = dims[3];
    int out_h = dims[4];
    int out_w = dims[5];
    int out_channels = dims[6];
    int kernel_h = dims[7];
    int kernel_w = dims[8];
    
    int total_neurons = batch_size * out_h * out_w * out_channels;
    if (gid >= total_neurons) return;
    
    // Decode output position
    int oc = gid % out_channels;
    int temp = gid / out_channels;
    int ox = temp % out_w;
    temp /= out_w;
    int oy = temp % out_h;
    int b = temp / out_h;
    
    // Compute convolution
    float current = 0.0f;
    int pad_h = kernel_h / 2;
    int pad_w = kernel_w / 2;
    
    for (int ky = 0; ky < kernel_h; ky++) {
        for (int kx = 0; kx < kernel_w; kx++) {
            int iy = oy + ky - pad_h;
            int ix = ox + kx - pad_w;
            
            if (iy < 0 || iy >= in_h || ix < 0 || ix >= in_w) continue;
            
            for (int ic = 0; ic < in_channels; ic++) {
                int input_idx = ((b * in_h + iy) * in_w + ix) * in_channels + ic;
                float spike = input_spikes[input_idx];
                
                int w_idx = ((oc * kernel_h + ky) * kernel_w + kx) * in_channels + ic;
                float w = weights[w_idx];
                
                current += spike * w;
            }
        }
    }
    
    // LIF dynamics
    int output_idx = gid;
    float v = membrane_v[output_idx];
    
    float dv = (-(v - v_reset) + current) / tau_m * dt;
    v += dv;
    
    if (v >= v_thresh) {
        output_spikes[output_idx] = 1.0f;
        v = v_reset;
    } else {
        output_spikes[output_idx] = 0.0f;
    }
    
    membrane_v[output_idx] = v;
}
"#;

/// Spike-based Max Pooling kernel
pub const SPIKE_MAXPOOL_KERNEL: &str = r#"
extern "C" __global__ void spike_maxpool(
    const float* input_spikes,
    float* output_spikes,
    const int batch_size,
    const int in_h,
    const int in_w,
    const int channels,
    const int pool_size,
    const int stride
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = (in_h - pool_size) / stride + 1;
    int out_w = (in_w - pool_size) / stride + 1;
    int total_outputs = batch_size * out_h * out_w * channels;
    
    if (gid >= total_outputs) return;
    
    int c = gid % channels;
    int temp = gid / channels;
    int ox = temp % out_w;
    temp /= out_w;
    int oy = temp % out_h;
    int b = temp / out_h;
    
    float max_spike = 0.0f;
    
    for (int py = 0; py < pool_size; py++) {
        for (int px = 0; px < pool_size; px++) {
            int iy = oy * stride + py;
            int ix = ox * stride + px;
            
            if (iy < in_h && ix < in_w) {
                int input_idx = ((b * in_h + iy) * in_w + ix) * channels + c;
                float spike = input_spikes[input_idx];
                if (spike > max_spike) {
                    max_spike = spike;
                }
            }
        }
    }
    
    output_spikes[gid] = max_spike;
}
"#;

/// GPU Spiking Convolutional Layer
#[derive(Debug)]
pub struct GpuSpikingConv2D {
    device: Arc<CudaDevice>,
    kernel: CudaFunction,

    // Parameters
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    in_h: usize,
    in_w: usize,

    // GPU buffers
    weights: CudaSlice<f32>,
    membrane_v: CudaSlice<f32>,
    output_spikes: CudaSlice<f32>,
    dims_gpu: CudaSlice<i32>,

    // LIF parameters
    dt: f32,
    tau_m: f32,
    v_thresh: f32,
    v_reset: f32,
}

impl GpuSpikingConv2D {
    pub fn new(
        device: Arc<CudaDevice>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        in_h: usize,
        in_w: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(SPIKING_CONV2D_KERNEL)?;
        device.load_ptx(ptx, "spiking_conv2d", &["spiking_conv2d"])?;
        let kernel = device.get_func("spiking_conv2d", "spiking_conv2d").unwrap();

        let kernel_h = kernel_size;
        let kernel_w = kernel_size;
        let out_h = in_h; // same padding
        let out_w = in_w;

        // Xavier init weights
        let n_weights = out_channels * kernel_h * kernel_w * in_channels;
        let xavier_std = (2.0 / (in_channels + out_channels) as f32).sqrt();
        let weights_host: Vec<f32> = (0..n_weights)
            .map(|i| {
                let x = ((i * 1103515245 + 12345) % 1000) as f32 / 1000.0;
                (x - 0.5) * xavier_std * 2.0
            })
            .collect();
        let weights = device.htod_copy(weights_host)?;

        // Allocate state buffers
        let n_outputs = out_h * out_w * out_channels;
        let membrane_v = device.alloc_zeros::<f32>(n_outputs)?;
        let output_spikes = device.alloc_zeros::<f32>(n_outputs)?;

        // Upload dimensions to GPU
        let dims_host: Vec<i32> = vec![
            1, // batch
            in_h as i32,
            in_w as i32,
            in_channels as i32,
            out_h as i32,
            out_w as i32,
            out_channels as i32,
            kernel_h as i32,
            kernel_w as i32,
        ];
        let dims_gpu = device.htod_copy(dims_host)?;

        log::info!(
            "GPU SpikingConv2D: {}x{}x{} -> {}x{}x{} ({} params)",
            in_h,
            in_w,
            in_channels,
            out_h,
            out_w,
            out_channels,
            n_weights
        );

        Ok(Self {
            device,
            kernel,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            in_h,
            in_w,
            weights,
            membrane_v,
            output_spikes,
            dims_gpu,
            dt: 1.0,
            tau_m: 20.0,
            v_thresh: -55.0,
            v_reset: -70.0,
        })
    }

    pub fn forward(
        &mut self,
        input_spikes: &CudaSlice<f32>,
    ) -> Result<&CudaSlice<f32>, Box<dyn std::error::Error>> {
        let out_h = self.in_h;
        let out_w = self.in_w;
        let total_neurons = out_h * out_w * self.out_channels;

        let threads = 256;
        let blocks = (total_neurons + threads - 1) / threads;

        // 9 params now (under 12 limit)
        let params = (
            input_spikes,
            &self.weights,
            &mut self.membrane_v,
            &mut self.output_spikes,
            &self.dims_gpu,
            self.dt,
            self.tau_m,
            self.v_thresh,
            self.v_reset,
        );

        unsafe {
            self.kernel.clone().launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                params,
            )?;
        }

        Ok(&self.output_spikes)
    }

    pub fn reset(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let n_outputs = self.in_h * self.in_w * self.out_channels;
        self.membrane_v = self.device.alloc_zeros::<f32>(n_outputs)?;
        Ok(())
    }

    pub fn out_h(&self) -> usize {
        self.in_h
    }
    pub fn out_w(&self) -> usize {
        self.in_w
    }
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}

/// GPU Spike Max Pooling
#[derive(Debug)]
pub struct GpuSpikeMaxPool {
    device: Arc<CudaDevice>,
    kernel: CudaFunction,
    pool_size: usize,
    stride: usize,
    in_h: usize,
    in_w: usize,
    channels: usize,
    output_spikes: CudaSlice<f32>,
}

impl GpuSpikeMaxPool {
    pub fn new(
        device: Arc<CudaDevice>,
        pool_size: usize,
        stride: usize,
        in_h: usize,
        in_w: usize,
        channels: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(SPIKE_MAXPOOL_KERNEL)?;
        device.load_ptx(ptx, "spike_maxpool", &["spike_maxpool"])?;
        let kernel = device.get_func("spike_maxpool", "spike_maxpool").unwrap();

        let out_h = (in_h - pool_size) / stride + 1;
        let out_w = (in_w - pool_size) / stride + 1;
        let n_outputs = out_h * out_w * channels;
        let output_spikes = device.alloc_zeros::<f32>(n_outputs)?;

        log::info!(
            "GPU SpikeMaxPool: {}x{}x{} -> {}x{}x{} (pool={})",
            in_h,
            in_w,
            channels,
            out_h,
            out_w,
            channels,
            pool_size
        );

        Ok(Self {
            device,
            kernel,
            pool_size,
            stride,
            in_h,
            in_w,
            channels,
            output_spikes,
        })
    }

    pub fn forward(
        &mut self,
        input_spikes: &CudaSlice<f32>,
    ) -> Result<&CudaSlice<f32>, Box<dyn std::error::Error>> {
        let out_h = (self.in_h - self.pool_size) / self.stride + 1;
        let out_w = (self.in_w - self.pool_size) / self.stride + 1;
        let total_outputs = out_h * out_w * self.channels;

        let threads = 256;
        let blocks = (total_outputs + threads - 1) / threads;

        // 8 params (under limit)
        let params = (
            input_spikes,
            &mut self.output_spikes,
            1i32, // batch
            self.in_h as i32,
            self.in_w as i32,
            self.channels as i32,
            self.pool_size as i32,
            self.stride as i32,
        );

        unsafe {
            self.kernel.clone().launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                params,
            )?;
        }

        Ok(&self.output_spikes)
    }

    pub fn out_h(&self) -> usize {
        (self.in_h - self.pool_size) / self.stride + 1
    }
    pub fn out_w(&self) -> usize {
        (self.in_w - self.pool_size) / self.stride + 1
    }
    pub fn out_channels(&self) -> usize {
        self.channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_spiking_conv2d() {
        match CudaDevice::new(0) {
            Ok(device) => {
                let mut conv = GpuSpikingConv2D::new(device.clone(), 1, 8, 3, 28, 28)
                    .expect("Failed to create conv");

                let input: Vec<f32> = (0..28 * 28)
                    .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
                    .collect();
                let input_gpu = device.htod_copy(input).unwrap();

                let output = conv.forward(&input_gpu).expect("Forward failed");

                let mut output_host = vec![0.0f32; 28 * 28 * 8];
                device
                    .dtoh_sync_copy_into(output, &mut output_host)
                    .unwrap();

                let non_zero = output_host.iter().filter(|&&x| x > 0.5).count();
                eprintln!("GPU SpikingConv2D: {} spikes", non_zero);
            }
            Err(_) => eprintln!("CUDA not available"),
        }
    }
}
