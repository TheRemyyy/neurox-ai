//! Simulation engine for spiking neural networks

pub mod event_queue;

pub use event_queue::{EventQueue, SpikeEvent, DelayBuffer};

use crate::cuda::{CudaContext, KernelConfig};
use crate::connectivity::SparseConnectivity;
use cudarc::driver::CudaSlice;
use std::sync::Arc;

/// Main simulator for GPU-accelerated SNN
pub struct Simulator {
    /// CUDA context
    cuda: Arc<CudaContext>,

    /// Number of neurons
    n_neurons: usize,

    /// Current timestep
    timestep: u64,

    /// Timestep duration (ms)
    dt: f32,

    // GPU buffers (need to be mutable for host-to-device copies)
    membrane_v: CudaSlice<f32>,
    thresholds: CudaSlice<f32>,
    tau_m: CudaSlice<f32>,
    v_reset: CudaSlice<f32>,
    refractory: CudaSlice<u8>,
    spike_flags: CudaSlice<f32>,
    input_currents: CudaSlice<f32>,

    // Preallocated synaptic currents buffer (optimization)
    synaptic_currents: CudaSlice<f32>,

    // Sparse synaptic connectivity (optional)
    connectivity: Option<SparseConnectivityGPU>,

    // Event-driven processing
    event_queue: EventQueue,
    delay_buffer: DelayBuffer,

    // Active neuron indices (for conditional execution)
    active_neurons: Vec<u32>,

    // Event-driven mode flag
    event_driven: bool,

    // Sparsity threshold (% active neurons to switch modes)
    sparsity_threshold: f32,
}

/// GPU-resident sparse connectivity
struct SparseConnectivityGPU {
    row_ptr: CudaSlice<i32>,
    col_idx: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    n_synapses: usize,
}

impl Simulator {
    /// Create new simulator
    pub fn new(
        n_neurons: usize,
        dt: f32,
        cuda: Arc<CudaContext>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("Initializing simulator with {} neurons", n_neurons);

        // Allocate GPU memory
        let mut membrane_v = cuda.allocate(n_neurons)?;
        let mut thresholds = cuda.allocate(n_neurons)?;
        let mut tau_m = cuda.allocate(n_neurons)?;
        let mut v_reset = cuda.allocate(n_neurons)?;
        let refractory = cuda.allocate(n_neurons)?;
        let spike_flags = cuda.allocate(n_neurons)?;
        let input_currents = cuda.allocate(n_neurons)?;
        let synaptic_currents = cuda.allocate(n_neurons)?; // Preallocate for spike propagation

        // Initialize with biological defaults
        let init_state = Self::create_initial_state(n_neurons);

        // Copy to GPU
        let device = cuda.device();
        device.htod_sync_copy_into(&init_state.membrane_v, &mut membrane_v)?;
        device.htod_sync_copy_into(&init_state.thresholds, &mut thresholds)?;
        device.htod_sync_copy_into(&init_state.tau_m, &mut tau_m)?;
        device.htod_sync_copy_into(&init_state.v_reset, &mut v_reset)?;

        log::info!("Simulator initialized successfully");

        // Initialize event-driven components
        let event_queue = EventQueue::new(n_neurons * 10); // 10× capacity for bursts
        let delay_buffer = DelayBuffer::new(20); // 20ms max delay (timesteps)

        Ok(Self {
            cuda,
            n_neurons,
            timestep: 0,
            dt,
            membrane_v,
            thresholds,
            tau_m,
            v_reset,
            refractory,
            spike_flags,
            input_currents,
            synaptic_currents,
            connectivity: None,
            event_queue,
            delay_buffer,
            active_neurons: Vec::with_capacity(n_neurons / 10), // Expect ~10% sparsity
            event_driven: true, // Enable by default
            sparsity_threshold: 0.15, // Switch to dense mode if >15% active
        })
    }

    /// Create simulator with sparse connectivity
    pub fn with_connectivity(
        n_neurons: usize,
        dt: f32,
        cuda: Arc<CudaContext>,
        connectivity: &SparseConnectivity,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("Initializing simulator with sparse connectivity");
        log::info!("  Neurons: {}", n_neurons);
        log::info!("  Synapses: {}", connectivity.nnz);
        log::info!("  Avg degree: {:.2}", connectivity.avg_degree());

        // Create base simulator
        let mut sim = Self::new(n_neurons, dt, cuda.clone())?;

        // Upload connectivity to GPU
        log::info!("Uploading connectivity to GPU...");
        let mut row_ptr = cuda.allocate(connectivity.row_ptr.len())?;
        let mut col_idx = cuda.allocate(connectivity.col_idx.len())?;
        let mut weights = cuda.allocate(connectivity.weights.len())?;

        let device = cuda.device();
        device.htod_sync_copy_into(&connectivity.row_ptr, &mut row_ptr)?;
        device.htod_sync_copy_into(&connectivity.col_idx, &mut col_idx)?;
        device.htod_sync_copy_into(&connectivity.weights, &mut weights)?;

        sim.connectivity = Some(SparseConnectivityGPU {
            row_ptr,
            col_idx,
            weights,
            n_synapses: connectivity.nnz,
        });

        log::info!("Sparse connectivity uploaded to GPU");

        Ok(sim)
    }

    /// Create initial neuron state
    fn create_initial_state(n_neurons: usize) -> InitialState {
        let membrane_v = vec![-70.0; n_neurons];
        let thresholds = vec![-55.0; n_neurons];
        let tau_m = vec![20.0; n_neurons];
        let v_reset = vec![-70.0; n_neurons];

        InitialState {
            membrane_v,
            thresholds,
            tau_m,
            v_reset,
        }
    }

    /// Step simulation forward by one timestep (event-driven)
    pub fn step(&mut self, external_input: Option<&[f32]>) -> Result<(), Box<dyn std::error::Error>> {
        // Update input currents if provided
        if let Some(input) = external_input {
            assert_eq!(input.len(), self.n_neurons);
            self.cuda
                .device()
                .htod_sync_copy_into(input, &mut self.input_currents)?;
        }

        // Retrieve delayed spikes from delay buffer
        let delayed_spikes = self.delay_buffer.get_current_spikes();
        for &neuron_id in delayed_spikes {
            // Add delayed spike to event queue
            self.event_queue.push(SpikeEvent::new(
                neuron_id,
                (self.timestep % 65536) as u16,
                0, // region_id
            ));
        }

        // Advance delay buffer to next timestep
        self.delay_buffer.advance();

        // Determine execution mode based on sparsity
        let active_count = self.event_queue.len();
        let sparsity = active_count as f32 / self.n_neurons as f32;

        // Use dense mode if:
        // - Event-driven is disabled
        // - No events yet (bootstrap phase)
        // - Activity above sparsity threshold
        if !self.event_driven || active_count == 0 || sparsity >= self.sparsity_threshold {
            // Dense mode: Process all neurons (traditional time-stepped)
            self.step_dense()?;
        } else {
            // Event-driven mode: Process only active neurons
            self.step_sparse()?;
        }

        self.timestep += 1;

        Ok(())
    }

    /// Dense time-stepped update (all neurons)
    fn step_dense(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let config = KernelConfig::for_neurons(self.n_neurons);

        self.cuda.lif_kernel().launch(
            config,
            &self.membrane_v,
            &self.thresholds,
            &self.tau_m,
            &self.v_reset,
            &mut self.refractory,
            &mut self.spike_flags,
            &self.input_currents,
            self.n_neurons as i32,
            self.dt,
            10.0, // r_m
        )?;

        // Collect spikes and propagate
        self.collect_and_propagate_spikes()?;

        Ok(())
    }

    /// Sparse event-driven update (active neurons only)
    fn step_sparse(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Extract active neuron IDs from event queue
        self.active_neurons.clear();
        while let Some(event) = self.event_queue.pop() {
            self.active_neurons.push(event.neuron_id);
        }

        if self.active_neurons.is_empty() {
            return Ok(()); // No active neurons, skip computation
        }

        // TODO: Launch sparse kernel for active neurons only
        // For now, fall back to dense mode
        // This requires a new CUDA kernel that processes only active_neurons indices
        self.step_dense()?;

        Ok(())
    }

    /// Collect spikes and propagate through synapses
    fn collect_and_propagate_spikes(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Optimization: Only download spikes every 100 timesteps (10ms @ 0.1ms/step)
        // This reduces GPU → CPU memory transfer overhead significantly
        if self.timestep % 100 == 0 {
            // Get spikes from GPU
            let spikes = self.get_spikes()?;

            // Add spiking neurons to event queue and delay buffer
            for (neuron_id, &spike_flag) in spikes.iter().enumerate() {
                if spike_flag > 0.5 {
                    // Add to delay buffer (synaptic delay of 1ms)
                    self.delay_buffer.add_spike(neuron_id as u32, 1);

                    // Add to event queue for immediate processing
                    self.event_queue.push(SpikeEvent::new(
                        neuron_id as u32,
                        (self.timestep % 65536) as u16,
                        0, // region_id
                    ));
                }
            }
        }

        // Propagate spikes through synapses if connectivity is present
        if let Some(ref conn) = self.connectivity {
            let config = KernelConfig::for_neurons(self.n_neurons);

            self.cuda.spike_kernel().launch(
                config,
                &conn.row_ptr,
                &conn.col_idx,
                &conn.weights,
                &self.spike_flags,
                &mut self.synaptic_currents,
                self.n_neurons as i32,
            )?;

            // TODO: Accumulate synaptic_currents into input_currents
        }

        Ok(())
    }

    /// Get spike flags from GPU
    pub fn get_spikes(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut spikes = vec![0.0; self.n_neurons];
        self.cuda
            .device()
            .dtoh_sync_copy_into(&self.spike_flags, &mut spikes)?;
        Ok(spikes)
    }

    /// Get membrane potentials
    pub fn get_voltages(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut voltages = vec![0.0; self.n_neurons];
        self.cuda
            .device()
            .dtoh_sync_copy_into(&self.membrane_v, &mut voltages)?;
        Ok(voltages)
    }

    /// Get current timestep
    pub fn timestep(&self) -> u64 {
        self.timestep
    }

    /// Get current simulation time (ms)
    pub fn time(&self) -> f32 {
        self.timestep as f32 * self.dt
    }

    /// Get number of neurons
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Get timestep duration (ms)
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Synchronize GPU
    pub fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.cuda.synchronize()
    }
}

struct InitialState {
    membrane_v: Vec<f32>,
    thresholds: Vec<f32>,
    tau_m: Vec<f32>,
    v_reset: Vec<f32>,
}
