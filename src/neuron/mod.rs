//! Neuron models and dynamics
//!
//! Implements biologically-inspired spiking neuron models with GPU-optimized data layouts.

pub mod lif;
pub mod hierarchical;
pub mod interneurons;
pub mod dendritic;

pub use lif::LIFNeuron;
pub use hierarchical::{NeuronLevel, RegionGroup, MeanFieldRegion, HierarchicalBrain, StreamingBuffer};
pub use interneurons::{
    PVInterneuron, SSTInterneuron, VIPInterneuron, InterneuronCircuit, InterneuronStats,
};
pub use dendritic::{
    DendriticBranch, DendriticNeuron, DendriticLayer, DendriticNeuronStats, DendriticLayerStats,
};

use serde::{Deserialize, Serialize};

/// Neuron state for GPU-efficient storage
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct NeuronState {
    /// Membrane potential (mV)
    pub v: f32,

    /// Spike threshold (mV)
    pub threshold: f32,

    /// Membrane time constant (ms)
    pub tau_m: f32,

    /// Reset potential (mV)
    pub v_reset: f32,

    /// Last spike time (timestep)
    pub last_spike: u16,

    /// Refractory period counter
    pub refractory_counter: u8,

    /// Neuron ID
    pub id: u32,

    /// Padding for alignment
    _padding: u8,
}

impl NeuronState {
    /// Create new neuron state with biological defaults
    pub fn new(id: u32) -> Self {
        Self {
            v: -70.0,              // Resting potential
            threshold: -55.0,      // Spike threshold
            tau_m: 20.0,          // 20ms membrane time constant
            v_reset: -70.0,       // Reset to resting
            last_spike: 0,
            refractory_counter: 0,
            id,
            _padding: 0,
        }
    }

    /// Check if neuron is in refractory period
    #[inline]
    pub fn is_refractory(&self) -> bool {
        self.refractory_counter > 0
    }

    /// Check if neuron should spike
    #[inline]
    pub fn should_spike(&self) -> bool {
        !self.is_refractory() && self.v >= self.threshold
    }
}

/// Trait for all neuron models
pub trait Neuron: Send + Sync {
    /// Update neuron state for one timestep
    fn update(&mut self, dt: f32, input_current: f32) -> bool;

    /// Reset neuron to resting state
    fn reset(&mut self);

    /// Get membrane potential
    fn voltage(&self) -> f32;

    /// Get neuron ID
    fn id(&self) -> u32;
}
