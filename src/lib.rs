//! # NeuroxAI - Whole-Brain Neuromorphic Architecture
//!
//! Biologically-inspired spiking neural network simulation with GPU acceleration.
//! Targets 1-10M neurons on RTX 3070 with state-of-the-art learning algorithms.

#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

pub mod neuron;
pub mod cuda;
pub mod connectivity;
pub mod simulation;
pub mod learning;
pub mod synapse;
pub mod utils;
pub mod datasets;
pub mod training;
pub mod serialization;

// Cognitive architecture modules
pub mod cortex;
pub mod attention;
pub mod memory;
pub mod language;
pub mod brain;

// New biological systems
pub mod basal_ganglia;
pub mod neuromodulation;
pub mod oscillations;
pub mod spatial;
pub mod semantics;

// GPU-accelerated neural processor (100% GPU, zero CPU bottlenecks)
pub mod neural_processor;

// Re-export key types
pub use neuron::{LIFNeuron, NeuronState};
pub use cuda::CudaContext;
pub use simulation::{Simulator, OptimizationStats};
pub use connectivity::{ProceduralConnectivity, SparseConnectivity, ConnectivityType};
pub use datasets::{MNISTDataset, MNISTImage};
pub use training::{TrainingConfig, MNISTTrainer, train_mnist};
pub use serialization::{NeuromorphicModel, ModelMetadata, NeuronParameters, PlasticityState};

// Re-export cognitive modules
pub use cortex::{
    WorkingMemory, WorkingMemoryStats, PredictiveHierarchy,
    EnhancedPredictiveHierarchy, EnhancedPredictiveStats,
};
pub use attention::{AttentionSystem, AttentionStats};
pub use memory::{Hippocampus, HippocampusStats};
pub use language::{
    DualStreamLanguage, DualStreamStats,
};
pub use brain::{NeuromorphicBrain, BrainStats};

// Re-export new biological systems
pub use basal_ganglia::{BasalGanglia, BasalGangliaStats, DopamineNeuron};
pub use neuromodulation::{NeuromodulationSystem, NeuromodulationStats};
pub use oscillations::{OscillatoryCircuit, OscillationStats};
pub use spatial::{SpatialSystem, PlaceCell, GridCell};
pub use semantics::{SemanticSystem, SemanticHub, EmbeddingLayer};

// Re-export new neuron types
pub use neuron::{
    PVInterneuron, SSTInterneuron, VIPInterneuron, InterneuronCircuit,
    DendriticNeuron, DendriticLayer,
};

// Re-export new learning mechanisms
pub use learning::{
    HomeostaticSystem, HomeostaticStats, BCMMetaplasticity,
    CriticalityHomeostasis,
};

// Re-export neural processor
pub use neural_processor::{NeuralProcessor, ProcessorStats};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Recommended timestep for biological accuracy (ms)
pub const DEFAULT_TIMESTEP: f32 = 0.1;

/// Target firing rate for homeostatic regulation (Hz)
pub const TARGET_FIRING_RATE: f32 = 5.0;
