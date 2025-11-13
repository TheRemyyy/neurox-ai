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

// Re-export key types
pub use neuron::{LIFNeuron, NeuronState};
pub use cuda::CudaContext;
pub use simulation::Simulator;
pub use connectivity::{ProceduralConnectivity, SparseConnectivity, ConnectivityType};
pub use datasets::{MNISTDataset, MNISTImage};
pub use training::{TrainingConfig, MNISTTrainer, train_mnist};
pub use serialization::{NeuromorphicModel, ModelMetadata, NeuronParameters, PlasticityState};

// Re-export cognitive modules
pub use cortex::{WorkingMemory, WorkingMemoryStats, PredictiveHierarchy};
pub use attention::{AttentionSystem, AttentionStats};
pub use memory::{Hippocampus, HippocampusStats};
pub use language::{LanguageSystem, LanguageStats};
pub use brain::{NeuromorphicBrain, BrainStats};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Recommended timestep for biological accuracy (ms)
pub const DEFAULT_TIMESTEP: f32 = 0.1;

/// Target firing rate for homeostatic regulation (Hz)
pub const TARGET_FIRING_RATE: f32 = 5.0;
