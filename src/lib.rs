//! # NeuroxAI - Whole-Brain Neuromorphic Architecture
//!
//! Biologically-inspired spiking neural network simulation with GPU acceleration.
//! Targets 1-10M neurons on RTX 3070 with state-of-the-art learning algorithms.

#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

// Brain module - contains all neural/cognitive systems
pub mod brain;

// Controller modules (CLI, plugins, config)
pub mod commands;
pub mod config;
pub mod plugin;

// Re-export key types directly from brain submodules
pub use brain::connectivity::{ConnectivityType, ProceduralConnectivity, SparseConnectivity};
pub use brain::cuda::CudaContext;
pub use brain::datasets::{download_mnist, MNISTDataset, MNISTImage};
pub use brain::neuron::{LIFNeuron, NeuronState};
pub use brain::serialization::{
    ModelMetadata, NeuromorphicModel, NeuronParameters, PlasticityState,
};
pub use brain::simulation::{OptimizationStats, Simulator};
pub use brain::training::{train_mnist, MNISTTrainer, TrainingConfig};

// Re-export cognitive modules from brain
pub use brain::attention::{AttentionStats, AttentionSystem};
pub use brain::cortex::{
    EnhancedPredictiveHierarchy, EnhancedPredictiveStats, PredictiveHierarchy, WorkingMemory,
    WorkingMemoryStats,
};
pub use brain::language::{DualStreamLanguage, DualStreamStats};
pub use brain::memory::{EnhancedEpisodicMemory, Hippocampus, HippocampusStats, KnowledgeGraph};
pub use brain::{BrainStats, NeuromorphicBrain};

// Re-export biological systems from brain
pub use brain::basal_ganglia::{BasalGanglia, BasalGangliaStats, DopamineNeuron};
pub use brain::neuromodulation::{NeuromodulationStats, NeuromodulationSystem};
pub use brain::oscillations::{OscillationStats, OscillatoryCircuit};
pub use brain::semantics::{EmbeddingLayer, SemanticHub, SemanticSystem};
pub use brain::spatial::{GridCell, PlaceCell, SpatialSystem};

// Re-export neuron types from brain
pub use brain::neuron::{
    DendriticLayer, DendriticNeuron, InterneuronCircuit, PVInterneuron, SSTInterneuron,
    VIPInterneuron,
};

// Re-export learning mechanisms from brain
pub use brain::learning::{
    BCMMetaplasticity, CriticalityHomeostasis, HomeostaticStats, HomeostaticSystem, STDPConfig,
};

// Re-export affect, cognition, motivation, reasoning from brain
pub use brain::affect::{
    Emotion, EmotionalState, EmotionalStateMachine, EmotionalStats, MoodState,
};
pub use brain::cognition::{
    AgentModel, BDIModel, BehavioralProfile, BeliefState, CapabilityModel, CognitiveStrategy,
    DialogueArbiter, InnerDialogue, Metacognition, MetacognitionStats, Perspective, SelfModel,
    TheoryOfMind,
};
pub use brain::motivation::{CuriosityDrive, CuriosityStats, InformationGain};
pub use brain::reasoning::{AbstractReasoning, AnalogyEngine, ReasoningChain};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Recommended timestep for biological accuracy (ms)
pub const DEFAULT_TIMESTEP: f32 = 0.1;

/// Target firing rate for homeostatic regulation (Hz)
pub const TARGET_FIRING_RATE: f32 = 5.0;
