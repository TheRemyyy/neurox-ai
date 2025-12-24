//! # NeuroxAI - Whole-Brain Neuromorphic Architecture
//!
//! Biologically-inspired spiking neural network simulation with GPU acceleration.
//! Targets 1-10M neurons on RTX 3070 with state-of-the-art learning algorithms.

#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

pub mod connectivity;
pub mod cuda;
pub mod datasets;
pub mod learning;
pub mod neuron;
pub mod serialization;
pub mod simulation;
pub mod synapse;
pub mod training;
pub mod utils;

// Cognitive architecture modules
pub mod attention;
pub mod brain;
pub mod cortex;
pub mod language;
pub mod memory;

// New biological systems
pub mod basal_ganglia;
pub mod neuromodulation;
pub mod oscillations;
pub mod semantics;
pub mod spatial;

// Human-limit upgrade modules (2025)
pub mod affect;
pub mod cognition;
pub mod motivation;
pub mod reasoning;

// Re-export key types
pub use connectivity::{ConnectivityType, ProceduralConnectivity, SparseConnectivity};
pub use cuda::CudaContext;
pub use datasets::{MNISTDataset, MNISTImage};
pub use neuron::{LIFNeuron, NeuronState};
pub use serialization::{ModelMetadata, NeuromorphicModel, NeuronParameters, PlasticityState};
pub use simulation::{OptimizationStats, Simulator};
pub use training::{train_mnist, MNISTTrainer, TrainingConfig};

// Re-export cognitive modules
pub use attention::{AttentionStats, AttentionSystem};
pub use brain::{BrainStats, NeuromorphicBrain};
pub use cortex::{
    EnhancedPredictiveHierarchy, EnhancedPredictiveStats, PredictiveHierarchy, WorkingMemory,
    WorkingMemoryStats,
};
pub use language::{DualStreamLanguage, DualStreamStats};
pub use memory::{EnhancedEpisodicMemory, Hippocampus, HippocampusStats, KnowledgeGraph};

// Re-export new biological systems
pub use basal_ganglia::{BasalGanglia, BasalGangliaStats, DopamineNeuron};
pub use neuromodulation::{NeuromodulationStats, NeuromodulationSystem};
pub use oscillations::{OscillationStats, OscillatoryCircuit};
pub use semantics::{EmbeddingLayer, SemanticHub, SemanticSystem};
pub use spatial::{GridCell, PlaceCell, SpatialSystem};

// Re-export new neuron types
pub use neuron::{
    DendriticLayer, DendriticNeuron, InterneuronCircuit, PVInterneuron, SSTInterneuron,
    VIPInterneuron,
};

// Re-export new learning mechanisms
pub use learning::{
    BCMMetaplasticity, CriticalityHomeostasis, HomeostaticStats, HomeostaticSystem,
};

// Re-export human-limit upgrade modules (2025)
pub use affect::{Emotion, EmotionalState, EmotionalStateMachine, EmotionalStats, MoodState};
pub use cognition::{
    AgentModel, BDIModel, BehavioralProfile, BeliefState, CapabilityModel, CognitiveStrategy,
    DialogueArbiter, InnerDialogue, Metacognition, MetacognitionStats, Perspective, SelfModel,
    TheoryOfMind,
};
pub use motivation::{CuriosityDrive, CuriosityStats, InformationGain};
pub use reasoning::{AbstractReasoning, AnalogyEngine, ReasoningChain};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Recommended timestep for biological accuracy (ms)
pub const DEFAULT_TIMESTEP: f32 = 0.1;

/// Target firing rate for homeostatic regulation (Hz)
pub const TARGET_FIRING_RATE: f32 = 5.0;
