//! Cortical Processing Modules
//!
//! High-level cortical functions including working memory, predictive coding,
//! sensory processing (V1 orientation, cochlea, barrel cortex), motion processing (MT-MST),
//! sleep consolidation, and more.

pub mod working_memory;
pub mod predictive_coding;
pub mod enhanced_predictive;
pub mod v1_orientation;
pub mod cochlea;
pub mod mt_mst;
pub mod barrel_cortex;
pub mod sleep;

pub use working_memory::{BistableNeuron, WorkingMemory, WorkingMemoryStats};
pub use predictive_coding::{
    PredictiveCodingLayer, PredictiveHierarchy,
};
pub use enhanced_predictive::{
    LaminarLayer, EnhancedPredictiveHierarchy, EnhancedPredictiveStats,
};
pub use v1_orientation::{
    V1OrientationSystem, V1Layer, SimpleCell, ComplexCell,
    RetinaLayer, RelayLayer,
};
pub use cochlea::{
    NeuromorphicCochlea, BasilarMembrane, InnerHairCell, AuditoryNerveFiber,
};
pub use mt_mst::{
    MotionProcessingSystem, MiddleTemporal, MedialSuperiorTemporalDorsal,
    MedialSuperiorTemporalVentral, MotionOutput, OpticFlow,
};
pub use barrel_cortex::{
    BarrelCortex, CorticalBarrel, PVInterneuron, SSTInterneuron, VIPInterneuron,
};
pub use sleep::{
    SleepConsolidation, Experience, SleepStage, ConsolidationResult, SleepStats,
};
