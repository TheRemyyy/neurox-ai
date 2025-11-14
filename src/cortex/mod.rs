//! Cortical Processing Modules
//!
//! High-level cortical functions including working memory, predictive coding,
//! sensory processing (V1 orientation, cochlea), and more.

pub mod working_memory;
pub mod predictive_coding;
pub mod enhanced_predictive;
pub mod v1_orientation;
pub mod cochlea;

pub use working_memory::{
    CircularBuffer, PersistentNeuron, WorkingMemory, WorkingMemoryStats,
};
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
