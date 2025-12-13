//! Cortical Processing Modules
//!
//! High-level cortical functions including working memory, predictive coding,
//! sensory processing (V1 orientation, cochlea, barrel cortex), motion processing (MT-MST),
//! sleep consolidation, and more.

pub mod barrel_cortex;
pub mod cochlea;
pub mod enhanced_predictive;
pub mod mt_mst;
pub mod predictive_coding;
pub mod sleep;
pub mod v1_orientation;
pub mod working_memory;

pub use barrel_cortex::BarrelCortex;
pub use cochlea::{GammatoneFilter, InnerHairCell, NeuromorphicCochlea};
pub use enhanced_predictive::{
    EnhancedPredictiveHierarchy, EnhancedPredictiveStats, LaminarLayer,
};
pub use mt_mst::{MotionOutput, MotionProcessingSystem, OpticFlow};
pub use predictive_coding::{PredictiveCodingLayer, PredictiveHierarchy};
pub use sleep::{SleepConsolidation, SleepStats};
pub use v1_orientation::V1OrientationSystem;
pub use working_memory::{BistableNeuron, WorkingMemory, WorkingMemoryStats};

