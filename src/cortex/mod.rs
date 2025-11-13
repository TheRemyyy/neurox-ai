//! Cortical Processing Modules
//!
//! High-level cortical functions including working memory and predictive coding.

pub mod working_memory;
pub mod predictive_coding;

pub use working_memory::{
    CircularBuffer, PersistentNeuron, WorkingMemory, WorkingMemoryStats,
};
pub use predictive_coding::{
    PredictiveCodingLayer, PredictiveHierarchy,
};
