//! Memory Systems
//!
//! Hippocampal episodic memory and consolidation.

pub mod hippocampus;

pub use hippocampus::{
    FeedforwardLayer, Hippocampus, HippocampusStats, MemoryTrace, RecurrentNetwork, SparseEncoder,
};
