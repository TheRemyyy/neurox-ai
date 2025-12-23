//! Attention and Routing Systems
//!
//! Thalamic attention mechanisms for selective information processing.

pub mod thalamic_routing;

pub use thalamic_routing::{
    AttentionStats, AttentionSystem, DynamicConnectivity, DynamicConnectivityStats,
};
