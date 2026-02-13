//! Synaptic dynamics and transmission
//!
//! Implements realistic synaptic vesicle cycles, calcium-dependent release,
//! memristive coupling, and short-term plasticity mechanisms.

pub mod memristive;
pub mod vesicles;

pub use memristive::{
    MemristiveNetwork, MemristiveNetworkStats, MemristiveState, MemristiveSynapse,
};
pub use vesicles::{VesiclePools, VesicleStats};
