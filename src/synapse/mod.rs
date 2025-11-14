//! Synaptic dynamics and transmission
//!
//! Implements realistic synaptic vesicle cycles, calcium-dependent release,
//! memristive coupling, and short-term plasticity mechanisms.

pub mod vesicles;
pub mod memristive;

pub use vesicles::{VesiclePools, VesicleStats};
pub use memristive::{MemristiveSynapse, MemristiveState, MemristiveNetwork, MemristiveNetworkStats};
