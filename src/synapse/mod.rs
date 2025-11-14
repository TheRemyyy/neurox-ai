//! Synaptic dynamics and transmission
//!
//! Implements realistic synaptic vesicle cycles, calcium-dependent release,
//! and short-term plasticity mechanisms.

pub mod vesicles;

pub use vesicles::{VesiclePools, VesicleStats};
