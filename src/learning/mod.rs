//! Learning rules and synaptic plasticity
//!
//! Implements STDP, homeostatic plasticity, and STP mechanisms.

pub mod stdp;

pub use stdp::{TripletSTDP, HomeostaticPlasticity};

use serde::{Deserialize, Serialize};

/// Spike-Timing-Dependent Plasticity (STDP) parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDPConfig {
    /// Pre-synaptic learning rate
    pub lr_pre: f32,

    /// Post-synaptic learning rate
    pub lr_post: f32,

    /// Pre-synaptic trace time constant (ms)
    pub tau_pre: f32,

    /// Post-synaptic trace time constant (ms)
    pub tau_post: f32,

    /// Weight bounds
    pub w_min: f32,
    pub w_max: f32,
}

impl Default for STDPConfig {
    fn default() -> Self {
        Self {
            lr_pre: 0.0001,
            lr_post: 0.01,
            tau_pre: 20.0,
            tau_post: 20.0,
            w_min: 0.0,
            w_max: 1.0,
        }
    }
}

/// Triplet STDP trace
#[derive(Debug, Clone)]
pub struct STDPTrace {
    pub a_pre: f32,
    pub a_post1: f32,
    pub a_post2: f32,
}

impl STDPTrace {
    pub fn new() -> Self {
        Self {
            a_pre: 0.0,
            a_post1: 0.0,
            a_post2: 0.0,
        }
    }

    /// Update trace on pre-synaptic spike
    pub fn update_pre(&mut self, dt: f32, tau_pre: f32) {
        self.a_pre *= (-dt / tau_pre).exp();
        self.a_pre += 1.0;
    }

    /// Update trace on post-synaptic spike
    pub fn update_post(&mut self, dt: f32, tau_post1: f32, tau_post2: f32) {
        self.a_post1 *= (-dt / tau_post1).exp();
        self.a_post2 *= (-dt / tau_post2).exp();
        self.a_post1 += 1.0;
        self.a_post2 += 1.0;
    }
}

/// Short-Term Plasticity (STP) dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STPDynamics {
    /// Utilization parameter
    pub u_s: f32,

    /// Available resources
    pub x_s: f32,

    /// Facilitation time constant (ms)
    pub tau_f: f32,

    /// Depression time constant (ms)
    pub tau_d: f32,
}

impl Default for STPDynamics {
    fn default() -> Self {
        Self {
            u_s: 0.5,
            x_s: 1.0,
            tau_f: 100.0,
            tau_d: 200.0,
        }
    }
}
