//! Learning rules and synaptic plasticity
//!
//! Implements STDP, homeostatic plasticity, STP, quantization, metaplasticity,
//! calcium-based plasticity, burst-dependent STDP, heterosynaptic plasticity, and ETDP.

pub mod stdp;
pub mod quantization;
pub mod metaplasticity;
pub mod heterosynaptic;
pub mod etdp;

pub use stdp::{
    TripletSTDP, HomeostaticPlasticity, CalciumBasedPlasticity, BurstDependentSTDP,
};
pub use quantization::{QuantizationConfig, QuantizedWeights, QATSimulator};
pub use metaplasticity::{
    BCMMetaplasticity, SynapticScaling, CriticalityHomeostasis, IntrinsicPlasticity,
    HomeostaticSystem, HomeostaticStats,
};
pub use heterosynaptic::{HeterosynapticPlasticity, HeterosynapticStats};
pub use etdp::{ETDP, ETDPStats};

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

/// Short-Term Plasticity (STP) dynamics (Nature SR 2025)
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

    /// Baseline utilization
    pub u_0: f32,

    /// STP scaling factor (k in paper: 8-10.5)
    pub k: f32,
}

impl Default for STPDynamics {
    fn default() -> Self {
        Self {
            u_s: 0.5,
            x_s: 1.0,
            tau_f: 100.0,  // 100ms facilitation
            tau_d: 200.0,  // 200ms depression
            u_0: 0.5,
            k: 9.0,        // Mid-range scaling factor
        }
    }
}

impl STPDynamics {
    /// Update STP state over time (continuous decay)
    pub fn update(&mut self, dt: f32) {
        // du_s/dt = (1-u_s)/τ_f
        let du = (self.u_0 - self.u_s) / self.tau_f * dt;
        self.u_s += du;

        // dx_s/dt = (1-x_s)/τ_d
        let dx = (1.0 - self.x_s) / self.tau_d * dt;
        self.x_s += dx;

        // Clamp to valid range
        self.u_s = self.u_s.clamp(0.0, 1.0);
        self.x_s = self.x_s.clamp(0.0, 1.0);
    }

    /// Process spike (update u_s and x_s)
    pub fn on_spike(&mut self) -> f32 {
        // Calculate release: r_s = u_s * x_s
        let r_s = self.u_s * self.x_s;

        // Update utilization (facilitation)
        self.u_s = self.u_s + self.u_0 * (1.0 - self.u_s);

        // Update resources (depression)
        self.x_s = self.x_s * (1.0 - self.u_s);

        r_s
    }

    /// Calculate effective synaptic weight with STP
    /// g_exc(t) = w_STDP + k * w_STDP * r_s(t)
    pub fn modulate_weight(&self, w_stdp: f32) -> f32 {
        let r_s = self.u_s * self.x_s;
        w_stdp + self.k * w_stdp * r_s
    }

    /// Create facilitation-dominant STP (for excitatory synapses)
    pub fn facilitation() -> Self {
        Self {
            u_s: 0.3,
            x_s: 1.0,
            tau_f: 50.0,   // Fast facilitation
            tau_d: 500.0,  // Slow depression
            u_0: 0.3,
            k: 10.0,       // Strong STP effect
        }
    }

    /// Create depression-dominant STP (for inhibitory synapses)
    pub fn depression() -> Self {
        Self {
            u_s: 0.8,
            x_s: 1.0,
            tau_f: 200.0,  // Slow facilitation
            tau_d: 100.0,  // Fast depression
            u_0: 0.8,
            k: 8.0,
        }
    }
}
