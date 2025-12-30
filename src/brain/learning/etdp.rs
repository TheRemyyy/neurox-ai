//! Event-driven Timing-Dependent Plasticity (ETDP)
//!
//! Voltage-dependent plasticity that unifies subthreshold and suprathreshold learning.
//! Unlike traditional STDP which only triggers on spikes, ETDP responds to all
//! significant voltage changes (events), enabling dendritic computation.
//!
//! # Key Features
//! - Voltage-threshold detection for events (not just spikes)
//! - Graded plasticity based on voltage amplitude
//! - Unifies subthreshold EPSP/IPSP learning and spike-based STDP
//! - Compatible with dendritic branching and nonlinear integration
//!
//! # Mathematical Model
//! ΔW = η · f(ΔV_pre) · g(ΔV_post) · h(Δt)
//! where:
//! - f(ΔV) is voltage-dependent pre-synaptic factor
//! - g(ΔV) is voltage-dependent post-synaptic factor
//! - h(Δt) is timing-dependent kernel
//!
//! # References
//! Brea et al. (2016) "Matching Recall and Storage in Sequence Learning with Spiking Neural Networks"
//! Bengio et al. (2015) "Event-driven random backpropagation"

use serde::{Deserialize, Serialize};

/// Event-driven timing-dependent plasticity system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ETDP {
    /// Learning rate
    pub eta: f32,

    /// Voltage threshold for event detection (mV above baseline)
    pub voltage_threshold: f32,

    /// Time constant for exponential timing kernel (ms)
    pub tau_plus: f32,
    pub tau_minus: f32,

    /// Maximum timing window (ms)
    pub window_size: f32,

    /// Voltage sensitivity parameters
    pub alpha_pre: f32,   // Pre-synaptic voltage scaling
    pub alpha_post: f32,  // Post-synaptic voltage scaling

    /// Nonlinearity exponent for voltage-dependent factors
    pub voltage_exponent: f32,

    /// Recent events buffer: (neuron_id, time, voltage_change)
    pub pre_events: Vec<(usize, f32, f32)>,
    pub post_events: Vec<(usize, f32, f32)>,

    /// Current simulation time (ms)
    pub current_time: f32,
}

impl ETDP {
    /// Create new ETDP system
    pub fn new(eta: f32) -> Self {
        Self {
            eta,
            voltage_threshold: 5.0,    // 5 mV above baseline triggers event
            tau_plus: 20.0,            // LTP time constant
            tau_minus: 20.0,           // LTD time constant
            window_size: 50.0,         // 50 ms learning window
            alpha_pre: 0.1,            // Pre-synaptic voltage scaling
            alpha_post: 0.2,           // Post-synaptic voltage scaling
            voltage_exponent: 2.0,     // Quadratic voltage dependence
            pre_events: Vec::new(),
            post_events: Vec::new(),
            current_time: 0.0,
        }
    }

    /// Detect voltage events and update buffer
    ///
    /// # Arguments
    /// - `neuron_id`: ID of the neuron
    /// - `voltage_change`: Voltage change from baseline (mV)
    /// - `is_presynaptic`: Whether this is pre-synaptic (true) or post-synaptic (false)
    pub fn detect_event(&mut self, neuron_id: usize, voltage_change: f32, is_presynaptic: bool) {
        // Only register event if voltage change exceeds threshold
        if voltage_change.abs() < self.voltage_threshold {
            return;
        }

        let event = (neuron_id, self.current_time, voltage_change);

        if is_presynaptic {
            self.pre_events.push(event);
        } else {
            self.post_events.push(event);
        }
    }

    /// Compute weight change for a synapse
    ///
    /// # Arguments
    /// - `pre_neuron_id`: Pre-synaptic neuron ID
    /// - `post_neuron_id`: Post-synaptic neuron ID
    ///
    /// # Returns
    /// Weight change ΔW
    pub fn compute_weight_change(&self, pre_neuron_id: usize, post_neuron_id: usize) -> f32 {
        let mut delta_w = 0.0;

        // For each pre-synaptic event
        for &(pre_id, pre_time, pre_voltage) in &self.pre_events {
            if pre_id != pre_neuron_id {
                continue;
            }

            // Check all post-synaptic events within window
            for &(post_id, post_time, post_voltage) in &self.post_events {
                if post_id != post_neuron_id {
                    continue;
                }

                let dt = post_time - pre_time;

                // Skip if outside learning window
                if dt.abs() > self.window_size {
                    continue;
                }

                // Voltage-dependent pre-synaptic factor
                let f_pre = self.voltage_factor(pre_voltage, self.alpha_pre);

                // Voltage-dependent post-synaptic factor
                let g_post = self.voltage_factor(post_voltage, self.alpha_post);

                // Timing-dependent kernel
                let h_timing = if dt > 0.0 {
                    // Causal: post after pre (LTP)
                    (-(dt / self.tau_plus)).exp()
                } else {
                    // Anti-causal: pre after post (LTD)
                    -((dt.abs() / self.tau_minus)).exp()
                };

                // Combine factors
                delta_w += self.eta * f_pre * g_post * h_timing;
            }
        }

        delta_w
    }

    /// Voltage-dependent factor with nonlinearity
    fn voltage_factor(&self, voltage_change: f32, alpha: f32) -> f32 {
        let normalized = alpha * voltage_change;
        normalized.abs().powf(self.voltage_exponent) * normalized.signum()
    }

    /// Update system for new timestep
    pub fn update(&mut self, dt: f32) {
        self.current_time += dt;

        // Remove old events outside learning window
        let cutoff_time = self.current_time - self.window_size;

        self.pre_events.retain(|&(_, time, _)| time >= cutoff_time);
        self.post_events.retain(|&(_, time, _)| time >= cutoff_time);
    }

    /// Clear all events (e.g., at trial boundaries)
    pub fn clear_events(&mut self) {
        self.pre_events.clear();
        self.post_events.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> ETDPStats {
        ETDPStats {
            num_pre_events: self.pre_events.len(),
            num_post_events: self.post_events.len(),
            current_time: self.current_time,
            window_size: self.window_size,
        }
    }
}

/// ETDP statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ETDPStats {
    pub num_pre_events: usize,
    pub num_post_events: usize,
    pub current_time: f32,
    pub window_size: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_etdp_creation() {
        let etdp = ETDP::new(0.001);
        assert_eq!(etdp.eta, 0.001);
        assert_eq!(etdp.voltage_threshold, 5.0);
        assert_eq!(etdp.pre_events.len(), 0);
        assert_eq!(etdp.post_events.len(), 0);
    }

    #[test]
    fn test_event_detection() {
        let mut etdp = ETDP::new(0.001);

        // Subthreshold event (should not register)
        etdp.detect_event(0, 2.0, true);
        assert_eq!(etdp.pre_events.len(), 0);

        // Suprathreshold event (should register)
        etdp.detect_event(0, 10.0, true);
        assert_eq!(etdp.pre_events.len(), 1);

        // Post-synaptic event
        etdp.detect_event(1, 15.0, false);
        assert_eq!(etdp.post_events.len(), 1);
    }

    #[test]
    fn test_causal_potentiation() {
        let mut etdp = ETDP::new(0.01);

        // Pre-synaptic event at t=0
        etdp.current_time = 0.0;
        etdp.detect_event(0, 20.0, true);  // Pre neuron 0

        // Post-synaptic event at t=10ms (causal)
        etdp.current_time = 10.0;
        etdp.detect_event(1, 25.0, false);  // Post neuron 1

        // Compute weight change
        let dw = etdp.compute_weight_change(0, 1);

        // Should be positive (LTP) for causal pairing
        assert!(dw > 0.0, "Causal pairing should cause LTP (dw={})", dw);
    }

    #[test]
    fn test_anticausal_depression() {
        let mut etdp = ETDP::new(0.01);

        // Post-synaptic event at t=0
        etdp.current_time = 0.0;
        etdp.detect_event(1, 25.0, false);  // Post neuron 1

        // Pre-synaptic event at t=10ms (anti-causal)
        etdp.current_time = 10.0;
        etdp.detect_event(0, 20.0, true);  // Pre neuron 0

        // Compute weight change
        let dw = etdp.compute_weight_change(0, 1);

        // Should be negative (LTD) for anti-causal pairing
        assert!(dw < 0.0, "Anti-causal pairing should cause LTD (dw={})", dw);
    }

    #[test]
    fn test_voltage_dependence() {
        let mut etdp = ETDP::new(0.01);

        // Weak voltage change
        etdp.current_time = 0.0;
        etdp.detect_event(0, 10.0, true);
        etdp.current_time = 5.0;
        etdp.detect_event(1, 10.0, false);
        let dw_weak = etdp.compute_weight_change(0, 1);

        // Reset
        etdp.clear_events();

        // Strong voltage change
        etdp.current_time = 0.0;
        etdp.detect_event(0, 30.0, true);
        etdp.current_time = 5.0;
        etdp.detect_event(1, 30.0, false);
        let dw_strong = etdp.compute_weight_change(0, 1);

        // Stronger voltage should cause larger weight change
        assert!(dw_strong.abs() > dw_weak.abs(),
            "Stronger voltage should cause larger plasticity (weak={}, strong={})",
            dw_weak, dw_strong);
    }

    #[test]
    fn test_window_pruning() {
        let mut etdp = ETDP::new(0.01);
        etdp.window_size = 20.0;  // 20ms window

        // Add event at t=0
        etdp.current_time = 0.0;
        etdp.detect_event(0, 15.0, true);
        assert_eq!(etdp.pre_events.len(), 1);

        // Update to t=25ms (outside window)
        etdp.update(25.0);

        // Old event should be pruned
        assert_eq!(etdp.pre_events.len(), 0, "Old events should be pruned");
    }

    #[test]
    fn test_graded_subthreshold() {
        let mut etdp = ETDP::new(0.01);

        // Subthreshold EPSP (just above threshold)
        etdp.current_time = 0.0;
        etdp.detect_event(0, 6.0, true);   // 6mV (just above 5mV threshold)
        etdp.current_time = 5.0;
        etdp.detect_event(1, 6.0, false);
        let dw_subthreshold = etdp.compute_weight_change(0, 1);

        // Reset
        etdp.clear_events();

        // Suprathreshold spike (strong)
        etdp.current_time = 0.0;
        etdp.detect_event(0, 50.0, true);  // 50mV (spike-like)
        etdp.current_time = 5.0;
        etdp.detect_event(1, 50.0, false);
        let dw_spike = etdp.compute_weight_change(0, 1);

        // Both should cause plasticity, but spike should be much stronger
        assert!(dw_subthreshold > 0.0, "Subthreshold should cause some LTP");
        assert!(dw_spike > dw_subthreshold * 5.0,
            "Spike should cause much stronger plasticity (subthreshold={}, spike={})",
            dw_subthreshold, dw_spike);
    }
}
