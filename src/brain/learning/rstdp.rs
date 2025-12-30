//! Reward-Modulated Spike-Timing-Dependent Plasticity (R-STDP)
//!
//! Implements meta-learning through eligibility traces and reward modulation.
//! Solves the temporal credit assignment problem by maintaining eligibility
//! traces that bridge the gap between synaptic activity and delayed rewards.
//!
//! # Features
//! - Eligibility traces with configurable time constants
//! - Dopamine-modulated learning (reward signals)
//! - Three-factor learning rule (pre, post, reward)
//! - Meta-learning: learning to learn through reward statistics
//! - Compatible with continuous and discrete rewards
//!
//! # Mathematical Model
//! e(t) = e(t-1)·exp(-dt/τ_e) + δ_pre·δ_post  (eligibility trace)
//! dw/dt = η·e(t)·R(t)                        (weight update)
//! R(t) = (r(t) - r̄) / σ_r                   (normalized reward)
//!
//! # References
//! - Frémaux & Gerstner (2016) "Neuromodulated STDP"
//! - Izhikevich (2007) "Solving the distal reward problem"
//! - October 2024 paper on meta-learning with eligibility traces

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Reward-modulated STDP synapse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSTDPSynapse {
    /// Synaptic weight
    pub weight: f32,

    /// Eligibility trace
    pub eligibility: f32,

    /// Pre-synaptic trace
    pub pre_trace: f32,

    /// Post-synaptic trace
    pub post_trace: f32,

    /// Cumulative reward received
    pub cumulative_reward: f32,

    /// Number of updates
    pub num_updates: u64,
}

impl RSTDPSynapse {
    pub fn new(initial_weight: f32) -> Self {
        Self {
            weight: initial_weight,
            eligibility: 0.0,
            pre_trace: 0.0,
            post_trace: 0.0,
            cumulative_reward: 0.0,
            num_updates: 0,
        }
    }
}

/// R-STDP learning system with meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSTDPSystem {
    /// Synapses: (pre_id, post_id) -> synapse
    pub synapses: HashMap<(usize, usize), RSTDPSynapse>,

    /// Learning rate
    pub eta: f32,

    /// Eligibility trace time constant (ms)
    pub tau_eligibility: f32,

    /// STDP trace time constants (ms)
    pub tau_pre: f32,
    pub tau_post: f32,

    /// STDP parameters
    pub a_plus: f32,   // LTP amplitude
    pub a_minus: f32,  // LTD amplitude

    /// Reward statistics (for meta-learning)
    pub reward_mean: f32,
    pub reward_std: f32,
    pub reward_history: Vec<f32>,
    pub reward_window: usize,

    /// Meta-learning: adaptive learning rate
    pub meta_learning_enabled: bool,
    pub eta_min: f32,
    pub eta_max: f32,

    /// Weight bounds
    pub w_min: f32,
    pub w_max: f32,

    /// Statistics
    pub total_updates: u64,
    pub total_reward: f32,
}

impl RSTDPSystem {
    /// Create new R-STDP system
    pub fn new(eta: f32) -> Self {
        Self {
            synapses: HashMap::new(),
            eta,
            tau_eligibility: 1000.0,  // 1s eligibility trace
            tau_pre: 20.0,             // 20ms pre-synaptic trace
            tau_post: 20.0,            // 20ms post-synaptic trace
            a_plus: 0.01,              // LTP amplitude
            a_minus: 0.012,            // LTD amplitude (slightly stronger)
            reward_mean: 0.0,
            reward_std: 1.0,
            reward_history: Vec::new(),
            reward_window: 100,        // Track last 100 rewards
            meta_learning_enabled: true,
            eta_min: 0.0001,
            eta_max: 0.01,
            w_min: 0.0,
            w_max: 1.0,
            total_updates: 0,
            total_reward: 0.0,
        }
    }

    /// Add synapse
    pub fn add_synapse(&mut self, pre_id: usize, post_id: usize, initial_weight: f32) {
        self.synapses.insert((pre_id, post_id), RSTDPSynapse::new(initial_weight));
    }

    /// Update traces on pre-synaptic spike
    pub fn on_pre_spike(&mut self, pre_id: usize, post_ids: &[usize], dt: f32) {
        for &post_id in post_ids {
            if let Some(synapse) = self.synapses.get_mut(&(pre_id, post_id)) {
                // Decay pre-synaptic trace
                synapse.pre_trace *= (-dt / self.tau_pre).exp();

                // Update eligibility trace: LTD component
                // e ← e - a_minus·post_trace
                synapse.eligibility -= self.a_minus * synapse.post_trace;

                // Increment pre-synaptic trace
                synapse.pre_trace += 1.0;
            }
        }
    }

    /// Update traces on post-synaptic spike
    pub fn on_post_spike(&mut self, post_id: usize, pre_ids: &[usize], dt: f32) {
        for &pre_id in pre_ids {
            if let Some(synapse) = self.synapses.get_mut(&(pre_id, post_id)) {
                // Decay post-synaptic trace
                synapse.post_trace *= (-dt / self.tau_post).exp();

                // Update eligibility trace: LTP component
                // e ← e + a_plus·pre_trace
                synapse.eligibility += self.a_plus * synapse.pre_trace;

                // Increment post-synaptic trace
                synapse.post_trace += 1.0;
            }
        }
    }

    /// Apply reward signal to update weights
    ///
    /// # Arguments
    /// - `reward`: Scalar reward value
    /// - `dt`: Timestep (ms)
    pub fn apply_reward(&mut self, reward: f32, dt: f32) {
        // Update reward statistics
        self.reward_history.push(reward);
        if self.reward_history.len() > self.reward_window {
            self.reward_history.remove(0);
        }

        // Calculate reward statistics
        let n = self.reward_history.len() as f32;
        self.reward_mean = self.reward_history.iter().sum::<f32>() / n;

        let variance: f32 = self.reward_history
            .iter()
            .map(|r| (r - self.reward_mean).powi(2))
            .sum::<f32>()
            / n;
        self.reward_std = variance.sqrt().max(0.01); // Avoid division by zero

        // Normalize reward (z-score)
        let normalized_reward = (reward - self.reward_mean) / self.reward_std;

        // Meta-learning: adapt learning rate based on reward uncertainty
        if self.meta_learning_enabled {
            // Higher uncertainty → higher learning rate
            let uncertainty = self.reward_std;
            self.eta = self.eta_min + (self.eta_max - self.eta_min) * (uncertainty / (1.0 + uncertainty));
        }

        // Update all synaptic weights using eligibility traces
        for synapse in self.synapses.values_mut() {
            // Decay eligibility trace
            synapse.eligibility *= (-dt / self.tau_eligibility).exp();

            // Three-factor learning rule: dw = η · e · R
            let dw = self.eta * synapse.eligibility * normalized_reward;
            synapse.weight += dw;

            // Clamp weight
            synapse.weight = synapse.weight.clamp(self.w_min, self.w_max);

            // Statistics
            synapse.cumulative_reward += reward;
            synapse.num_updates += 1;
        }

        self.total_updates += 1;
        self.total_reward += reward;
    }

    /// Update system (decay traces)
    pub fn update(&mut self, dt: f32) {
        for synapse in self.synapses.values_mut() {
            // Decay traces
            synapse.pre_trace *= (-dt / self.tau_pre).exp();
            synapse.post_trace *= (-dt / self.tau_post).exp();
            synapse.eligibility *= (-dt / self.tau_eligibility).exp();
        }
    }

    /// Get synapse weight
    pub fn get_weight(&self, pre_id: usize, post_id: usize) -> Option<f32> {
        self.synapses.get(&(pre_id, post_id)).map(|s| s.weight)
    }

    /// Get statistics
    pub fn stats(&self) -> RSTDPStats {
        if self.synapses.is_empty() {
            return RSTDPStats {
                num_synapses: 0,
                avg_weight: 0.0,
                avg_eligibility: 0.0,
                current_learning_rate: self.eta,
                reward_mean: self.reward_mean,
                reward_std: self.reward_std,
                total_reward: self.total_reward,
            };
        }

        let avg_weight = self.synapses.values().map(|s| s.weight).sum::<f32>() / self.synapses.len() as f32;
        let avg_eligibility = self.synapses.values().map(|s| s.eligibility).sum::<f32>() / self.synapses.len() as f32;

        RSTDPStats {
            num_synapses: self.synapses.len(),
            avg_weight,
            avg_eligibility,
            current_learning_rate: self.eta,
            reward_mean: self.reward_mean,
            reward_std: self.reward_std,
            total_reward: self.total_reward,
        }
    }
}

/// R-STDP statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSTDPStats {
    pub num_synapses: usize,
    pub avg_weight: f32,
    pub avg_eligibility: f32,
    pub current_learning_rate: f32,
    pub reward_mean: f32,
    pub reward_std: f32,
    pub total_reward: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rstdp_creation() {
        let rstdp = RSTDPSystem::new(0.01);
        assert_eq!(rstdp.eta, 0.01);
        assert_eq!(rstdp.synapses.len(), 0);
    }

    #[test]
    fn test_add_synapse() {
        let mut rstdp = RSTDPSystem::new(0.01);
        rstdp.add_synapse(0, 1, 0.5);

        assert_eq!(rstdp.synapses.len(), 1);
        assert_eq!(rstdp.get_weight(0, 1), Some(0.5));
    }

    #[test]
    fn test_eligibility_trace_formation() {
        let mut rstdp = RSTDPSystem::new(0.01);
        rstdp.add_synapse(0, 1, 0.5);

        // Pre-spike followed by post-spike (should create LTP eligibility)
        rstdp.on_pre_spike(0, &[1], 0.1);
        rstdp.on_post_spike(1, &[0], 10.0); // 10ms later

        let synapse = rstdp.synapses.get(&(0, 1)).unwrap();
        assert!(synapse.eligibility > 0.0, "Causal pairing should create positive eligibility");
    }

    #[test]
    fn test_reward_modulation() {
        let mut rstdp = RSTDPSystem::new(0.1);
        rstdp.add_synapse(0, 1, 0.5);

        // Build reward statistics first with neutral rewards
        for _ in 0..10 {
            rstdp.apply_reward(0.0, 0.1);
        }

        // Create eligibility trace
        rstdp.on_pre_spike(0, &[1], 0.1);
        rstdp.on_post_spike(1, &[0], 5.0);

        let initial_weight = rstdp.get_weight(0, 1).unwrap();

        // Apply positive reward
        rstdp.apply_reward(1.0, 0.1);

        let final_weight = rstdp.get_weight(0, 1).unwrap();

        // Positive eligibility + positive reward → weight increase
        assert!(final_weight > initial_weight,
            "Positive reward should increase weight (initial={}, final={})",
            initial_weight, final_weight);
    }

    #[test]
    fn test_reward_normalization() {
        let mut rstdp = RSTDPSystem::new(0.1);

        // Apply several rewards to build statistics
        for r in &[0.0, 0.5, 1.0, 0.5, 0.0, 1.0] {
            rstdp.apply_reward(*r, 0.1);
        }

        // Mean should be around 0.5
        assert!((rstdp.reward_mean - 0.5).abs() < 0.2, "Mean should be ~0.5");

        // Std should be non-zero
        assert!(rstdp.reward_std > 0.0, "Std should be positive");
    }

    #[test]
    fn test_meta_learning_adaptation() {
        let mut rstdp = RSTDPSystem::new(0.005);
        rstdp.meta_learning_enabled = true;
        rstdp.add_synapse(0, 1, 0.5);

        // High variance rewards → higher learning rate
        rstdp.apply_reward(10.0, 0.1);
        rstdp.apply_reward(-5.0, 0.1);
        rstdp.apply_reward(8.0, 0.1);

        let eta_high_variance = rstdp.eta;

        // Reset
        rstdp = RSTDPSystem::new(0.005);
        rstdp.meta_learning_enabled = true;
        rstdp.add_synapse(0, 1, 0.5);

        // Low variance rewards → lower learning rate
        rstdp.apply_reward(0.1, 0.1);
        rstdp.apply_reward(0.11, 0.1);
        rstdp.apply_reward(0.09, 0.1);

        let eta_low_variance = rstdp.eta;

        assert!(eta_high_variance > eta_low_variance,
            "High variance should increase learning rate (high={}, low={})",
            eta_high_variance, eta_low_variance);
    }

    #[test]
    fn test_eligibility_decay() {
        let mut rstdp = RSTDPSystem::new(0.01);
        rstdp.add_synapse(0, 1, 0.5);

        // Create eligibility
        rstdp.on_pre_spike(0, &[1], 0.1);
        rstdp.on_post_spike(1, &[0], 5.0);

        let initial_eligibility = rstdp.synapses.get(&(0, 1)).unwrap().eligibility;

        // Let time pass
        for _ in 0..100 {
            rstdp.update(10.0); // 10ms timesteps
        }

        let final_eligibility = rstdp.synapses.get(&(0, 1)).unwrap().eligibility;

        assert!(final_eligibility < initial_eligibility * 0.5,
            "Eligibility should decay over time");
    }

    #[test]
    fn test_weight_bounds() {
        let mut rstdp = RSTDPSystem::new(0.5);
        rstdp.add_synapse(0, 1, 0.9);

        // Create strong eligibility
        rstdp.on_pre_spike(0, &[1], 0.1);
        rstdp.on_post_spike(1, &[0], 5.0);

        // Apply many large positive rewards
        for _ in 0..100 {
            rstdp.apply_reward(10.0, 0.1);
        }

        let weight = rstdp.get_weight(0, 1).unwrap();

        // Weight should be clamped to max
        assert!(weight <= rstdp.w_max, "Weight should not exceed maximum");
        assert!(weight >= rstdp.w_min, "Weight should not go below minimum");
    }

    #[test]
    fn test_three_factor_learning() {
        let mut rstdp = RSTDPSystem::new(0.1);
        rstdp.tau_eligibility = 10000.0; // Very long eligibility trace for test
        rstdp.add_synapse(0, 1, 0.5);
        rstdp.add_synapse(0, 2, 0.5);

        // Build reward statistics
        for _ in 0..10 {
            rstdp.apply_reward(0.0, 0.1);
        }

        // Create eligibility for synapse 0→1 only
        rstdp.on_pre_spike(0, &[1], 0.1);
        rstdp.on_post_spike(1, &[0], 5.0);

        // No activity for synapse 0→2

        let weight_01_before = rstdp.get_weight(0, 1).unwrap();
        let weight_02_before = rstdp.get_weight(0, 2).unwrap();

        // Apply reward
        rstdp.apply_reward(1.0, 0.1);

        let weight_01_after = rstdp.get_weight(0, 1).unwrap();
        let weight_02_after = rstdp.get_weight(0, 2).unwrap();

        // Only synapse with eligibility should change
        assert!((weight_01_after - weight_01_before).abs() > 0.00001,
            "Synapse with eligibility should change (01_before={}, 01_after={}, diff={})",
            weight_01_before, weight_01_after, (weight_01_after - weight_01_before).abs());
        assert!((weight_02_after - weight_02_before).abs() < 0.00001,
            "Synapse without eligibility should not change");
    }
}
