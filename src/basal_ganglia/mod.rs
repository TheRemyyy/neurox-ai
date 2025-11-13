//! Basal Ganglia - Action Selection and Reinforcement Learning
//!
//! Implements dopamine-modulated temporal difference learning through competing pathways.
//! Critical for goal-directed behavior, habit formation, and reward-based learning.
//!
//! # Architecture
//! - Striatum with D1 (Go pathway) and D2 (NoGo pathway) neurons
//! - Direct pathway: Striatum D1 → GPi → Thalamus (disinhibits actions)
//! - Indirect pathway: Striatum D2 → GPe → STN → GPi (suppresses actions)
//! - Dopamine system computing temporal difference errors
//! - Eligibility traces for retroactive credit assignment
//!
//! # Biological Parameters
//! - 2.79M striatal neurons (50% D1, 50% D2)
//! - Baseline firing: ~1Hz, burst during selection: 40-60Hz
//! - Dopamine bursts: 15-30Hz (+300% baseline) for positive errors
//! - Dopamine pauses: <1Hz (-80%) for negative errors
//! - TD learning: δ(t) = r(t) + γV(t) - V(t-1)
//! - Learning rate α = 0.01-0.1, discount γ = 0.95-0.99

use crate::neuron::LIFNeuron;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Dopamine neuron encoding temporal difference errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DopamineNeuron {
    /// Current dopamine level (0.0-1.0, baseline ~0.3)
    pub dopamine_level: f32,

    /// Baseline firing rate (Hz)
    pub baseline_rate: f32,

    /// Current firing rate (Hz)
    pub firing_rate: f32,

    /// Value estimate V(t)
    pub value_estimate: f32,

    /// Learning rate for value function
    pub alpha: f32,

    /// Discount factor
    pub gamma: f32,

    /// Last reward received
    last_reward: f32,

    /// TD error history (for debugging)
    pub td_error_history: Vec<f32>,
}

impl DopamineNeuron {
    /// Create new dopamine neuron
    pub fn new(alpha: f32, gamma: f32) -> Self {
        Self {
            dopamine_level: 0.3,
            baseline_rate: 4.0,
            firing_rate: 4.0,
            value_estimate: 0.0,
            alpha,
            gamma,
            last_reward: 0.0,
            td_error_history: Vec::new(),
        }
    }

    /// Compute temporal difference error
    /// δ(t) = r(t) + γV(t) - V(t-1)
    pub fn compute_td_error(&mut self, reward: f32, next_state_value: f32) -> f32 {
        let td_error = reward + self.gamma * next_state_value - self.value_estimate;

        // Update value estimate
        self.value_estimate += self.alpha * td_error;

        // Store for analysis
        self.td_error_history.push(td_error);
        if self.td_error_history.len() > 1000 {
            self.td_error_history.remove(0);
        }

        self.last_reward = reward;
        td_error
    }

    /// Update dopamine signaling based on TD error
    ///
    /// Positive errors → burst (15-30Hz, +300% baseline)
    /// Negative errors → pause (<1Hz, -80% baseline)
    pub fn update_dopamine(&mut self, td_error: f32) {
        // Map TD error to dopamine signal
        if td_error > 0.0 {
            // Burst for positive prediction error
            let burst_strength = (td_error * 5.0).min(1.0);
            self.firing_rate = self.baseline_rate * (1.0 + 3.0 * burst_strength); // Up to +300%
            self.dopamine_level = 0.3 + 0.7 * burst_strength; // 0.3 to 1.0
        } else {
            // Pause for negative prediction error
            let pause_strength = (td_error.abs() * 5.0).min(1.0);
            self.firing_rate = self.baseline_rate * (1.0 - 0.8 * pause_strength); // Down to -80%
            self.dopamine_level = 0.3 * (1.0 - pause_strength); // 0.3 to 0.0
        }
    }

    /// Get dopamine modulation factor for learning
    /// Returns multiplier for learning rate (0.0-2.0)
    pub fn get_learning_modulation(&self) -> f32 {
        // Map dopamine level to learning modulation
        // Baseline (0.3) → 1.0x
        // High (1.0) → 2.0x
        // Low (0.0) → 0.0x
        (self.dopamine_level / 0.3).min(2.0)
    }
}

/// Striatal neuron (MSN - Medium Spiny Neuron)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StriatumNeuron {
    /// Base LIF neuron
    pub neuron: LIFNeuron,

    /// Receptor type: D1 (Go) or D2 (NoGo)
    pub receptor_type: ReceptorType,

    /// Eligibility trace for credit assignment (decays over 1-2 seconds)
    pub eligibility_trace: f32,

    /// Eligibility trace decay constant (1500ms for 1.5 second window)
    pub tau_eligibility: f32,

    /// Synaptic weights to output (thalamus/GPe)
    pub output_weights: Vec<f32>,
}

/// Dopamine receptor type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceptorType {
    /// D1 receptors - Direct "Go" pathway
    D1,
    /// D2 receptors - Indirect "NoGo" pathway
    D2,
}

impl StriatumNeuron {
    /// Create new striatal neuron
    pub fn new(id: u32, receptor_type: ReceptorType, n_outputs: usize) -> Self {
        Self {
            neuron: LIFNeuron::new(id),
            receptor_type,
            eligibility_trace: 0.0,
            tau_eligibility: 1500.0, // 1.5 seconds
            output_weights: vec![0.1; n_outputs],
        }
    }

    /// Update eligibility trace
    pub fn update_eligibility(&mut self, dt: f32, spiked: bool) {
        // Decay trace
        let decay = (-dt / self.tau_eligibility).exp();
        self.eligibility_trace *= decay;

        // Increase on spike (marks synapse as eligible for modification)
        if spiked {
            self.eligibility_trace += 1.0;
            self.eligibility_trace = self.eligibility_trace.min(1.0);
        }
    }

    /// Apply dopamine-modulated plasticity
    ///
    /// Three-factor learning: pre × post × dopamine
    /// ΔW = learning_rate × eligibility × dopamine
    pub fn apply_dopamine_plasticity(&mut self, dopamine_level: f32, base_learning_rate: f32) {
        for weight in &mut self.output_weights {
            // Three-factor rule
            let delta_w = base_learning_rate * self.eligibility_trace * (dopamine_level - 0.3);

            // D1 and D2 respond differently to dopamine
            let modulated_delta = match self.receptor_type {
                ReceptorType::D1 => delta_w,        // Potentiate when dopamine high
                ReceptorType::D2 => -delta_w,       // Potentiate when dopamine low
            };

            *weight += modulated_delta;
            *weight = weight.clamp(0.0, 1.0);
        }
    }
}

/// Globus Pallidus External segment (GPe)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPeNeuron {
    pub neuron: LIFNeuron,
    pub output_weights: Vec<f32>,
}

impl GPeNeuron {
    pub fn new(id: u32, n_outputs: usize) -> Self {
        Self {
            neuron: LIFNeuron::new(id),
            output_weights: vec![0.5; n_outputs],
        }
    }
}

/// Subthalamic Nucleus (STN)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STNNeuron {
    pub neuron: LIFNeuron,
    pub output_weights: Vec<f32>,
}

impl STNNeuron {
    pub fn new(id: u32, n_outputs: usize) -> Self {
        Self {
            neuron: LIFNeuron::new(id),
            output_weights: vec![0.8; n_outputs],
        }
    }
}

/// Globus Pallidus Internal segment (GPi) - output gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPiNeuron {
    pub neuron: LIFNeuron,
    pub output_weights: Vec<f32>,
}

impl GPiNeuron {
    pub fn new(id: u32, n_outputs: usize) -> Self {
        Self {
            neuron: LIFNeuron::new(id),
            output_weights: vec![0.5; n_outputs],
        }
    }
}

/// Basal Ganglia System
///
/// Implements action selection through competing Go/NoGo pathways
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasalGanglia {
    /// Striatal neurons (D1 + D2)
    pub striatum: Vec<StriatumNeuron>,

    /// GPe neurons (indirect pathway)
    pub gpe: Vec<GPeNeuron>,

    /// STN neurons (indirect pathway)
    pub stn: Vec<STNNeuron>,

    /// GPi neurons (output)
    pub gpi: Vec<GPiNeuron>,

    /// Dopamine system
    pub dopamine: DopamineNeuron,

    /// Number of striatal neurons
    pub n_striatum: usize,

    /// Number of actions
    pub n_actions: usize,

    /// Base learning rate
    pub learning_rate: f32,

    /// Action values (Q-values)
    #[serde(skip)]
    action_values: Arc<DashMap<usize, f32>>,

    /// Temperature for softmax action selection
    pub temperature: f32,
}

impl BasalGanglia {
    /// Create new basal ganglia circuit
    ///
    /// # Arguments
    /// - `n_striatum`: Number of striatal neurons (will be split 50/50 D1/D2)
    /// - `n_actions`: Number of possible actions
    /// - `learning_rate`: Base learning rate (0.01-0.1)
    /// - `gamma`: Discount factor (0.95-0.99)
    pub fn new(n_striatum: usize, n_actions: usize, learning_rate: f32, gamma: f32) -> Self {
        let n_d1 = n_striatum / 2;
        let n_d2 = n_striatum - n_d1;

        // Create striatal neurons (50% D1, 50% D2)
        let mut striatum = Vec::with_capacity(n_striatum);
        for i in 0..n_d1 {
            striatum.push(StriatumNeuron::new(i as u32, ReceptorType::D1, n_actions));
        }
        for i in n_d1..n_striatum {
            striatum.push(StriatumNeuron::new(i as u32, ReceptorType::D2, n_actions));
        }

        // Create GPe neurons (receive from D2)
        let n_gpe = n_d2 / 4;
        let gpe = (0..n_gpe)
            .map(|i| GPeNeuron::new((n_striatum + i) as u32, n_actions))
            .collect();

        // Create STN neurons (receive from GPe)
        let n_stn = n_gpe / 2;
        let stn = (0..n_stn)
            .map(|i| STNNeuron::new((n_striatum + n_gpe + i) as u32, n_actions))
            .collect();

        // Create GPi neurons (receive from D1 and STN)
        let gpi = (0..n_actions)
            .map(|i| GPiNeuron::new((n_striatum + n_gpe + n_stn + i) as u32, 1))
            .collect();

        // Create dopamine neuron
        let dopamine = DopamineNeuron::new(learning_rate, gamma);

        // Initialize action values
        let action_values = Arc::new(DashMap::new());
        for i in 0..n_actions {
            action_values.insert(i, 0.0);
        }

        Self {
            striatum,
            gpe,
            stn,
            gpi,
            dopamine,
            n_striatum,
            n_actions,
            learning_rate,
            action_values,
            temperature: 1.0,
        }
    }

    /// Process input and select action
    ///
    /// # Arguments
    /// - `state`: Current state representation
    /// - `dt`: Timestep (ms)
    ///
    /// # Returns
    /// Selected action index
    pub fn select_action(&mut self, state: &[f32], dt: f32) -> usize {
        // Compute Go and NoGo signals for each action
        let mut go_signals = vec![0.0; self.n_actions];
        let mut nogo_signals = vec![0.0; self.n_actions];

        // Process through striatum
        for (i, neuron) in self.striatum.iter_mut().enumerate() {
            // Input from state (simplified - in reality would be cortical input)
            let input_current = state.iter().sum::<f32>() / state.len() as f32;

            // Update neuron
            let spiked = neuron.neuron.update(dt, input_current);
            neuron.update_eligibility(dt, spiked);

            if spiked {
                // Accumulate pathway signals
                for (action_idx, weight) in neuron.output_weights.iter().enumerate() {
                    match neuron.receptor_type {
                        ReceptorType::D1 => go_signals[action_idx] += weight,
                        ReceptorType::D2 => nogo_signals[action_idx] += weight,
                    }
                }
            }
        }

        // Process indirect pathway (D2 → GPe → STN → GPi → suppression)
        let mut gpe_activity = vec![0.0; self.gpe.len()];
        for (i, neuron) in self.gpe.iter_mut().enumerate() {
            let input = nogo_signals.iter().sum::<f32>() / self.n_actions as f32;
            if neuron.neuron.update(dt, input) {
                gpe_activity[i] = 1.0;
            }
        }

        let mut stn_activity = vec![0.0; self.stn.len()];
        for (i, neuron) in self.stn.iter_mut().enumerate() {
            // STN inhibited by GPe
            let input = -gpe_activity.iter().sum::<f32>() / self.gpe.len() as f32;
            if neuron.neuron.update(dt, input) {
                stn_activity[i] = 1.0;
            }
        }

        // Process GPi (receives D1 inhibition and STN excitation)
        for (action_idx, neuron) in self.gpi.iter_mut().enumerate() {
            let d1_inhibition = -go_signals[action_idx];
            let stn_excitation = stn_activity.iter().sum::<f32>() / self.stn.len() as f32;
            let input = d1_inhibition + stn_excitation;
            neuron.neuron.update(dt, input);
        }

        // Compute action probabilities using softmax over (Go - NoGo)
        let mut action_scores: Vec<f32> = go_signals
            .iter()
            .zip(nogo_signals.iter())
            .map(|(go, nogo)| go - nogo)
            .collect();

        // Softmax selection
        let max_score = action_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = action_scores
            .iter()
            .map(|&s| ((s - max_score) / self.temperature).exp())
            .collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let probabilities: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Sample action
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probabilities.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }

        self.n_actions - 1
    }

    /// Update after receiving reward
    ///
    /// Computes TD error and applies dopamine-modulated plasticity
    pub fn update(&mut self, reward: f32, next_state_value: f32, dt: f32) {
        // Compute TD error
        let td_error = self.dopamine.compute_td_error(reward, next_state_value);

        // Update dopamine signaling
        self.dopamine.update_dopamine(td_error);

        // Apply dopamine-modulated plasticity to striatal neurons
        let dopamine_level = self.dopamine.dopamine_level;
        let modulated_lr = self.learning_rate * self.dopamine.get_learning_modulation();

        for neuron in &mut self.striatum {
            neuron.apply_dopamine_plasticity(dopamine_level, modulated_lr);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> BasalGangliaStats {
        let avg_d1_trace: f32 = self.striatum
            .iter()
            .filter(|n| n.receptor_type == ReceptorType::D1)
            .map(|n| n.eligibility_trace)
            .sum::<f32>() / (self.n_striatum / 2) as f32;

        let avg_d2_trace: f32 = self.striatum
            .iter()
            .filter(|n| n.receptor_type == ReceptorType::D2)
            .map(|n| n.eligibility_trace)
            .sum::<f32>() / (self.n_striatum / 2) as f32;

        BasalGangliaStats {
            dopamine_level: self.dopamine.dopamine_level,
            dopamine_firing_rate: self.dopamine.firing_rate,
            value_estimate: self.dopamine.value_estimate,
            avg_td_error: self.dopamine.td_error_history.iter().sum::<f32>()
                / self.dopamine.td_error_history.len().max(1) as f32,
            avg_d1_eligibility: avg_d1_trace,
            avg_d2_eligibility: avg_d2_trace,
            n_striatum: self.n_striatum,
            n_actions: self.n_actions,
        }
    }
}

/// Basal ganglia statistics
#[derive(Debug, Clone)]
pub struct BasalGangliaStats {
    pub dopamine_level: f32,
    pub dopamine_firing_rate: f32,
    pub value_estimate: f32,
    pub avg_td_error: f32,
    pub avg_d1_eligibility: f32,
    pub avg_d2_eligibility: f32,
    pub n_striatum: usize,
    pub n_actions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dopamine_neuron() {
        let mut da = DopamineNeuron::new(0.1, 0.95);

        // Positive prediction error
        let td_error = da.compute_td_error(1.0, 0.0);
        assert!(td_error > 0.0);

        da.update_dopamine(td_error);
        assert!(da.dopamine_level > 0.3); // Above baseline
        assert!(da.firing_rate > da.baseline_rate);
    }

    #[test]
    fn test_eligibility_trace() {
        let mut neuron = StriatumNeuron::new(0, ReceptorType::D1, 4);

        // Spike should increase trace
        neuron.update_eligibility(1.0, true);
        assert!(neuron.eligibility_trace > 0.0);

        // Should decay over time
        for _ in 0..100 {
            neuron.update_eligibility(10.0, false);
        }
        assert!(neuron.eligibility_trace < 0.5);
    }

    #[test]
    fn test_basal_ganglia_action_selection() {
        let mut bg = BasalGanglia::new(100, 4, 0.1, 0.95);

        let state = vec![0.5, 0.3, 0.8, 0.2];
        let action = bg.select_action(&state, 0.1);

        assert!(action < 4);
    }

    #[test]
    fn test_three_factor_learning() {
        let mut neuron = StriatumNeuron::new(0, ReceptorType::D1, 4);

        // Mark synapse as eligible
        neuron.update_eligibility(1.0, true);

        let initial_weight = neuron.output_weights[0];

        // High dopamine → potentiate D1
        neuron.apply_dopamine_plasticity(0.8, 0.1);
        assert!(neuron.output_weights[0] > initial_weight);
    }
}
