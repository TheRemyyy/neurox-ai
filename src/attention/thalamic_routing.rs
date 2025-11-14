//! Thalamic Attention and Routing System
//!
//! Spike-based attention mechanism with dynamic connectivity modulation.
//! Implements thalamic gating and cortical routing based on salience.
//!
//! # Biological Basis
//! - Thalamus as cortical router (Sherman & Guillery)
//! - Pulvinar attention system
//! - Top-down (task) and bottom-up (stimulus) signals
//! - Winner-take-all via lateral inhibition
//! - Dynamic gain modulation of existing connections

use crate::connectivity::SparseConnectivity;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Dynamic connectivity with gain modulation
///
/// Structural connectivity (base weights) is fixed,
/// but functional connectivity changes via gain modulation.
/// This implements the "connectivity fingerprint" concept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicConnectivity {
    /// Base structural connectivity (fixed)
    pub base: SparseConnectivity,

    /// Dynamic gain multipliers (one per synapse)
    /// Effective weight = base_weight * gain
    pub gains: Vec<f32>,

    /// Gain time constant (how fast gains change, ~50ms)
    tau_gain: f32,

    /// Maximum gain (prevents runaway)
    max_gain: f32,
}

impl DynamicConnectivity {
    /// Create new dynamic connectivity from base connectivity
    pub fn new(base: SparseConnectivity, tau_gain: f32, max_gain: f32) -> Self {
        let n_synapses = base.nnz;
        Self {
            base,
            gains: vec![1.0; n_synapses], // Start at neutral gain
            tau_gain,
            max_gain,
        }
    }

    /// Get effective weight for synapse
    pub fn effective_weight(&self, synapse_id: usize) -> f32 {
        self.base.weights[synapse_id] * self.gains[synapse_id]
    }

    /// Modulate gain for specific connection
    pub fn modulate_gain(&mut self, synapse_id: usize, target_gain: f32) {
        if synapse_id < self.gains.len() {
            self.gains[synapse_id] = target_gain.clamp(0.0, self.max_gain);
        }
    }

    /// Modulate gains for connections from source to target
    pub fn modulate_pathway(&mut self, source: usize, target: usize, gain: f32) {
        // Find synapses from source to target
        let start = self.base.row_ptr[target] as usize;
        let end = self.base.row_ptr[target + 1] as usize;

        for syn_id in start..end {
            if self.base.col_idx[syn_id] as usize == source {
                self.modulate_gain(syn_id, gain);
            }
        }
    }

    /// Update gains (decay towards 1.0)
    pub fn update_gains(&mut self, dt: f32) {
        let decay = (-dt / self.tau_gain).exp();
        for gain in &mut self.gains {
            // Decay towards neutral (1.0)
            *gain = 1.0 + (*gain - 1.0) * decay;
        }
    }

    /// Reset all gains to neutral
    pub fn reset_gains(&mut self) {
        self.gains.fill(1.0);
    }

    /// Get statistics
    pub fn stats(&self) -> DynamicConnectivityStats {
        let avg_gain = self.gains.iter().sum::<f32>() / self.gains.len() as f32;
        let max_gain = self.gains.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_gain = self.gains.iter().copied().fold(f32::INFINITY, f32::min);

        let modulated = self.gains.iter().filter(|&&g| (g - 1.0).abs() > 0.1).count();

        DynamicConnectivityStats {
            avg_gain,
            max_gain,
            min_gain,
            modulated_synapses: modulated,
            total_synapses: self.gains.len(),
        }
    }
}

/// Dynamic connectivity statistics
#[derive(Debug, Clone)]
pub struct DynamicConnectivityStats {
    pub avg_gain: f32,
    pub max_gain: f32,
    pub min_gain: f32,
    pub modulated_synapses: usize,
    pub total_synapses: usize,
}

/// Attention System implementing thalamic routing
///
/// Combines top-down (task-driven) and bottom-up (stimulus-driven)
/// attention signals to modulate information flow.
///
/// # Components
/// - Salience map: What's important right now
/// - Routing matrix: Dynamic connectivity
/// - Top-down signals: Task goals
/// - Bottom-up signals: Stimulus features
/// - Lateral inhibition: Winner-take-all
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSystem {
    /// Salience map (one value per neuron/location)
    salience: Vec<f32>,

    /// Dynamic connectivity matrix
    routing_matrix: DynamicConnectivity,

    /// Top-down attention signals (task-driven)
    top_down_signals: Vec<f32>,

    /// Bottom-up attention signals (stimulus-driven)
    bottom_up_signals: Vec<f32>,

    /// Lateral inhibition strength
    inhibition_strength: f32,

    /// Attention decay time constant (ms)
    tau_attention: f32,

    /// Winner-take-all threshold
    wta_threshold: f32,

    /// Number of locations/neurons
    n_locations: usize,
}

impl AttentionSystem {
    /// Create new attention system
    ///
    /// # Arguments
    /// - `n_locations`: Number of spatial locations or neuron groups
    /// - `connectivity`: Base structural connectivity
    /// - `inhibition_strength`: Lateral inhibition strength (typical: 2.0-5.0)
    pub fn new(
        n_locations: usize,
        connectivity: SparseConnectivity,
        inhibition_strength: f32,
    ) -> Self {
        Self {
            salience: vec![0.0; n_locations],
            routing_matrix: DynamicConnectivity::new(connectivity, 50.0, 5.0),
            top_down_signals: vec![0.0; n_locations],
            bottom_up_signals: vec![0.0; n_locations],
            inhibition_strength,
            tau_attention: 100.0, // 100ms attention time constant
            wta_threshold: 0.3,
            n_locations,
        }
    }

    /// Compute attention via spike-based correlation
    ///
    /// Implements transformer-like attention but with spikes:
    /// attention(Q, K) = spike_correlation(Q, K) with lateral inhibition
    ///
    /// # Returns
    /// Attention weights (sums to 1.0 after softmax)
    pub fn compute_attention(&mut self, query: &[f32], keys: &[Vec<f32>]) -> Vec<f32> {
        assert_eq!(query.len(), keys[0].len(), "Query/key dimension mismatch");

        let n_keys = keys.len();
        let mut scores = vec![0.0; n_keys];

        // Compute similarity scores (spike correlation)
        for (i, key) in keys.iter().enumerate() {
            scores[i] = Self::spike_correlation(query, key);
        }

        // Apply lateral inhibition (winner-take-all)
        self.apply_lateral_inhibition(&mut scores);

        // Softmax normalization
        Self::softmax(&mut scores);

        scores
    }

    /// Route information from source to target with specific strength
    ///
    /// Modulates the pathway gain to enhance/suppress specific connections.
    pub fn route(&mut self, source: usize, target: usize, strength: f32) {
        self.routing_matrix.modulate_pathway(source, target, strength);
    }

    /// Update salience map based on inputs
    ///
    /// Salience = α * bottom_up + β * top_down
    /// where α, β balance stimulus-driven vs task-driven attention
    pub fn update_salience(&mut self, inputs: &[f32]) {
        assert_eq!(inputs.len(), self.n_locations);

        // Bottom-up: Driven by stimulus intensity/novelty
        for (i, &input) in inputs.iter().enumerate() {
            self.bottom_up_signals[i] = input.abs();
        }

        // Combine top-down and bottom-up
        const ALPHA: f32 = 0.4; // Bottom-up weight
        const BETA: f32 = 0.6; // Top-down weight

        for i in 0..self.n_locations {
            self.salience[i] = ALPHA * self.bottom_up_signals[i] + BETA * self.top_down_signals[i];
        }

        // Normalize to [0, 1]
        let max_salience = self
            .salience
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if max_salience > 0.0 {
            for s in &mut self.salience {
                *s /= max_salience;
            }
        }
    }

    /// Set top-down attention signal (task goals)
    pub fn set_top_down(&mut self, location: usize, strength: f32) {
        if location < self.n_locations {
            self.top_down_signals[location] = strength.clamp(0.0, 1.0);
        }
    }

    /// Get current salience at location
    pub fn get_salience(&self, location: usize) -> f32 {
        *self.salience.get(location).unwrap_or(&0.0)
    }

    /// Get effective weight for synapse (base * gain)
    pub fn effective_weight(&self, synapse_id: usize) -> f32 {
        self.routing_matrix.effective_weight(synapse_id)
    }

    /// Update attention dynamics
    pub fn update(&mut self, dt: f32) {
        // Decay salience
        let decay = (-dt / self.tau_attention).exp();
        for s in &mut self.salience {
            *s *= decay;
        }

        // Decay gains towards neutral
        self.routing_matrix.update_gains(dt);

        // Decay top-down signals (task goals persist longer)
        let task_decay = (-dt / (self.tau_attention * 2.0)).exp();
        for td in &mut self.top_down_signals {
            *td *= task_decay;
        }
    }

    /// Apply lateral inhibition for winner-take-all
    ///
    /// Implements divisive normalization:
    /// output[i] = input[i] / (1 + inhibition_strength * sum(input))
    fn apply_lateral_inhibition(&self, scores: &mut [f32]) {
        let total: f32 = scores.iter().sum();

        if total > 0.0 {
            for score in scores.iter_mut() {
                *score /= 1.0 + self.inhibition_strength * total;
            }
        }

        // Apply winner-take-all threshold
        for score in scores.iter_mut() {
            if *score < self.wta_threshold {
                *score = 0.0;
            }
        }
    }

    /// Compute spike correlation (dot product normalized)
    fn spike_correlation(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());

        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Softmax normalization (spike-based approximation)
    fn softmax(scores: &mut [f32]) {
        // Find max for numerical stability
        let max_score = scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        // Exp
        for score in scores.iter_mut() {
            *score = (*score - max_score).exp();
        }

        // Normalize
        let sum: f32 = scores.iter().sum();
        if sum > 0.0 {
            for score in scores.iter_mut() {
                *score /= sum;
            }
        }
    }

    /// Focus attention on specific location
    ///
    /// Enhances routing to/from this location via gain modulation
    pub fn focus(&mut self, location: usize, strength: f32) {
        if location < self.n_locations {
            self.set_top_down(location, strength);

            // Boost salience
            self.salience[location] = strength.clamp(0.0, 1.0);
        }
    }

    /// Get attention statistics
    pub fn stats(&self) -> AttentionStats {
        let avg_salience = self.salience.iter().sum::<f32>() / self.n_locations as f32;
        let max_salience = self
            .salience
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let focused_locations = self
            .salience
            .iter()
            .filter(|&&s| s > self.wta_threshold)
            .count();

        let connectivity_stats = self.routing_matrix.stats();

        AttentionStats {
            avg_salience,
            max_salience,
            focused_locations,
            avg_top_down: self.top_down_signals.iter().sum::<f32>() / self.n_locations as f32,
            avg_bottom_up: self.bottom_up_signals.iter().sum::<f32>() / self.n_locations as f32,
            connectivity: connectivity_stats,
        }
    }
}

/// Attention statistics
#[derive(Debug, Clone)]
pub struct AttentionStats {
    pub avg_salience: f32,
    pub max_salience: f32,
    pub focused_locations: usize,
    pub avg_top_down: f32,
    pub avg_bottom_up: f32,
    pub connectivity: DynamicConnectivityStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_connectivity(n: usize) -> SparseConnectivity {
        // Simple all-to-all connectivity
        let mut row_ptr = vec![0; n + 1];
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();

        for i in 0..n {
            for j in 0..n {
                col_idx.push(j as i32);
                weights.push(0.5);
            }
            row_ptr[i + 1] = col_idx.len() as i32;
        }

        let nnz = col_idx.len();

        SparseConnectivity {
            row_ptr,
            col_idx,
            weights,
            nnz,
            n_neurons: n,
        }
    }

    #[test]
    fn test_dynamic_connectivity() {
        let base = create_test_connectivity(3);
        let mut dc = DynamicConnectivity::new(base, 50.0, 5.0);

        // Test gain modulation
        dc.modulate_gain(0, 2.0);
        assert!((dc.effective_weight(0) - 1.0).abs() < 0.01); // 0.5 * 2.0 = 1.0

        // Test decay
        dc.update_gains(50.0);
        assert!(dc.gains[0] < 2.0); // Should decay towards 1.0
    }

    #[test]
    fn test_attention_computation() {
        let conn = create_test_connectivity(5);
        let mut attention = AttentionSystem::new(5, conn, 2.0);

        let query = vec![1.0, 0.0, 0.0];
        let keys = vec![
            vec![1.0, 0.0, 0.0], // Perfect match
            vec![0.0, 1.0, 0.0], // Orthogonal
            vec![0.5, 0.5, 0.0], // Partial match
        ];

        let weights = attention.compute_attention(&query, &keys);

        // Weights should sum to 1.0
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // First key should have highest weight (perfect match)
        assert!(weights[0] > weights[1]);
        assert!(weights[0] > weights[2]);
    }

    #[test]
    fn test_salience_update() {
        let conn = create_test_connectivity(5);
        let mut attention = AttentionSystem::new(5, conn, 2.0);

        // Set top-down attention
        attention.set_top_down(2, 1.0);

        // Update with bottom-up input
        let inputs = vec![0.5, 0.0, 0.3, 0.0, 0.0];
        attention.update_salience(&inputs);

        // Location 2 should have high salience (top-down + bottom-up)
        assert!(attention.get_salience(2) > 0.5);
    }

    #[test]
    fn test_routing() {
        let conn = create_test_connectivity(3);
        let mut attention = AttentionSystem::new(3, conn, 2.0);

        // Route from 0 to 1 with strength 3.0
        attention.route(0, 1, 3.0);

        // Gains should be modulated
        let stats = attention.stats();
        assert!(stats.connectivity.max_gain >= 3.0);
    }

    #[test]
    fn test_focus() {
        let conn = create_test_connectivity(5);
        let mut attention = AttentionSystem::new(5, conn, 2.0);

        attention.focus(2, 0.9);

        assert!(attention.get_salience(2) > 0.8);
        assert!(attention.top_down_signals[2] > 0.8);
    }
}
