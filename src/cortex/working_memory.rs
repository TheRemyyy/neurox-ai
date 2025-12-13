//! Working Memory Module (Attractor Network)
//!
//! Biologically-inspired working memory based on continuous attractor dynamics
//! and NMDA-mediated bistability.
//!
//! # Biological Basis
//! - **Attractor Dynamics:** Memories are stable states (fixed points) of the recurrent network.
//! - **NMDA Plateaus:** Neurons exhibit bistability (UP/DOWN states) mediated by nonlinear dendritic integration.
//! - **Lateral Inhibition:** Competition between memory items limits capacity (Miller's 7±2).
//! - **Content Addressability:** Retrieval is auto-associative pattern completion.

use crate::neuron::NeuronState;
use serde::{Deserialize, Serialize};

/// Neuron with NMDA-mediated bistability (UP/DOWN states)
///
/// Models the "calcium plateau" mechanism that allows prefrontal cortex neurons
/// to maintain firing without continuous input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BistableNeuron {
    /// Soma state (spiking)
    pub state: NeuronState,

    /// Dendritic NMDA current (slow dynamics)
    pub nmda_current: f32,

    /// Bistability threshold (inputs above this trigger UP state)
    pub plateau_threshold: f32,

    /// Plateau maintenance current (self-excitation)
    pub plateau_current: f32,

    /// Is neuron in UP state (persistent firing)?
    pub in_up_state: bool,

    /// Adaptation (fatigue) to prevent infinite persistence
    pub adaptation: f32,
}

impl BistableNeuron {
    pub fn new(id: u32) -> Self {
        Self {
            state: NeuronState::new(id),
            nmda_current: 0.0,
            plateau_threshold: 0.5,
            plateau_current: 12.0, // Strong enough to drive spiking
            in_up_state: false,
            adaptation: 0.0,
        }
    }

    /// Update neuron dynamics
    ///
    /// # Dynamics
    /// 1. Integrate NMDA current (slow).
    /// 2. Check for UP state transition (bistability).
    /// 3. Add plateau current if in UP state.
    /// 4. Integrate somatic voltage.
    pub fn update(&mut self, dt: f32, synaptic_input: f32, inhibition: f32) -> bool {
        // 1. NMDA Dynamics (Slow integration of excitation)
        // dI_nmda/dt = -I_nmda/tau + Input
        let tau_nmda = 100.0; // 100ms
        self.nmda_current += ((-self.nmda_current + synaptic_input) / tau_nmda) * dt;

        // 2. Bistability Logic (Hysteresis)
        // Enter UP state if input is strong
        if self.nmda_current > self.plateau_threshold && !self.in_up_state {
            self.in_up_state = true;
        }
        
        // Exit UP state if strong inhibition or fatigue
        if (inhibition > 2.0 || self.adaptation > 1.0) && self.in_up_state {
            self.in_up_state = false;
        }

        // 3. Somatic Input
        let mut total_current = synaptic_input - inhibition;
        
        // Add self-sustaining plateau current if in UP state
        if self.in_up_state {
            total_current += self.plateau_current * (1.0 - self.adaptation);
        }

        // 4. Adaptation Dynamics (prevents items sticking forever)
        // dA/dt = (UP_state - A) / tau_adapt
        let tau_adapt = 5000.0; // 5 seconds persistence
        let target_adapt = if self.in_up_state { 1.0 } else { 0.0 };
        self.adaptation += ((target_adapt - self.adaptation) / tau_adapt) * dt;

        // 5. LIF Dynamics
        let spiked = self.state.v >= self.state.threshold;
        if spiked {
            self.state.v = self.state.v_reset;
            self.state.refractory_counter = 20; // 2ms
        } else if self.state.refractory_counter > 0 {
            self.state.refractory_counter -= 1;
        } else {
            // dV/dt = (-V + I) / tau
            let dv = ((-self.state.v - 70.0) + total_current) / self.state.tau_m * dt;
            self.state.v += dv;
        }

        spiked
    }

    /// Reset neuron
    pub fn reset(&mut self) {
        self.state.v = -70.0;
        self.nmda_current = 0.0;
        self.in_up_state = false;
        self.adaptation = 0.0;
    }
}

/// Working Memory Attractor Network
///
/// Stores patterns as stable reverberating states.
/// Capacity is limited by lateral inhibition (competition).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemory {
    /// Pool of bistable neurons
    neurons: Vec<BistableNeuron>,

    /// Recurrent weight matrix (flattened)
    /// W[i, j] = connection from j to i
    recurrent_weights: Vec<f32>,

    /// Pattern dimensionality
    pub pattern_dim: usize,

    /// Maximum capacity (number of "slots" or orthogonal patterns)
    pub capacity: usize,

    /// Global inhibition level (regulates total activity)
    global_inhibition: f32,

    /// Current number of active items (estimated)
    active_items: usize,

    /// Attention gating (0.0 = closed, 1.0 = open)
    input_gate: f32,
}

impl WorkingMemory {
    /// Create new Attractor Working Memory
    ///
    /// # Arguments
    /// - `capacity`: Approximate capacity (scales network size)
    /// - `pattern_dim`: Size of input patterns
    /// - `attention_threshold`: (Unused in attractor model, kept for API compatibility)
    pub fn new(capacity: usize, pattern_dim: usize, _attention_threshold: f32) -> Self {
        // We simulate a network large enough to hold 'capacity' orthogonal patterns
        // For simulation efficiency, we map 1:1 neuron to pattern dimension, 
        // but add "hidden" slots if needed. Here we assume pattern_dim IS the network size.
        let n_neurons = pattern_dim; 
        
        let neurons = (0..n_neurons).map(|i| BistableNeuron::new(i as u32)).collect();

        // Initialize weights as Identity (Auto-associative) - Self-excitation is handled internally
        // In a full Hopfield net, this would be Hebbian learned.
        // We start with zero recurrent weights and let 'store' update them (Fast Hebbian Plasticity).
        let recurrent_weights = vec![0.0; n_neurons * n_neurons];

        Self {
            neurons,
            recurrent_weights,
            pattern_dim,
            capacity,
            global_inhibition: 0.0,
            active_items: 0,
            input_gate: 0.0,
        }
    }

    /// Store pattern via Fast Hebbian Plasticity (One-shot learning)
    ///
    /// Implements: \Delta W = \eta * (x * x^T)
    /// This imprints the pattern as an attractor basin.
    ///
    /// # Returns
    /// `true` if stored (gate was open), `false` otherwise
    pub fn store(&mut self, pattern: &[f32], attention: f32) -> bool {
        if attention < 0.5 { return false; }

        assert_eq!(pattern.len(), self.neurons.len(), "Pattern dim mismatch");

        // 1. Fast Hebbian Plasticity: Imprint pattern weights
        // W_ij += rate * (p_i * p_j)
        // To prevent catastrophic forgetting, we implement a "palimpsest" memory 
        // where old weights decay slightly.
        let learning_rate = 0.5 * attention;
        let decay = 0.95; 

        for i in 0..self.pattern_dim {
            for j in 0..self.pattern_dim {
                if i == j { continue; } // No self-weights (handled by intrinsic bistability)
                
                let idx = i * self.pattern_dim + j;
                let delta = learning_rate * pattern[i] * pattern[j];
                
                self.recurrent_weights[idx] = self.recurrent_weights[idx] * decay + delta;
                
                // Bound weights
                self.recurrent_weights[idx] = self.recurrent_weights[idx].clamp(-1.0, 1.0);
            }
        }

        // 2. Direct Input Injection (Ignite the attractor)
        for (i, &val) in pattern.iter().enumerate() {
            if val > 0.5 {
                // Strong input pushes neuron into UP state
                self.neurons[i].nmda_current += 1.0; 
                self.neurons[i].in_up_state = true;
                self.neurons[i].adaptation = 0.0; // Reset fatigue (refresh memory)
            }
        }

        true
    }

    /// Retrieve pattern via Attractor Settling
    ///
    /// Given a cue, run the network dynamics for a few steps to let it settle
    /// into the nearest attractor (Pattern Completion).
    pub fn retrieve(&self, query: &[f32]) -> Option<Vec<f32>> {
        // Clone state for simulation
        let mut sim_neurons = self.neurons.clone();
        
        // Inject query
        for (i, &val) in query.iter().enumerate() {
            sim_neurons[i].nmda_current += val * 2.0; // Strong seed
        }

        // Run dynamics for 20 steps (settling)
        let dt = 1.0; // 1ms
        for _ in 0..20 {
            let n = sim_neurons.len();
            let mut inputs = vec![0.0; n];
            
            // Compute recurrent input
            for i in 0..n {
                for j in 0..n {
                    if sim_neurons[j].in_up_state { // Using state as proxy for rate
                         let weight = self.recurrent_weights[i * n + j];
                         inputs[i] += weight;
                    }
                }
            }

            // Global Inhibition (Competition)
            let active_count = sim_neurons.iter().filter(|n| n.in_up_state).count() as f32;
            let inhibition = active_count * 0.1;

            // Update
            for (i, neuron) in sim_neurons.iter_mut().enumerate() {
                neuron.update(dt, inputs[i], inhibition);
            }
        }

        // Read out state
        let result: Vec<f32> = sim_neurons.iter()
            .map(|n| if n.in_up_state { 1.0 } else { 0.0 })
            .collect();

        // If silence, return None
        if result.iter().sum::<f32>() < 0.1 {
            None
        } else {
            Some(result)
        }
    }

    /// Maintain reverberating activity (Update loop)
    pub fn maintain(&mut self, dt: f32) {
        let n = self.neurons.len();
        
        // 1. Calculate Recurrent Inputs (Matrix-Vector Multiply)
        // input_i = \sum_j W_ij * activity_j
        let mut recurrent_inputs = vec![0.0; n];
        let activity: Vec<f32> = self.neurons.iter()
            .map(|n| if n.in_up_state { 1.0 } else { 0.0 }) // Binary rate proxy
            .collect();

        // Optimization: In a real sparse system, we wouldn't do O(N^2)
        // But for WM (size ~1000), this is negligible.
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += self.recurrent_weights[i * n + j] * activity[j];
            }
            recurrent_inputs[i] = sum;
        }

        // 2. Calculate Global Inhibition (Negative Feedback)
        // Regulates network to prevent epilepsy and enforce capacity
        let total_activity: f32 = activity.iter().sum();
        
        // Target roughly 'capacity' active neurons (if distributed) 
        // or 'pattern_dim / capacity' if local. 
        // Simple regulation: Inhibition proportional to activity squared
        self.global_inhibition = total_activity * 0.05; 

        // 3. Update Neurons
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.update(dt, recurrent_inputs[i], self.global_inhibition);
        }

        // Estimate active items
        self.active_items = (total_activity / (self.pattern_dim as f32 / self.capacity as f32)).ceil() as usize;
    }

    /// Clear all memories (Global Reset)
    pub fn clear(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        self.recurrent_weights.fill(0.0);
    }

    /// Get current capacity utilization
    pub fn utilization(&self) -> f32 {
        let active = self.neurons.iter().filter(|n| n.in_up_state).count();
        // Assuming ~10% sparsity per item
        (active as f32 / (self.pattern_dim as f32 * 0.1)).min(1.0)
    }

    /// Get number of active items
    pub fn active_count(&self) -> usize {
        self.active_items
    }

    /// Get all patterns (Reconstruct from attractor state)
    /// (Approximation for visualization)
    pub fn get_all_patterns(&self) -> Vec<Vec<f32>> {
        // In an attractor net, patterns are superpositioned.
        // We return the current raw state vector.
        let state: Vec<f32> = self.neurons.iter()
            .map(|n| if n.in_up_state { 1.0 } else { 0.0 })
            .collect();
        vec![state] // Return as single "superposition" pattern
    }

    /// Get specific slot (Not applicable in attractor net, returns full state)
    pub fn get_pattern(&self, _slot: usize) -> Option<&Vec<f32>> {
        None // Attractor networks don't have discrete slots
    }

    /// Get stats
    pub fn stats(&self) -> WorkingMemoryStats {
        let active_neurons = self.neurons.iter().filter(|n| n.in_up_state).count();
        let avg_activity = active_neurons as f32 / self.neurons.len() as f32;
        
        WorkingMemoryStats {
            stored_patterns: self.active_items,
            capacity: self.capacity,
            utilization: self.utilization(),
            active_count: self.active_items,
            active_neurons,
            avg_activity,
            avg_attention: 1.0, // Deprecated field
        }
    }
}

/// Statistics
#[derive(Debug, Clone)]
pub struct WorkingMemoryStats {
    pub stored_patterns: usize,
    pub capacity: usize,
    pub utilization: f32,
    pub active_count: usize,
    pub active_neurons: usize,
    pub avg_activity: f32,
    pub avg_attention: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bistable_neuron_up_state() {
        let mut neuron = BistableNeuron::new(0);

        // Weak input -> No UP state
        neuron.update(1.0, 0.1, 0.0);
        assert!(!neuron.in_up_state);

        // Strong input -> UP state
        neuron.update(1.0, 1.0, 0.0);
        assert!(neuron.in_up_state);

        // Remove input -> Should persist (Bistability)
        neuron.update(1.0, 0.0, 0.0);
        assert!(neuron.in_up_state, "Neuron should maintain UP state without input");
    }

    #[test]
    fn test_bistable_neuron_reset_inhibition() {
        let mut neuron = BistableNeuron::new(0);
        neuron.in_up_state = true;
        neuron.plateau_current = 10.0;

        // Apply strong inhibition
        neuron.update(1.0, 0.0, 5.0);
        assert!(!neuron.in_up_state, "Strong inhibition should knock neuron out of UP state");
    }

    #[test]
    fn test_attractor_store_retrieve() {
        let mut wm = WorkingMemory::new(5, 10, 0.5);

        // Pattern 1: [1, 1, 0, 0, ...]
        let mut pattern = vec![0.0; 10];
        pattern[0] = 1.0;
        pattern[1] = 1.0;

        wm.store(&pattern, 1.0);

        // Verify storage (weights modified)
        // W_01 should be positive
        assert!(wm.recurrent_weights[0 * 10 + 1] > 0.0);

        // Retrieve with partial cue [1, 0, ...]
        let mut cue = vec![0.0; 10];
        cue[0] = 1.0;
        
        let retrieved = wm.retrieve(&cue).unwrap();
        
        // Should complete the pattern (index 1 should be active)
        assert!(retrieved[1] > 0.5, "Pattern completion failed");
    }

    #[test]
    fn test_capacity_competition() {
        // Small network, overload it
        let mut wm = WorkingMemory::new(2, 10, 0.5); // Capacity 2

        // Store 3 patterns
        for i in 0..3 {
            let mut pat = vec![0.0; 10];
            pat[i*2] = 1.0;
            wm.store(&pat, 1.0);
        }

        // Run maintenance
        wm.maintain(100.0);

        // Global inhibition should be high
        assert!(wm.global_inhibition > 0.0);
    }
}
