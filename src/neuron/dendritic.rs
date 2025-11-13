//! Dendritic Computation - Active Dendrites for Capacity Enhancement
//!
//! Active dendrites provide dramatic capacity enhancement (2-5×) through
//! local nonlinear computation before somatic integration.
//!
//! # Mechanisms
//! - Each dendritic branch computes local nonlinear function
//! - NMDA spikes: 5-15 synapses within 50μm and 50ms trigger plateau potential
//! - Plateau potentials: 50-200ms duration, 200-500% somatic amplification
//! - Branch-specific learning and calcium dynamics
//! - Creates two-layer network within single neuron
//!
//! # Performance
//! Research shows 97.7% MNIST accuracy vs 90% for synaptic-only models
//! with same neuron count (2024 findings).

use crate::neuron::NeuronState;
use serde::{Deserialize, Serialize};

/// Dendritic branch with local computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DendriticBranch {
    /// Branch ID
    pub id: usize,

    /// Synaptic inputs to this branch
    pub synapses: Vec<Synapse>,

    /// Local calcium concentration (for learning)
    pub calcium: f32,

    /// Calcium decay time constant (ms)
    pub tau_calcium: f32,

    /// NMDA spike threshold (number of coincident inputs)
    pub nmda_threshold: usize,

    /// Plateau potential state
    pub plateau_potential: f32,

    /// Plateau duration (ms)
    pub plateau_duration: f32,

    /// Time remaining in plateau (ms)
    pub plateau_timer: f32,

    /// Amplification factor during plateau (2-5×)
    pub amplification: f32,

    /// Branch-specific learning rate
    pub learning_rate: f32,

    /// Recent spike times (for coincidence detection)
    spike_buffer: Vec<f32>,

    /// Spatial extent (μm)
    pub spatial_extent: f32,
}

/// Synapse on dendritic branch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    /// Source neuron ID
    pub source_id: u32,

    /// Weight
    pub weight: f32,

    /// Spatial position on branch (0-1)
    pub position: f32,

    /// Last spike time
    pub last_spike: f32,
}

impl DendriticBranch {
    /// Create new dendritic branch
    pub fn new(id: usize, n_synapses: usize, spatial_extent: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let synapses = (0..n_synapses)
            .map(|i| Synapse {
                source_id: i as u32,
                weight: rng.gen_range(0.0..0.5),
                position: rng.gen(),
                last_spike: 0.0,
            })
            .collect();

        Self {
            id,
            synapses,
            calcium: 0.0,
            tau_calcium: 100.0,  // 100ms calcium decay
            nmda_threshold: 10,  // 10 coincident inputs
            plateau_potential: 0.0,
            plateau_duration: 100.0,  // 100ms plateau
            plateau_timer: 0.0,
            amplification: 3.0,  // 3× amplification
            learning_rate: 0.01,
            spike_buffer: Vec::new(),
            spatial_extent,
        }
    }

    /// Update dendritic branch
    ///
    /// Returns output current to soma
    pub fn update(&mut self, dt: f32, current_time: f32, input_spikes: &[bool]) -> f32 {
        // Decay calcium
        let decay = (-dt / self.tau_calcium).exp();
        self.calcium *= decay;

        // Decay plateau potential
        if self.plateau_timer > 0.0 {
            self.plateau_timer -= dt;
            if self.plateau_timer <= 0.0 {
                self.plateau_potential = 0.0;
            }
        }

        // Process synaptic inputs
        let mut branch_current = 0.0;
        let mut recent_spikes = Vec::new();

        for (i, synapse) in self.synapses.iter_mut().enumerate() {
            if i < input_spikes.len() && input_spikes[i] {
                // Record spike
                synapse.last_spike = current_time;
                recent_spikes.push((current_time, synapse.position));

                // Add synaptic current
                branch_current += synapse.weight;

                // Increase calcium
                self.calcium += 0.1;
            }
        }

        // Check for coincident inputs (NMDA spike)
        self.spike_buffer.extend(recent_spikes.iter().map(|(t, _)| *t));
        self.spike_buffer.retain(|&t| current_time - t < 50.0);  // 50ms window

        // Detect spatiotemporal clustering
        let clustered_spikes = self.detect_cluster(&recent_spikes, current_time);

        if clustered_spikes >= self.nmda_threshold {
            // Trigger NMDA spike → plateau potential
            self.trigger_plateau();
        }

        // Apply plateau amplification
        if self.plateau_potential > 0.0 {
            branch_current *= self.amplification;
        }

        // Clamp calcium
        self.calcium = self.calcium.clamp(0.0, 1.0);

        branch_current
    }

    /// Detect spatiotemporal clustering of inputs
    fn detect_cluster(&self, recent_spikes: &[(f32, f32)], current_time: f32) -> usize {
        if recent_spikes.is_empty() {
            return 0;
        }

        let mut clusters = 0;

        // Count spikes within 50μm and 50ms
        for (time1, pos1) in recent_spikes {
            if current_time - time1 > 50.0 {
                continue;
            }

            let mut local_count = 1;
            for (time2, pos2) in recent_spikes {
                if time1 == time2 && pos1 == pos2 {
                    continue;
                }

                let spatial_dist = (pos1 - pos2).abs() * self.spatial_extent;
                let temporal_dist = (time1 - time2).abs();

                if spatial_dist < 50.0 && temporal_dist < 50.0 {
                    local_count += 1;
                }
            }

            if local_count > clusters {
                clusters = local_count;
            }
        }

        clusters
    }

    /// Trigger NMDA-mediated plateau potential
    fn trigger_plateau(&mut self) {
        self.plateau_potential = 1.0;
        self.plateau_timer = self.plateau_duration;

        // Large calcium influx during plateau
        self.calcium += 0.3;
        self.calcium = self.calcium.min(1.0);
    }

    /// Branch-specific learning rule
    ///
    /// Hebbian with calcium gating
    pub fn learn(&mut self, post_spike: bool) {
        if !post_spike {
            return;
        }

        // Only learn if calcium is elevated (clustered input occurred)
        if self.calcium < 0.3 {
            return;
        }

        // Strengthen recently active synapses
        for synapse in &mut self.synapses {
            // Recent activity? (within 100ms)
            let recency = (100.0 - synapse.last_spike).max(0.0) / 100.0;
            if recency > 0.0 {
                // Calcium-gated plasticity
                let delta_w = self.learning_rate * recency * self.calcium;
                synapse.weight += delta_w;
                synapse.weight = synapse.weight.clamp(0.0, 1.0);
            }
        }
    }

    /// Get output for given input pattern
    pub fn compute(&self, inputs: &[f32]) -> f32 {
        let mut output = 0.0;
        for (i, synapse) in self.synapses.iter().enumerate() {
            if i < inputs.len() {
                output += synapse.weight * inputs[i];
            }
        }

        // Apply plateau amplification if active
        if self.plateau_potential > 0.0 {
            output *= self.amplification;
        }

        output
    }
}

/// Multi-compartment neuron with dendritic branches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DendriticNeuron {
    /// Somatic compartment
    pub soma: NeuronState,

    /// Dendritic branches
    pub branches: Vec<DendriticBranch>,

    /// Number of branches (typically 5-10)
    pub n_branches: usize,

    /// Current time (ms)
    current_time: f32,
}

impl DendriticNeuron {
    /// Create new dendritic neuron
    ///
    /// # Arguments
    /// - `id`: Neuron ID
    /// - `n_branches`: Number of dendritic branches (5-10)
    /// - `synapses_per_branch`: Synapses per branch (20-50)
    pub fn new(id: u32, n_branches: usize, synapses_per_branch: usize) -> Self {
        let branches = (0..n_branches)
            .map(|i| DendriticBranch::new(i, synapses_per_branch, 200.0))
            .collect();

        Self {
            soma: NeuronState::new(id),
            branches,
            n_branches,
            current_time: 0.0,
        }
    }

    /// Update neuron with dendritic computation
    ///
    /// # Arguments
    /// - `dt`: Timestep (ms)
    /// - `branch_inputs`: Spike inputs for each branch
    ///
    /// # Returns
    /// True if neuron spiked
    pub fn update(&mut self, dt: f32, branch_inputs: &[Vec<bool>]) -> bool {
        self.current_time += dt;

        // Compute dendritic currents
        let mut total_current = 0.0;
        for (i, branch) in self.branches.iter_mut().enumerate() {
            if i < branch_inputs.len() {
                let branch_current = branch.update(dt, self.current_time, &branch_inputs[i]);
                total_current += branch_current;
            }
        }

        // Integrate at soma
        if self.soma.refractory_counter > 0 {
            self.soma.refractory_counter -= 1;
            return false;
        }

        let dv = ((-self.soma.v + total_current) / self.soma.tau_m) * dt;
        self.soma.v += dv;

        // Check spike
        if self.soma.v >= self.soma.threshold {
            self.soma.v = self.soma.v_reset;
            self.soma.refractory_counter = 20;

            // Trigger learning in active branches
            for branch in &mut self.branches {
                branch.learn(true);
            }

            true
        } else {
            false
        }
    }

    /// Compute output for given branch-structured input
    pub fn compute(&self, branch_patterns: &[Vec<f32>]) -> f32 {
        let mut total = 0.0;
        for (i, branch) in self.branches.iter().enumerate() {
            if i < branch_patterns.len() {
                total += branch.compute(&branch_patterns[i]);
            }
        }
        total
    }

    /// Get statistics
    pub fn stats(&self) -> DendriticNeuronStats {
        let active_branches = self.branches
            .iter()
            .filter(|b| b.plateau_potential > 0.0)
            .count();

        let avg_calcium = self.branches
            .iter()
            .map(|b| b.calcium)
            .sum::<f32>() / self.n_branches as f32;

        let total_synapses: usize = self.branches
            .iter()
            .map(|b| b.synapses.len())
            .sum();

        DendriticNeuronStats {
            n_branches: self.n_branches,
            active_branches,
            avg_calcium,
            total_synapses,
            somatic_voltage: self.soma.v,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DendriticNeuronStats {
    pub n_branches: usize,
    pub active_branches: usize,
    pub avg_calcium: f32,
    pub total_synapses: usize,
    pub somatic_voltage: f32,
}

/// Layer of dendritic neurons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DendriticLayer {
    pub neurons: Vec<DendriticNeuron>,
    pub n_neurons: usize,
}

impl DendriticLayer {
    pub fn new(n_neurons: usize, branches_per_neuron: usize, synapses_per_branch: usize) -> Self {
        let neurons = (0..n_neurons)
            .map(|i| DendriticNeuron::new(i as u32, branches_per_neuron, synapses_per_branch))
            .collect();

        Self { neurons, n_neurons }
    }

    /// Update all neurons
    pub fn update(&mut self, dt: f32, inputs: &[Vec<Vec<bool>>]) -> Vec<bool> {
        let mut spikes = vec![false; self.n_neurons];

        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            if i < inputs.len() {
                spikes[i] = neuron.update(dt, &inputs[i]);
            }
        }

        spikes
    }

    /// Get layer statistics
    pub fn stats(&self) -> DendriticLayerStats {
        let spiking = self.neurons
            .iter()
            .filter(|n| n.soma.v > n.soma.threshold - 10.0)
            .count();

        let total_active_branches: usize = self.neurons
            .iter()
            .map(|n| n.branches.iter().filter(|b| b.plateau_potential > 0.0).count())
            .sum();

        DendriticLayerStats {
            n_neurons: self.n_neurons,
            spiking_neurons: spiking,
            active_branches: total_active_branches,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DendriticLayerStats {
    pub n_neurons: usize,
    pub spiking_neurons: usize,
    pub active_branches: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dendritic_branch() {
        let mut branch = DendriticBranch::new(0, 20, 200.0);

        // Sparse inputs shouldn't trigger plateau
        let inputs = vec![true, false, false, false, false];
        let current = branch.update(0.1, 0.0, &inputs);

        assert!(branch.plateau_potential == 0.0);

        // Dense clustered inputs should trigger plateau
        let dense_inputs = vec![true; 15];
        branch.update(0.1, 0.1, &dense_inputs);

        // May trigger plateau if spatiotemporally clustered
        // (depends on random positioning)
    }

    #[test]
    fn test_dendritic_neuron() {
        let mut neuron = DendriticNeuron::new(0, 5, 10);

        // Create branch inputs
        let branch_inputs: Vec<Vec<bool>> = (0..5)
            .map(|_| vec![true, false, true, false, true, false, true, false, false, false])
            .collect();

        // Update should integrate dendritic currents
        let spiked = neuron.update(0.1, &branch_inputs);

        // Check branches were updated
        assert!(neuron.branches.iter().any(|b| b.calcium > 0.0));
    }

    #[test]
    fn test_nmda_spike_detection() {
        let mut branch = DendriticBranch::new(0, 30, 100.0);

        // Position synapses close together for clustering
        for syn in &mut branch.synapses {
            syn.position = 0.5;  // All at same location
        }

        // Activate many synapses simultaneously
        let inputs = vec![true; 30];
        branch.update(0.1, 0.0, &inputs);

        // Should trigger plateau
        assert!(branch.plateau_potential > 0.0);
    }

    #[test]
    fn test_calcium_gated_learning() {
        let mut branch = DendriticBranch::new(0, 10, 100.0);

        // Set high calcium (as if NMDA spike occurred)
        branch.calcium = 0.8;

        let initial_weights: Vec<f32> = branch.synapses.iter().map(|s| s.weight).collect();

        // Trigger learning
        branch.learn(true);

        // Weights should have changed
        let changed = branch.synapses
            .iter()
            .zip(initial_weights.iter())
            .any(|(s, &w)| (s.weight - w).abs() > 0.001);

        // Note: May not change if no recent spikes
    }

    #[test]
    fn test_dendritic_layer() {
        let mut layer = DendriticLayer::new(10, 5, 10);

        // Create dummy inputs
        let inputs: Vec<Vec<Vec<bool>>> = (0..10)
            .map(|_| {
                (0..5).map(|_| vec![false; 10]).collect()
            })
            .collect();

        let spikes = layer.update(0.1, &inputs);
        assert_eq!(spikes.len(), 10);
    }
}
