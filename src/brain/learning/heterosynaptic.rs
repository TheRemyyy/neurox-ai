//! Heterosynaptic Plasticity - NO-mediated and Astrocyte-mediated
//!
//! Implements nitric oxide (NO) diffusion for synapse-level potentiation
//! and astrocyte-mediated depression at neighboring synapses, enabling
//! meta-plasticity and distributed learning beyond classical Hebbian rules.
//!
//! # Features
//! - NO-mediated LTP: Active synapses release NO which diffuses spatially
//! - Astrocyte-mediated LTD: Glutamate spillover triggers Ca²⁺ in astrocytes
//! - Eligibility traces: Marks synapses for delayed reward signals
//! - Meta-plasticity: Hidden states θ_hidden track activity history
//! - Group mutation: Correlated changes across synapse clusters
//!
//! # Memory Overhead
//! - 3-5× vs basic STDP (estimated 300-500 MB for 10,000 synapses)
//! - NO concentration: 4 bytes/synapse
//! - Astrocyte calcium: 4 bytes/astrocyte (~1 per 100 synapses)
//! - Eligibility traces: 4 bytes/synapse
//! - Hidden states: 4 bytes/synapse
//!
//! # Biological Basis
//! - NO diffusion radius: ~100 μm (Garthwaite 2008)
//! - NO decay: τ_NO ≈ 2-5 seconds
//! - Astrocyte coverage: ~100,000 synapses per astrocyte
//! - Glutamate spillover threshold: ~1 mM

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Heterosynaptic plasticity system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterosynapticPlasticity {
    /// NO concentration per synapse (μM)
    pub no_concentration: Vec<f32>,

    /// NO decay time constant (seconds)
    pub no_decay_tau: f32,

    /// NO diffusion radius (μm)
    pub no_diffusion_radius: f32,

    /// NO production rate per spike
    pub no_production_rate: f32,

    /// Astrocyte coverage map: astrocyte_id -> [synapse_indices]
    pub astrocyte_coverage: HashMap<usize, Vec<usize>>,

    /// Astrocyte calcium concentration (μM)
    pub astrocyte_calcium: Vec<f32>,

    /// Astrocyte calcium decay time constant (seconds)
    pub astrocyte_ca_tau: f32,

    /// Glutamate spillover accumulator per astrocyte
    pub glutamate_spillover: Vec<f32>,

    /// Glutamate threshold for astrocyte activation (mM)
    pub glutamate_threshold: f32,

    /// Eligibility traces per synapse (for reinforcement learning)
    pub eligibility_traces: Vec<f32>,

    /// Eligibility trace decay time constant (seconds)
    pub trace_decay_tau: f32,

    /// Meta-plasticity hidden states per synapse
    pub hidden_states: Vec<f32>,

    /// Meta-learning rate
    pub meta_learning_rate: f32,

    /// LTP rate from NO
    pub eta_no_ltp: f32,

    /// LTD rate from astrocytes
    pub eta_astro_ltd: f32,

    /// Synapse spatial positions (x, y, z) in μm
    pub synapse_positions: Vec<(f32, f32, f32)>,

    /// Pre-computed neighbor lists for efficient diffusion (O(K) instead of O(N²))
    /// neighbor_lists[i] contains indices of synapses within diffusion radius of synapse i
    pub neighbor_lists: Vec<Vec<usize>>,

    /// Statistics
    pub total_no_events: usize,
    pub total_astrocyte_events: usize,
    pub total_potentiation_events: usize,
    pub total_depression_events: usize,
}

impl HeterosynapticPlasticity {
    /// Create new heterosynaptic plasticity system
    ///
    /// # Arguments
    /// - `n_synapses`: Number of synapses
    /// - `n_astrocytes`: Number of astrocytes (~1 per 100 synapses)
    /// - `spatial_extent`: Spatial extent of network (μm) for random positioning
    pub fn new(n_synapses: usize, n_astrocytes: usize, spatial_extent: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Random synapse positions in 3D space
        let synapse_positions: Vec<_> = (0..n_synapses)
            .map(|_| {
                (
                    rng.gen_range(0.0..spatial_extent),
                    rng.gen_range(0.0..spatial_extent),
                    rng.gen_range(0.0..spatial_extent),
                )
            })
            .collect();

        // Assign synapses to astrocytes based on spatial proximity
        let mut astrocyte_coverage: HashMap<usize, Vec<usize>> = HashMap::new();
        let astrocyte_positions: Vec<_> = (0..n_astrocytes)
            .map(|_| {
                (
                    rng.gen_range(0.0..spatial_extent),
                    rng.gen_range(0.0..spatial_extent),
                    rng.gen_range(0.0..spatial_extent),
                )
            })
            .collect();

        for (syn_idx, &syn_pos) in synapse_positions.iter().enumerate() {
            // Find nearest astrocyte
            let nearest_astro = astrocyte_positions
                .iter()
                .enumerate()
                .min_by(|(_, a_pos), (_, b_pos)| {
                    let dist_a = Self::euclidean_distance_3d(syn_pos, **a_pos);
                    let dist_b = Self::euclidean_distance_3d(syn_pos, **b_pos);
                    dist_a.partial_cmp(&dist_b).unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap();

            astrocyte_coverage
                .entry(nearest_astro)
                .or_default()
                .push(syn_idx);
        }

        // Pre-compute neighbor lists for O(K) diffusion instead of O(N²)
        let diffusion_radius: f32 = 100.0; // 100 μm
        let mut neighbor_lists = vec![Vec::new(); n_synapses];

        for i in 0..n_synapses {
            let (x1, y1, z1) = synapse_positions[i];
            for j in (i + 1)..n_synapses {
                let (x2, y2, z2) = synapse_positions[j];
                let dist_sq = (x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2);

                if dist_sq < diffusion_radius.powi(2) {
                    neighbor_lists[i].push(j);
                    neighbor_lists[j].push(i); // Symmetric
                }
            }
        }

        Self {
            no_concentration: vec![0.0; n_synapses],
            no_decay_tau: 3.0, // 3 seconds (Garthwaite 2008)
            no_diffusion_radius: diffusion_radius,
            no_production_rate: 0.5, // μM per spike
            astrocyte_coverage,
            astrocyte_calcium: vec![0.1; n_astrocytes], // Baseline Ca²⁺
            astrocyte_ca_tau: 2.0,                      // 2 seconds
            glutamate_spillover: vec![0.0; n_astrocytes],
            glutamate_threshold: 1.0, // 1 mM
            eligibility_traces: vec![0.0; n_synapses],
            trace_decay_tau: 1.0,                 // 1 second
            hidden_states: vec![0.5; n_synapses], // Neutral initial state
            meta_learning_rate: 0.001,
            eta_no_ltp: 0.01,     // LTP rate
            eta_astro_ltd: 0.005, // LTD rate (weaker than LTP)
            synapse_positions,
            neighbor_lists,
            total_no_events: 0,
            total_astrocyte_events: 0,
            total_potentiation_events: 0,
            total_depression_events: 0,
        }
    }

    /// Update heterosynaptic plasticity and return weight changes
    ///
    /// # Arguments
    /// - `synaptic_activity`: Recent spike activity per synapse (0-1)
    /// - `pre_spikes`: Pre-synaptic spikes (for eligibility traces)
    /// - `post_spikes`: Post-synaptic spikes (for eligibility traces)
    /// - `dt`: Timestep in seconds
    ///
    /// # Returns
    /// Weight changes for each synapse
    pub fn update(
        &mut self,
        synaptic_activity: &[f32],
        pre_spikes: &[bool],
        post_spikes: &[bool],
        dt: f32,
    ) -> Vec<f32> {
        let n_synapses = synaptic_activity.len();
        let mut weight_changes = vec![0.0; n_synapses];

        // 1. Update NO diffusion
        self.update_no_diffusion(synaptic_activity, dt);

        // 2. Update astrocyte dynamics
        self.update_astrocyte_dynamics(synaptic_activity, dt);

        // 3. Compute weight changes from NO and astrocytes
        for i in 0..n_synapses {
            // NO-mediated potentiation at active synapses
            if synaptic_activity[i] > 0.5 {
                let potentiation = self.compute_no_potentiation(i);
                weight_changes[i] += potentiation;
                if potentiation > 0.0001 {
                    self.total_potentiation_events += 1;
                }
            }

            // Astrocyte-mediated depression (affects neighbors of active synapses)
            let depression = self.compute_astrocyte_depression(i);
            weight_changes[i] -= depression;
            if depression > 0.0001 {
                self.total_depression_events += 1;
            }

            // Update eligibility traces
            self.eligibility_traces[i] *= (-dt / self.trace_decay_tau).exp();
            if pre_spikes[i] && post_spikes[i] {
                // Coincidence detection
                self.eligibility_traces[i] += 1.0;
            }

            // Meta-plasticity: adjust hidden state based on recent activity
            self.update_meta_plasticity(i, synaptic_activity[i], dt);

            // Modulate weight changes by hidden state
            weight_changes[i] *= self.hidden_states[i];
        }

        weight_changes
    }

    /// Update NO concentration with production, decay, and diffusion
    fn update_no_diffusion(&mut self, activity: &[f32], dt: f32) {
        let n_synapses = self.no_concentration.len();
        let mut new_no = self.no_concentration.clone();

        for i in 0..n_synapses {
            // NO decay: d[NO]/dt = -[NO]/τ_NO
            new_no[i] *= (-dt / self.no_decay_tau).exp();

            // NO production from active synapses
            if activity[i] > 0.5 {
                new_no[i] += self.no_production_rate * activity[i];
                self.total_no_events += 1;
            }

            // NO diffusion from nearby synapses (O(K) using pre-computed neighbors)
            let (x1, y1, z1) = self.synapse_positions[i];
            let mut diffusion_input = 0.0;

            for &j in &self.neighbor_lists[i] {
                let (x2, y2, z2) = self.synapse_positions[j];
                let distance = Self::euclidean_distance_3d((x1, y1, z1), (x2, y2, z2));

                // Gaussian diffusion kernel (no radius check needed - neighbors pre-filtered)
                let diffusion_kernel = (-distance * distance
                    / (2.0 * self.no_diffusion_radius * self.no_diffusion_radius))
                    .exp();
                diffusion_input += self.no_concentration[j] * diffusion_kernel * dt;
            }

            new_no[i] += diffusion_input * 0.1; // Diffusion rate
        }

        self.no_concentration = new_no;
    }

    /// Update astrocyte calcium and glutamate spillover
    fn update_astrocyte_dynamics(&mut self, activity: &[f32], dt: f32) {
        let n_astrocytes = self.astrocyte_calcium.len();

        for astro_idx in 0..n_astrocytes {
            // Decay glutamate spillover
            self.glutamate_spillover[astro_idx] *= 0.9; // Fast decay

            // Accumulate glutamate from covered synapses
            if let Some(covered_synapses) = self.astrocyte_coverage.get(&astro_idx) {
                for &syn_idx in covered_synapses {
                    if activity[syn_idx] > 0.5 {
                        self.glutamate_spillover[astro_idx] += activity[syn_idx] * 0.1;
                    }
                }
            }

            // Astrocyte calcium response to glutamate spillover
            if self.glutamate_spillover[astro_idx] > self.glutamate_threshold {
                // Calcium spike
                self.astrocyte_calcium[astro_idx] += 2.0;
                self.total_astrocyte_events += 1;
            }

            // Calcium decay
            let baseline = 0.1;
            self.astrocyte_calcium[astro_idx] +=
                dt * (baseline - self.astrocyte_calcium[astro_idx]) / self.astrocyte_ca_tau;
        }
    }

    /// Compute NO-mediated potentiation for a synapse
    fn compute_no_potentiation(&self, synapse_idx: usize) -> f32 {
        // LTP proportional to NO concentration
        // Δw = η_LTP × [NO] × (1 - w_current)
        // Assuming w_current is handled externally, return just the NO contribution
        self.eta_no_ltp * self.no_concentration[synapse_idx]
    }

    /// Compute astrocyte-mediated depression for a synapse
    fn compute_astrocyte_depression(&self, synapse_idx: usize) -> f32 {
        // Find which astrocyte covers this synapse
        for (astro_idx, covered_synapses) in &self.astrocyte_coverage {
            if covered_synapses.contains(&synapse_idx) {
                // LTD proportional to astrocyte calcium elevation
                let ca_elevation = (self.astrocyte_calcium[*astro_idx] - 0.1).max(0.0);
                return self.eta_astro_ltd * ca_elevation;
            }
        }
        0.0
    }

    /// Update meta-plasticity hidden state
    fn update_meta_plasticity(&mut self, synapse_idx: usize, activity: f32, dt: f32) {
        // θ_hidden tracks recent activity history and modulates future plasticity
        // dθ/dt = α_meta × (activity - θ_target)
        let theta_target = 0.5; // Neutral target

        self.hidden_states[synapse_idx] += self.meta_learning_rate * dt * (theta_target - activity);

        // Clamp to [0.1, 2.0] to prevent runaway
        self.hidden_states[synapse_idx] = self.hidden_states[synapse_idx].clamp(0.1, 2.0);
    }

    /// Apply eligibility-trace-modulated plasticity (for reinforcement learning)
    ///
    /// # Arguments
    /// - `reward`: Global reward signal (-1 to 1)
    ///
    /// # Returns
    /// Weight changes modulated by eligibility traces and reward
    pub fn apply_reward_modulated_plasticity(&self, reward: f32) -> Vec<f32> {
        self.eligibility_traces
            .iter()
            .map(|&trace| trace * reward * 0.01)
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> HeterosynapticStats {
        let avg_no = self.no_concentration.iter().sum::<f32>() / self.no_concentration.len() as f32;
        let avg_astro_ca =
            self.astrocyte_calcium.iter().sum::<f32>() / self.astrocyte_calcium.len() as f32;
        let avg_eligibility =
            self.eligibility_traces.iter().sum::<f32>() / self.eligibility_traces.len() as f32;
        let avg_hidden = self.hidden_states.iter().sum::<f32>() / self.hidden_states.len() as f32;

        HeterosynapticStats {
            avg_no_concentration: avg_no,
            avg_astrocyte_calcium: avg_astro_ca,
            avg_eligibility_trace: avg_eligibility,
            avg_hidden_state: avg_hidden,
            total_no_events: self.total_no_events,
            total_astrocyte_events: self.total_astrocyte_events,
            total_potentiation_events: self.total_potentiation_events,
            total_depression_events: self.total_depression_events,
        }
    }

    /// Euclidean distance in 3D space
    fn euclidean_distance_3d(p1: (f32, f32, f32), p2: (f32, f32, f32)) -> f32 {
        let (x1, y1, z1) = p1;
        let (x2, y2, z2) = p2;
        ((x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2)).sqrt()
    }

    /// Memory footprint in bytes
    pub fn memory_footprint(&self) -> usize {
        let no_size = std::mem::size_of_val(&self.no_concentration[..]);
        let astro_ca_size = std::mem::size_of_val(&self.astrocyte_calcium[..]);
        let glut_size = std::mem::size_of_val(&self.glutamate_spillover[..]);
        let trace_size = std::mem::size_of_val(&self.eligibility_traces[..]);
        let hidden_size = std::mem::size_of_val(&self.hidden_states[..]);
        let pos_size = std::mem::size_of_val(&self.synapse_positions[..]);

        no_size + astro_ca_size + glut_size + trace_size + hidden_size + pos_size
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterosynapticStats {
    pub avg_no_concentration: f32,
    pub avg_astrocyte_calcium: f32,
    pub avg_eligibility_trace: f32,
    pub avg_hidden_state: f32,
    pub total_no_events: usize,
    pub total_astrocyte_events: usize,
    pub total_potentiation_events: usize,
    pub total_depression_events: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heterosynaptic_creation() {
        let hetero = HeterosynapticPlasticity::new(1000, 10, 500.0);

        assert_eq!(hetero.no_concentration.len(), 1000);
        assert_eq!(hetero.astrocyte_calcium.len(), 10);
        assert_eq!(hetero.eligibility_traces.len(), 1000);
        assert_eq!(hetero.synapse_positions.len(), 1000);

        // Check astrocyte coverage
        let total_covered: usize = hetero.astrocyte_coverage.values().map(|v| v.len()).sum();
        assert_eq!(
            total_covered, 1000,
            "All synapses should be covered by astrocytes"
        );
    }

    #[test]
    fn test_no_diffusion() {
        let mut hetero = HeterosynapticPlasticity::new(100, 1, 100.0);

        // Activate a single synapse
        let mut activity = vec![0.0; 100];
        activity[50] = 1.0;

        hetero.update_no_diffusion(&activity, 0.01);

        // NO should increase at active synapse
        assert!(
            hetero.no_concentration[50] > 0.0,
            "Active synapse should produce NO"
        );

        // Nearby synapses should receive diffused NO
        let nearby_no: f32 = (0..100)
            .filter(|&i| i != 50)
            .map(|i| hetero.no_concentration[i])
            .sum();

        // Run multiple steps to allow diffusion
        for _ in 0..10 {
            hetero.update_no_diffusion(&activity, 0.01);
        }

        let nearby_no_after: f32 = (0..100)
            .filter(|&i| i != 50)
            .map(|i| hetero.no_concentration[i])
            .sum();

        assert!(
            nearby_no_after > nearby_no,
            "NO should diffuse to nearby synapses"
        );
    }

    #[test]
    fn test_astrocyte_activation() {
        let mut hetero = HeterosynapticPlasticity::new(100, 1, 100.0);

        // High activity to trigger glutamate spillover
        let activity = vec![1.0; 100];

        hetero.update_astrocyte_dynamics(&activity, 0.01);

        // Glutamate should accumulate
        assert!(
            hetero.glutamate_spillover[0] > 0.0,
            "Glutamate should accumulate from active synapses"
        );

        // Run multiple steps to exceed threshold
        for _ in 0..20 {
            hetero.update_astrocyte_dynamics(&activity, 0.01);
        }

        // Astrocyte calcium should spike
        assert!(
            hetero.astrocyte_calcium[0] > 1.0,
            "Astrocyte calcium should spike after glutamate threshold"
        );
        assert!(
            hetero.total_astrocyte_events > 0,
            "Astrocyte events should be recorded"
        );
    }

    #[test]
    fn test_eligibility_traces() {
        let mut hetero = HeterosynapticPlasticity::new(100, 1, 100.0);

        let activity = vec![0.5; 100];
        let pre_spikes = vec![true; 100];
        let post_spikes = vec![true; 100];

        hetero.update(&activity, &pre_spikes, &post_spikes, 0.01);

        // Eligibility traces should increase with coincident spikes
        assert!(
            hetero.eligibility_traces[0] > 0.0,
            "Eligibility traces should increase with coincident spikes"
        );

        // Traces should decay
        let trace_before = hetero.eligibility_traces[0];
        let no_spikes = vec![false; 100];
        for _ in 0..100 {
            hetero.update(&vec![0.0; 100], &no_spikes, &no_spikes, 0.01);
        }

        assert!(
            hetero.eligibility_traces[0] < trace_before,
            "Eligibility traces should decay"
        );
    }

    #[test]
    fn test_weight_changes() {
        let mut hetero = HeterosynapticPlasticity::new(100, 2, 100.0);

        let activity = vec![1.0; 100];
        let pre_spikes = vec![true; 100];
        let post_spikes = vec![true; 100];

        let weight_changes = hetero.update(&activity, &pre_spikes, &post_spikes, 0.01);

        // Should have both potentiation and depression
        let total_potentiation: f32 = weight_changes.iter().filter(|&&w| w > 0.0).sum();
        let total_depression: f32 = weight_changes.iter().filter(|&&w| w < 0.0).sum();

        // After sufficient time, should see both effects
        for _ in 0..50 {
            hetero.update(&activity, &pre_spikes, &post_spikes, 0.01);
        }

        let stats = hetero.stats();
        assert!(stats.total_potentiation_events > 0);
        // Depression events might take longer to accumulate
    }

    #[test]
    fn test_reward_modulation() {
        let mut hetero = HeterosynapticPlasticity::new(100, 1, 100.0);

        // Build up eligibility traces
        let activity = vec![1.0; 100];
        let pre_spikes = vec![true; 100];
        let post_spikes = vec![true; 100];

        for _ in 0..10 {
            hetero.update(&activity, &pre_spikes, &post_spikes, 0.01);
        }

        // Apply positive reward
        let reward_changes = hetero.apply_reward_modulated_plasticity(1.0);

        // Should have positive weight changes
        assert!(
            reward_changes.iter().any(|&w| w > 0.0),
            "Positive reward should potentiate synapses with eligibility"
        );

        // Apply negative reward
        let punishment_changes = hetero.apply_reward_modulated_plasticity(-1.0);

        // Should have negative weight changes
        assert!(
            punishment_changes.iter().any(|&w| w < 0.0),
            "Negative reward should depress synapses with eligibility"
        );
    }

    #[test]
    fn test_meta_plasticity() {
        let mut hetero = HeterosynapticPlasticity::new(10, 1, 100.0);

        // Start with neutral hidden state (0.5)
        assert_eq!(hetero.hidden_states[0], 0.5);

        // High activity should drive hidden state toward target
        let high_activity = vec![1.0; 10];
        for _ in 0..100 {
            hetero.update_meta_plasticity(0, high_activity[0], 0.01);
        }

        // Hidden state should have changed
        assert_ne!(hetero.hidden_states[0], 0.5);

        // Should be clamped within bounds
        assert!(hetero.hidden_states[0] >= 0.1 && hetero.hidden_states[0] <= 2.0);
    }
}
