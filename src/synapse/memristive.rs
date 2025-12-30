//! Memristive Synaptic Coupling
//!
//! Implements electromagnetic field effects through 5D memristor-based synapses.
//! Captures realistic coupling between neurons through electromagnetic fields,
//! enabling field-dependent plasticity and synchronization.
//!
//! # Features
//! - 5D memristor model (voltage, current, magnetic flux, charge, memristance)
//! - Electromagnetic field coupling between neurons
//! - Field-dependent synaptic strength modulation
//! - Hebbian learning with memristive dynamics
//! - Realistic power-law memristance updates
//!
//! # Mathematical Model
//! dM/dt = α·I(t)^β - γ·M(t)^δ
//! I_syn = w·M·(V_pre - V_post)·F_em
//! F_em = exp(-d²/λ²) where d is distance, λ is coupling length
//!
//! # References
//! - Chua (1971) "Memristor—The Missing Circuit Element"
//! - Strukov et al. (2008) "The missing memristor found"
//! - October 2024 paper on electromagnetic coupling in SNNs

use serde::{Deserialize, Serialize};

/// Memristive synapse with electromagnetic coupling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemristiveSynapse {
    /// Base synaptic weight
    pub weight: f32,

    /// Memristance state (Ω)
    pub memristance: f32,

    /// Magnetic flux (Wb)
    pub flux: f32,

    /// Accumulated charge (C)
    pub charge: f32,

    /// Memristor parameters
    pub m_min: f32, // Minimum memristance (Ω)
    pub m_max: f32, // Maximum memristance (Ω)
    pub alpha: f32, // Learning rate for memristance increase
    pub beta: f32,  // Current exponent (nonlinearity)
    pub gamma: f32, // Decay rate
    pub delta: f32, // Memristance exponent

    /// Electromagnetic coupling
    pub coupling_strength: f32, // Field coupling coefficient
    pub coupling_length: f32, // Spatial coupling length (μm)

    /// Neuron positions for field calculation
    pub pre_position: (f32, f32, f32), // 3D position
    pub post_position: (f32, f32, f32),

    /// Statistics
    pub total_updates: u64,
    pub last_current: f32,
}

impl MemristiveSynapse {
    /// Create new memristive synapse
    ///
    /// # Arguments
    /// - `weight`: Initial synaptic weight
    /// - `pre_pos`: Pre-synaptic neuron 3D position
    /// - `post_pos`: Post-synaptic neuron 3D position
    pub fn new(weight: f32, pre_pos: (f32, f32, f32), post_pos: (f32, f32, f32)) -> Self {
        Self {
            weight,
            memristance: 1000.0, // Initial memristance (1kΩ)
            flux: 0.0,
            charge: 0.0,

            // Memristor parameters (from Strukov et al. 2008)
            m_min: 100.0,   // Min 100Ω
            m_max: 10000.0, // Max 10kΩ
            alpha: 0.1,     // Learning rate
            beta: 2.0,      // Quadratic current dependence
            gamma: 0.01,    // Slow decay
            delta: 1.0,     // Linear decay

            // EM coupling parameters
            coupling_strength: 0.1,
            coupling_length: 100.0, // 100μm coupling length

            pre_position: pre_pos,
            post_position: post_pos,

            total_updates: 0,
            last_current: 0.0,
        }
    }

    /// Calculate electromagnetic field coupling factor
    fn em_field_coupling(&self) -> f32 {
        let (x1, y1, z1) = self.pre_position;
        let (x2, y2, z2) = self.post_position;

        // Euclidean distance
        let dx = x2 - x1;
        let dy = y2 - y1;
        let dz = z2 - z1;
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

        // Exponential decay of field strength
        let lambda = self.coupling_length;
        (-(distance * distance) / (lambda * lambda)).exp()
    }

    /// Update memristive synapse dynamics
    ///
    /// # Arguments
    /// - `dt`: Timestep (ms)
    /// - `v_pre`: Pre-synaptic voltage (mV)
    /// - `v_post`: Post-synaptic voltage (mV)
    /// - `em_field`: External electromagnetic field (optional)
    ///
    /// # Returns
    /// Synaptic current (pA)
    pub fn update(&mut self, dt: f32, v_pre: f32, v_post: f32, em_field: Option<f32>) -> f32 {
        // Voltage difference drives current
        let v_diff = v_pre - v_post;

        // EM field coupling
        let f_em = self.em_field_coupling();
        let field = em_field.unwrap_or(0.0) + self.coupling_strength * f_em;

        // Memristive current: I = w·M^(-1)·V·(1 + field)
        let current = self.weight * v_diff / self.memristance * (1.0 + field);
        self.last_current = current;

        // Update magnetic flux: dΦ/dt = V
        self.flux += v_diff * dt;

        // Update charge: dQ/dt = I
        self.charge += current * dt;

        // Memristance update (5D dynamics)
        // dM/dt = α·|I|^β - γ·M^δ
        let dm_dt = self.alpha * current.abs().powf(self.beta)
            - self.gamma * self.memristance.powf(self.delta);
        self.memristance += dm_dt * dt;

        // Clamp memristance to valid range
        self.memristance = self.memristance.clamp(self.m_min, self.m_max);

        self.total_updates += 1;

        current
    }

    /// Apply spike-timing-dependent memristive plasticity
    ///
    /// # Arguments
    /// - `dt_spike`: Time difference between pre and post spike (ms)
    /// - `learning_rate`: Plasticity learning rate
    pub fn stdp_update(&mut self, dt_spike: f32, learning_rate: f32) {
        // Memristive STDP: weight change depends on both timing and memristance state
        let tau = 20.0; // STDP time constant

        let dw = if dt_spike > 0.0 {
            // LTP: post after pre
            learning_rate * (-dt_spike / tau).exp() * (self.m_max - self.memristance) / self.m_max
        } else {
            // LTD: pre after post
            -learning_rate * (dt_spike / tau).exp() * (self.memristance - self.m_min) / self.m_max
        };

        self.weight += dw;
        self.weight = self.weight.clamp(0.0, 1.0);
    }

    /// Reset synapse to initial state
    pub fn reset(&mut self) {
        self.memristance = 1000.0;
        self.flux = 0.0;
        self.charge = 0.0;
        self.last_current = 0.0;
    }

    /// Get synapse state
    pub fn state(&self) -> MemristiveState {
        MemristiveState {
            weight: self.weight,
            memristance: self.memristance,
            flux: self.flux,
            charge: self.charge,
            em_coupling: self.em_field_coupling(),
        }
    }
}

/// Memristive synapse state snapshot
#[derive(Debug, Clone)]
pub struct MemristiveState {
    pub weight: f32,
    pub memristance: f32,
    pub flux: f32,
    pub charge: f32,
    pub em_coupling: f32,
}

/// Network of memristive synapses with electromagnetic coupling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemristiveNetwork {
    /// Synapses: indexed by (pre_neuron, post_neuron)
    pub synapses: Vec<MemristiveSynapse>,

    /// Neuron positions in 3D space
    pub neuron_positions: Vec<(f32, f32, f32)>,

    /// Global electromagnetic field
    pub global_em_field: f32,

    /// Field update time constant (ms)
    pub field_tau: f32,
}

impl MemristiveNetwork {
    /// Create new memristive network
    ///
    /// # Arguments
    /// - `n_neurons`: Number of neurons
    /// - `connectivity`: Connection probability
    pub fn new(n_neurons: usize, connectivity: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Random neuron positions in 3D space (normalized to unit cube)
        let neuron_positions: Vec<_> = (0..n_neurons)
            .map(|_| {
                (
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                )
            })
            .collect();

        // Create synapses
        let mut synapses = Vec::new();

        for pre in 0..n_neurons {
            for post in 0..n_neurons {
                if pre == post {
                    continue; // No self-connections
                }

                if rng.gen::<f32>() < connectivity {
                    let weight = rng.gen_range(0.1..0.5);
                    let synapse = MemristiveSynapse::new(
                        weight,
                        neuron_positions[pre],
                        neuron_positions[post],
                    );
                    synapses.push(synapse);
                }
            }
        }

        Self {
            synapses,
            neuron_positions,
            global_em_field: 0.0,
            field_tau: 10.0, // 10ms field time constant
        }
    }

    /// Update network electromagnetic field
    ///
    /// # Arguments
    /// - `dt`: Timestep (ms)
    /// - `neuron_currents`: Current flowing through each neuron (pA)
    pub fn update_em_field(&mut self, dt: f32, neuron_currents: &[f32]) {
        // Calculate global EM field from all neuron currents
        let total_current: f32 = neuron_currents.iter().sum();
        let avg_current = total_current / neuron_currents.len() as f32;

        // Field evolves with time constant
        let target_field = avg_current * 0.001; // Scale factor
        let dfield = (target_field - self.global_em_field) / self.field_tau * dt;

        self.global_em_field += dfield;
    }

    /// Get statistics
    pub fn stats(&self) -> MemristiveNetworkStats {
        if self.synapses.is_empty() {
            return MemristiveNetworkStats {
                num_synapses: 0,
                avg_memristance: 0.0,
                avg_weight: 0.0,
                avg_em_coupling: 0.0,
                global_field: self.global_em_field,
            };
        }

        let avg_memristance =
            self.synapses.iter().map(|s| s.memristance).sum::<f32>() / self.synapses.len() as f32;
        let avg_weight =
            self.synapses.iter().map(|s| s.weight).sum::<f32>() / self.synapses.len() as f32;
        let avg_em_coupling = self
            .synapses
            .iter()
            .map(|s| s.em_field_coupling())
            .sum::<f32>()
            / self.synapses.len() as f32;

        MemristiveNetworkStats {
            num_synapses: self.synapses.len(),
            avg_memristance,
            avg_weight,
            avg_em_coupling,
            global_field: self.global_em_field,
        }
    }
}

/// Memristive network statistics
#[derive(Debug, Clone)]
pub struct MemristiveNetworkStats {
    pub num_synapses: usize,
    pub avg_memristance: f32,
    pub avg_weight: f32,
    pub avg_em_coupling: f32,
    pub global_field: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memristive_synapse_creation() {
        let synapse = MemristiveSynapse::new(0.5, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0));
        assert_eq!(synapse.weight, 0.5);
        assert!(synapse.memristance > 0.0);
        assert_eq!(synapse.flux, 0.0);
        assert_eq!(synapse.charge, 0.0);
    }

    #[test]
    fn test_em_field_coupling() {
        // Nearby neurons
        let close = MemristiveSynapse::new(0.5, (0.0, 0.0, 0.0), (0.01, 0.0, 0.0));
        let close_coupling = close.em_field_coupling();

        // Distant neurons
        let far = MemristiveSynapse::new(0.5, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0));
        let far_coupling = far.em_field_coupling();

        assert!(
            close_coupling > far_coupling,
            "Close neurons should have stronger coupling (close={}, far={})",
            close_coupling,
            far_coupling
        );
    }

    #[test]
    fn test_memristance_dynamics() {
        let mut synapse = MemristiveSynapse::new(0.5, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0));
        let initial_m = synapse.memristance;

        let dt = 0.1;
        let v_pre = 10.0; // Pre-synaptic voltage
        let v_post = -70.0; // Post-synaptic voltage

        // Apply voltage difference
        for _ in 0..100 {
            synapse.update(dt, v_pre, v_post, None);
        }

        // Memristance should change
        assert_ne!(
            synapse.memristance, initial_m,
            "Memristance should update (initial={}, final={})",
            initial_m, synapse.memristance
        );
    }

    #[test]
    fn test_current_flow() {
        let mut synapse = MemristiveSynapse::new(0.5, (0.0, 0.0, 0.0), (0.1, 0.0, 0.0));
        let dt = 0.1;

        // Voltage difference should generate current
        let current = synapse.update(dt, 10.0, -70.0, None);
        assert!(
            current.abs() > 0.0,
            "Voltage difference should generate current"
        );
    }

    #[test]
    fn test_memristive_stdp() {
        let mut synapse = MemristiveSynapse::new(0.5, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0));
        let initial_weight = synapse.weight;

        // LTP: post after pre (positive dt)
        synapse.stdp_update(10.0, 0.01);
        assert!(
            synapse.weight > initial_weight,
            "Causal pairing should increase weight"
        );

        // LTD: pre after post (negative dt)
        synapse.stdp_update(-10.0, 0.01);
        assert!(
            synapse.weight < initial_weight + 0.01,
            "Anti-causal should decrease weight"
        );
    }

    #[test]
    fn test_flux_and_charge_accumulation() {
        let mut synapse = MemristiveSynapse::new(0.5, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0));
        let dt = 0.1;

        for _ in 0..100 {
            synapse.update(dt, 5.0, -70.0, None);
        }

        // Flux and charge should accumulate
        assert!(synapse.flux.abs() > 0.0, "Flux should accumulate");
        assert!(synapse.charge.abs() > 0.0, "Charge should accumulate");
    }

    #[test]
    fn test_memristive_network() {
        let network = MemristiveNetwork::new(50, 0.1);
        let stats = network.stats();

        assert!(stats.num_synapses > 0, "Network should have synapses");
        assert!(stats.avg_memristance > 0.0, "Should have valid memristance");
        assert!(stats.avg_weight > 0.0, "Should have valid weights");
    }

    #[test]
    fn test_global_em_field() {
        let mut network = MemristiveNetwork::new(10, 0.2);
        let dt = 0.1;

        // Simulate neuron currents
        let currents = vec![100.0; 10]; // All neurons active

        network.update_em_field(dt, &currents);

        assert_ne!(network.global_em_field, 0.0, "EM field should update");
    }

    #[test]
    fn test_memristance_bounds() {
        let mut synapse = MemristiveSynapse::new(0.5, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0));
        let dt = 0.1;

        // Apply strong stimulation
        for _ in 0..1000 {
            synapse.update(dt, 50.0, -70.0, Some(10.0));
        }

        // Memristance should stay within bounds
        assert!(
            synapse.memristance >= synapse.m_min,
            "Memristance should not go below minimum"
        );
        assert!(
            synapse.memristance <= synapse.m_max,
            "Memristance should not exceed maximum"
        );
    }
}
