//! Synaptic connectivity and sparse matrix management
//!
//! Implements procedural connectivity generation, CSR sparse storage,
//! and structural plasticity optimized for neuromorphic computing
//! with biological realism.

pub mod structural;

pub use structural::{StructuralPlasticity, Synapse, StructuralPlasticityStats};

use serde::{Deserialize, Serialize};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Connectivity topology type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConnectivityType {
    /// Random connections with fixed probability
    Random,
    /// Distance-dependent (for spatial networks)
    DistanceDependent { sigma: f32 },
    /// Small-world topology (biological cortex)
    SmallWorld { k: usize, beta: f64 },
    /// All-to-all (fully connected)
    AllToAll,
}

/// Procedural connectivity generator
///
/// Generates synaptic connections on-the-fly from seeds and rules,
/// reducing memory from petabytes to kilobytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralConnectivity {
    /// Random seed for reproducibility
    pub seed: u64,

    /// Connection probability (for Random topology)
    pub connection_prob: f64,

    /// Weight distribution mean (biological: 0.5-2.0 for exc, -1.0 to -0.5 for inh)
    pub weight_mean: f32,

    /// Weight distribution std dev
    pub weight_std: f32,

    /// Topology type
    pub topology: ConnectivityType,

    /// Excitatory/inhibitory ratio (biological: ~80/20)
    pub exc_ratio: f32,
}

impl ProceduralConnectivity {
    /// Create new procedural connectivity with biological defaults
    pub fn new(seed: u64, connection_prob: f64, weight_mean: f32, weight_std: f32) -> Self {
        Self {
            seed,
            connection_prob,
            weight_mean,
            weight_std,
            topology: ConnectivityType::Random,
            exc_ratio: 0.8, // 80% excitatory, 20% inhibitory (Dale's principle)
        }
    }

    /// Create with specific topology
    pub fn with_topology(mut self, topology: ConnectivityType) -> Self {
        self.topology = topology;
        self
    }

    /// Create cortical-like small-world connectivity
    pub fn cortical(seed: u64, k: usize, rewiring_prob: f64) -> Self {
        Self {
            seed,
            connection_prob: 0.1, // Typical cortical sparsity
            weight_mean: 1.0,
            weight_std: 0.2,
            topology: ConnectivityType::SmallWorld {
                k,
                beta: rewiring_prob,
            },
            exc_ratio: 0.8,
        }
    }

    /// Generate connections for a source neuron (optimized)
    pub fn generate_connections(
        &self,
        source_id: usize,
        target_range: std::ops::Range<usize>,
    ) -> Vec<(usize, f32)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed ^ (source_id as u64));
        let weight_dist = Normal::new(self.weight_mean, self.weight_std).unwrap();

        let mut connections = Vec::new();
        let n_targets = target_range.len();

        // Determine if source is excitatory or inhibitory
        let is_excitatory = (source_id as f32 / n_targets as f32) < self.exc_ratio;

        match self.topology {
            ConnectivityType::Random => {
                for target_id in target_range {
                    if target_id != source_id && rng.gen::<f64>() < self.connection_prob {
                        let mut weight = weight_dist.sample(&mut rng);
                        // Dale's principle: inhibitory neurons have negative weights
                        if !is_excitatory {
                            weight = -weight.abs();
                        }
                        connections.push((target_id, weight));
                    }
                }
            }
            ConnectivityType::AllToAll => {
                for target_id in target_range {
                    if target_id != source_id {
                        let mut weight = weight_dist.sample(&mut rng);
                        if !is_excitatory {
                            weight = -weight.abs();
                        }
                        connections.push((target_id, weight));
                    }
                }
            }
            ConnectivityType::SmallWorld { k, beta } => {
                // Ring lattice + rewiring (Watts-Strogatz model)
                let n = n_targets;
                for j in 1..=k {
                    let target = (source_id + j) % n;
                    if rng.gen::<f64>() < beta {
                        // Rewire to random target
                        let random_target = rng.gen_range(0..n);
                        if random_target != source_id {
                            let mut weight = weight_dist.sample(&mut rng);
                            if !is_excitatory {
                                weight = -weight.abs();
                            }
                            connections.push((random_target, weight));
                        }
                    } else {
                        // Keep regular connection
                        let mut weight = weight_dist.sample(&mut rng);
                        if !is_excitatory {
                            weight = -weight.abs();
                        }
                        connections.push((target, weight));
                    }
                }
            }
            ConnectivityType::DistanceDependent { sigma } => {
                // Gaussian probability based on distance
                for target_id in target_range.clone() {
                    if target_id != source_id {
                        let distance = (target_id as f32 - source_id as f32).abs() / n_targets as f32;
                        let prob = (-distance * distance / (2.0 * sigma * sigma)).exp();
                        if rng.gen::<f64>() < prob as f64 {
                            let mut weight = weight_dist.sample(&mut rng);
                            if !is_excitatory {
                                weight = -weight.abs();
                            }
                            connections.push((target_id, weight));
                        }
                    }
                }
            }
        }

        connections
    }

    /// Estimate number of synapses for memory planning
    pub fn estimate_synapses(&self, n_neurons: usize) -> usize {
        match self.topology {
            ConnectivityType::Random => {
                (n_neurons as f64 * n_neurons as f64 * self.connection_prob) as usize
            }
            ConnectivityType::AllToAll => n_neurons * (n_neurons - 1),
            ConnectivityType::SmallWorld { k, .. } => n_neurons * k,
            ConnectivityType::DistanceDependent { sigma } => {
                // Approximate using Gaussian integral
                (n_neurons as f64 * n_neurons as f64 * 0.5 * sigma as f64) as usize
            }
        }
    }
}

/// CSR (Compressed Sparse Row) matrix for synaptic connectivity
///
/// Memory-efficient representation: ~12 bytes per synapse vs 8 bytes
/// for dense (but only for non-zero entries).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseConnectivity {
    /// Row pointers (n_neurons + 1)
    pub row_ptr: Vec<i32>,

    /// Column indices (nnz)
    pub col_idx: Vec<i32>,

    /// Weights (nnz)
    pub weights: Vec<f32>,

    /// Number of neurons
    pub n_neurons: usize,

    /// Number of non-zero entries
    pub nnz: usize,
}

impl SparseConnectivity {
    /// Create empty sparse matrix
    pub fn empty(n_neurons: usize) -> Self {
        Self {
            row_ptr: vec![0; n_neurons + 1],
            col_idx: Vec::new(),
            weights: Vec::new(),
            n_neurons,
            nnz: 0,
        }
    }

    /// Build sparse matrix from procedural connectivity
    pub fn from_procedural(
        n_neurons: usize,
        proc_conn: &ProceduralConnectivity,
    ) -> Self {
        log::info!("Building sparse connectivity matrix for {} neurons", n_neurons);
        log::info!("Topology: {:?}", proc_conn.topology);

        let estimated_nnz = proc_conn.estimate_synapses(n_neurons);
        log::info!("Estimated synapses: {}", estimated_nnz);

        let mut row_ptr = vec![0i32; n_neurons + 1];
        let mut col_idx = Vec::with_capacity(estimated_nnz);
        let mut weights = Vec::with_capacity(estimated_nnz);

        // Generate connections for each neuron
        let mut current_ptr = 0;
        for source_id in 0..n_neurons {
            let connections = proc_conn.generate_connections(source_id, 0..n_neurons);

            for (target_id, weight) in connections {
                col_idx.push(target_id as i32);
                weights.push(weight);
                current_ptr += 1;
            }

            row_ptr[source_id + 1] = current_ptr;

            if source_id % 10000 == 0 && source_id > 0 {
                log::info!("  Generated {} / {} neurons ({:.1}%)",
                    source_id, n_neurons,
                    100.0 * source_id as f64 / n_neurons as f64
                );
            }
        }

        let nnz = col_idx.len();
        log::info!("Sparse matrix built: {} synapses ({:.2}% sparsity)",
            nnz,
            100.0 * (1.0 - nnz as f64 / (n_neurons * n_neurons) as f64)
        );

        Self {
            row_ptr,
            col_idx,
            weights,
            n_neurons,
            nnz,
        }
    }

    /// Get connections for a specific neuron (row)
    pub fn get_row(&self, neuron_id: usize) -> (&[i32], &[f32]) {
        let start = self.row_ptr[neuron_id] as usize;
        let end = self.row_ptr[neuron_id + 1] as usize;
        (&self.col_idx[start..end], &self.weights[start..end])
    }

    /// Average synapses per neuron
    pub fn avg_degree(&self) -> f32 {
        self.nnz as f32 / self.n_neurons as f32
    }

    /// Memory footprint in bytes
    pub fn memory_footprint(&self) -> usize {
        std::mem::size_of_val(&self.row_ptr[..])
            + std::mem::size_of_val(&self.col_idx[..])
            + std::mem::size_of_val(&self.weights[..])
    }

    /// Sparsity percentage (0-100)
    pub fn sparsity(&self) -> f32 {
        100.0 * (1.0 - self.nnz as f64 / (self.n_neurons * self.n_neurons) as f64) as f32
    }
}
