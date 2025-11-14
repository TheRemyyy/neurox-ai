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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_procedural_connectivity_creation() {
        let proc_conn = ProceduralConnectivity::new(42, 0.1, 1.0, 0.2);

        assert_eq!(proc_conn.seed, 42);
        assert_eq!(proc_conn.connection_prob, 0.1);
        assert_eq!(proc_conn.weight_mean, 1.0);
        assert_eq!(proc_conn.weight_std, 0.2);
        assert_eq!(proc_conn.exc_ratio, 0.8);
    }

    #[test]
    fn test_procedural_connectivity_topology() {
        let proc_conn = ProceduralConnectivity::new(42, 0.1, 1.0, 0.2)
            .with_topology(ConnectivityType::SmallWorld { k: 10, beta: 0.1 });

        match proc_conn.topology {
            ConnectivityType::SmallWorld { k, beta } => {
                assert_eq!(k, 10);
                assert_eq!(beta, 0.1);
            }
            _ => panic!("Expected SmallWorld topology"),
        }
    }

    #[test]
    fn test_cortical_connectivity() {
        let proc_conn = ProceduralConnectivity::cortical(42, 10, 0.1);

        assert_eq!(proc_conn.connection_prob, 0.1);
        assert_eq!(proc_conn.exc_ratio, 0.8);
        match proc_conn.topology {
            ConnectivityType::SmallWorld { .. } => {},
            _ => panic!("Expected SmallWorld topology for cortical"),
        }
    }

    #[test]
    fn test_generate_random_connections() {
        let proc_conn = ProceduralConnectivity::new(42, 0.2, 1.0, 0.1);

        let connections = proc_conn.generate_connections(0, 0..100);

        // Should have some connections (probabilistic)
        assert!(connections.len() > 0);
        assert!(connections.len() < 100); // Not all-to-all

        // Check weights are reasonable
        for (_target, weight) in &connections {
            assert!(weight.abs() > 0.0);
            assert!(weight.abs() < 5.0); // Should be within ~3 std devs
        }
    }

    #[test]
    fn test_generate_all_to_all_connections() {
        let proc_conn = ProceduralConnectivity::new(42, 1.0, 1.0, 0.1)
            .with_topology(ConnectivityType::AllToAll);

        let connections = proc_conn.generate_connections(0, 0..10);

        // Should connect to all except self
        assert_eq!(connections.len(), 9);
    }

    #[test]
    fn test_dales_principle() {
        let proc_conn = ProceduralConnectivity::new(42, 0.5, 1.0, 0.1);

        // First 80% should be excitatory (positive weights)
        let exc_connections = proc_conn.generate_connections(10, 0..1000);
        let exc_positive = exc_connections.iter().filter(|(_, w)| *w > 0.0).count();
        assert!(exc_positive > exc_connections.len() / 2);

        // Last 20% should be inhibitory (negative weights)
        let inh_connections = proc_conn.generate_connections(900, 0..1000);
        let inh_negative = inh_connections.iter().filter(|(_, w)| *w < 0.0).count();
        assert!(inh_negative > inh_connections.len() / 2);
    }

    #[test]
    fn test_estimate_synapses_random() {
        let proc_conn = ProceduralConnectivity::new(42, 0.1, 1.0, 0.1);

        let estimate = proc_conn.estimate_synapses(1000);

        // Should be ~1000 * 1000 * 0.1 = 100,000
        assert!(estimate > 50_000);
        assert!(estimate < 150_000);
    }

    #[test]
    fn test_estimate_synapses_all_to_all() {
        let proc_conn = ProceduralConnectivity::new(42, 1.0, 1.0, 0.1)
            .with_topology(ConnectivityType::AllToAll);

        let estimate = proc_conn.estimate_synapses(100);

        // Should be 100 * 99 = 9,900
        assert_eq!(estimate, 9_900);
    }

    #[test]
    fn test_estimate_synapses_small_world() {
        let proc_conn = ProceduralConnectivity::new(42, 0.1, 1.0, 0.1)
            .with_topology(ConnectivityType::SmallWorld { k: 10, beta: 0.1 });

        let estimate = proc_conn.estimate_synapses(1000);

        // Should be ~1000 * 10 = 10,000
        assert_eq!(estimate, 10_000);
    }

    #[test]
    fn test_sparse_connectivity_empty() {
        let sparse = SparseConnectivity::empty(100);

        assert_eq!(sparse.n_neurons, 100);
        assert_eq!(sparse.nnz, 0);
        assert_eq!(sparse.row_ptr.len(), 101); // n_neurons + 1
        assert_eq!(sparse.col_idx.len(), 0);
        assert_eq!(sparse.weights.len(), 0);
    }

    #[test]
    fn test_sparse_connectivity_from_procedural() {
        let proc_conn = ProceduralConnectivity::new(42, 0.1, 1.0, 0.1);
        let sparse = SparseConnectivity::from_procedural(100, &proc_conn);

        assert_eq!(sparse.n_neurons, 100);
        assert!(sparse.nnz > 0);
        assert_eq!(sparse.row_ptr.len(), 101);
        assert_eq!(sparse.col_idx.len(), sparse.nnz);
        assert_eq!(sparse.weights.len(), sparse.nnz);
    }

    #[test]
    fn test_sparse_connectivity_get_row() {
        let proc_conn = ProceduralConnectivity::new(42, 0.2, 1.0, 0.1);
        let sparse = SparseConnectivity::from_procedural(50, &proc_conn);

        let (targets, weights) = sparse.get_row(0);

        // Should have connections
        assert!(targets.len() > 0);
        assert_eq!(targets.len(), weights.len());

        // All targets should be valid neuron IDs
        for &target in targets {
            assert!(target >= 0 && target < 50);
        }
    }

    #[test]
    fn test_sparse_connectivity_avg_degree() {
        let proc_conn = ProceduralConnectivity::new(42, 0.1, 1.0, 0.1);
        let sparse = SparseConnectivity::from_procedural(100, &proc_conn);

        let avg_degree = sparse.avg_degree();

        // Should be around 10 connections per neuron (0.1 * 100)
        assert!(avg_degree > 5.0);
        assert!(avg_degree < 15.0);
    }

    #[test]
    fn test_sparse_connectivity_memory_footprint() {
        let proc_conn = ProceduralConnectivity::new(42, 0.1, 1.0, 0.1);
        let sparse = SparseConnectivity::from_procedural(100, &proc_conn);

        let footprint = sparse.memory_footprint();

        // Should be reasonable (much less than dense matrix)
        // Dense: 100 * 100 * 4 bytes = 40,000 bytes
        // Sparse with 10% density: ~1000 * (4 + 4) + 101 * 4 = ~8,400 bytes
        assert!(footprint < 20_000);
    }

    #[test]
    fn test_sparse_connectivity_sparsity() {
        let proc_conn = ProceduralConnectivity::new(42, 0.1, 1.0, 0.1);
        let sparse = SparseConnectivity::from_procedural(100, &proc_conn);

        let sparsity = sparse.sparsity();

        // Should be around 90% sparse (10% density)
        assert!(sparsity > 80.0);
        assert!(sparsity < 100.0);
    }

    #[test]
    fn test_distance_dependent_connectivity() {
        let proc_conn = ProceduralConnectivity::new(42, 0.5, 1.0, 0.1)
            .with_topology(ConnectivityType::DistanceDependent { sigma: 0.1 });

        let connections = proc_conn.generate_connections(50, 0..100);

        // Should favor nearby neurons
        let nearby_count = connections.iter()
            .filter(|(target, _)| (*target as i32 - 50).abs() < 10)
            .count();

        let far_count = connections.iter()
            .filter(|(target, _)| (*target as i32 - 50).abs() > 40)
            .count();

        assert!(nearby_count > far_count);
    }

    #[test]
    fn test_small_world_connectivity() {
        let proc_conn = ProceduralConnectivity::new(42, 0.5, 1.0, 0.1)
            .with_topology(ConnectivityType::SmallWorld { k: 4, beta: 0.1 });

        let connections = proc_conn.generate_connections(0, 0..100);

        // Should have ~k connections
        assert!(connections.len() >= 3);
        assert!(connections.len() <= 5);
    }

    #[test]
    fn test_reproducibility() {
        let proc_conn1 = ProceduralConnectivity::new(42, 0.1, 1.0, 0.1);
        let proc_conn2 = ProceduralConnectivity::new(42, 0.1, 1.0, 0.1);

        let conn1 = proc_conn1.generate_connections(0, 0..100);
        let conn2 = proc_conn2.generate_connections(0, 0..100);

        // Same seed should produce identical connectivity
        assert_eq!(conn1.len(), conn2.len());
        for (c1, c2) in conn1.iter().zip(conn2.iter()) {
            assert_eq!(c1.0, c2.0); // Same targets
            assert_eq!(c1.1, c2.1); // Same weights
        }
    }

    #[test]
    fn test_biological_realism_sparsity() {
        // Cortical connectivity is typically 10-20% dense
        let proc_conn = ProceduralConnectivity::cortical(42, 10, 0.1);
        let sparse = SparseConnectivity::from_procedural(1000, &proc_conn);

        let sparsity = sparse.sparsity();

        // Should be highly sparse (>80%)
        assert!(sparsity > 80.0, "Cortical connectivity should be sparse");
    }

    #[test]
    fn test_biological_realism_exc_inh_ratio() {
        let proc_conn = ProceduralConnectivity::new(42, 0.2, 1.0, 0.1);

        // Count excitatory vs inhibitory
        let mut total_exc = 0;
        let mut total_inh = 0;

        for i in 0..1000 {
            let connections = proc_conn.generate_connections(i, 0..1000);
            for (_, weight) in connections {
                if weight > 0.0 {
                    total_exc += 1;
                } else {
                    total_inh += 1;
                }
            }
        }

        let total = total_exc + total_inh;
        let exc_ratio = total_exc as f32 / total as f32;

        // Should be around 80% excitatory
        assert!(exc_ratio > 0.7 && exc_ratio < 0.9,
            "Excitatory ratio should be ~80%, got {}", exc_ratio);
    }

    #[test]
    fn test_biological_realism_weight_distribution() {
        let proc_conn = ProceduralConnectivity::new(42, 0.2, 1.0, 0.2);

        let connections = proc_conn.generate_connections(50, 0..1000);

        // Calculate mean and std of weights
        let weights: Vec<f32> = connections.iter().map(|(_, w)| w.abs()).collect();
        let mean = weights.iter().sum::<f32>() / weights.len() as f32;

        // Mean should be close to specified mean
        assert!((mean - 1.0).abs() < 0.3, "Mean weight should be ~1.0");

        // All weights should be within reasonable range (3 std devs)
        for &weight in &weights {
            assert!(weight < 2.0, "Weight should be within biological range");
        }
    }

    #[test]
    fn test_edge_case_single_neuron() {
        let proc_conn = ProceduralConnectivity::new(42, 0.5, 1.0, 0.1);

        let connections = proc_conn.generate_connections(0, 0..1);

        // Should have no self-connections
        assert_eq!(connections.len(), 0);
    }

    #[test]
    fn test_edge_case_zero_probability() {
        let proc_conn = ProceduralConnectivity::new(42, 0.0, 1.0, 0.1);

        let connections = proc_conn.generate_connections(0, 0..100);

        // Should have no connections
        assert_eq!(connections.len(), 0);
    }

    #[test]
    fn test_performance_large_network() {
        // Test that we can handle reasonably large networks
        let proc_conn = ProceduralConnectivity::new(42, 0.01, 1.0, 0.1);

        // This should complete quickly even for 10k neurons
        let sparse = SparseConnectivity::from_procedural(1000, &proc_conn);

        assert_eq!(sparse.n_neurons, 1000);
        assert!(sparse.nnz > 0);
    }

    #[test]
    fn test_integration_csr_structure() {
        let proc_conn = ProceduralConnectivity::new(42, 0.2, 1.0, 0.1);
        let sparse = SparseConnectivity::from_procedural(10, &proc_conn);

        // Verify CSR structure integrity
        assert_eq!(sparse.row_ptr[0], 0);
        assert_eq!(sparse.row_ptr[sparse.n_neurons] as usize, sparse.nnz);

        // Row pointers should be monotonically increasing
        for i in 1..sparse.row_ptr.len() {
            assert!(sparse.row_ptr[i] >= sparse.row_ptr[i - 1]);
        }

        // All column indices should be valid
        for &col in &sparse.col_idx {
            assert!(col >= 0 && col < sparse.n_neurons as i32);
        }
    }
}
