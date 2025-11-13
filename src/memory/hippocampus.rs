//! Hippocampal Memory System
//!
//! Fast one-shot learning with pattern separation and completion.
//! Implements dentate gyrus (DG), CA3, and CA1 circuitry.
//!
//! # Biological Basis
//! - DG: Pattern separation via sparse random projection (2-5% active)
//! - CA3: Recurrent network for pattern completion
//! - CA1: Output comparator
//! - Replay: Memory consolidation during "sleep"
//! - Prioritized replay based on prediction error

use crate::connectivity::SparseConnectivity;
use crate::neuron::{LIFNeuron, NeuronState};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Memory trace for episodic storage
///
/// Represents single experience with input, output, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrace {
    /// Input pattern
    pub input: Vec<f32>,

    /// Output/target pattern
    pub output: Vec<f32>,

    /// Timestamp (biological time in ms)
    pub time: f32,

    /// Priority for replay (higher = more important)
    pub priority: f32,

    /// Number of times replayed
    replay_count: usize,

    /// Prediction error (for prioritization)
    error: f32,
}

impl MemoryTrace {
    /// Create new memory trace
    pub fn new(input: Vec<f32>, output: Vec<f32>, time: f32, error: f32) -> Self {
        // Priority based on prediction error (surprise)
        let priority = error * error; // Square emphasizes large errors

        Self {
            input,
            output,
            time,
            priority,
            replay_count: 0,
            error,
        }
    }

    /// Update priority (e.g., after replay)
    pub fn update_priority(&mut self, new_error: f32) {
        self.error = new_error;
        self.priority = new_error * new_error;
        self.replay_count += 1;

        // Decay priority with replay count (avoid over-rehearsal)
        self.priority /= (1.0 + self.replay_count as f32 * 0.1);
    }
}

// For priority queue (max heap)
impl Eq for MemoryTrace {}

impl PartialEq for MemoryTrace {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl PartialOrd for MemoryTrace {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.priority.partial_cmp(&other.priority)
    }
}

impl Ord for MemoryTrace {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Sparse encoder for pattern separation
///
/// Implements dentate gyrus (DG) functionality:
/// - Random sparse projection
/// - Winner-take-all (top-k activation)
/// - Decorrelates similar inputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseEncoder {
    /// Random projection matrix (sparse)
    pub projection: SparseConnectivity,

    /// Sparsity level (fraction of active neurons, typically 0.02-0.05)
    sparsity: f32,

    /// Input dimension
    input_dim: usize,

    /// Output dimension (DG size, typically 10× larger than input)
    output_dim: usize,
}

impl SparseEncoder {
    /// Create new sparse encoder
    ///
    /// # Arguments
    /// - `input_dim`: Input dimensionality
    /// - `output_dim`: Output dimensionality (DG size, typically 10× input)
    /// - `sparsity`: Fraction of active neurons (0.02-0.05)
    /// - `connection_prob`: Probability of connection (0.1-0.2)
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        sparsity: f32,
        connection_prob: f32,
    ) -> Self {
        // Create random sparse projection
        let projection = Self::create_random_projection(input_dim, output_dim, connection_prob);

        Self {
            projection,
            sparsity,
            input_dim,
            output_dim,
        }
    }

    /// Encode input via sparse projection + winner-take-all
    ///
    /// Returns sparse code (most active k neurons)
    pub fn encode(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim);

        // Project input
        let mut projected = vec![0.0; self.output_dim];

        for target in 0..self.output_dim {
            let start = self.projection.row_ptr[target] as usize;
            let end = self.projection.row_ptr[target + 1] as usize;

            for syn_id in start..end {
                let source = self.projection.col_idx[syn_id] as usize;
                let weight = self.projection.weights[syn_id];
                projected[target] += input[source] * weight;
            }
        }

        // Winner-take-all: keep only top k
        let k = (self.output_dim as f32 * self.sparsity) as usize;
        Self::top_k(&mut projected, k);

        projected
    }

    /// Winner-take-all: zero out all but top k values
    fn top_k(values: &mut [f32], k: usize) {
        let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Zero out all except top k
        for (i, _) in indexed.iter().skip(k) {
            values[*i] = 0.0;
        }

        // Normalize top k
        let sum: f32 = values.iter().sum();
        if sum > 0.0 {
            for v in values.iter_mut() {
                if *v > 0.0 {
                    *v /= sum;
                }
            }
        }
    }

    /// Create random sparse projection matrix
    fn create_random_projection(
        input_dim: usize,
        output_dim: usize,
        connection_prob: f32,
    ) -> SparseConnectivity {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut row_ptr = vec![0; output_dim + 1];
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();

        for target in 0..output_dim {
            for source in 0..input_dim {
                if rng.gen::<f32>() < connection_prob {
                    col_idx.push(source as i32);
                    // Random weight from Gaussian
                    weights.push(rng.gen_range(-1.0..1.0));
                }
            }
            row_ptr[target + 1] = col_idx.len() as i32;
        }

        SparseConnectivity {
            row_ptr,
            col_idx,
            weights,
            nnz: col_idx.len(),
            n_neurons: output_dim,
        }
    }

    /// Get sparsity level
    pub fn get_sparsity(&self) -> f32 {
        self.sparsity
    }
}

/// Recurrent network for pattern completion (CA3)
///
/// Auto-associative memory that can recall full pattern from partial cue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentNetwork {
    /// Neurons
    neurons: Vec<LIFNeuron>,

    /// Recurrent connectivity (auto-associative)
    connectivity: SparseConnectivity,

    /// Hebbian weight matrix (for one-shot learning)
    hebbian_weights: Vec<f32>,

    /// Number of neurons
    n_neurons: usize,

    /// Learning rate for Hebbian learning
    learning_rate: f32,
}

impl RecurrentNetwork {
    /// Create new recurrent network
    pub fn new(n_neurons: usize, connection_prob: f32, learning_rate: f32) -> Self {
        // Create recurrent connectivity
        let connectivity = Self::create_recurrent_connectivity(n_neurons, connection_prob);

        let neurons = vec![LIFNeuron::default(); n_neurons];
        let hebbian_weights = vec![0.0; connectivity.nnz];

        Self {
            neurons,
            connectivity,
            hebbian_weights,
            n_neurons,
            learning_rate,
        }
    }

    /// Store pattern via Hebbian learning (one-shot)
    ///
    /// Implements Hebbian rule: Δw_ij = η * x_i * x_j
    pub fn store_pattern(&mut self, pattern: &[f32]) {
        assert_eq!(pattern.len(), self.n_neurons);

        // Hebbian learning: strengthen connections between co-active neurons
        for target in 0..self.n_neurons {
            let start = self.connectivity.row_ptr[target] as usize;
            let end = self.connectivity.row_ptr[target + 1] as usize;

            for syn_id in start..end {
                let source = self.connectivity.col_idx[syn_id] as usize;

                // Hebbian update
                let delta_w = self.learning_rate * pattern[source] * pattern[target];
                self.hebbian_weights[syn_id] += delta_w;

                // Clip to prevent runaway
                self.hebbian_weights[syn_id] = self.hebbian_weights[syn_id].clamp(-1.0, 1.0);
            }
        }
    }

    /// Recall pattern (pattern completion)
    ///
    /// Iteratively activates network until convergence.
    pub fn recall(&self, partial: &[f32], iterations: usize) -> Vec<f32> {
        assert_eq!(partial.len(), self.n_neurons);

        let mut state = partial.to_vec();

        // Iterative recall
        for _ in 0..iterations {
            let mut new_state = vec![0.0; self.n_neurons];

            // Recurrent activation
            for target in 0..self.n_neurons {
                let start = self.connectivity.row_ptr[target] as usize;
                let end = self.connectivity.row_ptr[target + 1] as usize;

                let mut activation = 0.0;
                for syn_id in start..end {
                    let source = self.connectivity.col_idx[syn_id] as usize;
                    let weight =
                        self.connectivity.weights[syn_id] + self.hebbian_weights[syn_id];
                    activation += state[source] * weight;
                }

                new_state[target] = activation.tanh(); // Nonlinearity
            }

            state = new_state;
        }

        state
    }

    /// Create recurrent connectivity matrix
    fn create_recurrent_connectivity(n_neurons: usize, connection_prob: f32) -> SparseConnectivity {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut row_ptr = vec![0; n_neurons + 1];
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();

        for target in 0..n_neurons {
            for source in 0..n_neurons {
                if source != target && rng.gen::<f32>() < connection_prob {
                    col_idx.push(source as i32);
                    weights.push(rng.gen_range(-0.1..0.1)); // Small initial weights
                }
            }
            row_ptr[target + 1] = col_idx.len() as i32;
        }

        SparseConnectivity {
            row_ptr,
            col_idx,
            weights,
            nnz: col_idx.len(),
            n_neurons,
        }
    }
}

/// Feedforward layer (CA1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedforwardLayer {
    neurons: Vec<LIFNeuron>,
    n_neurons: usize,
}

impl FeedforwardLayer {
    pub fn new(n_neurons: usize) -> Self {
        Self {
            neurons: vec![LIFNeuron::default(); n_neurons],
            n_neurons,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Simple linear transformation for now
        input.to_vec()
    }
}

/// Hippocampal Memory System
///
/// Integrates DG, CA3, CA1 for fast episodic learning.
///
/// # Architecture
/// Input → DG (pattern separation) → CA3 (pattern completion) → CA1 (output)
///
/// # Features
/// - One-shot binding
/// - Pattern separation (decorrelation)
/// - Pattern completion (associative recall)
/// - Prioritized experience replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hippocampus {
    /// CA3 recurrent network
    ca3: RecurrentNetwork,

    /// CA1 output layer
    ca1: FeedforwardLayer,

    /// Dentate gyrus sparse encoder
    dg: SparseEncoder,

    /// Experience replay buffer (priority queue)
    #[serde(skip)]
    replay_buffer: BinaryHeap<MemoryTrace>,

    /// Maximum replay buffer size
    max_replay_buffer: usize,

    /// Input dimension
    input_dim: usize,

    /// DG dimension (expanded representation)
    dg_dim: usize,
}

impl Hippocampus {
    /// Create new hippocampus
    ///
    /// # Arguments
    /// - `input_dim`: Input dimensionality
    /// - `dg_expansion`: DG expansion factor (typically 10)
    /// - `sparsity`: DG sparsity (0.02-0.05)
    /// - `max_replay_buffer`: Maximum replay buffer size
    pub fn new(
        input_dim: usize,
        dg_expansion: usize,
        sparsity: f32,
        max_replay_buffer: usize,
    ) -> Self {
        let dg_dim = input_dim * dg_expansion;

        let dg = SparseEncoder::new(input_dim, dg_dim, sparsity, 0.15);
        let ca3 = RecurrentNetwork::new(dg_dim, 0.1, 0.01);
        let ca1 = FeedforwardLayer::new(input_dim);

        Self {
            ca3,
            ca1,
            dg,
            replay_buffer: BinaryHeap::new(),
            max_replay_buffer,
            input_dim,
            dg_dim,
        }
    }

    /// Encode pattern (one-shot learning)
    ///
    /// Stores pattern in CA3 and adds to replay buffer.
    ///
    /// # Returns
    /// Memory ID (for tracking)
    pub fn encode(&mut self, pattern: &[f32]) -> usize {
        assert_eq!(pattern.len(), self.input_dim);

        // 1. Pattern separation (DG)
        let sparse_code = self.dg.encode(pattern);

        // 2. Store in CA3 (one-shot Hebbian)
        self.ca3.store_pattern(&sparse_code);

        // 3. Add to replay buffer
        let trace = MemoryTrace::new(pattern.to_vec(), sparse_code.clone(), 0.0, 1.0);

        self.replay_buffer.push(trace);

        // Limit buffer size
        if self.replay_buffer.len() > self.max_replay_buffer {
            // Remove lowest priority
            let mut temp: Vec<_> = self.replay_buffer.drain().collect();
            temp.sort_by(|a, b| b.partial_cmp(a).unwrap());
            temp.truncate(self.max_replay_buffer);
            self.replay_buffer = temp.into_iter().collect();
        }

        self.replay_buffer.len()
    }

    /// Recall pattern from partial cue
    ///
    /// # Pattern Completion
    /// Can retrieve full pattern from incomplete/noisy input.
    pub fn recall(&self, partial: &[f32]) -> Vec<f32> {
        assert_eq!(partial.len(), self.input_dim);

        // 1. Encode partial cue
        let sparse_partial = self.dg.encode(partial);

        // 2. Pattern completion in CA3
        let completed = self.ca3.recall(&sparse_partial, 10); // 10 iterations

        // 3. Decode via CA1
        let output = self.ca1.forward(&completed);

        // Resize to match input (decode from DG space)
        let decoded = self.decode_from_dg(&completed);

        decoded
    }

    /// Consolidate memories via replay
    ///
    /// Replays high-priority memories for transfer to neocortex.
    ///
    /// # Arguments
    /// - `n_replays`: Number of memories to replay
    ///
    /// # Returns
    /// Replayed memories (for neocortical consolidation)
    pub fn consolidate(&mut self, n_replays: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut replayed = Vec::new();

        let available = n_replays.min(self.replay_buffer.len());

        // Replay top-priority memories
        let mut temp_buffer: Vec<_> = self.replay_buffer.drain().collect();
        temp_buffer.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Highest priority first

        for trace in temp_buffer.iter_mut().take(available) {
            replayed.push((trace.input.clone(), trace.output.clone()));

            // Update priority (decays with replay)
            trace.update_priority(trace.error * 0.9);
        }

        // Restore buffer
        self.replay_buffer = temp_buffer.into_iter().collect();

        replayed
    }

    /// Decode from DG space back to input space
    ///
    /// Simple projection (in reality would be learned)
    fn decode_from_dg(&self, dg_pattern: &[f32]) -> Vec<f32> {
        // Average pooling to decode
        let pool_size = dg_pattern.len() / self.input_dim;
        let mut decoded = vec![0.0; self.input_dim];

        for (i, chunk) in dg_pattern.chunks(pool_size).enumerate() {
            if i < self.input_dim {
                decoded[i] = chunk.iter().sum::<f32>() / chunk.len() as f32;
            }
        }

        decoded
    }

    /// Get buffer statistics
    pub fn stats(&self) -> HippocampusStats {
        let avg_priority = if !self.replay_buffer.is_empty() {
            self.replay_buffer.iter().map(|t| t.priority).sum::<f32>()
                / self.replay_buffer.len() as f32
        } else {
            0.0
        };

        HippocampusStats {
            buffer_size: self.replay_buffer.len(),
            max_buffer: self.max_replay_buffer,
            avg_priority,
            dg_sparsity: self.dg.get_sparsity(),
            dg_dim: self.dg_dim,
        }
    }
}

/// Hippocampus statistics
#[derive(Debug, Clone)]
pub struct HippocampusStats {
    pub buffer_size: usize,
    pub max_buffer: usize,
    pub avg_priority: f32,
    pub dg_sparsity: f32,
    pub dg_dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_encoder() {
        let encoder = SparseEncoder::new(100, 1000, 0.05, 0.15);
        let input = vec![0.5; 100];
        let sparse = encoder.encode(&input);

        // Check sparsity
        let active = sparse.iter().filter(|&&x| x > 0.0).count();
        let expected_active = (1000.0 * 0.05) as usize;
        assert!((active as i32 - expected_active as i32).abs() < 10);
    }

    #[test]
    fn test_recurrent_network_storage() {
        let mut ca3 = RecurrentNetwork::new(100, 0.1, 0.01);
        let pattern = vec![1.0, 0.0, 1.0, 0.0].repeat(25); // 100 dims

        ca3.store_pattern(&pattern);

        // Recall should retrieve similar pattern
        let recalled = ca3.recall(&pattern, 5);
        assert_eq!(recalled.len(), 100);
    }

    #[test]
    fn test_hippocampus_encode_recall() {
        let mut hippo = Hippocampus::new(50, 10, 0.05, 1000);

        let pattern = vec![1.0, 0.0, 0.5, 0.8].repeat(12); // 48 dims
        let mut full_pattern = pattern.clone();
        full_pattern.resize(50, 0.0);

        hippo.encode(&full_pattern);

        // Partial recall
        let mut partial = full_pattern.clone();
        for i in 25..50 {
            partial[i] = 0.0; // Corrupt second half
        }

        let recalled = hippo.recall(&partial);
        assert_eq!(recalled.len(), 50);
    }

    #[test]
    fn test_consolidation() {
        let mut hippo = Hippocampus::new(20, 10, 0.05, 100);

        // Store multiple patterns
        for i in 0..5 {
            let pattern = vec![i as f32 / 5.0; 20];
            hippo.encode(&pattern);
        }

        // Consolidate
        let replayed = hippo.consolidate(3);
        assert_eq!(replayed.len(), 3);
    }

    #[test]
    fn test_memory_trace_priority() {
        let trace1 = MemoryTrace::new(vec![1.0], vec![1.0], 0.0, 0.5);
        let trace2 = MemoryTrace::new(vec![1.0], vec![1.0], 0.0, 1.0);

        // Higher error = higher priority
        assert!(trace2.priority > trace1.priority);
    }
}
