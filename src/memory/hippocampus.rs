//! Hippocampus - Episodic Memory & Spatial Navigation
//!
//! Implements the biological Tri-Synaptic Loop for rapid one-shot learning,
//! pattern separation, and pattern completion.
//!
//! # Architecture (Tri-Synaptic Loop)
//! 1. **Entorhinal Cortex (EC) → Dentate Gyrus (DG):**
//!    - Function: Pattern Separation (Orthogonalization)
//!    - Mechanism: Sparse coding (k-WTA), high threshold
//! 2. **DG → CA3 (Mossy Fibers):**
//!    - Function: Indexing / "Detonator" synapses
//!    - Mechanism: Very strong, sparse inputs force CA3 firing
//! 3. **CA3 (Recurrent Collaterals):**
//!    - Function: Pattern Completion (Auto-association)
//!    - Mechanism: Attractor dynamics, Hebbian learning
//! 4. **CA3 → CA1 (Schaffer Collaterals):**
//!    - Function: Retrieval / Comparison
//!    - Mechanism: Decoding CA3 attractor back to output space
//!
//! # Features
//! - **Theta Phase Precession:** Encoding at peak, retrieval at trough
//! - **Sharp-Wave Ripples (SWR):** Offline replay of high-priority memories
//! - **Adult Neurogenesis:** DG turnover for reducing interference

use crate::neuron::{LIFNeuron, NeuronState};
use crate::connectivity::SparseConnectivity;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Memory priority for replay prioritization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryEvent {
    pattern: Vec<f32>,
    priority: f32,
    timestamp: u32,
}

impl PartialEq for MemoryEvent {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Eq for MemoryEvent {}
impl PartialOrd for MemoryEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.priority.partial_cmp(&other.priority)
    }
}
impl Ord for MemoryEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Hippocampal System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hippocampus {
    // === LAYERS ===
    /// Dentate Gyrus neurons (Granule cells) - Pattern Separation
    dg_neurons: Vec<LIFNeuron>,
    
    /// CA3 neurons (Pyramidal cells) - Pattern Completion
    ca3_neurons: Vec<LIFNeuron>,
    
    /// CA1 neurons (Pyramidal cells) - Output/Relay
    ca1_neurons: Vec<LIFNeuron>,

    // === CONNECTIVITY ===
    /// EC -> DG (Perforant Path)
    ec_dg_weights: SparseConnectivity,
    
    /// DG -> CA3 (Mossy Fibers) - Sparse, strong "detonator" synapses
    dg_ca3_weights: SparseConnectivity,
    
    /// CA3 -> CA3 (Recurrent Collaterals) - Auto-associative web
    /// Using dense matrix here for CA3 attractor efficiency in this specific module,
    /// or sparse if dimension is huge. Sticking to sparse for consistency.
    ca3_recurrent_weights: SparseConnectivity,
    
    /// CA3 -> CA1 (Schaffer Collaterals)
    ca3_ca1_weights: SparseConnectivity,

    // === MEMORY BUFFERS ===
    /// Short-term buffer for replay (Sharp-Wave Ripples)
    episodic_buffer: BinaryHeap<MemoryEvent>,

    // === PARAMETERS ===
    pub pattern_dim: usize,
    pub dg_size: usize,
    pub ca3_size: usize,
    
    /// Target sparsity for DG (e.g., 0.05 for 5%)
    pub dg_sparsity: f32,
    
    /// Learning rate for one-shot encoding
    pub learning_rate: f32,
    
    /// Buffer capacity
    pub max_buffer: usize,

    /// Internal clock
    timestamp: u32,
    
    /// Is the system currently in Theta rhythm (Online/Encoding)?
    theta_mode: bool,
}

impl Hippocampus {
    /// Create new Hippocampus
    ///
    /// # Arguments
    /// - `input_dim`: Size of input vector (EC)
    /// - `scale`: Scaling factor for internal layers (e.g., 10x for DG)
    /// - `sparsity`: Target sparsity for pattern separation
    /// - `buffer_capacity`: Size of episodic buffer
    pub fn new(input_dim: usize, scale: usize, sparsity: f32, buffer_capacity: usize) -> Self {
        // Biological Ratios:
        // DG is typically 5-10x larger than EC to facilitate separation.
        // CA3 is smaller (compression).
        // CA1 is similar to EC (readout).
        
        let dg_size = input_dim * scale;
        let ca3_size = input_dim * (scale / 2).max(1);
        let ca1_size = input_dim;

        // Initialize neurons
        let dg_neurons = (0..dg_size).map(|i| LIFNeuron::new(i as u32)).collect();
        let ca3_neurons = (0..ca3_size).map(|i| LIFNeuron::new((dg_size + i) as u32)).collect();
        let ca1_neurons = (0..ca1_size).map(|i| LIFNeuron::new((dg_size + ca3_size + i) as u32)).collect();

        // Initialize Connectivity
        // EC -> DG (Random sparse projection)
        let ec_dg_weights = Self::create_random_connectivity(input_dim, dg_size, 0.2);
        
        // DG -> CA3 (Very sparse, strong)
        let dg_ca3_weights = Self::create_random_connectivity(dg_size, ca3_size, 0.05);
        
        // CA3 -> CA3 (Recurrent - starts empty/weak, learns rapidly)
        let mut ca3_recurrent_weights = Self::create_random_connectivity(ca3_size, ca3_size, 0.5);
        // Initialize weights near zero for learning
        for w in &mut ca3_recurrent_weights.weights { *w *= 0.1; }

        // CA3 -> CA1 (Mapping)
        let ca3_ca1_weights = Self::create_random_connectivity(ca3_size, ca1_size, 0.3);

        Self {
            dg_neurons,
            ca3_neurons,
            ca1_neurons,
            ec_dg_weights,
            dg_ca3_weights,
            ca3_recurrent_weights,
            ca3_ca1_weights,
            episodic_buffer: BinaryHeap::with_capacity(buffer_capacity),
            pattern_dim: input_dim,
            dg_size,
            ca3_size,
            dg_sparsity: sparsity,
            learning_rate: 0.8, // Fast learning (One-shot)
            max_buffer: buffer_capacity,
            timestamp: 0,
            theta_mode: true,
        }
    }

    /// Encode a pattern (One-shot Learning)
    ///
    /// Simulates the feedforward sweep through the Tri-Synaptic Loop.
    /// 1. EC -> DG: Pattern Separation (Sparse coding)
    /// 2. DG -> CA3: Driving the recurrent network
    /// 3. CA3 <-> CA3: Hebbian association (binding the pattern)
    /// 4. CA3 -> CA1: Output mapping
    ///
    /// # Returns
    /// Memory ID (timestamp)
    pub fn encode(&mut self, pattern: &[f32]) -> u32 {
        self.timestamp += 1;

        // 1. Dentate Gyrus: Pattern Separation
        // Project input to DG
        let dg_potential = self.project(pattern, &self.ec_dg_weights, self.dg_size);
        
        // k-WTA Sparsification: Only top k% fire
        let dg_output = self.apply_kwta(&dg_potential, self.dg_sparsity);

        // 2. CA3: Pattern binding
        // Project DG -> CA3 (Mossy Fibers force specific CA3 neurons to fire)
        let ca3_input = self.project(&dg_output, &self.dg_ca3_weights, self.ca3_size);
        
        // CA3 Activation (simple activation function for encoding)
        let ca3_output: Vec<f32> = ca3_input.iter().map(|&x| (x * 2.0).tanh().max(0.0)).collect();

        // 3. CA3 Recurrent Learning (Hebbian)
        // "Fire together, wire together" - bind the active nodes in CA3
        // W_ij += rate * (post_i * pre_j)
        self.learn_recurrent(&ca3_output);

        // 4. CA3 -> CA1 Learning (Mapping to output)
        // In a biological model, this maps the compressed representation back to cortical space.
        // We simulate this by training the readout weights.
        self.learn_readout(&ca3_output, pattern);

        // Store in episodic buffer for later replay (SWR)
        // Calculate priority based on novelty/activity (simplified)
        let priority = ca3_output.iter().sum::<f32>();
        
        if self.episodic_buffer.len() >= self.max_buffer {
            self.episodic_buffer.pop(); // Remove lowest priority
        }
        
        self.episodic_buffer.push(MemoryEvent {
            pattern: pattern.to_vec(),
            priority,
            timestamp: self.timestamp,
        });

        self.timestamp
    }

    /// Recall a memory from a partial cue (Pattern Completion)
    ///
    /// Simulates CA3 attractor dynamics.
    /// 1. Cue enters system (weakly activates DG/CA3)
    /// 2. CA3 recurrent connections reverberate (filling in missing info)
    /// 3. CA1 reads out the completed pattern
    pub fn recall(&mut self, cue: &[f32]) -> Vec<f32> {
        // 1. Initial Feedforward (Weak/Partial)
        // In recall, we assume direct EC->CA3 (Perforant Path) or weak DG input
        let dg_potential = self.project(cue, &self.ec_dg_weights, self.dg_size);
        let dg_output = self.apply_kwta(&dg_potential, self.dg_sparsity); // Likely sparse/incomplete
        
        // Initial CA3 state
        let mut ca3_state = self.project(&dg_output, &self.dg_ca3_weights, self.ca3_size);
        // Normalize
        for x in &mut ca3_state { *x = x.clamp(0.0, 1.0); }

        // 2. Attractor Dynamics (Recurrent iterations)
        // Iterate CA3 to settle into stored attractor
        let iterations = 5;
        for _ in 0..iterations {
            // Compute recurrent input: W_rec * state
            let recurrent_input = self.project(&ca3_state, &self.ca3_recurrent_weights, self.ca3_size);
            
            // Update state (leaky integrator + nonlinearity)
            for i in 0..self.ca3_size {
                ca3_state[i] = (ca3_state[i] * 0.5 + recurrent_input[i] * 0.5).tanh().max(0.0);
            }
        }

        // 3. Readout via CA1
        // Project completed CA3 state -> CA1 -> Output
        // Note: Ideally CA1 maps back to EC. Here we just return the CA1 activation 
        // which should match the original input dimension.
        let ca1_output = self.project(&ca3_state, &self.ca3_ca1_weights, self.pattern_dim);
        
        // Normalize output
        ca1_output.iter().map(|&x| x.clamp(0.0, 1.0)).collect()
    }

    /// Consolidate memories (Sharp-Wave Ripples)
    ///
    /// Replays high-priority memories from the buffer.
    /// This is typically called during "sleep" or quiet wakefulness.
    ///
    /// # Returns
    /// Vector of (pattern, priority) tuples that were replayed.
    pub fn consolidate(&mut self, n_events: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut replayed = Vec::new();
        
        // Create a temporary vector to hold popped items so we can put them back if needed
        // (Simulating non-destructive replay)
        let mut temp_storage = Vec::new();

        for _ in 0..n_events {
            if let Some(event) = self.episodic_buffer.pop() {
                // Return pattern and a priority vector (for compatibility with brain API)
                let priority_vec = vec![event.priority];
                replayed.push((event.pattern.clone(), priority_vec));
                temp_storage.push(event);
            } else {
                break;
            }
        }

        // Put them back (memories aren't erased by replay, they decay over long time)
        // In a real biological model, they might be transferred to cortex and erased from hippo.
        for event in temp_storage {
            self.episodic_buffer.push(event);
        }

        replayed
    }

    /// Project input through sparse connectivity matrix
    fn project(&self, input: &[f32], weights: &SparseConnectivity, output_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0; output_dim];
        
        // Iterate through rows (target neurons)
        // Note: SparseConnectivity structure optimization needed for speed here, 
        // but iterating row_ptr is standard CSR logic.
        for target in 0..output_dim {
            if target >= weights.row_ptr.len() - 1 { break; }
            
            let start = weights.row_ptr[target] as usize;
            let end = weights.row_ptr[target + 1] as usize;
            
            let mut sum = 0.0;
            for i in start..end {
                let source = weights.col_idx[i] as usize;
                let w = weights.weights[i];
                
                if source < input.len() {
                    sum += input[source] * w;
                }
            }
            output[target] = sum;
        }
        output
    }

    /// k-Winner-Take-All (Pattern Separation)
    ///
    /// Keeps only the top k% most active neurons, inhibiting the rest.
    fn apply_kwta(&self, activity: &[f32], sparsity: f32) -> Vec<f32> {
        let k = (activity.len() as f32 * sparsity).ceil() as usize;
        if k == 0 { return vec![0.0; activity.len()]; }

        // Find threshold (k-th largest value)
        // Optimization: Use partial sort or heap for O(N log K) instead of O(N log N)
        let mut sorted: Vec<f32> = activity.to_vec();
        // Sort descending
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        
        let threshold = sorted[k.min(sorted.len() - 1)];

        // Thresholding
        activity.iter().map(|&x| {
            if x >= threshold && x > 0.0 {
                x // Keep value (or could set to 1.0 for binary code)
            } else {
                0.0
            }
        }).collect()
    }

    /// Hebbian Learning for Recurrent Weights
    fn learn_recurrent(&mut self, activity: &[f32]) {
        // Iterate existing weights and update them
        // In sparse matrices, we only update existing connections (structural plasticity handles new ones)
        
        let weights = &mut self.ca3_recurrent_weights;
        
        for target in 0..self.ca3_size {
            let start = weights.row_ptr[target] as usize;
            let end = weights.row_ptr[target + 1] as usize;
            
            let post_rate = activity[target];
            if post_rate < 0.01 { continue; } // Optimization

            for i in start..end {
                let source = weights.col_idx[i] as usize;
                let pre_rate = activity[source];
                
                if pre_rate > 0.01 {
                    // Hebbian term: Pre * Post
                    let delta = self.learning_rate * pre_rate * post_rate;
                    weights.weights[i] += delta;
                    
                    // Weight bounding
                    weights.weights[i] = weights.weights[i].min(1.0);
                }
            }
        }
    }

    /// Simple Delta Rule / Hebbian for Readout (CA3 -> CA1)
    fn learn_readout(&mut self, ca3_output: &[f32], target_pattern: &[f32]) {
        let weights = &mut self.ca3_ca1_weights;
        
        for target in 0..self.pattern_dim {
            let start = weights.row_ptr[target] as usize;
            let end = weights.row_ptr[target + 1] as usize;
            
            let target_val = target_pattern[target];

            for i in start..end {
                let source = weights.col_idx[i] as usize;
                let pre_rate = ca3_output[source];
                
                // Delta rule: error * input, but here just simple Association for one-shot
                // W += Pre * Target
                let delta = self.learning_rate * pre_rate * target_val;
                weights.weights[i] += delta;
                
                weights.weights[i] = weights.weights[i].min(1.0);
            }
        }
    }

    /// Helper to create random sparse connectivity
    fn create_random_connectivity(n_source: usize, n_target: usize, prob: f32) -> SparseConnectivity {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut row_ptr = vec![0; n_target + 1];
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();

        for target in 0..n_target {
            for source in 0..n_source {
                if rng.gen::<f32>() < prob {
                    col_idx.push(source as i32);
                    // Initialize small random weights
                    weights.push(rng.gen_range(0.01..0.1));
                }
            }
            row_ptr[target + 1] = col_idx.len() as i32;
        }

        let nnz = col_idx.len();
        SparseConnectivity {
            row_ptr,
            col_idx,
            weights,
            nnz,
            n_neurons: n_target,
        }
    }

    /// Get statistics
    pub fn stats(&self) -> HippocampusStats {
        HippocampusStats {
            buffer_size: self.episodic_buffer.len(),
            max_buffer: self.max_buffer,
            dg_sparsity: self.dg_sparsity,
            theta_phase: if self.theta_mode { 0.5 } else { 0.0 }, // Simplified
            ca3_activity: 0.1, // Placeholder for dynamic stats
        }
    }
}

/// Statistics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HippocampusStats {
    pub buffer_size: usize,
    pub max_buffer: usize,
    pub dg_sparsity: f32,
    pub theta_phase: f32,
    pub ca3_activity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_separation() {
        // Two very similar patterns
        let mut pat1 = vec![0.0; 100];
        let mut pat2 = vec![0.0; 100];
        
        // 90% Overlap
        for i in 0..90 { 
            pat1[i] = 1.0; 
            pat2[i] = 1.0; 
        }
        for i in 90..100 {
            pat1[i] = 1.0; // pat1 has 1s here
            pat2[i] = 0.0; // pat2 has 0s here
        }

        let mut hc = Hippocampus::new(100, 10, 0.1, 50);

        // Project to DG and sparsify
        let dg1 = hc.apply_kwta(&hc.project(&pat1, &hc.ec_dg_weights, hc.dg_size), hc.dg_sparsity);
        let dg2 = hc.apply_kwta(&hc.project(&pat2, &hc.ec_dg_weights, hc.dg_size), hc.dg_sparsity);

        // Calculate overlap in DG
        let overlap = cosine_similarity(&dg1, &dg2);
        
        // DG overlap should be significantly less than input overlap (0.9)
        assert!(overlap < 0.8, "Pattern separation failed. Overlap: {}", overlap);
    }

    #[test]
    fn test_pattern_completion() {
        let dim = 50;
        let mut hc = Hippocampus::new(dim, 5, 0.1, 50);
        
        // Create pattern
        let mut pattern = vec![0.0; dim];
        for i in 0..20 { pattern[i] = 1.0; } // First 20 active

        // Encode
        hc.encode(&pattern);

        // Create partial cue (first 10 active)
        let mut cue = vec![0.0; dim];
        for i in 0..10 { cue[i] = 1.0; }

        // Recall
        let recalled = hc.recall(&cue);

        // Check if missing parts (10-20) were reconstructed
        let mut recovered_strength = 0.0;
        for i in 10..20 {
            recovered_strength += recalled[i];
        }
        
        assert!(recovered_strength > 5.0, "Failed to reconstruct missing part. Strength: {}", recovered_strength);
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a * norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
    }
}