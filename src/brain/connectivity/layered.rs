//! Layered Feedforward Connectivity for Deep SNNs
//!
//! Implements structured layer-by-layer connectivity:
//! Input(784) → Hidden1(512) → Hidden2(256) → Output(10)
//!
//! This replaces chaotic sparse connectivity with proper feedforward architecture
//! essential for supervised learning.

use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Layer definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub name: String,
    pub start_idx: usize,
    pub size: usize,
}

impl Layer {
    pub fn end_idx(&self) -> usize {
        self.start_idx + self.size
    }

    pub fn contains(&self, idx: usize) -> bool {
        idx >= self.start_idx && idx < self.end_idx()
    }
}

/// Layered feedforward connectivity
///
/// Explicit layer-to-layer connections for deep SNN training.
/// Each layer is fully connected to the next layer only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredConnectivity {
    pub layers: Vec<Layer>,
    /// Weight matrices between adjacent layers
    /// weights[i] connects layer[i] to layer[i+1]
    /// Shape: [layer[i].size * layer[i+1].size]
    pub weights: Vec<Vec<f32>>,
    /// Total number of neurons
    pub n_neurons: usize,
}

impl LayeredConnectivity {
    /// Create new layered connectivity with default MNIST architecture
    ///
    /// Input(784) → Hidden1(512) → Hidden2(256) → Output(10)
    pub fn new_mnist(seed: u64) -> Self {
        let layer_sizes = vec![784, 512, 256, 10];
        Self::from_layer_sizes(&layer_sizes, seed)
    }

    /// Create from custom layer sizes
    pub fn from_layer_sizes(sizes: &[usize], seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut layers = Vec::with_capacity(sizes.len());
        let mut weights = Vec::with_capacity(sizes.len() - 1);

        let mut current_idx = 0;
        for (i, &size) in sizes.iter().enumerate() {
            layers.push(Layer {
                name: match i {
                    0 => "input".to_string(),
                    x if x == sizes.len() - 1 => "output".to_string(),
                    _ => format!("hidden{}", i),
                },
                start_idx: current_idx,
                size,
            });
            current_idx += size;
        }

        // Initialize weight matrices with Xavier initialization
        for i in 0..layers.len() - 1 {
            let fan_in = layers[i].size;
            let fan_out = layers[i + 1].size;
            let xavier_std = (2.0 / (fan_in + fan_out) as f32).sqrt();

            let n_weights = fan_in * fan_out;
            let mut w = Vec::with_capacity(n_weights);

            for _ in 0..n_weights {
                // Xavier uniform initialization
                let val: f32 = rng.gen_range(-xavier_std..xavier_std);
                // Ensure weights are in valid range [0, 1] for SNN
                w.push((val + 0.5).clamp(0.0, 1.0));
            }
            weights.push(w);
        }

        let n_neurons = current_idx;

        Self {
            layers,
            weights,
            n_neurons,
        }
    }

    /// Get weight between two neurons
    /// Returns None if not connected (not in adjacent layers)
    pub fn get_weight(&self, pre: usize, post: usize) -> Option<f32> {
        // Find which layers they belong to
        let pre_layer_idx = self.find_layer(pre)?;
        let post_layer_idx = self.find_layer(post)?;

        // Must be in adjacent layers
        if post_layer_idx != pre_layer_idx + 1 {
            return None;
        }

        let pre_layer = &self.layers[pre_layer_idx];
        let post_layer = &self.layers[post_layer_idx];

        let local_pre = pre - pre_layer.start_idx;
        let local_post = post - post_layer.start_idx;

        let weight_idx = local_pre * post_layer.size + local_post;
        Some(self.weights[pre_layer_idx][weight_idx])
    }

    /// Set weight between two neurons
    pub fn set_weight(&mut self, pre: usize, post: usize, w: f32) -> bool {
        let pre_layer_idx = match self.find_layer(pre) {
            Some(i) => i,
            None => return false,
        };
        let post_layer_idx = match self.find_layer(post) {
            Some(i) => i,
            None => return false,
        };

        if post_layer_idx != pre_layer_idx + 1 {
            return false;
        }

        let pre_layer = &self.layers[pre_layer_idx];
        let post_layer = &self.layers[post_layer_idx];

        let local_pre = pre - pre_layer.start_idx;
        let local_post = post - post_layer.start_idx;

        let weight_idx = local_pre * post_layer.size + local_post;
        self.weights[pre_layer_idx][weight_idx] = w.clamp(0.0, 1.0);
        true
    }

    /// Find which layer an index belongs to
    fn find_layer(&self, idx: usize) -> Option<usize> {
        self.layers.iter().position(|l| l.contains(idx))
    }

    /// Get layer by name
    pub fn get_layer(&self, name: &str) -> Option<&Layer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Get input layer
    pub fn input_layer(&self) -> &Layer {
        &self.layers[0]
    }

    /// Get output layer
    pub fn output_layer(&self) -> &Layer {
        self.layers.last().unwrap()
    }

    /// Forward pass: compute currents for next layer given input spikes
    pub fn forward(&self, spikes: &[f32]) -> Vec<f32> {
        let mut currents = vec![0.0f32; self.n_neurons];

        // For each layer pair
        for layer_idx in 0..self.layers.len() - 1 {
            let pre_layer = &self.layers[layer_idx];
            let post_layer = &self.layers[layer_idx + 1];
            let weights = &self.weights[layer_idx];

            // Compute weighted sum for each post neuron
            for post_local in 0..post_layer.size {
                let post_global = post_layer.start_idx + post_local;
                let mut current = 0.0f32;

                for pre_local in 0..pre_layer.size {
                    let pre_global = pre_layer.start_idx + pre_local;
                    let spike = spikes[pre_global];

                    if spike > 0.5 {
                        let w_idx = pre_local * post_layer.size + post_local;
                        current += weights[w_idx];
                    }
                }

                currents[post_global] = current;
            }
        }

        currents
    }

    /// Get all incoming connections for a neuron
    pub fn get_incoming(&self, post: usize) -> Vec<(usize, usize)> {
        let mut incoming = Vec::new();

        let post_layer_idx = match self.find_layer(post) {
            Some(i) if i > 0 => i,
            _ => return incoming,
        };

        let pre_layer = &self.layers[post_layer_idx - 1];
        let post_layer = &self.layers[post_layer_idx];
        let local_post = post - post_layer.start_idx;

        for pre_local in 0..pre_layer.size {
            let pre_global = pre_layer.start_idx + pre_local;
            let w_idx = pre_local * post_layer.size + local_post;
            // (pre_id, weight_index in the layer's weight matrix)
            // We encode layer_idx * 1000000 + w_idx to make it unique
            let global_w_idx = (post_layer_idx - 1) * 10_000_000 + w_idx;
            incoming.push((pre_global, global_w_idx));
        }

        incoming
    }

    /// Get all outgoing connections for a neuron
    pub fn get_outgoing(&self, pre: usize) -> Vec<(usize, usize)> {
        let mut outgoing = Vec::new();

        let pre_layer_idx = match self.find_layer(pre) {
            Some(i) if i < self.layers.len() - 1 => i,
            _ => return outgoing,
        };

        let pre_layer = &self.layers[pre_layer_idx];
        let post_layer = &self.layers[pre_layer_idx + 1];
        let local_pre = pre - pre_layer.start_idx;

        for post_local in 0..post_layer.size {
            let post_global = post_layer.start_idx + post_local;
            let w_idx = local_pre * post_layer.size + post_local;
            let global_w_idx = pre_layer_idx * 10_000_000 + w_idx;
            outgoing.push((post_global, global_w_idx));
        }

        outgoing
    }

    /// Apply weight update by global index
    pub fn apply_weight_update(&mut self, global_w_idx: usize, delta: f32) {
        let layer_idx = global_w_idx / 10_000_000;
        let local_idx = global_w_idx % 10_000_000;

        if layer_idx < self.weights.len() && local_idx < self.weights[layer_idx].len() {
            let new_w = (self.weights[layer_idx][local_idx] + delta).clamp(0.0, 1.0);
            self.weights[layer_idx][local_idx] = new_w;
        }
    }

    /// Get total number of weights (synapses)
    pub fn total_weights(&self) -> usize {
        self.weights.iter().map(|w| w.len()).sum()
    }

    /// Get mean weight across all layers
    pub fn mean_weight(&self) -> f32 {
        let total: f32 = self.weights.iter().flat_map(|w| w.iter()).sum();
        total / self.total_weights() as f32
    }

    /// Convert to CSR format for GPU simulation
    ///
    /// Returns (row_ptr, col_idx, weights) suitable for SparseConnectivity upload
    pub fn to_sparse_csr(&self) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let mut row_ptr = vec![0i32; self.n_neurons + 1];
        let mut col_idx = Vec::new();
        let mut weights_flat = Vec::new();

        // For each neuron, find all its outgoing connections
        for neuron_id in 0..self.n_neurons {
            let start_nnz = col_idx.len();

            // Find which layer this neuron is in
            if let Some(layer_idx) = self.find_layer(neuron_id) {
                // Only if not the output layer (no outgoing)
                if layer_idx < self.layers.len() - 1 {
                    let pre_layer = &self.layers[layer_idx];
                    let post_layer = &self.layers[layer_idx + 1];
                    let local_pre = neuron_id - pre_layer.start_idx;

                    // This neuron connects to all neurons in the next layer
                    for post_local in 0..post_layer.size {
                        let post_global = post_layer.start_idx + post_local;
                        let w_idx = local_pre * post_layer.size + post_local;
                        let weight = self.weights[layer_idx][w_idx];

                        col_idx.push(post_global as i32);
                        weights_flat.push(weight);
                    }
                }
            }

            row_ptr[neuron_id + 1] = col_idx.len() as i32;
        }

        log::info!(
            "LayeredConnectivity -> CSR: {} neurons, {} synapses",
            self.n_neurons,
            weights_flat.len()
        );

        (row_ptr, col_idx, weights_flat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_structure() {
        let conn = LayeredConnectivity::new_mnist(42);

        assert_eq!(conn.layers.len(), 4);
        assert_eq!(conn.layers[0].name, "input");
        assert_eq!(conn.layers[0].size, 784);
        assert_eq!(conn.layers[1].name, "hidden1");
        assert_eq!(conn.layers[1].size, 512);
        assert_eq!(conn.layers[2].name, "hidden2");
        assert_eq!(conn.layers[2].size, 256);
        assert_eq!(conn.layers[3].name, "output");
        assert_eq!(conn.layers[3].size, 10);

        assert_eq!(conn.n_neurons, 784 + 512 + 256 + 10);
    }

    #[test]
    fn test_weight_access() {
        let conn = LayeredConnectivity::new_mnist(42);

        // Input[0] -> Hidden1[784] should have a weight
        let w = conn.get_weight(0, 784);
        assert!(w.is_some());

        // Input[0] -> Hidden2[1296] should NOT have a weight (not adjacent)
        let w2 = conn.get_weight(0, 1296);
        assert!(w2.is_none());
    }

    #[test]
    fn test_forward_pass() {
        let conn = LayeredConnectivity::new_mnist(42);

        // Create input spikes (all ones for simplicity)
        let mut spikes = vec![0.0f32; conn.n_neurons];
        for i in 0..784 {
            spikes[i] = 1.0;
        }

        let currents = conn.forward(&spikes);

        // Hidden1 neurons should have non-zero currents
        for i in 784..784 + 512 {
            assert!(currents[i] > 0.0, "Hidden1[{}] should receive current", i);
        }
    }
}
