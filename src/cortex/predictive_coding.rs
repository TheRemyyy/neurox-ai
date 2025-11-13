//! Predictive Coding Hierarchy
//!
//! Hierarchical prediction with error propagation.
//! Only prediction errors propagate (efficient sparse communication).
//!
//! # Biological Basis
//! - Cortical hierarchy predicts lower layers
//! - Top-down predictions (forward models)
//! - Bottom-up error signals (surprise)
//! - Bidirectional information flow
//! - Predictive processing (free energy principle)

use crate::connectivity::SparseConnectivity;
use crate::neuron::LIFNeuron;
use serde::{Deserialize, Serialize};

/// Single predictive coding layer
///
/// Contains prediction units (forward model) and error units (mismatch detectors).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveCodingLayer {
    /// Prediction neurons (forward model)
    prediction: Vec<LIFNeuron>,

    /// Error neurons (mismatch detection)
    error: Vec<LIFNeuron>,

    /// Top-down weights (predictions from higher layer)
    top_down_weights: SparseConnectivity,

    /// Bottom-up weights (errors to higher layer)
    bottom_up_weights: SparseConnectivity,

    /// Layer size
    n_neurons: usize,

    /// Prediction precision (inverse variance)
    precision: f32,
}

impl PredictiveCodingLayer {
    /// Create new predictive coding layer
    pub fn new(
        n_neurons: usize,
        top_down_weights: SparseConnectivity,
        bottom_up_weights: SparseConnectivity,
        precision: f32,
    ) -> Self {
        Self {
            prediction: vec![LIFNeuron::default(); n_neurons],
            error: vec![LIFNeuron::default(); n_neurons],
            top_down_weights,
            bottom_up_weights,
            n_neurons,
            precision,
        }
    }

    /// Forward pass: compute prediction and error
    ///
    /// # Returns
    /// (prediction, error)
    pub fn forward(&mut self, input: &[f32], dt: f32) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(input.len(), self.n_neurons);

        // Compute prediction from top-down
        let mut prediction_vals = vec![0.0; self.n_neurons];

        // Top-down prediction (would come from higher layer)
        // For now, use simple dynamics
        for i in 0..self.n_neurons {
            prediction_vals[i] = self.prediction[i].state.v;
        }

        // Compute prediction error
        let mut error_vals = vec![0.0; self.n_neurons];
        for i in 0..self.n_neurons {
            error_vals[i] = (input[i] - prediction_vals[i]) * self.precision;
        }

        (prediction_vals, error_vals)
    }

    /// Backward pass: update predictions from higher layer
    pub fn backward(&mut self, top_down: &[f32]) {
        assert_eq!(top_down.len(), self.n_neurons);

        // Update predictions based on top-down signal
        for i in 0..self.n_neurons {
            self.prediction[i].state.v = top_down[i];
        }
    }

    /// Get prediction
    pub fn get_prediction(&self) -> Vec<f32> {
        self.prediction.iter().map(|n| n.state.v).collect()
    }

    /// Get error
    pub fn get_error(&self) -> Vec<f32> {
        self.error.iter().map(|n| n.state.v).collect()
    }
}

/// Hierarchical predictive coding network
///
/// Stack of layers where each predicts the layer below.
/// Errors propagate upward, predictions propagate downward.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveHierarchy {
    /// Layers (bottom to top)
    layers: Vec<PredictiveCodingLayer>,

    /// Number of layers
    n_layers: usize,

    /// Learning rate for prediction update
    learning_rate: f32,
}

impl PredictiveHierarchy {
    /// Create new predictive hierarchy
    pub fn new(layer_sizes: &[usize], learning_rate: f32) -> Self {
        let n_layers = layer_sizes.len() - 1;
        let mut layers = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            let bottom_size = layer_sizes[i];
            let top_size = layer_sizes[i + 1];

            // Create connectivity
            let top_down = Self::create_connectivity(top_size, bottom_size, 0.2);
            let bottom_up = Self::create_connectivity(bottom_size, top_size, 0.2);

            layers.push(PredictiveCodingLayer::new(
                bottom_size, top_down, bottom_up, 1.0,
            ));
        }

        Self {
            layers,
            n_layers,
            learning_rate,
        }
    }

    /// Process input through hierarchy
    ///
    /// Returns hierarchical errors (one per layer)
    pub fn process(&mut self, input: &[f32], dt: f32) -> Vec<Vec<f32>> {
        let mut errors = Vec::with_capacity(self.n_layers);
        let mut current_input = input.to_vec();

        // Bottom-up pass
        for layer in &mut self.layers {
            let (_, error) = layer.forward(&current_input, dt);
            errors.push(error.clone());
            current_input = error; // Errors propagate upward
        }

        // Top-down pass
        for i in (0..self.n_layers - 1).rev() {
            let top_prediction = self.layers[i + 1].get_prediction();
            self.layers[i].backward(&top_prediction);
        }

        errors
    }

    /// Create random connectivity
    fn create_connectivity(
        n_source: usize,
        n_target: usize,
        prob: f32,
    ) -> SparseConnectivity {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut row_ptr = vec![0; n_target + 1];
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();

        for target in 0..n_target {
            for source in 0..n_source {
                if rng.gen::<f32>() < prob {
                    col_idx.push(source as i32);
                    weights.push(rng.gen_range(-0.5..0.5));
                }
            }
            row_ptr[target + 1] = col_idx.len() as i32;
        }

        SparseConnectivity {
            row_ptr,
            col_idx,
            weights,
            nnz: col_idx.len(),
            n_neurons: n_target,
        }
    }

    /// Get prediction at level
    pub fn get_prediction(&self, level: usize) -> Option<Vec<f32>> {
        self.layers.get(level).map(|l| l.get_prediction())
    }

    /// Get total prediction error
    pub fn total_error(&self) -> f32 {
        self.layers
            .iter()
            .flat_map(|l| l.get_error())
            .map(|e| e * e)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_forward() {
        let top_down = PredictiveHierarchy::create_connectivity(10, 10, 0.2);
        let bottom_up = PredictiveHierarchy::create_connectivity(10, 10, 0.2);

        let mut layer = PredictiveCodingLayer::new(10, top_down, bottom_up, 1.0);

        let input = vec![0.5; 10];
        let (pred, err) = layer.forward(&input, 0.1);

        assert_eq!(pred.len(), 10);
        assert_eq!(err.len(), 10);
    }

    #[test]
    fn test_hierarchy() {
        let mut hierarchy = PredictiveHierarchy::new(&[20, 10, 5], 0.01);

        let input = vec![0.5; 20];
        let errors = hierarchy.process(&input, 0.1);

        assert_eq!(errors.len(), 2); // 2 layers
        assert_eq!(errors[0].len(), 20); // Bottom layer
    }
}
