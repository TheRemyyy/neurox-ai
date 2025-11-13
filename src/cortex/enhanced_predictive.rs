//! Enhanced Predictive Coding Hierarchy (5-7 Levels with Laminar Structure)
//!
//! Expands from 3 to 5-7 hierarchical levels implementing proper cortical laminar architecture:
//! - Layer 4: Input processing, generates initial predictions
//! - Layer 2/3: Error neurons computing precision-weighted prediction errors (sparse)
//! - Layer 5: Prediction neurons maintaining generative models (dense)
//! - Layer 6: Precision/modulation units adjusting confidence weights
//! - Lateral: Contextual integration within level
//!
//! # Hierarchy Structure
//! Level 1 (V1): 512 neurons, 5×5 RF, 50ms timescale - edges/orientations
//! Level 2 (V2): 384 neurons, 10×10 RF, 100ms timescale - textures/corners
//! Level 3 (V4): 256 neurons, 20×20 RF, 200ms timescale - object parts
//! Level 4 (IT): 128 neurons, 40×40 RF, 400ms timescale - whole objects
//! Level 5 (PFC): 64 neurons, 80×80 RF, 800ms timescale - categories/context
//!
//! # Expected Improvements
//! - 10-100× error reduction (from 940,800 to 10,000-100,000)
//! - Abstract representations at higher levels
//! - Temporal predictions supporting sequence learning
//! - Precision weighting implementing attention
//! - Oscillatory coordination for flexible routing

use crate::connectivity::SparseConnectivity;
use crate::neuron::{LIFNeuron, Neuron};
use crate::oscillations::OscillatoryCircuit;
use serde::{Deserialize, Serialize};

/// Cortical layer with laminar structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaminarLayer {
    /// Layer 4: Input layer
    pub layer4: Vec<LIFNeuron>,

    /// Layer 2/3: Error neurons (sparse)
    pub layer2_3: Vec<LIFNeuron>,

    /// Layer 5: Prediction neurons (dense)
    pub layer5: Vec<LIFNeuron>,

    /// Layer 6: Precision modulation
    pub layer6: Vec<LIFNeuron>,

    /// Lateral connections (within layer)
    pub lateral_connections: SparseConnectivity,

    /// Number of neurons per sublayer
    pub n_neurons: usize,

    /// Precision weights (one per prediction)
    pub precision: Vec<f32>,

    /// Learning rate (layer-specific)
    pub learning_rate: f32,
}

impl LaminarLayer {
    pub fn new(n_neurons: usize, learning_rate: f32) -> Self {
        let layer4 = (0..n_neurons).map(|i| LIFNeuron::new(i as u32)).collect();
        let layer2_3 = (0..n_neurons).map(|i| LIFNeuron::new((n_neurons + i) as u32)).collect();
        let layer5 = (0..n_neurons).map(|i| LIFNeuron::new((2 * n_neurons + i) as u32)).collect();
        let layer6 = (0..n_neurons/2).map(|i| LIFNeuron::new((3 * n_neurons + i) as u32)).collect();

        // Create lateral connections (sparse)
        let lateral_connections = Self::create_lateral(n_neurons, 0.1);

        let precision = vec![1.0; n_neurons];

        Self {
            layer4,
            layer2_3,
            layer5,
            layer6,
            lateral_connections,
            n_neurons,
            precision,
            learning_rate,
        }
    }

    /// Create lateral connections for contextual integration
    fn create_lateral(n_neurons: usize, connection_prob: f32) -> SparseConnectivity {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut row_ptr = vec![0; n_neurons + 1];
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();

        for target in 0..n_neurons {
            for source in 0..n_neurons {
                if source != target && rng.gen::<f32>() < connection_prob {
                    col_idx.push(source as i32);
                    weights.push(rng.gen_range(-0.2..0.2));
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
            n_neurons,
        }
    }

    /// Forward pass: compute prediction and error
    ///
    /// # Process
    /// 1. L4 receives bottom-up input
    /// 2. L5 generates prediction
    /// 3. L2/3 computes precision-weighted error
    /// 4. L6 modulates precision
    pub fn forward(&mut self, input: &[f32], top_down: &[f32], dt: f32) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(input.len(), self.n_neurons);

        // Layer 4: Process input
        let mut l4_output = vec![0.0; self.n_neurons];
        for (i, neuron) in self.layer4.iter_mut().enumerate() {
            if neuron.update(dt, input[i]) {
                l4_output[i] = 1.0;
            }
        }

        // Layer 5: Generate prediction (from top-down)
        let mut prediction = vec![0.0; self.n_neurons];
        for (i, neuron) in self.layer5.iter_mut().enumerate() {
            let td_input = if i < top_down.len() { top_down[i] } else { 0.0 };
            if neuron.update(dt, td_input) {
                prediction[i] = neuron.state.v;
            }
        }

        // Layer 2/3: Compute precision-weighted error
        let mut error = vec![0.0; self.n_neurons];
        for i in 0..self.n_neurons {
            // Prediction error
            let raw_error = input[i] - prediction[i];

            // Precision weighting (attention)
            error[i] = self.precision[i] * raw_error;

            // Update error neuron
            self.layer2_3[i].update(dt, error[i].abs());
        }

        // Layer 6: Update precision based on error statistics
        for (i, neuron) in self.layer6.iter_mut().enumerate() {
            if i * 2 < self.n_neurons {
                // Precision inversely related to recent error variance
                let local_error = error[i * 2..(i * 2 + 2).min(self.n_neurons)].iter().sum::<f32>();
                neuron.update(dt, local_error.abs());

                // Update precision (inverse of error variance)
                let error_var = local_error * local_error;
                self.precision[i * 2] = (1.0 / (error_var + 0.1)).min(10.0);
                if i * 2 + 1 < self.n_neurons {
                    self.precision[i * 2 + 1] = self.precision[i * 2];
                }
            }
        }

        // Lateral integration (contextual)
        let lateral_output = self.apply_lateral(&l4_output);
        for (i, &lat) in lateral_output.iter().enumerate() {
            if i < self.n_neurons {
                error[i] += lat * 0.2;  // 20% lateral contribution
            }
        }

        (prediction, error)
    }

    /// Apply lateral connections for context
    fn apply_lateral(&self, activity: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.n_neurons];

        for target in 0..self.n_neurons {
            let start = self.lateral_connections.row_ptr[target] as usize;
            let end = self.lateral_connections.row_ptr[target + 1] as usize;

            for syn_id in start..end {
                let source = self.lateral_connections.col_idx[syn_id] as usize;
                let weight = self.lateral_connections.weights[syn_id];

                if source < activity.len() {
                    output[target] += weight * activity[source];
                }
            }
        }

        output
    }

    /// Backward pass: update predictions
    pub fn backward(&mut self, error_from_above: &[f32]) {
        // Layer 5 adjusts predictions based on higher-level error
        for (i, neuron) in self.layer5.iter_mut().enumerate() {
            if i < error_from_above.len() {
                let adjustment = -self.learning_rate * error_from_above[i];
                neuron.state.v += adjustment;
            }
        }
    }

    /// Get average error
    pub fn average_error(&self) -> f32 {
        self.layer2_3.iter().map(|n| n.state.v.abs()).sum::<f32>() / self.n_neurons as f32
    }
}

/// Enhanced hierarchical predictive coding network (5-7 levels)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedPredictiveHierarchy {
    /// Hierarchical levels (bottom to top)
    pub levels: Vec<LaminarLayer>,

    /// Number of levels
    pub n_levels: usize,

    /// Receptive field sizes
    pub receptive_fields: Vec<usize>,

    /// Timescales per level (ms)
    pub timescales: Vec<f32>,

    /// Oscillatory circuit for coordination
    #[serde(skip)]
    pub oscillations: Option<OscillatoryCircuit>,

    /// Global learning rate
    pub learning_rate: f32,
}

impl EnhancedPredictiveHierarchy {
    /// Create 5-level hierarchy (can extend to 7)
    ///
    /// # Default Architecture
    /// - Level 0: 512 neurons, 5×5 RF, 50ms (V1 - edges)
    /// - Level 1: 384 neurons, 10×10 RF, 100ms (V2 - textures)
    /// - Level 2: 256 neurons, 20×20 RF, 200ms (V4 - parts)
    /// - Level 3: 128 neurons, 40×40 RF, 400ms (IT - objects)
    /// - Level 4: 64 neurons, 80×80 RF, 800ms (PFC - categories)
    pub fn new_default() -> Self {
        let layer_sizes = vec![512, 384, 256, 128, 64];
        let receptive_fields = vec![5, 10, 20, 40, 80];
        let timescales = vec![50.0, 100.0, 200.0, 400.0, 800.0];

        Self::new(&layer_sizes, &receptive_fields, &timescales, 0.001)
    }

    /// Create custom hierarchy
    pub fn new(
        layer_sizes: &[usize],
        receptive_fields: &[usize],
        timescales: &[f32],
        learning_rate: f32,
    ) -> Self {
        assert_eq!(layer_sizes.len(), receptive_fields.len());
        assert_eq!(layer_sizes.len(), timescales.len());

        let n_levels = layer_sizes.len();
        let mut levels = Vec::with_capacity(n_levels);

        for (i, &size) in layer_sizes.iter().enumerate() {
            // Layer-specific learning rate (slower for higher levels)
            let layer_lr = learning_rate / (i + 1) as f32;
            levels.push(LaminarLayer::new(size, layer_lr));
        }

        Self {
            levels,
            n_levels,
            receptive_fields: receptive_fields.to_vec(),
            timescales: timescales.to_vec(),
            oscillations: Some(OscillatoryCircuit::new()),
            learning_rate,
        }
    }

    /// Process input through hierarchy
    ///
    /// Returns hierarchical errors (one per level)
    pub fn process(&mut self, input: &[f32], dt: f32) -> Vec<Vec<f32>> {
        let mut errors = Vec::with_capacity(self.n_levels);
        let mut current_input = input.to_vec();

        // Update oscillations
        if let Some(ref mut osc) = self.oscillations {
            osc.update(dt, 0.5);  // 0.5 = medium prediction strength
        }

        // Bottom-up pass (compute errors at each level)
        for i in 0..self.n_levels {
            // Get top-down prediction from level above (if exists)
            let top_down = if i + 1 < self.n_levels {
                self.levels[i + 1].layer5.iter().map(|n| n.state.v).collect()
            } else {
                vec![0.0; self.levels[i].n_neurons]
            };

            // Forward pass at this level
            let (prediction, error) = self.levels[i].forward(&current_input, &top_down, dt);

            errors.push(error.clone());

            // Errors propagate upward (compressed)
            current_input = error;

            // Pad or truncate to next level size
            if i + 1 < self.n_levels {
                let next_size = self.levels[i + 1].n_neurons;
                if current_input.len() < next_size {
                    current_input.resize(next_size, 0.0);
                } else if current_input.len() > next_size {
                    current_input.truncate(next_size);
                }
            }
        }

        // Top-down pass (update predictions)
        for i in (0..self.n_levels - 1).rev() {
            let error_from_above = if i + 1 < errors.len() {
                errors[i + 1].clone()
            } else {
                vec![0.0; self.levels[i].n_neurons]
            };

            self.levels[i].backward(&error_from_above);
        }

        errors
    }

    /// Get prediction at specific level
    pub fn get_prediction(&self, level: usize) -> Option<Vec<f32>> {
        self.levels.get(level).map(|l| {
            l.layer5.iter().map(|n| n.state.v).collect()
        })
    }

    /// Get total prediction error (across all levels)
    pub fn total_error(&self) -> f32 {
        self.levels
            .iter()
            .map(|l| l.average_error())
            .sum()
    }

    /// Get level-specific error
    pub fn level_error(&self, level: usize) -> f32 {
        self.levels
            .get(level)
            .map(|l| l.average_error())
            .unwrap_or(0.0)
    }

    /// Get precision at level
    pub fn get_precision(&self, level: usize) -> Option<&Vec<f32>> {
        self.levels.get(level).map(|l| &l.precision)
    }

    /// Set attention (modulates precision)
    pub fn set_attention(&mut self, level: usize, boost: f32) {
        if let Some(layer) = self.levels.get_mut(level) {
            for p in &mut layer.precision {
                *p *= boost;
                *p = p.clamp(0.1, 10.0);
            }
        }
    }

    /// Get hierarchy statistics
    pub fn stats(&self) -> EnhancedPredictiveStats {
        let level_errors: Vec<f32> = self.levels.iter().map(|l| l.average_error()).collect();
        let total_error = level_errors.iter().sum();

        let avg_precision: Vec<f32> = self.levels
            .iter()
            .map(|l| l.precision.iter().sum::<f32>() / l.precision.len() as f32)
            .collect();

        EnhancedPredictiveStats {
            n_levels: self.n_levels,
            level_errors,
            total_error,
            avg_precision,
            timescales: self.timescales.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnhancedPredictiveStats {
    pub n_levels: usize,
    pub level_errors: Vec<f32>,
    pub total_error: f32,
    pub avg_precision: Vec<f32>,
    pub timescales: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laminar_layer() {
        let mut layer = LaminarLayer::new(50, 0.01);

        let input = vec![0.5; 50];
        let top_down = vec![0.3; 50];

        let (pred, error) = layer.forward(&input, &top_down, 0.1);

        assert_eq!(pred.len(), 50);
        assert_eq!(error.len(), 50);
    }

    #[test]
    fn test_enhanced_hierarchy() {
        let mut hierarchy = EnhancedPredictiveHierarchy::new_default();

        assert_eq!(hierarchy.n_levels, 5);
        assert_eq!(hierarchy.levels[0].n_neurons, 512);
        assert_eq!(hierarchy.levels[4].n_neurons, 64);
    }

    #[test]
    fn test_processing() {
        let mut hierarchy = EnhancedPredictiveHierarchy::new_default();

        let input = vec![0.5; 512];
        let errors = hierarchy.process(&input, 0.1);

        assert_eq!(errors.len(), 5);
    }

    #[test]
    fn test_precision_weighting() {
        let mut layer = LaminarLayer::new(10, 0.01);

        // Set high precision for first half
        for i in 0..5 {
            layer.precision[i] = 5.0;
        }

        let input = vec![1.0; 10];
        let top_down = vec![0.5; 10];

        let (_, error) = layer.forward(&input, &top_down, 0.1);

        // First half should have larger errors (high precision amplifies)
        let first_half_err: f32 = error[0..5].iter().sum();
        let second_half_err: f32 = error[5..10].iter().sum();

        assert!(first_half_err > second_half_err);
    }

    #[test]
    fn test_layer_specific_learning_rates() {
        let hierarchy = EnhancedPredictiveHierarchy::new_default();

        // Higher levels should have slower learning rates
        for i in 1..hierarchy.n_levels {
            assert!(hierarchy.levels[i].learning_rate < hierarchy.levels[i - 1].learning_rate);
        }
    }
}
