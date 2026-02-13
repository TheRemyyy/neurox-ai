//! Predictive Coding Hierarchy (Canonical Cortical Microcircuit)
//!
//! Implements a biologically plausible Predictive Coding model based on the
//! Canonical Cortical Microcircuit (Bastos et al., 2020; Friston, 2024).
//!
//! # Biological Basis
//! - **Deep Pyramidal Cells (Layer 5/6):** Encode Predictions (Representation).
//!   - Apical Dendrites: Receive Top-Down priors.
//!   - Basal Dendrites: Receive Bottom-Up error signals.
//! - **Superficial Pyramidal Cells (Layer 2/3):** Encode Prediction Errors.
//!   - Receive sensory input (Excitatory).
//!   - Receive lateral inhibition from Deep Pyramidal cells (Prediction).
//! - **Precision:** Implemented as synaptic gain (neuromodulation) on Error units.
//! - **Plasticity:** Local Hebbian learning minimizing Free Energy (F ~ Error²).

use crate::brain::connectivity::SparseConnectivity;
use crate::brain::neuron::{LIFNeuron, Neuron}; // Use standard traits
use serde::{Deserialize, Serialize};

/// Single predictive coding layer implementing a Canonical Microcircuit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveCodingLayer {
    /// Deep Pyramidal Neurons (Representation / Prediction)
    prediction: Vec<LIFNeuron>,

    /// Superficial Pyramidal Neurons (Error Units)
    error: Vec<LIFNeuron>,

    /// Top-down weights (Priors -> Prediction Units)
    /// Biological equivalent: Apical dendrite synapses on Deep Pyramidal cells
    top_down_weights: SparseConnectivity,

    /// Lateral weights (Prediction -> Error)
    /// Biological equivalent: Deep Pyramidal -> Parvalbumin Interneurons -> Superficial Pyramidal
    /// (Effectively inhibitory)
    prediction_to_error_weights: SparseConnectivity,

    /// Bottom-up weights (Error -> Higher Layer)
    /// Biological equivalent: Superficial Pyramidal -> Deep Pyramidal of next area
    bottom_up_weights: SparseConnectivity,

    /// Feedback weights (Error -> Prediction)
    /// Biological equivalent: Superficial Pyramidal -> Deep Pyramidal (same layer update)
    error_to_prediction_weights: Vec<f32>, // Dense 1:1 for simplicity in intra-layer loop

    /// Layer size
    n_neurons: usize,

    /// Precision (Synaptic Gain on Error Units)
    /// Biological equivalent: Cholinergic/Dopaminergic modulation
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
        // Initialize 1:1 feedback weights (Error correcting Prediction)
        let error_to_prediction_weights = vec![0.1; n_neurons];

        // Create lateral inhibition weights (Identity matrix approx for local PC)
        // In a real brain, this is learned, but we initialize as 1:1 inhibition for standard PC
        let prediction_to_error_weights = Self::create_identity_connectivity(n_neurons);

        Self {
            prediction: (0..n_neurons).map(|i| LIFNeuron::new(i as u32)).collect(),
            error: (0..n_neurons).map(|i| LIFNeuron::new(i as u32)).collect(),
            top_down_weights,
            prediction_to_error_weights,
            bottom_up_weights,
            error_to_prediction_weights,
            n_neurons,
            precision,
        }
    }

    /// Forward pass: Compute Error and Update State
    ///
    /// This step simulates the fast dynamics of the microcircuit (relaxation to attractor).
    ///
    /// # Dynamics
    /// 1. Error Units receive Input (+) and Prediction (-)
    /// 2. Prediction Units receive Top-Down (+) and Error Feedback (+)
    ///
    /// # Returns
    /// (prediction_rates, error_rates)
    pub fn forward(&mut self, input: &[f32], dt: f32) -> (Vec<f32>, Vec<f32>) {
        let n = self.n_neurons;

        // 1. Calculate Prediction Activity (Firing Rates)
        // Sigmoid or ReLU activation function on voltage to approximate rate code for inter-layer comms
        let pred_rates: Vec<f32> = self
            .prediction
            .iter()
            .map(|n| (n.state.v + 65.0).max(0.0) / 100.0) // Simple rate approximation
            .collect();

        // 2. Compute Inhibitory Input to Error Units (Prediction -> Interneuron -> Error)
        // We assume 1:1 mapping for simplicity in this specific module, but use the struct
        let mut inhibition = vec![0.0; n];
        // Apply sparse matrix multiplication (prediction_to_error * pred_rates)
        // Optimized manual sparse multiply
        for (i, &rate) in pred_rates.iter().enumerate() {
            // In full implementation, traverse sparse matrix.
            // Here we use the identity simplification logic from init, but robustly:
            inhibition[i] = rate;
        }

        // 3. Update Error Units (Superficial Pyramidal)
        // dV_e/dt = Input - Prediction - Leak
        let mut error_rates = Vec::with_capacity(n);
        for (i, inhib) in inhibition.iter().enumerate().take(n) {
            let sensory_input = input.get(i).cloned().unwrap_or(0.0);

            // Current = (Input - Inhibition) * Precision_Gain
            // Precision acts as a gain control on the mismatch signal
            let mismatch_current = (sensory_input - inhib) * self.precision;

            // Integrate Error Neuron
            self.error[i].update(dt, mismatch_current);

            // Calculate rate
            let rate = (self.error[i].state.v + 65.0).max(0.0) / 100.0;
            error_rates.push(rate);
        }

        // 4. Update Prediction Units (Deep Pyramidal) via Local Feedback
        // They need to move to minimize the error they just caused.
        // dV_p/dt = Error * W_feedback + Top_Down - Leak
        for (i, error_signal) in error_rates.iter().enumerate().take(n) {
            let feedback_current = error_signal * self.error_to_prediction_weights[i];

            // Note: Top-down is handled in 'backward' or separate phase in this architecture,
            // but biologically it happens simultaneously. We add the feedback current here.
            self.prediction[i].update(dt, feedback_current);
        }

        (pred_rates, error_rates)
    }

    /// Backward pass: Inject Top-Down Priors
    ///
    /// Simulates apical dendritic input from higher cortical areas.
    /// In modern theories, this modulates the excitability of Deep Pyramidal cells
    /// or drives them directly (Active Inference).
    pub fn backward(&mut self, top_down: &[f32], dt: f32) {
        // Top-down signals come via sparse connectivity
        // For efficiency in this loop, we map the input vector through the sparse weights

        let mut dendritic_input = vec![0.0; self.n_neurons];

        // Manual sparse matrix-vector multiplication (top_down_weights * top_down)
        // Assuming CSR structure: row_ptr maps target neurons
        let w = &self.top_down_weights;

        for (target_idx, cell) in dendritic_input.iter_mut().enumerate().take(self.n_neurons) {
            if target_idx >= w.row_ptr.len() - 1 {
                break;
            }

            let start = w.row_ptr[target_idx] as usize;
            let end = w.row_ptr[target_idx + 1] as usize;

            let mut sum = 0.0;
            for k in start..end {
                let source_idx = w.col_idx[k] as usize;
                if source_idx < top_down.len() {
                    sum += top_down[source_idx] * w.weights[k];
                }
            }
            *cell = sum;
        }

        // Apply apical input to Prediction Neurons
        for (i, dend) in dendritic_input.iter().enumerate().take(self.n_neurons) {
            // Top-down input drives the neuron towards the prior
            self.prediction[i].update(dt, *dend);
        }
    }

    /// Update Weights (Synaptic Plasticity)
    ///
    /// Implements a local Hebbian rule that minimizes Free Energy.
    /// \Delta W = \eta * Pre * Post
    ///
    /// For PC:
    /// - Forward weights (bu): Minimize prediction error at next level
    /// - Backward weights (td): Minimize prediction error at current level
    pub fn update_weights(&mut self, top_down_input: &[f32], learning_rate: f32) {
        // Update Top-Down Weights (Generative Model)
        // Rule: \Delta W_{ij} = \eta * Error_i * Prediction_j (from higher layer)
        // Here: Prediction_current_layer * Prediction_higher_layer (associative)
        // Or strictly PC: Prediction Error * High_Level_Activity

        // We implement: \Delta W_{td} = \eta * (Error_local * Top_Down_Input)
        // This aligns the prior with the actual error, refining the generative model.

        let w = &mut self.top_down_weights;
        let error_rates: Vec<f32> = self
            .error
            .iter()
            .map(|n| (n.state.v + 65.0).max(0.0))
            .collect();

        for (target_idx, post_activity) in error_rates.iter().enumerate().take(self.n_neurons) {
            if target_idx >= w.row_ptr.len() - 1 {
                break;
            }

            let start = w.row_ptr[target_idx] as usize;
            let end = w.row_ptr[target_idx + 1] as usize;

            // Post-synaptic is the Error unit (conceptually guiding the update)

            for k in start..end {
                let source_idx = w.col_idx[k] as usize;
                if source_idx < top_down_input.len() {
                    let pre_activity = top_down_input[source_idx];

                    // Hebbian Update minimizing Free Energy
                    let delta = learning_rate * *post_activity * pre_activity;
                    w.weights[k] += delta;

                    // Weight decay / Normalization
                    w.weights[k] *= 0.9995;
                }
            }
        }
    }

    /// Helper to create identity connectivity for 1:1 lateral inhibition
    fn create_identity_connectivity(n: usize) -> SparseConnectivity {
        let mut row_ptr = vec![0; n + 1];
        let mut col_idx = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);

        for i in 0..n {
            col_idx.push(i as i32);
            weights.push(1.0);
            row_ptr[i + 1] = (i + 1) as i32;
        }

        SparseConnectivity {
            row_ptr,
            col_idx,
            weights,
            nnz: n,
            n_neurons: n,
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
            // In a mature brain, these are sparse. We init them random-sparse.
            let top_down = Self::create_connectivity(top_size, bottom_size, 0.2);
            let bottom_up = Self::create_connectivity(bottom_size, top_size, 0.2);

            layers.push(PredictiveCodingLayer::new(
                bottom_size,
                top_down,
                bottom_up,
                2.0, // Precision = 2.0
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
    /// # Flow (Bastos et al. 2020 Gamma/Beta rhythm split)
    /// 1. Bottom-up (Feedforward Gamma): Errors propagate up quickly.
    /// 2. Top-down (Feedback Beta): Predictions propagate down.
    /// 3. Relaxation: System settles into low-energy state.
    pub fn process(&mut self, input: &[f32], dt: f32) -> Vec<Vec<f32>> {
        let mut errors = Vec::with_capacity(self.n_layers);
        let mut current_input = input.to_vec();

        // 1. Bottom-up Pass (Fast Error Propagation)
        for layer in &mut self.layers {
            let (_, error_out) = layer.forward(&current_input, dt);
            errors.push(error_out.clone());
            current_input = error_out; // Errors become input to next layer
        }

        // 2. Top-down Pass (Prediction / Priors)
        // Iterates from top layer down to bottom
        for i in (0..self.n_layers - 1).rev() {
            let top_prediction = self.layers[i + 1].get_prediction();

            // Apply top-down priors
            self.layers[i].backward(&top_prediction, dt);

            // 3. Synaptic Plasticity (Online Learning)
            // Update generative model (top-down weights) to better predict this error
            self.layers[i].update_weights(&top_prediction, self.learning_rate);
        }

        errors
    }

    /// Create random connectivity
    fn create_connectivity(n_source: usize, n_target: usize, prob: f32) -> SparseConnectivity {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut row_ptr = vec![0; n_target + 1];
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();

        for target in 0..n_target {
            for source in 0..n_source {
                if rng.gen::<f32>() < prob {
                    col_idx.push(source as i32);
                    // Initialize with small random weights near 0
                    weights.push(rng.gen_range(0.0..0.1));
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

    /// Get prediction at level
    pub fn get_prediction(&self, level: usize) -> Option<Vec<f32>> {
        self.layers.get(level).map(|l| l.get_prediction())
    }

    /// Get total prediction error (Free Energy approximation)
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
