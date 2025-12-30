//! Superior Colliculus - Eye Movement Control
//!
//! Implements visuomotor transformation for saccadic eye movements and
//! visual attention. The SC is a layered midbrain structure that maps
//! visual space to motor commands for rapid eye movements (saccades).
//!
//! # Architecture
//! - Superficial layers (SGS, SO, SAI): Visual processing
//! - Intermediate layers (SGI): Visuomotor integration
//! - Deep layers (SGP, SAP): Motor output
//! - Topographic motor map: position → saccade vector
//!
//! # Features
//! - Retinotopic visual input processing
//! - Log-polar motor map for saccade generation
//! - Burst neurons for saccade initiation
//! - Omnipause neurons for fixation
//! - Inhibition of return
//!
//! # Mathematical Model
//! Motor map: (x, y) → (r, θ) where r = k·log(e + √(x² + y²))
//! Burst activity: B(t) = A·exp(-(t-t₀)²/2σ²) for saccade
//!
//! # References
//! - Sparks & Mays (1990) "Signal transformations required for saccades"
//! - Gandhi & Katnani (2011) "Motor functions of the superior colliculus"
//! - October 2024 paper on SC topographic mapping

use serde::{Deserialize, Serialize};

/// Superior Colliculus layers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SCLayer {
    Superficial,  // Visual input (SGS, SO, SAI)
    Intermediate, // Visuomotor (SGI)
    Deep,         // Motor output (SGP, SAP)
}

/// Superior Colliculus neuron
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCNeuron {
    /// Neuron ID
    pub id: usize,

    /// Layer
    pub layer: u8, // 0=superficial, 1=intermediate, 2=deep

    /// Position in topographic map (normalized 0-1)
    pub map_x: f32,
    pub map_y: f32,

    /// Receptive field center (visual degrees)
    pub rf_x: f32,
    pub rf_y: f32,

    /// Receptive field size (visual degrees)
    pub rf_size: f32,

    /// Current activity
    pub activity: f32,

    /// Spike threshold
    pub threshold: f32,

    /// Refractory counter
    pub refractory: u8,
}

impl SCNeuron {
    pub fn new(id: usize, layer: u8, map_x: f32, map_y: f32) -> Self {
        // Map position to visual field using log-polar mapping
        // r = k·log(e + √(x² + y²))
        let k = 10.0; // Scaling factor
        let x_centered = (map_x - 0.5) * 2.0; // -1 to 1
        let y_centered = (map_y - 0.5) * 2.0;

        let r = k
            * (std::f32::consts::E + (x_centered * x_centered + y_centered * y_centered).sqrt())
                .ln();
        let theta = y_centered.atan2(x_centered);

        let rf_x = r * theta.cos();
        let rf_y = r * theta.sin();

        // RF size increases with eccentricity
        let eccentricity = (rf_x * rf_x + rf_y * rf_y).sqrt();
        let rf_size = 2.0 + eccentricity * 0.5;

        Self {
            id,
            layer,
            map_x,
            map_y,
            rf_x,
            rf_y,
            rf_size,
            activity: 0.0,
            threshold: 50.0,
            refractory: 0,
        }
    }

    /// Check if visual stimulus is in receptive field
    pub fn in_receptive_field(&self, visual_x: f32, visual_y: f32) -> f32 {
        let dx = visual_x - self.rf_x;
        let dy = visual_y - self.rf_y;
        let distance = (dx * dx + dy * dy).sqrt();

        // Gaussian receptive field
        let response = (-(distance * distance) / (2.0 * self.rf_size * self.rf_size)).exp();
        response
    }

    /// Update neuron activity
    pub fn update(&mut self, input: f32, dt: f32) -> bool {
        if self.refractory > 0 {
            self.refractory -= 1;
            self.activity *= 0.9; // Decay during refractory
            return false;
        }

        // Integrate input
        let tau = 10.0; // ms
        let decay = (-dt / tau).exp();
        self.activity = self.activity * decay + input * (1.0 - decay);

        // Check for spike
        if self.activity >= self.threshold {
            self.refractory = 20; // 2ms @ 0.1ms timestep
            self.activity = 0.0;
            true
        } else {
            false
        }
    }
}

/// Superior Colliculus system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperiorColliculus {
    /// Neurons organized by layer
    pub superficial_neurons: Vec<SCNeuron>,
    pub intermediate_neurons: Vec<SCNeuron>,
    pub deep_neurons: Vec<SCNeuron>,

    /// Map resolution
    pub map_width: usize,
    pub map_height: usize,

    /// Omnipause neurons (fixation control)
    pub omnipause_activity: f32,
    pub fixation_mode: bool,

    /// Saccade state
    pub saccade_active: bool,
    pub saccade_vector: (f32, f32), // (amplitude, direction) in degrees
    pub saccade_progress: f32,      // 0 to 1

    /// Inhibition of return (recently visited locations)
    pub ior_map: Vec<Vec<f32>>,
    pub ior_decay: f32,

    /// Statistics
    pub total_saccades: u64,
    pub avg_saccade_amplitude: f32,
}

impl SuperiorColliculus {
    /// Create new Superior Colliculus
    ///
    /// # Arguments
    /// - `map_width`: Width of topographic map (e.g., 32)
    /// - `map_height`: Height of topographic map (e.g., 32)
    pub fn new(map_width: usize, map_height: usize) -> Self {
        let mut superficial_neurons = Vec::new();
        let mut intermediate_neurons = Vec::new();
        let mut deep_neurons = Vec::new();

        let mut id = 0;

        // Create superficial layer (visual input)
        for i in 0..map_height {
            for j in 0..map_width {
                let map_x = j as f32 / map_width as f32;
                let map_y = i as f32 / map_height as f32;
                superficial_neurons.push(SCNeuron::new(id, 0, map_x, map_y));
                id += 1;
            }
        }

        // Create intermediate layer (visuomotor)
        for i in 0..map_height {
            for j in 0..map_width {
                let map_x = j as f32 / map_width as f32;
                let map_y = i as f32 / map_height as f32;
                intermediate_neurons.push(SCNeuron::new(id, 1, map_x, map_y));
                id += 1;
            }
        }

        // Create deep layer (motor output)
        for i in 0..map_height {
            for j in 0..map_width {
                let map_x = j as f32 / map_width as f32;
                let map_y = i as f32 / map_height as f32;
                deep_neurons.push(SCNeuron::new(id, 2, map_x, map_y));
                id += 1;
            }
        }

        // Initialize inhibition of return map
        let ior_map = vec![vec![0.0; map_width]; map_height];

        Self {
            superficial_neurons,
            intermediate_neurons,
            deep_neurons,
            map_width,
            map_height,
            omnipause_activity: 1.0, // Active during fixation
            fixation_mode: true,
            saccade_active: false,
            saccade_vector: (0.0, 0.0),
            saccade_progress: 0.0,
            ior_map,
            ior_decay: 0.995,
            total_saccades: 0,
            avg_saccade_amplitude: 0.0,
        }
    }

    /// Process visual input
    ///
    /// # Arguments
    /// - `visual_x`: Visual stimulus x position (degrees)
    /// - `visual_y`: Visual stimulus y position (degrees)
    /// - `salience`: Stimulus salience (0-1)
    pub fn process_visual_input(&mut self, visual_x: f32, visual_y: f32, salience: f32) {
        // Activate superficial layer neurons based on RF overlap
        for neuron in &mut self.superficial_neurons {
            let rf_response = neuron.in_receptive_field(visual_x, visual_y);
            let input = rf_response * salience * 100.0;
            neuron.update(input, 0.1);
        }
    }

    /// Update visuomotor transformation
    ///
    /// # Arguments
    /// - `dt`: Timestep (ms)
    pub fn update(&mut self, dt: f32) {
        // Update omnipause neurons
        if self.fixation_mode {
            self.omnipause_activity += (1.0 - self.omnipause_activity) * 0.1;
        } else {
            self.omnipause_activity *= 0.9;
        }

        // Superficial → Intermediate feedforward
        for i in 0..self.intermediate_neurons.len() {
            let input = self.superficial_neurons[i].activity * 0.5;
            self.intermediate_neurons[i].update(input, dt);
        }

        // Intermediate → Deep with IOR modulation
        for i in 0..self.deep_neurons.len() {
            let row = i / self.map_width;
            let col = i % self.map_width;

            let ior_suppression = self.ior_map[row][col];
            let input = self.intermediate_neurons[i].activity
                * (1.0 - ior_suppression)
                * (1.0 - self.omnipause_activity);

            self.deep_neurons[i].update(input, dt);
        }

        // Decay IOR map
        for row in &mut self.ior_map {
            for val in row {
                *val *= self.ior_decay;
            }
        }

        // Update saccade if active
        if self.saccade_active {
            self.saccade_progress += 0.02; // Saccade duration ~50ms

            if self.saccade_progress >= 1.0 {
                self.saccade_active = false;
                self.saccade_progress = 0.0;
                self.fixation_mode = true;

                // Add IOR at saccade target
                self.add_inhibition_of_return(self.saccade_vector.0, self.saccade_vector.1);
            }
        }
    }

    /// Initiate saccade to target
    ///
    /// # Arguments
    /// - `target_x`: Target x position (degrees)
    /// - `target_y`: Target y position (degrees)
    pub fn initiate_saccade(&mut self, target_x: f32, target_y: f32) {
        if self.saccade_active {
            return; // Can't initiate during active saccade
        }

        // Calculate saccade vector
        let amplitude = (target_x * target_x + target_y * target_y).sqrt();
        let direction = target_y.atan2(target_x);

        self.saccade_vector = (amplitude, direction);
        self.saccade_active = true;
        self.saccade_progress = 0.0;
        self.fixation_mode = false;

        // Update statistics
        self.total_saccades += 1;
        self.avg_saccade_amplitude =
            (self.avg_saccade_amplitude * (self.total_saccades - 1) as f32 + amplitude)
                / self.total_saccades as f32;
    }

    /// Trigger saccade from peak deep layer activity
    pub fn trigger_saccade_from_activity(&mut self) -> Option<(f32, f32)> {
        if !self.fixation_mode || self.saccade_active {
            return None;
        }

        // Find peak activity in deep layer
        let mut max_activity = 0.0;
        let mut max_idx = 0;

        for (i, neuron) in self.deep_neurons.iter().enumerate() {
            if neuron.activity > max_activity {
                max_activity = neuron.activity;
                max_idx = i;
            }
        }

        // Threshold for saccade initiation
        if max_activity > 30.0 {
            let rf_x = self.deep_neurons[max_idx].rf_x;
            let rf_y = self.deep_neurons[max_idx].rf_y;
            self.initiate_saccade(rf_x, rf_y);
            Some((rf_x, rf_y))
        } else {
            None
        }
    }

    /// Add inhibition of return at location
    fn add_inhibition_of_return(&mut self, x: f32, y: f32) {
        let k = 10.0;

        for i in 0..self.map_height {
            for j in 0..self.map_width {
                let map_x = j as f32 / self.map_width as f32;
                let map_y = i as f32 / self.map_height as f32;

                // Convert to RF coordinates
                let x_centered = (map_x - 0.5) * 2.0;
                let y_centered = (map_y - 0.5) * 2.0;
                let r = k
                    * (std::f32::consts::E
                        + (x_centered * x_centered + y_centered * y_centered).sqrt())
                    .ln();
                let theta = y_centered.atan2(x_centered);
                let rf_x = r * theta.cos();
                let rf_y = r * theta.sin();

                // Distance to saccade target
                let dx = rf_x - x;
                let dy = rf_y - y;
                let distance = (dx * dx + dy * dy).sqrt();

                // Add IOR with Gaussian profile
                let ior_strength = 0.8 * (-(distance * distance) / 50.0).exp();
                self.ior_map[i][j] = (self.ior_map[i][j] + ior_strength).min(1.0);
            }
        }
    }

    /// Get current gaze position (eye position)
    pub fn get_gaze_position(&self) -> (f32, f32) {
        if self.saccade_active {
            // Interpolate during saccade
            let (amplitude, direction) = self.saccade_vector;
            let current_amplitude = amplitude * self.saccade_progress;
            (
                current_amplitude * direction.cos(),
                current_amplitude * direction.sin(),
            )
        } else {
            (0.0, 0.0) // Fixation at center
        }
    }

    /// Get statistics
    pub fn stats(&self) -> SCStats {
        let total_neurons = self.superficial_neurons.len()
            + self.intermediate_neurons.len()
            + self.deep_neurons.len();

        let superficial_activity = self
            .superficial_neurons
            .iter()
            .map(|n| n.activity)
            .sum::<f32>()
            / self.superficial_neurons.len() as f32;
        let deep_activity = self.deep_neurons.iter().map(|n| n.activity).sum::<f32>()
            / self.deep_neurons.len() as f32;

        SCStats {
            total_neurons,
            superficial_activity,
            deep_activity,
            omnipause_activity: self.omnipause_activity,
            saccade_active: self.saccade_active,
            total_saccades: self.total_saccades,
            avg_saccade_amplitude: self.avg_saccade_amplitude,
        }
    }
}

/// Superior Colliculus statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SCStats {
    pub total_neurons: usize,
    pub superficial_activity: f32,
    pub deep_activity: f32,
    pub omnipause_activity: f32,
    pub saccade_active: bool,
    pub total_saccades: u64,
    pub avg_saccade_amplitude: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sc_creation() {
        let sc = SuperiorColliculus::new(16, 16);

        assert_eq!(sc.superficial_neurons.len(), 256);
        assert_eq!(sc.intermediate_neurons.len(), 256);
        assert_eq!(sc.deep_neurons.len(), 256);
        assert!(sc.fixation_mode);
    }

    #[test]
    fn test_visual_input_processing() {
        let mut sc = SuperiorColliculus::new(16, 16);

        // Present visual stimulus
        sc.process_visual_input(5.0, 5.0, 1.0);

        // Some superficial neurons should respond
        let max_activity = sc
            .superficial_neurons
            .iter()
            .map(|n| n.activity)
            .fold(0.0, f32::max);
        assert!(max_activity > 0.0, "Visual input should activate neurons");
    }

    #[test]
    fn test_saccade_initiation() {
        let mut sc = SuperiorColliculus::new(16, 16);

        // Initiate saccade
        sc.initiate_saccade(10.0, 5.0);

        assert!(sc.saccade_active, "Saccade should be active");
        assert!(!sc.fixation_mode, "Should not be in fixation mode");
        assert_eq!(sc.total_saccades, 1);
    }

    #[test]
    fn test_saccade_completion() {
        let mut sc = SuperiorColliculus::new(16, 16);

        sc.initiate_saccade(10.0, 5.0);

        // Update until saccade completes
        for _ in 0..100 {
            sc.update(0.1);
        }

        assert!(!sc.saccade_active, "Saccade should complete");
        assert!(sc.fixation_mode, "Should return to fixation");
    }

    #[test]
    fn test_inhibition_of_return() {
        let mut sc = SuperiorColliculus::new(16, 16);

        // Saccade to location
        sc.initiate_saccade(5.0, 5.0);

        // Complete saccade
        for _ in 0..100 {
            sc.update(0.1);
        }

        // IOR map should have some inhibition
        let total_ior: f32 = sc.ior_map.iter().flatten().sum();
        assert!(total_ior > 0.0, "IOR should be present after saccade");
    }

    #[test]
    fn test_omnipause_neurons() {
        let mut sc = SuperiorColliculus::new(16, 16);

        // In fixation mode, omnipause should be active
        assert!(sc.omnipause_activity > 0.5);

        // During saccade, omnipause should decrease
        sc.initiate_saccade(10.0, 0.0);
        for _ in 0..10 {
            sc.update(0.1);
        }

        assert!(
            sc.omnipause_activity < 0.5,
            "Omnipause should decrease during saccade"
        );
    }

    #[test]
    fn test_receptive_field() {
        let neuron = SCNeuron::new(0, 0, 0.5, 0.5);

        // Stimulus at RF center should give strong response
        let response_center = neuron.in_receptive_field(neuron.rf_x, neuron.rf_y);
        assert!(response_center > 0.9, "RF center should respond strongly");

        // Distant stimulus should give weak response
        let response_far = neuron.in_receptive_field(neuron.rf_x + 50.0, neuron.rf_y + 50.0);
        assert!(response_far < 0.1, "Distant stimulus should respond weakly");
    }

    #[test]
    fn test_gaze_position() {
        let mut sc = SuperiorColliculus::new(16, 16);

        // Initially at fixation
        let (x, y) = sc.get_gaze_position();
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);

        // During saccade, gaze should move
        sc.initiate_saccade(10.0, 0.0);
        sc.update(0.1);

        let (x2, y2) = sc.get_gaze_position();
        assert!(x2 > 0.0, "Gaze should move during saccade");
    }
}
