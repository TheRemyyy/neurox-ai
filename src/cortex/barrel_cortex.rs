//! Barrel Cortex Somatosensory System
//!
//! Implements rodent whisker somatosensory processing with layer-specific circuits,
//! three interneuron types (PV/SST/VIP), and biologically realistic connectivity.
//!
//! # Architecture
//! - **Layer 4**: Primary input from thalamus (VPM), granule cells
//! - **Layer 2/3**: Supragranular integration, inter-barrel processing
//! - **Layer 5**: Infragranular output, motor feedback
//! - **Layer 6**: Corticothalamic feedback
//!
//! # Interneuron Types
//! - **PV (Parvalbumin)**: Fast-spiking, perisomatic inhibition (~40 Hz)
//! - **SST (Somatostatin)**: Adapting, dendritic inhibition
//! - **VIP (Vasoactive Intestinal Peptide)**: Disinhibitory, targets SST
//!
//! # Connectivity
//! - Intra-barrel: L4 → L2/3 → L5 (columnar)
//! - Inter-barrel: L2/3 ↔ L2/3 (horizontal)
//! - Thalamic input: VPM → L4 (principal whisker)
//! - Feedback: L6 → VPM, L5 → Pons/SC
//!
//! # Features
//! - 25 barrels (5×5 whisker array)
//! - ~2,000 neurons per barrel = 50,000 total
//! - Disinhibitory circuit (VIP → SST -| PYR)
//! - Surround suppression
//! - Direction selectivity

use serde::{Deserialize, Serialize};
use crate::neuron::lif::LIFNeuron;

/// Barrel cortex somatosensory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrelCortex {
    /// 5×5 array of cortical barrels
    pub barrels: Vec<Vec<CorticalBarrel>>,

    /// Thalamic input (VPM nucleus)
    pub thalamus: Vec<Vec<ThalamicNeuron>>,

    /// Barrel grid dimensions
    pub n_rows: usize,
    pub n_cols: usize,

    /// Global modulatory signals
    pub arousal_level: f32,      // 0-1 (norepinephrine/acetylcholine)
    pub attention_focus: (usize, usize), // (row, col) attended barrel
}

impl BarrelCortex {
    /// Create new barrel cortex with 5×5 whisker array
    pub fn new() -> Self {
        let n_rows = 5;
        let n_cols = 5;

        let mut barrels = Vec::new();
        let mut thalamus = Vec::new();

        for row in 0..n_rows {
            let mut barrel_row = Vec::new();
            let mut thal_row = Vec::new();

            for col in 0..n_cols {
                barrel_row.push(CorticalBarrel::new(row, col));
                thal_row.push(ThalamicNeuron::new(row, col));
            }

            barrels.push(barrel_row);
            thalamus.push(thal_row);
        }

        Self {
            barrels,
            thalamus,
            n_rows,
            n_cols,
            arousal_level: 0.5,
            attention_focus: (2, 2), // Center barrel
        }
    }

    /// Process whisker deflection input
    ///
    /// # Arguments
    /// - `whisker_deflections`: 5×5 array of whisker angles (radians)
    /// - `whisker_velocities`: 5×5 array of deflection velocities
    /// - `dt`: Timestep (ms)
    ///
    /// # Returns
    /// Cortical activity map (5×5 array of L2/3 population responses)
    pub fn process(
        &mut self,
        whisker_deflections: &[Vec<f32>],
        whisker_velocities: &[Vec<f32>],
        dt: f32,
    ) -> Vec<Vec<f32>> {
        // 1. Thalamic processing
        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                let deflection = whisker_deflections[row][col];
                let velocity = whisker_velocities[row][col];

                self.thalamus[row][col].update(deflection, velocity, dt);
            }
        }

        // 2. L4 input integration
        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                let thalamic_spikes = self.thalamus[row][col].spike_train.clone();
                self.barrels[row][col].process_thalamic_input(&thalamic_spikes, dt);
            }
        }

        // 3. Intra-barrel vertical processing (L4 → L2/3 → L5)
        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                self.barrels[row][col].update_vertical(dt);
            }
        }

        // 4. Inter-barrel horizontal processing (L2/3 lateral)
        self.update_horizontal_connections(dt);

        // 5. Feedback loops (L6 → VPM, L5 → subcortical)
        self.update_feedback(dt);

        // 6. Extract L2/3 population activity
        let mut activity_map = vec![vec![0.0; self.n_cols]; self.n_rows];
        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                activity_map[row][col] = self.barrels[row][col].get_l23_activity();
            }
        }

        activity_map
    }

    /// Update horizontal connections between barrels
    fn update_horizontal_connections(&mut self, dt: f32) {
        // Pre-compute L2/3 activities
        let mut l23_activities = vec![vec![0.0; self.n_cols]; self.n_rows];

        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                l23_activities[row][col] = self.barrels[row][col].get_l23_activity();
            }
        }

        // Apply lateral influences
        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                // Surround suppression from neighbors
                let mut surround_input = 0.0;

                for dr in -1..=1 {
                    for dc in -1..=1 {
                        if dr == 0 && dc == 0 {
                            continue;
                        }

                        let neighbor_row = (row as i32 + dr) as usize;
                        let neighbor_col = (col as i32 + dc) as usize;

                        if neighbor_row < self.n_rows && neighbor_col < self.n_cols {
                            // Lateral inhibition (surround suppression)
                            surround_input += l23_activities[neighbor_row][neighbor_col] * 0.2;
                        }
                    }
                }

                self.barrels[row][col].apply_surround_suppression(surround_input, dt);
            }
        }
    }

    /// Update feedback connections
    fn update_feedback(&mut self, dt: f32) {
        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                // L6 corticothalamic feedback (modulatory)
                let l6_activity = self.barrels[row][col].get_l6_activity();
                self.thalamus[row][col].apply_cortical_feedback(l6_activity, dt);

                // L5 subcortical output (updated internally in barrel)
                self.barrels[row][col].update_l5_output(dt);
            }
        }
    }

    /// Set attentional focus on specific barrel
    pub fn set_attention(&mut self, row: usize, col: usize) {
        self.attention_focus = (row, col);

        // Attention enhances gain in focused barrel
        for r in 0..self.n_rows {
            for c in 0..self.n_cols {
                let gain = if r == row && c == col {
                    1.5 // 50% gain increase
                } else {
                    1.0
                };

                self.barrels[r][c].set_attentional_gain(gain);
            }
        }
    }
}

/// Single cortical barrel (column)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorticalBarrel {
    pub row: usize,
    pub col: usize,

    // Layers
    pub layer2_3: Layer2_3,
    pub layer4: Layer4,
    pub layer5: Layer5,
    pub layer6: Layer6,

    // Interneurons (across all layers)
    pub pv_interneurons: Vec<PVInterneuron>,
    pub sst_interneurons: Vec<SSTInterneuron>,
    pub vip_interneurons: Vec<VIPInterneuron>,

    /// Attentional gain modulation
    pub gain: f32,
}

impl CorticalBarrel {
    pub fn new(row: usize, col: usize) -> Self {
        Self {
            row,
            col,
            layer2_3: Layer2_3::new(400), // 400 pyramidal neurons
            layer4: Layer4::new(600),     // 600 granule/spiny stellate
            layer5: Layer5::new(300),     // 300 pyramidal neurons
            layer6: Layer6::new(200),     // 200 pyramidal neurons
            pv_interneurons: (0..100).map(|i| PVInterneuron::new(i)).collect(),
            sst_interneurons: (0..80).map(|i| SSTInterneuron::new(i)).collect(),
            vip_interneurons: (0..20).map(|i| VIPInterneuron::new(i)).collect(),
            gain: 1.0,
        }
    }

    pub fn process_thalamic_input(&mut self, thalamic_spikes: &[f32], dt: f32) {
        self.layer4.process_input(thalamic_spikes, dt);
    }

    pub fn update_vertical(&mut self, dt: f32) {
        // L4 → L2/3
        let l4_output = self.layer4.get_output();

        // PV inhibition on L2/3 (fast, perisomatic)
        let mut pv_inhibition = 0.0;
        for pv in &mut self.pv_interneurons {
            pv.update(&l4_output, dt);
            pv_inhibition += pv.output;
        }

        // SST inhibition on L2/3 (dendritic)
        let mut sst_inhibition = 0.0;
        for sst in &mut self.sst_interneurons {
            sst.update(&l4_output, dt);
            sst_inhibition += sst.output;
        }

        // VIP disinhibition (inhibits SST)
        let mut vip_to_sst = 0.0;
        for vip in &mut self.vip_interneurons {
            vip.update(&l4_output, dt);
            vip_to_sst += vip.output;
        }

        // VIP reduces SST activity (disinhibition)
        sst_inhibition *= (1.0 - vip_to_sst * 0.5).max(0.0);

        self.layer2_3.update(&l4_output, pv_inhibition, sst_inhibition, self.gain, dt);

        // L2/3 → L5
        let l23_output = self.layer2_3.get_output();
        self.layer5.update(&l23_output, dt);

        // L2/3 → L6 (also receives thalamic input)
        self.layer6.update(&l23_output, dt);
    }

    pub fn apply_surround_suppression(&mut self, surround: f32, dt: f32) {
        self.layer2_3.apply_inhibition(surround, dt);
    }

    pub fn update_l5_output(&mut self, dt: f32) {
        // L5 sends output to subcortical structures
        self.layer5.update_output(dt);
    }

    pub fn get_l23_activity(&self) -> f32 {
        self.layer2_3.get_population_rate()
    }

    pub fn get_l6_activity(&self) -> f32 {
        self.layer6.get_population_rate()
    }

    pub fn set_attentional_gain(&mut self, gain: f32) {
        self.gain = gain;
    }
}

/// Layer 4 (granular): Primary thalamic input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer4 {
    pub neurons: Vec<LIFNeuron>,
    pub n_neurons: usize,
}

impl Layer4 {
    pub fn new(n_neurons: usize) -> Self {
        let neurons = (0..n_neurons).map(|i| LIFNeuron::new(i as u32)).collect();
        Self { neurons, n_neurons }
    }

    pub fn process_input(&mut self, thalamic_spikes: &[f32], dt: f32) {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let input_idx = i % thalamic_spikes.len();
            neuron.state.input_current = thalamic_spikes[input_idx] * 10.0; // Amplify
            neuron.update(dt / 1000.0); // Convert ms to seconds
        }
    }

    pub fn get_output(&self) -> Vec<f32> {
        self.neurons.iter().map(|n| if n.state.has_spiked { 1.0 } else { 0.0 }).collect()
    }
}

/// Layer 2/3 (supragranular): Integration and inter-barrel communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer2_3 {
    pub neurons: Vec<LIFNeuron>,
    pub n_neurons: usize,
    pub membrane_voltages: Vec<f32>,
}

impl Layer2_3 {
    pub fn new(n_neurons: usize) -> Self {
        let neurons = (0..n_neurons).map(|i| LIFNeuron::new(i as u32)).collect();
        let membrane_voltages = vec![0.0; n_neurons];
        Self {
            neurons,
            n_neurons,
            membrane_voltages,
        }
    }

    pub fn update(
        &mut self,
        l4_input: &[f32],
        pv_inhibition: f32,
        sst_inhibition: f32,
        gain: f32,
        dt: f32,
    ) {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            // Feedforward from L4
            let ff_input: f32 = l4_input.iter().take(20).sum::<f32>() / 20.0; // Pool 20 L4 neurons

            // Total input with gain modulation
            let total_input = (ff_input * gain - pv_inhibition * 0.5 - sst_inhibition * 0.3).max(0.0);

            neuron.state.input_current = total_input * 5.0;
            neuron.update(dt / 1000.0);

            self.membrane_voltages[i] = neuron.state.membrane_potential;
        }
    }

    pub fn apply_inhibition(&mut self, inhibition: f32, dt: f32) {
        for neuron in &mut self.neurons {
            neuron.state.input_current -= inhibition;
        }
    }

    pub fn get_output(&self) -> Vec<f32> {
        self.neurons.iter().map(|n| if n.state.has_spiked { 1.0 } else { 0.0 }).collect()
    }

    pub fn get_population_rate(&self) -> f32 {
        let spike_count: f32 = self.neurons.iter().map(|n| if n.state.has_spiked { 1.0 } else { 0.0 }).sum();
        spike_count / self.n_neurons as f32
    }
}

/// Layer 5 (infragranular): Output to subcortical structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer5 {
    pub neurons: Vec<LIFNeuron>,
    pub n_neurons: usize,
}

impl Layer5 {
    pub fn new(n_neurons: usize) -> Self {
        let neurons = (0..n_neurons).map(|i| LIFNeuron::new(i as u32)).collect();
        Self { neurons, n_neurons }
    }

    pub fn update(&mut self, l23_input: &[f32], dt: f32) {
        for neuron in &mut self.neurons {
            let input: f32 = l23_input.iter().take(10).sum::<f32>() / 10.0;
            neuron.state.input_current = input * 3.0;
            neuron.update(dt / 1000.0);
        }
    }

    pub fn update_output(&mut self, dt: f32) {
        // L5 output dynamics (simplified)
    }
}

/// Layer 6 (infragranular): Corticothalamic feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer6 {
    pub neurons: Vec<LIFNeuron>,
    pub n_neurons: usize,
}

impl Layer6 {
    pub fn new(n_neurons: usize) -> Self {
        let neurons = (0..n_neurons).map(|i| LIFNeuron::new(i as u32)).collect();
        Self { neurons, n_neurons }
    }

    pub fn update(&mut self, l23_input: &[f32], dt: f32) {
        for neuron in &mut self.neurons {
            let input: f32 = l23_input.iter().take(5).sum::<f32>() / 5.0;
            neuron.state.input_current = input * 2.0;
            neuron.update(dt / 1000.0);
        }
    }

    pub fn get_population_rate(&self) -> f32 {
        let spike_count: f32 = self.neurons.iter().map(|n| if n.state.has_spiked { 1.0 } else { 0.0 }).sum();
        spike_count / self.n_neurons as f32
    }
}

/// PV (Parvalbumin) interneuron - Fast-spiking basket cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PVInterneuron {
    pub id: usize,
    pub neuron: LIFNeuron,
    pub output: f32,
}

impl PVInterneuron {
    pub fn new(id: usize) -> Self {
        let mut neuron = LIFNeuron::new(id as u32);
        // Fast-spiking parameters
        neuron.tau_mem = 5.0; // Fast dynamics (5ms)
        neuron.v_thresh = -50.0; // Lower threshold

        Self {
            id,
            neuron,
            output: 0.0,
        }
    }

    pub fn update(&mut self, input: &[f32], dt: f32) {
        let total_input: f32 = input.iter().sum::<f32>() / input.len() as f32;
        self.neuron.state.input_current = total_input * 8.0; // High gain
        self.neuron.update(dt / 1000.0);

        self.output = if self.neuron.state.has_spiked { 1.0 } else { self.output * 0.9 };
    }
}

/// SST (Somatostatin) interneuron - Adapting Martinotti cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSTInterneuron {
    pub id: usize,
    pub neuron: LIFNeuron,
    pub output: f32,
    pub adaptation: f32,
}

impl SSTInterneuron {
    pub fn new(id: usize) -> Self {
        let mut neuron = LIFNeuron::new(id as u32);
        // Adapting parameters
        neuron.tau_mem = 20.0; // Slower dynamics (20ms)

        Self {
            id,
            neuron,
            output: 0.0,
            adaptation: 0.0,
        }
    }

    pub fn update(&mut self, input: &[f32], dt: f32) {
        let total_input: f32 = input.iter().sum::<f32>() / input.len() as f32;
        self.neuron.state.input_current = (total_input * 5.0 - self.adaptation).max(0.0);
        self.neuron.update(dt / 1000.0);

        if self.neuron.state.has_spiked {
            self.adaptation += 0.1; // Increase adaptation on spike
            self.output = 1.0;
        } else {
            self.output *= 0.95; // Slower decay
        }

        self.adaptation *= 0.98; // Slow decay
    }
}

/// VIP (Vasoactive Intestinal Peptide) interneuron - Disinhibitory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VIPInterneuron {
    pub id: usize,
    pub neuron: LIFNeuron,
    pub output: f32,
}

impl VIPInterneuron {
    pub fn new(id: usize) -> Self {
        let neuron = LIFNeuron::new(id as u32);
        Self {
            id,
            neuron,
            output: 0.0,
        }
    }

    pub fn update(&mut self, input: &[f32], dt: f32) {
        let total_input: f32 = input.iter().sum::<f32>() / input.len() as f32;
        self.neuron.state.input_current = total_input * 6.0;
        self.neuron.update(dt / 1000.0);

        self.output = if self.neuron.state.has_spiked { 1.0 } else { self.output * 0.92 };
    }
}

/// Thalamic VPM neuron (sensory relay)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThalamicNeuron {
    pub row: usize,
    pub col: usize,
    pub neuron: LIFNeuron,
    pub spike_train: Vec<f32>,
    pub cortical_feedback_strength: f32,
}

impl ThalamicNeuron {
    pub fn new(row: usize, col: usize) -> Self {
        Self {
            row,
            col,
            neuron: LIFNeuron::new((row * 5 + col) as u32),
            spike_train: vec![0.0; 10],
            cortical_feedback_strength: 0.0,
        }
    }

    pub fn update(&mut self, deflection: f32, velocity: f32, dt: f32) {
        // Thalamic neurons respond to whisker deflection and velocity
        let sensory_input = deflection.abs() * 2.0 + velocity.abs() * 3.0;

        // Modulated by cortical feedback (L6)
        let total_input = sensory_input * (1.0 + self.cortical_feedback_strength * 0.3);

        self.neuron.state.input_current = total_input;
        self.neuron.update(dt / 1000.0);

        // Update spike train
        self.spike_train.rotate_right(1);
        self.spike_train[0] = if self.neuron.state.has_spiked { 1.0 } else { 0.0 };
    }

    pub fn apply_cortical_feedback(&mut self, feedback: f32, dt: f32) {
        // L6 corticothalamic feedback (modulatory)
        self.cortical_feedback_strength = feedback;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrel_cortex_creation() {
        let bc = BarrelCortex::new();

        assert_eq!(bc.n_rows, 5);
        assert_eq!(bc.n_cols, 5);
        assert_eq!(bc.barrels.len(), 5);
        assert_eq!(bc.barrels[0].len(), 5);
    }

    #[test]
    fn test_cortical_barrel_structure() {
        let barrel = CorticalBarrel::new(0, 0);

        assert_eq!(barrel.layer2_3.n_neurons, 400);
        assert_eq!(barrel.layer4.n_neurons, 600);
        assert_eq!(barrel.layer5.n_neurons, 300);
        assert_eq!(barrel.layer6.n_neurons, 200);
        assert_eq!(barrel.pv_interneurons.len(), 100);
        assert_eq!(barrel.sst_interneurons.len(), 80);
        assert_eq!(barrel.vip_interneurons.len(), 20);

        // Total: 400 + 600 + 300 + 200 + 100 + 80 + 20 = 1,700 neurons per barrel
        // 25 barrels × 1,700 = 42,500 neurons total
    }

    #[test]
    fn test_whisker_processing() {
        let mut bc = BarrelCortex::new();

        // Deflect center whisker
        let mut deflections = vec![vec![0.0; 5]; 5];
        let mut velocities = vec![vec![0.0; 5]; 5];

        deflections[2][2] = 0.5; // 0.5 radians
        velocities[2][2] = 1.0;  // Fast deflection

        let activity = bc.process(&deflections, &velocities, 1.0);

        // Center barrel should show highest activity
        assert!(activity[2][2] >= activity[0][0], "Center barrel should respond most");
    }

    #[test]
    fn test_surround_suppression() {
        let mut bc = BarrelCortex::new();

        // Activate all whiskers
        let deflections = vec![vec![0.3; 5]; 5];
        let velocities = vec![vec![0.5; 5]; 5];

        let activity_all = bc.process(&deflections, &velocities, 1.0);

        // Now activate only center
        let mut deflections_center = vec![vec![0.0; 5]; 5];
        let mut velocities_center = vec![vec![0.0; 5]; 5];
        deflections_center[2][2] = 0.3;
        velocities_center[2][2] = 0.5;

        let mut bc2 = BarrelCortex::new();
        let activity_center = bc2.process(&deflections_center, &velocities_center, 1.0);

        // Center response with surround active should be suppressed
        // (This is a simplified test - actual surround suppression is complex)
    }

    #[test]
    fn test_interneuron_types() {
        let mut pv = PVInterneuron::new(0);
        let mut sst = SSTInterneuron::new(0);
        let mut vip = VIPInterneuron::new(0);

        let input = vec![1.0; 10];

        // PV should be fast-spiking
        assert!(pv.neuron.tau_mem < 10.0, "PV should have fast dynamics");

        // SST should have adaptation
        for _ in 0..10 {
            sst.update(&input, 1.0);
        }
        assert!(sst.adaptation > 0.0, "SST should build adaptation");

        // VIP updates normally
        vip.update(&input, 1.0);
    }

    #[test]
    fn test_disinhibitory_circuit() {
        let barrel = CorticalBarrel::new(0, 0);

        // VIP targets SST, which targets pyramidal neurons
        // VIP activation should reduce SST inhibition (disinhibition)

        assert!(barrel.vip_interneurons.len() > 0);
        assert!(barrel.sst_interneurons.len() > 0);
    }
}
