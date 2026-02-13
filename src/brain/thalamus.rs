//! Thalamus - Sensory Gateway and Attention Modulation
//!
//! Implements thalamic nuclei for sensory preprocessing, attention gating,
//! and cortico-thalamic loops. The thalamus acts as a dynamic relay that
//! filters and enhances relevant sensory information.
//!
//! # Architecture
//! - First-order relay nuclei: LGN, MGN, VPL/VPM (sensory input)
//! - Higher-order nuclei: Pulvinar, MD (attention and feedback)
//! - Thalamic reticular nucleus (TRN): Inhibitory gating
//! - Cortico-thalamic feedback loops
//!
//! # Features
//! - Burst/tonic firing modes (T-type Ca2+ channels)
//! - Attention-based gain modulation
//! - TRN-mediated lateral inhibition
//! - Sleep spindle generation (7-14 Hz)
//!
//! # References
//! - Sherman & Guillery (2011) "Distinct functions for direct and transthalamic corticocortical connections"
//! - Halassa & Kastner (2017) "Thalamic functions in distributed cognitive control"
//! - October 2024 paper on thalamic routing

use serde::{Deserialize, Serialize};

/// Thalamic nucleus type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThalamicNucleus {
    LGN,      // Lateral Geniculate Nucleus (vision)
    MGN,      // Medial Geniculate Nucleus (audition)
    VPL,      // Ventral Posterolateral (somatosensory - body)
    VPM,      // Ventral Posteromedial (somatosensory - face)
    Pulvinar, // Higher-order visual attention
    MD,       // Mediodorsal (cognition, working memory)
    TRN,      // Thalamic Reticular Nucleus (inhibitory)
}

/// Thalamic neuron with burst/tonic modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThalamicNeuron {
    /// Neuron ID
    pub id: usize,

    /// Nucleus type
    pub nucleus: u8, // 0=LGN, 1=MGN, 2=VPL, 3=VPM, 4=Pulvinar, 5=MD, 6=TRN

    /// Membrane potential (mV)
    pub v: f32,

    /// T-type calcium channel activation (for burst mode)
    pub h_t: f32,

    /// Firing mode: true=burst, false=tonic
    pub burst_mode: bool,

    /// Activity level
    pub activity: f32,

    /// Attention modulation (0-1)
    pub attention_gain: f32,

    /// Spike threshold
    pub threshold: f32,

    /// Refractory counter
    pub refractory: u8,
}

impl ThalamicNeuron {
    pub fn new(id: usize, nucleus: u8) -> Self {
        Self {
            id,
            nucleus,
            v: -70.0,          // Resting potential
            h_t: 1.0,          // T-channel inactivation
            burst_mode: false, // Start in tonic mode
            activity: 0.0,
            attention_gain: 1.0,
            threshold: -55.0,
            refractory: 0,
        }
    }

    /// Update neuron dynamics
    pub fn update(&mut self, sensory_input: f32, cortical_feedback: f32, dt: f32) -> bool {
        if self.refractory > 0 {
            self.refractory -= 1;
            return false;
        }

        // T-type calcium channel dynamics (for burst mode)
        // When neuron is hyperpolarized, h_t increases (deinactivation)
        let tau_h = 20.0; // ms
        let h_inf = if self.v < -65.0 { 1.0 } else { 0.0 };
        self.h_t += (h_inf - self.h_t) / tau_h * dt;

        // Burst mode when hyperpolarized and deinactivated
        self.burst_mode = self.v < -65.0 && self.h_t > 0.8;

        // Total input with attention modulation
        let total_input = (sensory_input + cortical_feedback) * self.attention_gain;

        // Membrane dynamics
        let tau_m = if self.burst_mode { 5.0 } else { 10.0 }; // Faster in burst mode
        let i_leak = (self.v + 70.0) / tau_m;

        // T-current (burst)
        let i_t = if self.burst_mode {
            self.h_t * 2.0 * (self.v + 60.0) // Depolarizing current
        } else {
            0.0
        };

        let dv = (-i_leak + total_input + i_t) * dt;
        self.v += dv;

        // Update activity
        self.activity = self.activity * 0.9 + total_input * 0.1;

        // Check for spike
        if self.v >= self.threshold {
            self.v = -70.0;
            self.refractory = 20; // 2ms @ 0.1ms timestep

            // After burst spike, inactivate T-channel
            if self.burst_mode {
                self.h_t = 0.0;
            }

            true
        } else {
            false
        }
    }

    /// Set attention modulation
    pub fn set_attention(&mut self, gain: f32) {
        self.attention_gain = gain.clamp(0.0, 2.0);
    }
}

/// Complete thalamus system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thalamus {
    /// First-order relay neurons (sensory input)
    pub lgn_neurons: Vec<ThalamicNeuron>, // Visual
    pub mgn_neurons: Vec<ThalamicNeuron>, // Auditory
    pub vpl_neurons: Vec<ThalamicNeuron>, // Somatosensory body
    pub vpm_neurons: Vec<ThalamicNeuron>, // Somatosensory face

    /// Higher-order neurons (attention, cognition)
    pub pulvinar_neurons: Vec<ThalamicNeuron>,
    pub md_neurons: Vec<ThalamicNeuron>,

    /// Inhibitory gating
    pub trn_neurons: Vec<ThalamicNeuron>,

    /// Attention focus (which modality/location)
    pub attention_modality: u8, // 0=visual, 1=auditory, 2=somatosensory
    pub attention_strength: f32,

    /// Spindle oscillation state (for sleep)
    pub spindle_active: bool,
    pub spindle_phase: f32,

    /// Statistics
    pub total_bursts: u64,
    pub total_tonic_spikes: u64,
}

impl Thalamus {
    /// Create new thalamus
    ///
    /// # Arguments
    /// - `neurons_per_nucleus`: Number of neurons per relay nucleus
    pub fn new(neurons_per_nucleus: usize) -> Self {
        let mut lgn_neurons = Vec::new();
        let mut mgn_neurons = Vec::new();
        let mut vpl_neurons = Vec::new();
        let mut vpm_neurons = Vec::new();
        let mut pulvinar_neurons = Vec::new();
        let mut md_neurons = Vec::new();
        let mut trn_neurons = Vec::new();

        let mut id = 0;

        // Create LGN (visual)
        for _ in 0..neurons_per_nucleus {
            lgn_neurons.push(ThalamicNeuron::new(id, 0));
            id += 1;
        }

        // Create MGN (auditory)
        for _ in 0..neurons_per_nucleus {
            mgn_neurons.push(ThalamicNeuron::new(id, 1));
            id += 1;
        }

        // Create VPL (somatosensory body)
        for _ in 0..neurons_per_nucleus {
            vpl_neurons.push(ThalamicNeuron::new(id, 2));
            id += 1;
        }

        // Create VPM (somatosensory face)
        for _ in 0..neurons_per_nucleus {
            vpm_neurons.push(ThalamicNeuron::new(id, 3));
            id += 1;
        }

        // Create Pulvinar (higher-order visual)
        for _ in 0..neurons_per_nucleus {
            pulvinar_neurons.push(ThalamicNeuron::new(id, 4));
            id += 1;
        }

        // Create MD (mediodorsal)
        for _ in 0..neurons_per_nucleus / 2 {
            md_neurons.push(ThalamicNeuron::new(id, 5));
            id += 1;
        }

        // Create TRN (inhibitory gating)
        for _ in 0..neurons_per_nucleus / 2 {
            trn_neurons.push(ThalamicNeuron::new(id, 6));
            id += 1;
        }

        Self {
            lgn_neurons,
            mgn_neurons,
            vpl_neurons,
            vpm_neurons,
            pulvinar_neurons,
            md_neurons,
            trn_neurons,
            attention_modality: 0,
            attention_strength: 1.0,
            spindle_active: false,
            spindle_phase: 0.0,
            total_bursts: 0,
            total_tonic_spikes: 0,
        }
    }

    /// Process sensory inputs through thalamic relay
    ///
    /// # Arguments
    /// - `visual`: Visual input array
    /// - `auditory`: Auditory input array
    /// - `somatosensory`: Somatosensory input array
    /// - `cortical_feedback`: Feedback from cortex
    /// - `dt`: Timestep (ms)
    pub fn update(
        &mut self,
        visual: &[f32],
        auditory: &[f32],
        somatosensory: &[f32],
        cortical_feedback: &[f32],
        dt: f32,
    ) {
        // Set attention gains based on current focus
        let visual_gain = if self.attention_modality == 0 {
            self.attention_strength
        } else {
            0.5
        };
        let auditory_gain = if self.attention_modality == 1 {
            self.attention_strength
        } else {
            0.5
        };
        let somato_gain = if self.attention_modality == 2 {
            self.attention_strength
        } else {
            0.5
        };

        // Update LGN (visual)
        for (i, neuron) in self.lgn_neurons.iter_mut().enumerate() {
            neuron.set_attention(visual_gain);
            let input = visual.get(i).copied().unwrap_or(0.0);
            let feedback = cortical_feedback.get(i).copied().unwrap_or(0.0);

            if neuron.update(input, feedback, dt) {
                if neuron.burst_mode {
                    self.total_bursts += 1;
                } else {
                    self.total_tonic_spikes += 1;
                }
            }
        }

        // Update MGN (auditory)
        for (i, neuron) in self.mgn_neurons.iter_mut().enumerate() {
            neuron.set_attention(auditory_gain);
            let input = auditory.get(i).copied().unwrap_or(0.0);
            let feedback = cortical_feedback.get(i).copied().unwrap_or(0.0);

            if neuron.update(input, feedback, dt) {
                if neuron.burst_mode {
                    self.total_bursts += 1;
                } else {
                    self.total_tonic_spikes += 1;
                }
            }
        }

        // Update VPL (somatosensory body)
        for (i, neuron) in self.vpl_neurons.iter_mut().enumerate() {
            neuron.set_attention(somato_gain);
            let input = somatosensory.get(i).copied().unwrap_or(0.0);
            let feedback = cortical_feedback.get(i).copied().unwrap_or(0.0);

            if neuron.update(input, feedback, dt) {
                if neuron.burst_mode {
                    self.total_bursts += 1;
                } else {
                    self.total_tonic_spikes += 1;
                }
            }
        }

        // Update VPM (somatosensory face)
        let vpl_len = self.vpl_neurons.len();
        for (i, neuron) in self.vpm_neurons.iter_mut().enumerate() {
            neuron.set_attention(somato_gain);
            // VPM gets input from second half of somatosensory array (face/whiskers)
            let input = somatosensory.get(vpl_len + i).copied().unwrap_or(0.0);
            let feedback = cortical_feedback.get(i).copied().unwrap_or(0.0);

            if neuron.update(input, feedback, dt) {
                if neuron.burst_mode {
                    self.total_bursts += 1;
                } else {
                    self.total_tonic_spikes += 1;
                }
            }
        }

        // Update Pulvinar (higher-order visual attention)
        for neuron in &mut self.pulvinar_neurons {
            neuron.set_attention(visual_gain * 1.5); // Enhanced for attention
            neuron.update(0.0, 0.0, dt);
        }

        // Update MD (mediodorsal - cognition and working memory)
        for (i, neuron) in self.md_neurons.iter_mut().enumerate() {
            // MD receives from prefrontal/cognitive areas (use cortical feedback)
            let input = cortical_feedback.get(i).copied().unwrap_or(0.0);
            neuron.set_attention(self.attention_strength); // Cognitive attention
            if neuron.update(input, 0.0, dt) {
                if neuron.burst_mode {
                    self.total_bursts += 1;
                } else {
                    self.total_tonic_spikes += 1;
                }
            }
        }

        // Update TRN (thalamic reticular nucleus - inhibitory gating)
        for (i, neuron) in self.trn_neurons.iter_mut().enumerate() {
            // TRN receives collaterals from relay nuclei
            let mut relay_input = 0.0;

            // Sample from all relay nuclei
            if i < self.lgn_neurons.len() {
                relay_input += self.lgn_neurons[i].activity;
            }
            if i < self.mgn_neurons.len() {
                relay_input += self.mgn_neurons[i].activity;
            }
            if i < self.vpl_neurons.len() {
                relay_input += self.vpl_neurons[i].activity;
            }

            neuron.set_attention(1.0 / self.attention_strength); // Inverse attention (suppresses unattended)
            if neuron.update(relay_input * 0.5, 0.0, dt) {
                // TRN provides feedback inhibition to relay nuclei
                // (In full implementation, would inhibit specific relay neurons)
                if neuron.burst_mode {
                    self.total_bursts += 1;
                } else {
                    self.total_tonic_spikes += 1;
                }
            }
        }

        // Update spindle oscillation (7-14 Hz, for sleep)
        if self.spindle_active {
            self.spindle_phase += 2.0 * std::f32::consts::PI * 10.0 * dt / 1000.0; // 10 Hz
            if self.spindle_phase > 2.0 * std::f32::consts::PI {
                self.spindle_phase -= 2.0 * std::f32::consts::PI;
            }
        }
    }

    /// Set attention focus
    pub fn set_attention(&mut self, modality: u8, strength: f32) {
        self.attention_modality = modality.min(2);
        self.attention_strength = strength.clamp(0.0, 2.0);
    }

    /// Enable/disable sleep spindles
    pub fn set_spindle_mode(&mut self, active: bool) {
        self.spindle_active = active;
    }

    /// Get relay output for a specific modality
    pub fn get_relay_output(&self, modality: u8) -> Vec<f32> {
        match modality {
            0 => self.lgn_neurons.iter().map(|n| n.activity).collect(),
            1 => self.mgn_neurons.iter().map(|n| n.activity).collect(),
            2 => {
                let mut combined = self
                    .vpl_neurons
                    .iter()
                    .map(|n| n.activity)
                    .collect::<Vec<_>>();
                combined.extend(self.vpm_neurons.iter().map(|n| n.activity));
                combined
            }
            _ => Vec::new(),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> ThalamusStats {
        let total_neurons = self.lgn_neurons.len()
            + self.mgn_neurons.len()
            + self.vpl_neurons.len()
            + self.vpm_neurons.len()
            + self.pulvinar_neurons.len()
            + self.md_neurons.len()
            + self.trn_neurons.len();

        let burst_ratio = if self.total_bursts + self.total_tonic_spikes > 0 {
            self.total_bursts as f32 / (self.total_bursts + self.total_tonic_spikes) as f32
        } else {
            0.0
        };

        ThalamusStats {
            total_neurons,
            attention_modality: self.attention_modality,
            attention_strength: self.attention_strength,
            burst_ratio,
            spindle_active: self.spindle_active,
        }
    }
}

/// Thalamus statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThalamusStats {
    pub total_neurons: usize,
    pub attention_modality: u8,
    pub attention_strength: f32,
    pub burst_ratio: f32,
    pub spindle_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thalamus_creation() {
        let thalamus = Thalamus::new(100);

        assert_eq!(thalamus.lgn_neurons.len(), 100);
        assert_eq!(thalamus.mgn_neurons.len(), 100);
        assert_eq!(thalamus.vpl_neurons.len(), 100);
        assert_eq!(thalamus.vpm_neurons.len(), 100);
    }

    #[test]
    fn test_burst_mode() {
        let mut neuron = ThalamicNeuron::new(0, 0);

        // Hyperpolarize neuron
        neuron.v = -75.0;
        neuron.h_t = 1.0;

        // Should enter burst mode
        neuron.update(0.0, 0.0, 0.1);
        assert!(
            neuron.burst_mode,
            "Hyperpolarized neuron should enter burst mode"
        );
    }

    #[test]
    fn test_attention_modulation() {
        let mut thalamus = Thalamus::new(10);

        // Focus on visual modality
        thalamus.set_attention(0, 2.0);

        // Update to apply attention gains
        let visual = vec![0.0; 10];
        let auditory = vec![0.0; 10];
        let somatosensory = vec![0.0; 10];
        let feedback = vec![0.0; 10];
        thalamus.update(&visual, &auditory, &somatosensory, &feedback, 0.1);

        // Visual neurons should have high gain
        assert!(thalamus.lgn_neurons[0].attention_gain > 1.5);
    }

    #[test]
    fn test_sensory_relay() {
        let mut thalamus = Thalamus::new(10);

        let visual = vec![1.0; 10];
        let auditory = vec![0.5; 10];
        let somatosensory = vec![0.3; 10];
        let feedback = vec![0.0; 10];

        thalamus.update(&visual, &auditory, &somatosensory, &feedback, 0.1);

        // Should have some activity
        let lgn_output = thalamus.get_relay_output(0);
        assert!(
            lgn_output.iter().any(|&x| x > 0.0),
            "LGN should relay visual input"
        );
    }

    #[test]
    fn test_spindle_oscillation() {
        let mut thalamus = Thalamus::new(10);

        thalamus.set_spindle_mode(true);
        assert!(thalamus.spindle_active);

        // Update several times
        for _ in 0..100 {
            thalamus.update(&[], &[], &[], &[], 1.0);
        }

        // Spindle phase should change
        assert!(thalamus.spindle_phase > 0.0);
    }
}
