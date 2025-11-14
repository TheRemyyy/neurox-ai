//! Neural Oscillations and Communication Through Coherence
//!
//! Implements rhythmic dynamics pervading real circuits:
//! - **Theta (4-8Hz)**: Organizes sequences and working memory
//! - **Gamma (30-100Hz)**: Segments processing, encoding/retrieval
//!   - Slow gamma (30-70Hz): Retrieval
//!   - Fast gamma (60-100Hz): Encoding
//! - **Alpha/Beta (8-20Hz)**: Top-down predictions and feedback
//! - **Cross-frequency coupling**: Theta phase modulates gamma amplitude
//!
//! # Communication Through Coherence
//! Synchronized regions communicate effectively (3ms windows of high excitability).
//! Non-coherent regions functionally disconnect despite anatomical connections.

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Theta oscillator (4-8Hz) for sequential organization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThetaOscillator {
    /// Frequency (Hz)
    pub frequency: f32,

    /// Current phase (0-1)
    pub phase: f32,

    /// Amplitude (0-1)
    pub amplitude: f32,

    /// Phase reset on events (for sequence segmentation)
    pub phase_reset_enabled: bool,
}

impl ThetaOscillator {
    pub fn new(frequency: f32) -> Self {
        Self {
            frequency,
            phase: 0.0,
            amplitude: 0.5,
            phase_reset_enabled: true,
        }
    }

    /// Update theta phase
    pub fn update(&mut self, dt: f32) {
        let delta_phase = self.frequency * dt / 1000.0;  // dt in ms
        self.phase += delta_phase;
        self.phase %= 1.0;
    }

    /// Get current theta value
    pub fn value(&self) -> f32 {
        self.amplitude * (self.phase * 2.0 * PI).sin()
    }

    /// Reset phase (e.g., at sequence boundaries)
    pub fn reset_phase(&mut self) {
        if self.phase_reset_enabled {
            self.phase = 0.0;
        }
    }

    /// Get phase (0-1)
    pub fn get_phase(&self) -> f32 {
        self.phase
    }

    /// Check if at peak (good time for encoding)
    pub fn at_peak(&self) -> bool {
        (self.phase - 0.25).abs() < 0.05  // Peak at phase=0.25
    }

    /// Check if at trough (good time for retrieval)
    pub fn at_trough(&self) -> bool {
        (self.phase - 0.75).abs() < 0.05  // Trough at phase=0.75
    }
}

/// Gamma oscillator (30-100Hz) for processing cycles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaOscillator {
    /// Frequency (Hz) - modulated by behavioral state
    pub frequency: f32,

    /// Current phase (0-1)
    pub phase: f32,

    /// Amplitude (modulated by theta)
    pub amplitude: f32,

    /// Baseline amplitude
    pub baseline_amplitude: f32,

    /// Gamma type (slow for retrieval, fast for encoding)
    pub gamma_type: GammaType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GammaType {
    /// Slow gamma 30-70Hz (retrieval)
    Slow,
    /// Fast gamma 60-100Hz (encoding)
    Fast,
}

impl GammaOscillator {
    pub fn new(gamma_type: GammaType) -> Self {
        let frequency = match gamma_type {
            GammaType::Slow => 40.0,   // 40Hz
            GammaType::Fast => 80.0,   // 80Hz
        };

        Self {
            frequency,
            phase: 0.0,
            amplitude: 0.3,
            baseline_amplitude: 0.3,
            gamma_type,
        }
    }

    /// Update gamma phase
    pub fn update(&mut self, dt: f32) {
        let delta_phase = self.frequency * dt / 1000.0;
        self.phase += delta_phase;
        self.phase %= 1.0;
    }

    /// Get current gamma value
    pub fn value(&self) -> f32 {
        self.amplitude * (self.phase * 2.0 * PI).sin()
    }

    /// Modulate amplitude by theta phase (cross-frequency coupling)
    pub fn modulate_by_theta(&mut self, theta_phase: f32) {
        // Gamma amplitude peaks at specific theta phases
        let theta_modulation = match self.gamma_type {
            GammaType::Slow => {
                // Slow gamma peaks at theta trough (retrieval)
                (theta_phase * 2.0 * PI + PI).cos()  // Peak at trough
            }
            GammaType::Fast => {
                // Fast gamma peaks at theta peak (encoding)
                (theta_phase * 2.0 * PI).cos()  // Peak at peak
            }
        };

        self.amplitude = self.baseline_amplitude * (1.0 + 0.5 * theta_modulation);
    }

    /// Get excitability window (spike arrival effectiveness)
    pub fn excitability(&self) -> f32 {
        // High excitability during rising phase
        if self.phase < 0.5 {
            self.phase * 2.0  // 0 to 1
        } else {
            2.0 - self.phase * 2.0  // 1 to 0
        }
    }

    /// Get phase
    pub fn get_phase(&self) -> f32 {
        self.phase
    }
}

/// Alpha/Beta oscillator (8-20Hz) for top-down predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaBetaOscillator {
    /// Frequency (Hz)
    pub frequency: f32,

    /// Current phase (0-1)
    pub phase: f32,

    /// Amplitude (modulated by attention/prediction strength)
    pub amplitude: f32,

    /// Baseline amplitude
    pub baseline_amplitude: f32,
}

impl AlphaBetaOscillator {
    pub fn new(frequency: f32) -> Self {
        Self {
            frequency,
            phase: 0.0,
            amplitude: 0.4,
            baseline_amplitude: 0.4,
        }
    }

    /// Update phase
    pub fn update(&mut self, dt: f32) {
        let delta_phase = self.frequency * dt / 1000.0;
        self.phase += delta_phase;
        self.phase %= 1.0;
    }

    /// Get current value
    pub fn value(&self) -> f32 {
        self.amplitude * (self.phase * 2.0 * PI).sin()
    }

    /// Modulate amplitude by prediction strength
    pub fn modulate_by_prediction(&mut self, prediction_strength: f32) {
        // Strong predictions → strong alpha/beta
        self.amplitude = self.baseline_amplitude * (0.5 + 0.5 * prediction_strength);
    }

    /// Get phase
    pub fn get_phase(&self) -> f32 {
        self.phase
    }
}

/// Integrated oscillatory system with cross-frequency coupling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryCircuit {
    /// Theta oscillator
    pub theta: ThetaOscillator,

    /// Slow gamma (retrieval)
    pub gamma_slow: GammaOscillator,

    /// Fast gamma (encoding)
    pub gamma_fast: GammaOscillator,

    /// Alpha oscillator
    pub alpha: AlphaBetaOscillator,

    /// Beta oscillator
    pub beta: AlphaBetaOscillator,

    /// Current behavioral state
    pub encoding_mode: bool,

    /// Coherence tracking (for communication)
    coherence_buffer: Vec<f32>,
}

impl OscillatoryCircuit {
    pub fn new() -> Self {
        Self {
            theta: ThetaOscillator::new(6.0),  // 6Hz theta
            gamma_slow: GammaOscillator::new(GammaType::Slow),
            gamma_fast: GammaOscillator::new(GammaType::Fast),
            alpha: AlphaBetaOscillator::new(10.0),  // 10Hz alpha
            beta: AlphaBetaOscillator::new(15.0),   // 15Hz beta
            encoding_mode: true,
            coherence_buffer: Vec::new(),
        }
    }

    /// Update all oscillators
    pub fn update(&mut self, dt: f32, prediction_strength: f32) {
        // Update base oscillators
        self.theta.update(dt);
        self.gamma_slow.update(dt);
        self.gamma_fast.update(dt);
        self.alpha.update(dt);
        self.beta.update(dt);

        // Cross-frequency coupling: theta modulates gamma
        let theta_phase = self.theta.get_phase();
        self.gamma_slow.modulate_by_theta(theta_phase);
        self.gamma_fast.modulate_by_theta(theta_phase);

        // Prediction strength modulates alpha/beta
        self.alpha.modulate_by_prediction(prediction_strength);
        self.beta.modulate_by_prediction(prediction_strength);
    }

    /// Get active gamma based on encoding mode
    pub fn active_gamma(&self) -> &GammaOscillator {
        if self.encoding_mode {
            &self.gamma_fast
        } else {
            &self.gamma_slow
        }
    }

    /// Compute phase-dependent gain (communication through coherence)
    ///
    /// Spikes arriving during high excitability get amplified 5-10×
    pub fn phase_dependent_gain(&self, source_phase: f32) -> f32 {
        let gamma = self.active_gamma();
        let phase_diff = (gamma.get_phase() - source_phase).abs();

        // High gain when phases are aligned (<0.1 phase difference)
        if phase_diff < 0.1 {
            5.0  // 5× amplification
        } else if phase_diff < 0.3 {
            2.0  // 2× amplification
        } else {
            0.5  // Suppression when misaligned
        }
    }

    /// Compute coherence between two oscillators
    pub fn coherence(&self, phase1: f32, phase2: f32) -> f32 {
        // Phase locking value (0-1)
        // Map cos(phase_diff) from [-1, 1] to [0, 1]
        // 0 = completely out of phase, 1 = perfectly in phase
        let phase_diff = (phase1 - phase2) * 2.0 * PI;
        (phase_diff.cos() + 1.0) / 2.0
    }

    /// Set behavioral state (affects gamma type)
    pub fn set_encoding_mode(&mut self, encoding: bool) {
        self.encoding_mode = encoding;

        // Reset theta phase at state transitions
        if encoding {
            self.theta.reset_phase();
        }
    }

    /// Get temporal segmentation windows (theta cycles)
    ///
    /// Each theta cycle processes one "chunk"
    pub fn get_theta_cycle(&self) -> usize {
        (self.theta.get_phase() * 7.0) as usize  // ~7 items per cycle (Miller's law)
    }

    /// Compute cross-frequency coupling strength
    pub fn theta_gamma_coupling(&self) -> f32 {
        let gamma = self.active_gamma();
        // Measure amplitude correlation
        let theta_val = self.theta.value();
        let gamma_amp = gamma.amplitude;

        theta_val.abs() * gamma_amp
    }

    /// Get oscillatory modulation for neural excitability
    pub fn neural_modulation(&self) -> f32 {
        let gamma = self.active_gamma();

        // Combined modulation from all rhythms
        let gamma_contrib = gamma.value() * 0.5;
        let theta_contrib = self.theta.value() * 0.3;
        let alpha_contrib = self.alpha.value() * 0.2;

        gamma_contrib + theta_contrib + alpha_contrib
    }

    /// Get statistics
    pub fn stats(&self) -> OscillationStats {
        let active_gamma = self.active_gamma();

        OscillationStats {
            theta_phase: self.theta.get_phase(),
            theta_freq: self.theta.frequency,
            gamma_phase: active_gamma.get_phase(),
            gamma_freq: active_gamma.frequency,
            gamma_type: active_gamma.gamma_type,
            alpha_phase: self.alpha.get_phase(),
            beta_phase: self.beta.get_phase(),
            theta_gamma_coupling: self.theta_gamma_coupling(),
            encoding_mode: self.encoding_mode,
        }
    }
}

impl Default for OscillatoryCircuit {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct OscillationStats {
    pub theta_phase: f32,
    pub theta_freq: f32,
    pub gamma_phase: f32,
    pub gamma_freq: f32,
    pub gamma_type: GammaType,
    pub alpha_phase: f32,
    pub beta_phase: f32,
    pub theta_gamma_coupling: f32,
    pub encoding_mode: bool,
}

/// Phase precession for sequence prediction
///
/// Spikes shift from late to early theta phases during traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasePrecession {
    /// Somatic oscillator (8Hz)
    somatic_freq: f32,

    /// Dendritic oscillator (9.5Hz - creates beat frequency)
    dendritic_freq: f32,

    /// Somatic phase
    somatic_phase: f32,

    /// Dendritic phase
    dendritic_phase: f32,

    /// Position in sequence (0-1)
    position: f32,
}

impl PhasePrecession {
    pub fn new() -> Self {
        Self {
            somatic_freq: 8.0,
            dendritic_freq: 9.5,  // Interference creates ~180° precession
            somatic_phase: 0.0,
            dendritic_phase: 0.0,
            position: 0.0,
        }
    }

    /// Update oscillators
    pub fn update(&mut self, dt: f32, position: f32) {
        self.position = position.clamp(0.0, 1.0);

        // Update both oscillators
        self.somatic_phase += self.somatic_freq * dt / 1000.0;
        self.dendritic_phase += self.dendritic_freq * dt / 1000.0;

        self.somatic_phase %= 1.0;
        self.dendritic_phase %= 1.0;
    }

    /// Get spike phase (shifts with position)
    pub fn spike_phase(&self) -> f32 {
        // Interference between somatic and dendritic
        // Creates ~180° phase advance over full traversal
        let interference = (self.somatic_phase + self.dendritic_phase) / 2.0;
        let precession_shift = self.position * 0.5;  // 180° = 0.5 cycles

        (interference - precession_shift + 1.0) % 1.0
    }

    /// Check if should spike based on phase
    pub fn should_spike(&self, theta_phase: f32) -> bool {
        let spike_phase = self.spike_phase();
        (theta_phase - spike_phase).abs() < 0.05
    }
}

impl Default for PhasePrecession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theta_oscillation() {
        let mut theta = ThetaOscillator::new(6.0);

        // Full cycle at 6Hz should take ~167ms
        for _ in 0..167 {
            theta.update(1.0);
        }

        // Should be near phase 0 after full cycle
        assert!((theta.get_phase() - 0.0).abs() < 0.1 || (theta.get_phase() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_theta_gamma_coupling() {
        let mut circuit = OscillatoryCircuit::new();

        // Update and check coupling
        for _ in 0..100 {
            circuit.update(1.0, 0.5);
        }

        let coupling = circuit.theta_gamma_coupling();
        assert!(coupling >= 0.0);
    }

    #[test]
    fn test_gamma_types() {
        let slow = GammaOscillator::new(GammaType::Slow);
        let fast = GammaOscillator::new(GammaType::Fast);

        assert!(fast.frequency > slow.frequency);
        assert_eq!(slow.gamma_type, GammaType::Slow);
        assert_eq!(fast.gamma_type, GammaType::Fast);
    }

    #[test]
    fn test_phase_dependent_gain() {
        let circuit = OscillatoryCircuit::new();

        // Aligned phases → high gain
        let high_gain = circuit.phase_dependent_gain(circuit.active_gamma().get_phase());
        assert!(high_gain > 1.0);

        // Misaligned phases → low gain
        let low_gain = circuit.phase_dependent_gain(
            (circuit.active_gamma().get_phase() + 0.5) % 1.0
        );
        assert!(low_gain < 1.0);
    }

    #[test]
    fn test_phase_precession() {
        let mut precession = PhasePrecession::new();

        // At start of sequence
        precession.update(10.0, 0.0);
        let start_phase = precession.spike_phase();

        // At end of sequence
        precession.update(10.0, 1.0);
        let end_phase = precession.spike_phase();

        // Phase should have advanced
        assert_ne!(start_phase, end_phase);
    }

    #[test]
    fn test_coherence() {
        let circuit = OscillatoryCircuit::new();

        // Perfect coherence
        let coh1 = circuit.coherence(0.0, 0.0);
        assert!((coh1 - 1.0).abs() < 0.01);

        // No coherence
        let coh2 = circuit.coherence(0.0, 0.5);
        assert!(coh2 < 0.5);
    }
}
