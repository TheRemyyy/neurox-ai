//! Neuromorphic Cochlea - Auditory Transduction
//!
//! Biologically plausible auditory front-end using Gammatone filterbanks
//! and Inner Hair Cell (IHC) transduction models.
//!
//! # Processing Pipeline
//! 1. **Gammatone Filterbank:** Simulates basilar membrane motion (frequency analysis).
//!    - 4th order filters
//!    - ERB (Equivalent Rectangular Bandwidth) spacing
//! 2. **IHC Transduction:** Converts motion to receptor potential.
//!    - Half-wave rectification
//!    - Non-linear compression (active amplification)
//!    - Low-pass filtering (membrane leakage)
//! 3. **Spiral Ganglion Neurons:** Converts potential to spikes.
//!    - Phase-locking to fine structure (< 2-3 kHz)
//!    - Stochastic firing based on neurotransmitter release

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Single Gammatone Filter (4th order)
///
/// Impulse response: g(t) = t^(n-1) * exp(-2*pi*b*t) * cos(2*pi*f_c*t)
/// Implemented as a digital IIR filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammatoneFilter {
    /// Center frequency (Hz)
    pub center_freq: f32,

    /// Bandwidth (Hz)
    pub bandwidth: f32,

    /// Filter coefficients (for 4 cascaded 1st-order sections)
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,

    /// Filter state (delay lines for 4 stages)
    state_x: [[f32; 2]; 4],
    state_y: [[f32; 2]; 4],

    /// Gain normalization
    gain: f32,
}

impl GammatoneFilter {
    /// Create new Gammatone filter
    pub fn new(center_freq: f32, sample_rate: f32) -> Self {
        // ERB (Equivalent Rectangular Bandwidth) scale (Glasberg & Moore 1990)
        let erb = 24.7 * (4.37 * center_freq / 1000.0 + 1.0);
        let b = 1.019 * erb; // Bandwidth parameter

        // Digital filter design (Impulse Invariant Transformation)
        let t = 1.0 / sample_rate;
        let two_pi_b_t = 2.0 * PI * b * t;
        let two_pi_f_t = 2.0 * PI * center_freq * t;

        // Coefficients for complex resonator
        let exp_b = (-two_pi_b_t).exp();
        let cos_f = two_pi_f_t.cos();
        let _sin_f = two_pi_f_t.sin();

        // We implement 4th order gammatone as cascade of 4 complex one-pole filters
        // Real-valued implementation approximates this.
        // Simplified IIR coefficients for stability:
        let a1 = -2.0 * exp_b * cos_f;
        let a2 = exp_b * exp_b;

        let b0 = t;
        let b1 = 0.0;
        let b2 = 0.0;

        // Calculate gain at center frequency to normalize to unity
        // H(z) at f_c should be 1.0.
        // For simplicity in this simulation, we use an empirical gain factor scaling with frequency
        let gain = 2.0 * (PI * b).powi(4) / sample_rate.powi(4); // Scaling factor

        Self {
            center_freq,
            bandwidth: b,
            a1,
            a2,
            b0,
            b1,
            b2,
            state_x: [[0.0; 2]; 4],
            state_y: [[0.0; 2]; 4],
            gain,
        }
    }

    /// Process one sample
    /// Returns filtered output (basilar membrane displacement)
    pub fn process(&mut self, sample: f32) -> f32 {
        let mut input = sample;

        // Cascade 4 filter stages
        for i in 0..4 {
            // Difference equation: y[n] = b0*x[n] - a1*y[n-1] - a2*y[n-2]
            // Note: Simple resonator implementation
            let output =
                self.b0 * input - self.a1 * self.state_y[i][0] - self.a2 * self.state_y[i][1];

            // Update state
            self.state_x[i][1] = self.state_x[i][0];
            self.state_x[i][0] = input;

            self.state_y[i][1] = self.state_y[i][0];
            self.state_y[i][0] = output;

            input = output; // Output of stage i is input to stage i+1
        }

        input / self.gain
    }
}

/// Inner Hair Cell (IHC) Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnerHairCell {
    /// Receptor potential
    pub potential: f32,

    /// Neurotransmitter vesicle pool (0.0 - 1.0)
    pub transmitter: f32,

    /// Adaptation state
    adaptation: f32,
}

impl Default for InnerHairCell {
    fn default() -> Self {
        Self::new()
    }
}

impl InnerHairCell {
    pub fn new() -> Self {
        Self {
            potential: -60.0, // Resting potential (mV)
            transmitter: 1.0,
            adaptation: 0.0,
        }
    }

    /// Transduce basilar membrane motion to neurotransmitter release probability
    pub fn process(&mut self, displacement: f32, dt: f32) -> f32 {
        // 1. Mechano-electrical transduction (MET)
        // Sigmoidal function of displacement (half-wave rectification + saturation)
        // Boltsmann function: 1 / (1 + exp(-(x - x0)/s))
        let conductance = 1.0 / (1.0 + (-10.0 * displacement).exp());

        // 2. Receptor Potential (RC circuit)
        // C * dV/dt = -G_leak * (V - E_leak) - G_trans * (V - E_trans)
        let g_leak = 10.0; // nS
        let e_leak = -60.0; // mV
        let g_trans = 30.0 * conductance; // nS
        let e_trans = 0.0; // mV (Endocochlear potential)
        let c_m = 10.0; // pF

        let dv = (-g_leak * (self.potential - e_leak) - g_trans * (self.potential - e_trans)) / c_m;
        self.potential += dv * dt;

        // 3. Calcium / Neurotransmitter Release
        // Voltage dependent calcium channels
        let ca_open = 1.0 / (1.0 + (-(self.potential + 40.0) / 5.0).exp());

        // Transmitter depletion
        let release_rate = 5.0 * ca_open * self.transmitter; // kHz
        let replenishment = 0.5 * (1.0 - self.transmitter);

        let d_transmitter = (replenishment - release_rate * 0.001) * dt; // dt in ms
        self.transmitter += d_transmitter;
        self.transmitter = self.transmitter.clamp(0.0, 1.0);

        // Return release probability for synapse
        release_rate * 0.001 * dt // Prob per step
    }
}

/// Neuromorphic Cochlea System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicCochlea {
    /// Filterbank channels
    filters: Vec<GammatoneFilter>,

    /// Inner Hair Cells (one per filter)
    ihcs: Vec<InnerHairCell>,

    /// Parameters
    pub n_channels: usize,
    pub sample_rate: f32,
    pub min_freq: f32,
    pub max_freq: f32,
}

impl NeuromorphicCochlea {
    /// Create new cochlea
    pub fn new(n_channels: usize, sample_rate: f32, min_freq: f32, max_freq: f32) -> Self {
        let mut filters = Vec::with_capacity(n_channels);
        let mut ihcs = Vec::with_capacity(n_channels);

        // Generate ERB-spaced frequencies
        // Greenwood function or similar log-like spacing
        for i in 0..n_channels {
            // Linear spacing on ERB scale
            // simplified log spacing for now:
            let f = min_freq * (max_freq / min_freq).powf(i as f32 / (n_channels - 1) as f32);

            filters.push(GammatoneFilter::new(f, sample_rate));
            ihcs.push(InnerHairCell::new());
        }

        Self {
            filters,
            ihcs,
            n_channels,
            sample_rate,
            min_freq,
            max_freq,
        }
    }

    /// Process audio sample and return spikes
    ///
    /// # Arguments
    /// - `sample`: Normalized audio sample (-1.0 to 1.0)
    /// - `dt`: Simulation timestep (ms)
    ///
    /// # Returns
    /// Spike train (true = spike) for each channel
    pub fn process(&mut self, sample: f32, dt: f32) -> Vec<bool> {
        let mut spikes = Vec::with_capacity(self.n_channels);
        let mut rng = rand::thread_rng();
        use rand::Rng;

        for i in 0..self.n_channels {
            // 1. Filter (Basilar Membrane)
            let displacement = self.filters[i].process(sample);

            // 2. Transduction (IHC)
            let release_prob = self.ihcs[i].process(displacement, dt);

            // 3. Spike Generation (Spiral Ganglion)
            // Stochastic firing driven by vesicle release
            let spiked = rng.gen::<f32>() < release_prob;
            spikes.push(spiked);
        }

        spikes
    }

    /// Get center frequencies of channels
    pub fn frequencies(&self) -> Vec<f32> {
        self.filters.iter().map(|f| f.center_freq).collect()
    }

    /// Get current energy in each band (for visualization/stats)
    pub fn energy(&self) -> Vec<f32> {
        self.ihcs
            .iter()
            .map(|ihc| (ihc.potential + 60.0).max(0.0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gammatone_filter() {
        let mut filter = GammatoneFilter::new(1000.0, 44100.0);

        // Sine wave at center freq
        let mut max_out = 0.0;
        for i in 0..1000 {
            let t = i as f32 / 44100.0;
            let input = (2.0 * PI * 1000.0 * t).sin();
            let output = filter.process(input);
            if output.abs() > max_out {
                max_out = output.abs();
            }
        }

        // Should resonate
        assert!(max_out > 0.1, "Filter should resonate at center freq");
    }

    #[test]
    fn test_ihc_transduction() {
        let mut ihc = InnerHairCell::new();

        // Silence
        let prob_silence = ihc.process(0.0, 0.1);

        // Loud sound (rectification check)
        // Positive displacement
        let prob_loud_pos = ihc.process(1.0, 0.1);

        // Negative displacement (should be lower due to rectification)
        ihc = InnerHairCell::new(); // reset
        let prob_loud_neg = ihc.process(-1.0, 0.1);

        assert!(
            prob_loud_pos > prob_silence,
            "Sound should increase firing prob"
        );
        assert!(
            prob_loud_pos > prob_loud_neg,
            "Rectification should favor one direction"
        );
    }

    #[test]
    fn test_cochlea_channels() {
        let mut cochlea = NeuromorphicCochlea::new(10, 16000.0, 100.0, 5000.0);

        assert_eq!(cochlea.filters.len(), 10);
        assert_eq!(cochlea.ihcs.len(), 10);

        // Low frequency tone
        for i in 0..100 {
            let t = i as f32 / 16000.0;
            let input = (2.0 * PI * 150.0 * t).sin();
            let spikes = cochlea.process(input, 0.1);
            assert_eq!(spikes.len(), 10);
        }

        // Check frequencies are ordered
        let freqs = cochlea.frequencies();
        assert!(freqs[0] < freqs[9]);
        assert!(freqs[0] >= 100.0);
        assert!(freqs[9] <= 5000.0);
    }
}
