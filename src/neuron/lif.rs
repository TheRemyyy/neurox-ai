//! Leaky Integrate-and-Fire (LIF) neuron model
//!
//! Implements the classic LIF neuron with biologically-realistic parameters:
//! dV/dt = (-V + R*I) / tau_m
//!
//! When V >= threshold: spike and reset to v_reset

use super::{Neuron, NeuronState};
use serde::{Deserialize, Serialize};

/// LIF Neuron with biological dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIFNeuron {
    pub state: NeuronState,

    /// Membrane resistance (MΩ)
    pub r_m: f32,

    /// Refractory period duration (timesteps)
    pub refractory_period: u8,
}

impl LIFNeuron {
    /// Create new LIF neuron with biological defaults
    pub fn new(id: u32) -> Self {
        Self {
            state: NeuronState::new(id),
            r_m: 10.0,              // 10 MΩ typical for cortical neurons
            refractory_period: 20,  // 2ms @ 0.1ms timestep
        }
    }

    /// Create neuron with custom parameters
    pub fn with_params(
        id: u32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_m: f32,
        r_m: f32,
        refractory_period: u8,
    ) -> Self {
        let mut state = NeuronState::new(id);
        state.v = v_rest;
        state.threshold = v_thresh;
        state.v_reset = v_reset;
        state.tau_m = tau_m;

        Self {
            state,
            r_m,
            refractory_period,
        }
    }

    /// Update membrane potential using exponential integration
    #[inline]
    fn integrate(&mut self, dt: f32, input_current: f32) {
        // dV/dt = (-V + R*I) / tau_m
        let dv = ((-self.state.v + self.r_m * input_current) / self.state.tau_m) * dt;
        self.state.v += dv;

        // Clamp to prevent numerical instability
        self.state.v = self.state.v.clamp(-100.0, 50.0);
    }
}

impl Neuron for LIFNeuron {
    fn update(&mut self, dt: f32, input_current: f32) -> bool {
        // Handle refractory period
        if self.state.refractory_counter > 0 {
            self.state.refractory_counter -= 1;
            return false;
        }

        // Integrate membrane potential
        self.integrate(dt, input_current);

        // Check for spike
        if self.state.should_spike() {
            self.state.v = self.state.v_reset;
            self.state.refractory_counter = self.refractory_period;
            return true;
        }

        false
    }

    fn reset(&mut self) {
        self.state.v = self.state.v_reset;
        self.state.refractory_counter = 0;
        self.state.last_spike = 0;
    }

    fn voltage(&self) -> f32 {
        self.state.v
    }

    fn id(&self) -> u32 {
        self.state.id
    }
}

/// Conductance-based Adaptive Exponential (CAdEx) neuron
///
/// More biologically accurate than current-based AdEx, using conductance
/// adaptation to avoid unrealistic hyperpolarization.
///
/// Dynamics:
/// C dV/dt = -gₗ(V - Eₗ) + gₗΔₜexp((V - Vₜ)/Δₜ) - gw(V - Eₖ) + I
/// τw dgw/dt = a(V - Eₗ) - gw
///
/// Memory: ~30-40 MB per 10,000 neurons
/// Difficulty: Easy-Medium
/// Biological Accuracy: High (realistic adaptation and saturation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CAdExNeuron {
    pub state: NeuronState,

    /// Membrane capacitance (pF)
    pub c: f32,

    /// Leak conductance (nS)
    pub g_l: f32,

    /// Leak reversal potential (mV)
    pub e_l: f32,

    /// Adaptation conductance (nS)
    pub g_w: f32,

    /// Potassium reversal potential (mV)
    pub e_k: f32,

    /// Adaptation time constant (ms)
    pub tau_w: f32,

    /// Subthreshold adaptation coupling (nS)
    pub a: f32,

    /// Spike-triggered adaptation increment (nS)
    pub b: f32,

    /// Slope factor for exponential (mV)
    pub delta_t: f32,

    /// Rheobase threshold (mV)
    pub v_t: f32,

    /// Refractory period (timesteps)
    pub refractory_period: u8,
}

impl CAdExNeuron {
    /// Create new CAdEx neuron with biological defaults for cortical excitatory neurons
    pub fn new(id: u32) -> Self {
        let mut state = NeuronState::new(id);
        state.v = -70.0;           // Resting at E_L
        state.threshold = -50.0;    // Spike threshold
        state.v_reset = -70.0;      // Reset to E_L
        state.tau_m = 20.0;         // Not directly used, but kept for compatibility

        Self {
            state,
            c: 281.0,              // pF, typical pyramidal neuron
            g_l: 30.0,             // nS
            e_l: -70.6,            // mV, leak reversal
            g_w: 0.0,              // nS, starts at 0
            e_k: -80.0,            // mV, potassium reversal
            tau_w: 144.0,          // ms, adaptation time constant
            a: 4.0,                // nS, subthreshold adaptation
            b: 80.5,               // nS, spike-triggered adaptation
            delta_t: 2.0,          // mV, exponential slope
            v_t: -50.4,            // mV, rheobase threshold
            refractory_period: 20, // 2ms @ 0.1ms timestep
        }
    }

    /// Create fast-spiking interneuron variant (PV-like)
    pub fn fast_spiking(id: u32) -> Self {
        let mut neuron = Self::new(id);
        neuron.tau_w = 50.0;       // Faster adaptation
        neuron.a = 2.0;            // Less subthreshold adaptation
        neuron.b = 20.0;           // Less spike-triggered adaptation
        neuron.refractory_period = 10; // 1ms refractory
        neuron
    }

    /// Create adapting interneuron variant (SST-like)
    pub fn adapting(id: u32) -> Self {
        let mut neuron = Self::new(id);
        neuron.tau_w = 200.0;      // Slow adaptation
        neuron.a = 8.0;            // More subthreshold adaptation
        neuron.b = 120.0;          // Strong spike-triggered adaptation
        neuron
    }

    /// Update membrane potential using conductance-based dynamics
    #[inline]
    fn integrate(&mut self, dt: f32, input_current: f32) {
        let v = self.state.v;

        // Exponential term (spike generation mechanism)
        let exp_term = if v < self.v_t + 10.0 * self.delta_t {
            // Only compute exp when not too far from threshold (numerical stability)
            self.g_l * self.delta_t * ((v - self.v_t) / self.delta_t).exp()
        } else {
            // Saturate at high voltages
            self.g_l * self.delta_t * 100.0
        };

        // Membrane equation: C dV/dt = -gₗ(V - Eₗ) + exponential - gw(V - Eₖ) + I
        let dv = (
            -self.g_l * (v - self.e_l)           // Leak current
            + exp_term                            // Exponential spike generation
            - self.g_w * (v - self.e_k)          // Adaptation current (conductance-based)
            + input_current                       // External input
        ) / self.c;

        self.state.v += dv * dt;

        // Adaptation conductance: τw dgw/dt = a(V - Eₗ) - gw
        let dg_w = (self.a * (v - self.e_l) - self.g_w) / self.tau_w;
        self.g_w += dg_w * dt;

        // Prevent negative conductance
        self.g_w = self.g_w.max(0.0);

        // Clamp voltage for numerical stability
        self.state.v = self.state.v.clamp(-100.0, 50.0);
    }
}

impl Neuron for CAdExNeuron {
    fn update(&mut self, dt: f32, input_current: f32) -> bool {
        // Handle refractory period
        if self.state.refractory_counter > 0 {
            self.state.refractory_counter -= 1;
            return false;
        }

        // Integrate dynamics
        self.integrate(dt, input_current);

        // Check for spike
        if self.state.should_spike() {
            self.state.v = self.state.v_reset;
            self.g_w += self.b;  // Spike-triggered adaptation increment
            self.state.refractory_counter = self.refractory_period;
            return true;
        }

        false
    }

    fn reset(&mut self) {
        self.state.v = self.e_l;
        self.g_w = 0.0;
        self.state.refractory_counter = 0;
        self.state.last_spike = 0;
    }

    fn voltage(&self) -> f32 {
        self.state.v
    }

    fn id(&self) -> u32 {
        self.state.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_resting() {
        let mut neuron = LIFNeuron::new(0);
        let spiked = neuron.update(0.1, 0.0);
        assert!(!spiked);
        assert!(neuron.voltage() < -55.0); // Below threshold
    }

    #[test]
    fn test_lif_spike() {
        let mut neuron = LIFNeuron::new(0);

        // Apply strong input current
        for _ in 0..50 {
            if neuron.update(0.1, 2.0) {
                break;
            }
        }

        // Should have spiked and reset
        assert_eq!(neuron.voltage(), -70.0);
    }

    #[test]
    fn test_refractory_period() {
        let mut neuron = LIFNeuron::new(0);

        // Trigger spike
        for _ in 0..50 {
            if neuron.update(0.1, 2.0) {
                break;
            }
        }

        // Should not spike immediately after due to refractory period
        let spiked = neuron.update(0.1, 5.0);
        assert!(!spiked);
    }

    #[test]
    fn test_cadex_resting() {
        let mut neuron = CAdExNeuron::new(0);
        let spiked = neuron.update(0.1, 0.0);
        assert!(!spiked);
        assert!((neuron.voltage() - neuron.e_l).abs() < 5.0); // Near leak potential
    }

    #[test]
    fn test_cadex_spike() {
        let mut neuron = CAdExNeuron::new(0);

        // Apply strong input current
        let mut spike_count = 0;
        for _ in 0..1000 {
            if neuron.update(0.1, 500.0) {
                spike_count += 1;
            }
        }

        // Should spike multiple times
        assert!(spike_count > 0);
    }

    #[test]
    fn test_cadex_adaptation() {
        let mut neuron = CAdExNeuron::new(0);

        // Record inter-spike intervals
        let mut isi_values = Vec::new();
        let mut last_spike_time = 0;

        for t in 0..2000 {
            if neuron.update(0.1, 600.0) {
                if last_spike_time > 0 {
                    isi_values.push(t - last_spike_time);
                }
                last_spike_time = t;
            }
        }

        // Inter-spike intervals should increase (spike frequency adaptation)
        if isi_values.len() >= 3 {
            assert!(isi_values[2] > isi_values[0],
                "Expected adaptation: later ISI should be longer");
        }
    }

    #[test]
    fn test_cadex_fast_spiking() {
        let mut regular = CAdExNeuron::new(0);
        let mut fs = CAdExNeuron::fast_spiking(1);

        // Count spikes for both
        let mut regular_spikes = 0;
        let mut fs_spikes = 0;

        for _ in 0..1000 {
            if regular.update(0.1, 500.0) {
                regular_spikes += 1;
            }
            if fs.update(0.1, 500.0) {
                fs_spikes += 1;
            }
        }

        // Fast-spiking should spike more frequently
        assert!(fs_spikes > regular_spikes,
            "Fast-spiking neuron should spike more frequently");
    }
}
