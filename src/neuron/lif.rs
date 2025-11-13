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
}
