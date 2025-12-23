//! Interneuron Diversity - PV, SST, and VIP Neurons
//!
//! Three major inhibitory classes with profoundly different functions:
//! - PV (40%): Fast-spiking perisomatic inhibition for sparse coding and gamma
//! - SST (30%): Adapting dendritic inhibition for feedback control
//! - VIP (15%): Disinhibitory cells targeting PV/SST for behavioral state modulation
//!
//! # Biological Parameters
//! - PV: τ=10ms, >100Hz non-adapting, somatic targeting
//! - SST: τ=50ms, 20-40Hz adapting, dendritic targeting
//! - VIP: τ=30ms, targets other interneurons (disinhibition)
//!
//! # Functions
//! - PV: Winner-take-all, sparse coding (10-20× capacity), gamma oscillations
//! - SST: Gating learning phases (encoding vs retrieval), feedback control
//! - VIP: Behavioral state modulation (attention, arousal)

use crate::neuron::NeuronState;
use serde::{Deserialize, Serialize};

/// Parvalbumin (PV) fast-spiking interneuron
///
/// Fast perisomatic inhibition for sparse coding and gamma oscillations.
/// Fires >100Hz without adaptation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PVInterneuron {
    pub state: NeuronState,

    /// Fast membrane time constant (10ms)
    pub tau_m: f32,

    /// High firing threshold (rapid activation)
    pub high_threshold: f32,

    /// Targets soma (strong, fast inhibition)
    pub somatic_weight: f32,

    /// No adaptation current
    pub adaptation: f32,

    /// Spike count in gamma cycle
    pub spike_count: usize,

    /// Last gamma phase
    pub last_gamma_phase: f32,
}

impl PVInterneuron {
    pub fn new(id: u32) -> Self {
        let mut state = NeuronState::new(id);
        state.tau_m = 10.0;  // Fast
        state.threshold = -50.0;  // High threshold
        state.v_reset = -65.0;

        Self {
            state,
            tau_m: 10.0,
            high_threshold: -50.0,
            somatic_weight: 1.5,  // Strong inhibition
            adaptation: 0.0,
            spike_count: 0,
            last_gamma_phase: 0.0,
        }
    }

    /// Update PV neuron (fast, non-adapting)
    pub fn update(&mut self, dt: f32, input_current: f32) -> bool {
        // Fast LIF dynamics
        if self.state.refractory_counter > 0 {
            self.state.refractory_counter -= 1;
            return false;
        }

        // Integrate: dV/dt = (-V + I) / tau_m
        let dv = ((-self.state.v + input_current) / self.tau_m) * dt;
        self.state.v += dv;

        // Check spike
        if self.state.v >= self.high_threshold {
            self.state.v = self.state.v_reset;
            self.state.refractory_counter = 10;  // 1ms @ 0.1ms timestep
            self.spike_count += 1;
            true
        } else {
            false
        }
    }

    /// Compute gamma-phase dependent inhibition
    /// PV creates ~40Hz gamma rhythm through feedback inhibition
    pub fn gamma_inhibition(&self, gamma_phase: f32) -> f32 {
        // Strong inhibition during gamma peak
        let phase_modulation = (gamma_phase * 2.0 * std::f32::consts::PI).cos();
        self.somatic_weight * (1.0 + 0.5 * phase_modulation)
    }

    /// Winner-take-all: suppress all but top-k neurons
    pub fn apply_wta_inhibition(&self, neuron_activities: &mut [f32], k: usize) {
        let mut indexed: Vec<(usize, f32)> = neuron_activities
            .iter()
            .copied()
            .enumerate()
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Zero out all except top k
        for (i, _) in indexed.iter().skip(k) {
            neuron_activities[*i] = 0.0;
        }
    }
}

/// Somatostatin (SST) adapting interneuron
///
/// Dendritic inhibition for feedback control and plasticity gating.
/// Fires 20-40Hz with strong adaptation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSTInterneuron {
    pub state: NeuronState,

    /// Slower membrane time constant (50ms)
    pub tau_m: f32,

    /// Adaptation current (builds up with spiking)
    pub adaptation_current: f32,

    /// Adaptation time constant (200ms)
    pub tau_adaptation: f32,

    /// Adaptation strength
    pub adaptation_strength: f32,

    /// Targets dendrites (modulates plasticity)
    pub dendritic_weight: f32,

    /// Learning gate (0.0 = consolidation, 1.0 = encoding)
    pub learning_gate: f32,
}

impl SSTInterneuron {
    pub fn new(id: u32) -> Self {
        let mut state = NeuronState::new(id);
        state.tau_m = 50.0;  // Slower
        state.threshold = -55.0;
        state.v_reset = -70.0;

        Self {
            state,
            tau_m: 50.0,
            adaptation_current: 0.0,
            tau_adaptation: 200.0,
            adaptation_strength: 0.1,
            dendritic_weight: 1.0,
            learning_gate: 1.0,
        }
    }

    /// Update SST neuron (adapting)
    pub fn update(&mut self, dt: f32, input_current: f32) -> bool {
        // Handle refractory
        if self.state.refractory_counter > 0 {
            self.state.refractory_counter -= 1;
            return false;
        }

        // Decay adaptation
        let decay = (-dt / self.tau_adaptation).exp();
        self.adaptation_current *= decay;

        // Integrate with adaptation
        let effective_current = input_current - self.adaptation_current;
        let dv = ((-self.state.v + effective_current) / self.tau_m) * dt;
        self.state.v += dv;

        // Check spike
        if self.state.v >= self.state.threshold {
            self.state.v = self.state.v_reset;
            self.state.refractory_counter = 20;  // 2ms

            // Increase adaptation (causes firing rate adaptation)
            self.adaptation_current += self.adaptation_strength;

            true
        } else {
            false
        }
    }

    /// Gate learning based on SST activity
    /// High SST = suppress learning (retrieval mode)
    /// Low SST = allow learning (encoding mode)
    pub fn modulate_learning_rate(&self, base_lr: f32) -> f32 {
        // Inverse relationship: more SST activity = less learning
        let sst_suppression = self.adaptation_current;
        base_lr * (1.0 - sst_suppression).max(0.1)
    }

    /// Set behavioral state (encoding vs retrieval)
    pub fn set_behavioral_state(&mut self, encoding: bool) {
        self.learning_gate = if encoding { 1.0 } else { 0.3 };
    }
}

/// Vasoactive Intestinal Peptide (VIP) disinhibitory interneuron
///
/// Targets other interneurons (PV and SST) to disinhibit pyramidal cells.
/// Critical for behavioral state modulation and attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VIPInterneuron {
    pub state: NeuronState,

    /// Medium time constant (30ms)
    pub tau_m: f32,

    /// Targets other interneurons
    pub targets_interneurons: bool,

    /// Disinhibition strength
    pub disinhibition_strength: f32,

    /// Attention modulation (0.0-1.0)
    pub attention_level: f32,

    /// Arousal modulation (0.0-1.0)
    pub arousal_level: f32,
}

impl VIPInterneuron {
    pub fn new(id: u32) -> Self {
        let mut state = NeuronState::new(id);
        state.tau_m = 30.0;
        state.threshold = -55.0;
        state.v_reset = -70.0;

        Self {
            state,
            tau_m: 30.0,
            targets_interneurons: true,
            disinhibition_strength: 0.8,
            attention_level: 0.5,
            arousal_level: 0.5,
        }
    }

    /// Update VIP neuron
    pub fn update(&mut self, dt: f32, input_current: f32) -> bool {
        if self.state.refractory_counter > 0 {
            self.state.refractory_counter -= 1;
            return false;
        }

        // Integrate
        let dv = ((-self.state.v + input_current) / self.tau_m) * dt;
        self.state.v += dv;

        // Check spike
        if self.state.v >= self.state.threshold {
            self.state.v = self.state.v_reset;
            self.state.refractory_counter = 15;
            true
        } else {
            false
        }
    }

    /// Apply disinhibition to pyramidal cells
    /// VIP → SST/PV → Pyramidal (two-step disinhibition)
    pub fn apply_disinhibition(
        &self,
        pv_inhibition: f32,
        sst_inhibition: f32,
        vip_active: bool,
    ) -> f32 {
        if vip_active {
            // VIP suppresses SST and PV, reducing their inhibition
            let reduced_pv = pv_inhibition * (1.0 - self.disinhibition_strength);
            let reduced_sst = sst_inhibition * (1.0 - self.disinhibition_strength);
            -(reduced_pv + reduced_sst)  // Net disinhibition
        } else {
            -(pv_inhibition + sst_inhibition)  // Full inhibition
        }
    }

    /// Modulate based on attention
    pub fn set_attention(&mut self, attention: f32) {
        self.attention_level = attention.clamp(0.0, 1.0);
        // High attention increases VIP activity → disinhibition
    }

    /// Modulate based on arousal
    pub fn set_arousal(&mut self, arousal: f32) {
        self.arousal_level = arousal.clamp(0.0, 1.0);
    }

    /// Get effective disinhibition based on state
    pub fn get_effective_disinhibition(&self) -> f32 {
        self.disinhibition_strength * self.attention_level * self.arousal_level
    }
}

/// Circuit combining all three interneuron types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterneuronCircuit {
    pub pv_neurons: Vec<PVInterneuron>,
    pub sst_neurons: Vec<SSTInterneuron>,
    pub vip_neurons: Vec<VIPInterneuron>,

    /// Gamma oscillation phase (0-1)
    pub gamma_phase: f32,

    /// Gamma frequency (Hz)
    pub gamma_freq: f32,
}

impl InterneuronCircuit {
    /// Create new interneuron circuit
    ///
    /// Maintains biological ratios: PV:SST:VIP ≈ 40:30:15
    pub fn new(n_pyramidal: usize) -> Self {
        let n_pv = (n_pyramidal as f32 * 0.4) as usize;
        let n_sst = (n_pyramidal as f32 * 0.3) as usize;
        let n_vip = (n_pyramidal as f32 * 0.15) as usize;

        let pv_neurons = (0..n_pv).map(|i| PVInterneuron::new(i as u32)).collect();
        let sst_neurons = (0..n_sst).map(|i| SSTInterneuron::new(i as u32)).collect();
        let vip_neurons = (0..n_vip).map(|i| VIPInterneuron::new(i as u32)).collect();

        Self {
            pv_neurons,
            sst_neurons,
            vip_neurons,
            gamma_phase: 0.0,
            gamma_freq: 40.0,  // 40Hz gamma
        }
    }

    /// Update gamma oscillation
    pub fn update_gamma(&mut self, dt: f32) {
        let delta_phase = self.gamma_freq * dt / 1000.0;  // dt in ms
        self.gamma_phase += delta_phase;
        self.gamma_phase %= 1.0;
    }

    /// Compute total inhibition to pyramidal neurons
    pub fn compute_inhibition(&self, pyramidal_activity: &[f32]) -> Vec<f32> {
        let mut inhibition = vec![0.0; pyramidal_activity.len()];

        // PV: Fast somatic inhibition
        let pv_active = self.pv_neurons.iter().filter(|n| n.state.v > -60.0).count();
        let pv_inhibition = (pv_active as f32 / self.pv_neurons.len() as f32) * 1.5;

        // SST: Dendritic inhibition
        let sst_active = self.sst_neurons.iter().filter(|n| n.state.v > -60.0).count();
        let sst_inhibition = (sst_active as f32 / self.sst_neurons.len() as f32) * 1.0;

        // VIP: Disinhibition
        let vip_active = self.vip_neurons.iter().any(|n| n.state.v > -60.0);

        // Apply to each pyramidal neuron
        for (i, inh) in inhibition.iter_mut().enumerate() {
            // PV creates gamma-phase dependent inhibition
            let gamma_modulated_pv = pv_inhibition *
                (1.0 + 0.5 * (self.gamma_phase * 2.0 * std::f32::consts::PI).cos());

            // VIP disinhibits by suppressing PV and SST
            if vip_active {
                *inh = -(gamma_modulated_pv * 0.5 + sst_inhibition * 0.5);
            } else {
                *inh = -(gamma_modulated_pv + sst_inhibition);
            }
        }

        inhibition
    }

    /// Apply winner-take-all via PV neurons
    pub fn apply_sparse_coding(&self, activities: &mut [f32], sparsity: f32) {
        let k = (activities.len() as f32 * sparsity) as usize;
        if let Some(pv) = self.pv_neurons.first() {
            pv.apply_wta_inhibition(activities, k);
        }
    }

    /// Get circuit statistics
    pub fn stats(&self) -> InterneuronStats {
        let pv_active = self.pv_neurons.iter().filter(|n| n.state.v > -60.0).count();
        let sst_active = self.sst_neurons.iter().filter(|n| n.state.v > -60.0).count();
        let vip_active = self.vip_neurons.iter().filter(|n| n.state.v > -60.0).count();

        InterneuronStats {
            n_pv: self.pv_neurons.len(),
            n_sst: self.sst_neurons.len(),
            n_vip: self.vip_neurons.len(),
            pv_active,
            sst_active,
            vip_active,
            gamma_phase: self.gamma_phase,
            gamma_freq: self.gamma_freq,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterneuronStats {
    pub n_pv: usize,
    pub n_sst: usize,
    pub n_vip: usize,
    pub pv_active: usize,
    pub sst_active: usize,
    pub vip_active: usize,
    pub gamma_phase: f32,
    pub gamma_freq: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pv_fast_spiking() {
        let mut pv = PVInterneuron::new(0);

        // Should spike rapidly with strong input
        let mut spike_count = 0;
        for _ in 0..1000 {
            if pv.update(0.1, 3.0) {
                spike_count += 1;
            }
        }

        // Should achieve >100Hz with strong input
        assert!(spike_count > 10);  // >100Hz for 100ms
    }

    #[test]
    fn test_sst_adaptation() {
        let mut sst = SSTInterneuron::new(0);

        // Initial spike
        for _ in 0..100 {
            if sst.update(0.1, 2.0) {
                break;
            }
        }

        let initial_adaptation = sst.adaptation_current;

        // More spikes should increase adaptation
        for _ in 0..500 {
            sst.update(0.1, 2.0);
        }

        assert!(sst.adaptation_current >= initial_adaptation);
    }

    #[test]
    fn test_vip_disinhibition() {
        let vip = VIPInterneuron::new(0);

        let pv_inh = 1.0;
        let sst_inh = 1.0;

        // Without VIP: full inhibition
        let full_inh = vip.apply_disinhibition(pv_inh, sst_inh, false);

        // With VIP: reduced inhibition (disinhibition)
        let reduced_inh = vip.apply_disinhibition(pv_inh, sst_inh, true);

        assert!(reduced_inh.abs() < full_inh.abs());
    }

    #[test]
    fn test_circuit_ratios() {
        let circuit = InterneuronCircuit::new(100);

        // Check biological ratios (40:30:15 per 100 pyramidal)
        assert_eq!(circuit.pv_neurons.len(), 40);
        assert_eq!(circuit.sst_neurons.len(), 30);
        assert_eq!(circuit.vip_neurons.len(), 15);
    }
}
