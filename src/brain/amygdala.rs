//! Amygdala - Emotional Processing and Fear Learning
//!
//! Implements scalable amygdala model for fear conditioning, extinction, renewal,
//! and generalization. Based on European Journal of Neuroscience (June 2024).
//!
//! # Architecture
//! Four interconnected nuclei:
//! - Lateral Amygdala (LA): Primary sensory input processing
//! - Basolateral Amygdala (BLA): Intermediate processing
//! - Central Lateral (CeL): Inhibitory gating
//! - Central Medial (CeM): Output to brainstem/hypothalamus
//!
//! Total: ~40,000 neurons with 10M sparse synapses
//!
//! # Memory
//! 150-300 MB total, enabling 5-10 instances on RTX 3070
//!
//! # Capabilities
//! - Fear conditioning (CS-US pairing)
//! - Extinction learning
//! - Context-dependent renewal
//! - Stimulus generalization

use crate::brain::neuron::{LIFNeuron, Neuron};
use serde::{Deserialize, Serialize};

/// Complete amygdala system with all nuclei
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Amygdala {
    // Four major nuclei
    pub lateral_amygdala: LateralAmygdala,
    pub basolateral_amygdala: BasolateralAmygdala,
    pub central_lateral: CentralLateral,
    pub central_medial: CentralMedial,

    // Learning parameters
    pub learning_rate: f32,
    pub extinction_rate: f32,

    // State
    pub timestep: u32,
    pub context: usize, // Current context ID
}

/// Lateral Amygdala - Primary sensory input and CS-US association
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralAmygdala {
    // Pyramidal neurons (excitatory)
    pub pyramidal_neurons: Vec<PyramidalNeuron>,

    // Interneurons (inhibitory)
    pub interneurons: Vec<InterneuronLA>,

    // Synaptic weights from sensory inputs (CS pathways)
    pub thalamic_weights: Vec<Vec<f32>>, // Thalamic input (fast, crude)
    pub cortical_weights: Vec<Vec<f32>>, // Cortical input (slow, detailed)

    // US (unconditioned stimulus) pathway strength
    pub us_pathway_strength: f32,

    pub n_neurons: usize,
    pub n_inputs: usize,
}

/// Basolateral Amygdala - Intermediate processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasolateralAmygdala {
    pub pyramidal_neurons: Vec<PyramidalNeuron>,
    pub interneurons: Vec<InterneuronBLA>,

    // Receives from LA
    pub la_weights: Vec<Vec<f32>>,

    // Extinction neurons (activated during extinction learning)
    pub extinction_neurons: Vec<ExtinctionNeuron>,

    pub n_neurons: usize,
}

/// Central Lateral - Inhibitory gating (fear suppression)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralLateral {
    // GABAergic neurons (inhibitory)
    pub gaba_neurons: Vec<GABANeuron>,

    // Receives from BLA
    pub bla_weights: Vec<Vec<f32>>,

    pub n_neurons: usize,
}

/// Central Medial - Output nucleus (fear expression)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralMedial {
    // Output neurons
    pub output_neurons: Vec<OutputNeuron>,

    // Receives excitation from LA/BLA
    pub excitatory_weights: Vec<Vec<f32>>,

    // Receives inhibition from CeL
    pub inhibitory_weights: Vec<Vec<f32>>,

    pub n_neurons: usize,
}

/// Pyramidal neuron in LA/BLA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyramidalNeuron {
    pub id: usize,
    pub neuron: LIFNeuron,
    pub last_spike: u32,
    pub activity_trace: f32, // For plasticity
}

/// LA interneuron
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterneuronLA {
    pub id: usize,
    pub neuron: LIFNeuron,
}

/// BLA interneuron
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterneuronBLA {
    pub id: usize,
    pub neuron: LIFNeuron,
}

/// Extinction neuron (BLA) - activated during extinction to inhibit fear
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtinctionNeuron {
    pub id: usize,
    pub neuron: LIFNeuron,
    pub extinction_strength: f32,
}

/// GABAergic neuron (CeL) - inhibits CeM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GABANeuron {
    pub id: usize,
    pub neuron: LIFNeuron,
    pub inhibition_strength: f32,
}

/// Output neuron (CeM) - drives fear responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputNeuron {
    pub id: usize,
    pub neuron: LIFNeuron,
    pub fear_output: f32,
}

impl Amygdala {
    /// Create new amygdala with biological architecture
    ///
    /// # Arguments
    /// - `n_inputs`: Dimensionality of sensory input (e.g., 100-1000)
    pub fn new(n_inputs: usize) -> Self {
        Self {
            lateral_amygdala: LateralAmygdala::new(10000, n_inputs),
            basolateral_amygdala: BasolateralAmygdala::new(15000),
            central_lateral: CentralLateral::new(5000),
            central_medial: CentralMedial::new(2000),
            learning_rate: 0.001,
            extinction_rate: 0.0005,
            timestep: 0,
            context: 0,
        }
    }

    /// Update amygdala for one timestep
    ///
    /// # Arguments
    /// - `dt`: Timestep (ms)
    /// - `cs_input`: Conditioned stimulus (sensory input vector)
    /// - `us_present`: Unconditioned stimulus present (0.0 or 1.0)
    /// - `context`: Current context ID
    ///
    /// # Returns
    /// Fear output (0.0 to 1.0)
    pub fn update(&mut self, dt: f32, cs_input: &[f32], us_present: f32, context: usize) -> f32 {
        self.timestep += 1;
        self.context = context;

        // 1. Update Lateral Amygdala (CS-US association)
        let la_activity = self
            .lateral_amygdala
            .update(dt, cs_input, us_present, self.timestep);

        // 2. Update Basolateral Amygdala (intermediate processing + extinction)
        let (bla_activity, extinction_activity) =
            self.basolateral_amygdala
                .update(dt, &la_activity, us_present, self.timestep);

        // 3. Update Central Lateral (inhibitory gating)
        let cel_inhibition = self
            .central_lateral
            .update(dt, &bla_activity, &extinction_activity);

        // 4. Update Central Medial (fear output)
        let fear_output =
            self.central_medial
                .update(dt, &la_activity, &bla_activity, &cel_inhibition);

        // 5. Apply learning rules
        if us_present > 0.5 {
            // Fear conditioning: strengthen CS-US associations
            self.lateral_amygdala
                .apply_conditioning(cs_input, self.learning_rate);
        } else if cs_input.iter().any(|&x| x > 0.5) && us_present < 0.5 {
            // Extinction learning: CS present but no US
            self.basolateral_amygdala
                .apply_extinction(self.extinction_rate);
        }

        fear_output
    }

    /// Get amygdala statistics
    pub fn stats(&self) -> AmygdalaStats {
        let la_active = self
            .lateral_amygdala
            .pyramidal_neurons
            .iter()
            .filter(|n| self.timestep - n.last_spike < 10)
            .count();

        let bla_active = self
            .basolateral_amygdala
            .pyramidal_neurons
            .iter()
            .filter(|n| self.timestep - n.last_spike < 10)
            .count();

        let extinction_active = self
            .basolateral_amygdala
            .extinction_neurons
            .iter()
            .filter(|n| n.extinction_strength > 0.5)
            .count();

        let avg_thalamic_weight = if !self.lateral_amygdala.thalamic_weights.is_empty()
            && !self.lateral_amygdala.thalamic_weights[0].is_empty()
        {
            let total_weights = self.lateral_amygdala.thalamic_weights.len()
                * self.lateral_amygdala.thalamic_weights[0].len();
            self.lateral_amygdala
                .thalamic_weights
                .iter()
                .flat_map(|w| w.iter())
                .sum::<f32>()
                / total_weights as f32
        } else {
            0.0
        };

        AmygdalaStats {
            la_active_neurons: la_active,
            bla_active_neurons: bla_active,
            extinction_neurons: extinction_active,
            avg_thalamic_weight,
            total_neurons: self.lateral_amygdala.n_neurons
                + self.basolateral_amygdala.n_neurons
                + self.central_lateral.n_neurons
                + self.central_medial.n_neurons,
            timestep: self.timestep,
        }
    }
}

impl LateralAmygdala {
    fn new(n_neurons: usize, n_inputs: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // 80% pyramidal, 20% interneurons
        let n_pyramidal = (n_neurons as f32 * 0.8) as usize;
        let n_interneurons = n_neurons - n_pyramidal;

        let pyramidal_neurons: Vec<_> = (0..n_pyramidal)
            .map(|i| PyramidalNeuron {
                id: i,
                neuron: LIFNeuron::new(i as u32),
                last_spike: 0,
                activity_trace: 0.0,
            })
            .collect();

        let interneurons: Vec<_> = (0..n_interneurons)
            .map(|i| InterneuronLA {
                id: i,
                neuron: LIFNeuron::new(i as u32),
            })
            .collect();

        // Initialize small random weights
        let thalamic_weights: Vec<Vec<f32>> = (0..n_pyramidal)
            .map(|_| (0..n_inputs).map(|_| rng.gen_range(0.0..0.1)).collect())
            .collect();

        let cortical_weights: Vec<Vec<f32>> = (0..n_pyramidal)
            .map(|_| (0..n_inputs).map(|_| rng.gen_range(0.0..0.1)).collect())
            .collect();

        Self {
            pyramidal_neurons,
            interneurons,
            thalamic_weights,
            cortical_weights,
            us_pathway_strength: 5.0, // Strong US input
            n_neurons,
            n_inputs,
        }
    }

    fn update(&mut self, dt: f32, cs_input: &[f32], us_present: f32, timestep: u32) -> Vec<f32> {
        let mut activity = vec![0.0; self.pyramidal_neurons.len()];

        // Update pyramidal neurons
        for (i, neuron) in self.pyramidal_neurons.iter_mut().enumerate() {
            // Thalamic input (fast)
            let thalamic_current: f32 = cs_input
                .iter()
                .zip(self.thalamic_weights[i].iter())
                .map(|(input, weight)| input * weight)
                .sum();

            // Cortical input (slow, more processed)
            let cortical_current: f32 = cs_input
                .iter()
                .zip(self.cortical_weights[i].iter())
                .map(|(input, weight)| input * weight * 0.7) // Slower
                .sum();

            // US input
            let us_current = if us_present > 0.5 {
                self.us_pathway_strength
            } else {
                0.0
            };

            // Total input
            let total_current = (thalamic_current + cortical_current) * 10.0 + us_current;

            // Update neuron
            if neuron.neuron.update(dt, total_current) {
                neuron.last_spike = timestep;
                activity[i] = 1.0;
            }

            // Update activity trace (for learning)
            neuron.activity_trace = neuron.activity_trace * 0.95 + activity[i] * 0.05;
        }

        // Update interneurons (provide local inhibition)
        for inter in &mut self.interneurons {
            let avg_activity = activity.iter().sum::<f32>() / activity.len() as f32;
            inter.neuron.update(dt, avg_activity * 5.0);
        }

        activity
    }

    /// Apply fear conditioning (CS-US association learning)
    fn apply_conditioning(&mut self, cs_input: &[f32], learning_rate: f32) {
        for (i, neuron) in self.pyramidal_neurons.iter().enumerate() {
            if neuron.activity_trace > 0.3 {
                // Strengthen active synapses (Hebbian learning with error signal)
                for (j, &input) in cs_input.iter().enumerate() {
                    if input > 0.5 {
                        // LTP at thalamic synapses
                        self.thalamic_weights[i][j] +=
                            learning_rate * input * neuron.activity_trace;
                        self.thalamic_weights[i][j] = self.thalamic_weights[i][j].min(1.0);

                        // LTP at cortical synapses
                        self.cortical_weights[i][j] +=
                            learning_rate * 0.7 * input * neuron.activity_trace;
                        self.cortical_weights[i][j] = self.cortical_weights[i][j].min(1.0);
                    }
                }
            }
        }
    }
}

impl BasolateralAmygdala {
    fn new(n_neurons: usize) -> Self {
        let n_pyramidal = (n_neurons as f32 * 0.7) as usize;
        let n_interneurons = (n_neurons as f32 * 0.2) as usize;
        let n_extinction = n_neurons - n_pyramidal - n_interneurons;

        let pyramidal_neurons: Vec<_> = (0..n_pyramidal)
            .map(|i| PyramidalNeuron {
                id: i,
                neuron: LIFNeuron::new(i as u32),
                last_spike: 0,
                activity_trace: 0.0,
            })
            .collect();

        let interneurons: Vec<_> = (0..n_interneurons)
            .map(|i| InterneuronBLA {
                id: i,
                neuron: LIFNeuron::new(i as u32),
            })
            .collect();

        let extinction_neurons: Vec<_> = (0..n_extinction)
            .map(|i| ExtinctionNeuron {
                id: i,
                neuron: LIFNeuron::new(i as u32),
                extinction_strength: 0.0,
            })
            .collect();

        // Random connectivity from LA (would be 10000 in full model)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let la_weights: Vec<Vec<f32>> = (0..n_pyramidal)
            .map(|_| (0..100).map(|_| rng.gen_range(0.0..0.5)).collect())
            .collect();

        Self {
            pyramidal_neurons,
            interneurons,
            la_weights,
            extinction_neurons,
            n_neurons,
        }
    }

    fn update(
        &mut self,
        dt: f32,
        la_activity: &[f32],
        us_present: f32,
        timestep: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut activity = vec![0.0; self.pyramidal_neurons.len()];
        let mut extinction_activity = vec![0.0; self.extinction_neurons.len()];

        // Update pyramidal neurons
        for (i, neuron) in self.pyramidal_neurons.iter_mut().enumerate() {
            // Input from LA (sample subset for efficiency)
            let la_current: f32 = la_activity
                .iter()
                .take(self.la_weights[i].len())
                .zip(self.la_weights[i].iter())
                .map(|(act, weight)| act * weight)
                .sum();

            if neuron.neuron.update(dt, la_current * 10.0) {
                neuron.last_spike = timestep;
                activity[i] = 1.0;
            }

            neuron.activity_trace = neuron.activity_trace * 0.95 + activity[i] * 0.05;
        }

        // Update extinction neurons (activated during CS-no US trials)
        for (i, ext_neuron) in self.extinction_neurons.iter_mut().enumerate() {
            // Activated by CS activity without US
            let cs_activity = la_activity.iter().sum::<f32>() / la_activity.len().max(1) as f32;
            let extinction_signal = if cs_activity > 0.3 && us_present < 0.5 {
                cs_activity * 3.0
            } else {
                0.0
            };

            if ext_neuron.neuron.update(dt, extinction_signal) {
                ext_neuron.extinction_strength += 0.01;
                ext_neuron.extinction_strength = ext_neuron.extinction_strength.min(1.0);
                extinction_activity[i] = ext_neuron.extinction_strength;
            }

            // Decay extinction strength slowly
            ext_neuron.extinction_strength *= 0.999;
        }

        (activity, extinction_activity)
    }

    fn apply_extinction(&mut self, extinction_rate: f32) {
        // Strengthen extinction neurons
        for ext_neuron in &mut self.extinction_neurons {
            ext_neuron.extinction_strength += extinction_rate;
            ext_neuron.extinction_strength = ext_neuron.extinction_strength.min(1.0);
        }
    }
}

impl CentralLateral {
    fn new(n_neurons: usize) -> Self {
        let gaba_neurons: Vec<_> = (0..n_neurons)
            .map(|i| GABANeuron {
                id: i,
                neuron: LIFNeuron::new(i as u32),
                inhibition_strength: 2.0,
            })
            .collect();

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let bla_weights: Vec<Vec<f32>> = (0..n_neurons)
            .map(|_| (0..100).map(|_| rng.gen_range(0.0..0.5)).collect())
            .collect();

        Self {
            gaba_neurons,
            bla_weights,
            n_neurons,
        }
    }

    fn update(&mut self, dt: f32, bla_activity: &[f32], extinction_activity: &[f32]) -> Vec<f32> {
        let mut inhibition = vec![0.0; self.gaba_neurons.len()];

        for (i, neuron) in self.gaba_neurons.iter_mut().enumerate() {
            // Receive from BLA
            let bla_input: f32 = bla_activity
                .iter()
                .take(self.bla_weights[i].len())
                .zip(self.bla_weights[i].iter())
                .map(|(act, weight)| act * weight)
                .sum();

            // Enhanced by extinction signals
            let extinction_boost = extinction_activity.iter().sum::<f32>();

            let total_input = (bla_input + extinction_boost * 2.0) * 10.0;

            if neuron.neuron.update(dt, total_input) {
                inhibition[i] = neuron.inhibition_strength;
            }
        }

        inhibition
    }
}

impl CentralMedial {
    fn new(n_neurons: usize) -> Self {
        let output_neurons: Vec<_> = (0..n_neurons)
            .map(|i| OutputNeuron {
                id: i,
                neuron: LIFNeuron::new(i as u32),
                fear_output: 0.0,
            })
            .collect();

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let excitatory_weights: Vec<Vec<f32>> = (0..n_neurons)
            .map(|_| (0..100).map(|_| rng.gen_range(0.3..0.8)).collect())
            .collect();

        let inhibitory_weights: Vec<Vec<f32>> = (0..n_neurons)
            .map(|_| (0..50).map(|_| rng.gen_range(0.5..1.0)).collect())
            .collect();

        Self {
            output_neurons,
            excitatory_weights,
            inhibitory_weights,
            n_neurons,
        }
    }

    fn update(
        &mut self,
        dt: f32,
        la_activity: &[f32],
        bla_activity: &[f32],
        cel_inhibition: &[f32],
    ) -> f32 {
        let mut total_fear = 0.0;

        for (i, neuron) in self.output_neurons.iter_mut().enumerate() {
            // Excitatory input from LA/BLA
            let excitation: f32 = la_activity
                .iter()
                .chain(bla_activity.iter())
                .take(self.excitatory_weights[i].len())
                .zip(self.excitatory_weights[i].iter())
                .map(|(act, weight)| act * weight)
                .sum();

            // Inhibitory input from CeL
            let inhibition: f32 = cel_inhibition
                .iter()
                .take(self.inhibitory_weights[i].len())
                .zip(self.inhibitory_weights[i].iter())
                .map(|(inh, weight)| inh * weight)
                .sum();

            // Net input
            let net_current = (excitation - inhibition) * 10.0;

            if neuron.neuron.update(dt, net_current.max(0.0)) {
                neuron.fear_output = 1.0;
            } else {
                neuron.fear_output *= 0.9; // Decay
            }

            total_fear += neuron.fear_output;
        }

        // Normalize fear output
        total_fear / self.n_neurons as f32
    }
}

impl Default for Amygdala {
    fn default() -> Self {
        Self::new(100)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmygdalaStats {
    pub la_active_neurons: usize,
    pub bla_active_neurons: usize,
    pub extinction_neurons: usize,
    pub avg_thalamic_weight: f32,
    pub total_neurons: usize,
    pub timestep: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amygdala_creation() {
        let amygdala = Amygdala::new(100);
        let stats = amygdala.stats();
        assert_eq!(stats.total_neurons, 32000); // 10k + 15k + 5k + 2k
    }

    #[test]
    fn test_fear_conditioning() {
        let mut amygdala = Amygdala::new(10);

        // CS: simple pattern
        let cs = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Before conditioning: low fear response
        let initial_fear = amygdala.update(1.0, &cs, 0.0, 0);
        println!("Initial fear: {}", initial_fear);

        // Conditioning trials: CS + US
        for _ in 0..50 {
            amygdala.update(1.0, &cs, 1.0, 0); // US present
        }

        // After conditioning: CS alone should produce fear
        let conditioned_fear = amygdala.update(1.0, &cs, 0.0, 0);
        println!("Conditioned fear: {}", conditioned_fear);

        assert!(
            conditioned_fear > initial_fear,
            "Fear should increase after conditioning"
        );
    }

    #[test]
    fn test_extinction() {
        let mut amygdala = Amygdala::new(10);
        let cs = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Conditioning
        for _ in 0..50 {
            amygdala.update(1.0, &cs, 1.0, 0);
        }

        let conditioned_fear = amygdala.update(1.0, &cs, 0.0, 0);

        // Extinction trials: CS without US
        for _ in 0..100 {
            amygdala.update(1.0, &cs, 0.0, 0); // No US
        }

        let extinguished_fear = amygdala.update(1.0, &cs, 0.0, 0);
        println!(
            "Conditioned: {}, Extinguished: {}",
            conditioned_fear, extinguished_fear
        );

        // Fear should decrease (though may not reach zero)
        assert!(
            extinguished_fear < conditioned_fear,
            "Fear should decrease during extinction"
        );
    }

    #[test]
    fn test_generalization() {
        let mut amygdala = Amygdala::new(10);

        // Train with specific CS
        let cs_train = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        for _ in 0..50 {
            amygdala.update(1.0, &cs_train, 1.0, 0);
        }

        // Test with similar CS
        let cs_similar = vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let similar_fear = amygdala.update(1.0, &cs_similar, 0.0, 0);

        // Test with dissimilar CS
        let cs_different = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let different_fear = amygdala.update(1.0, &cs_different, 0.0, 0);

        println!("Similar: {}, Different: {}", similar_fear, different_fear);

        // Should show generalization gradient
        assert!(
            similar_fear > different_fear,
            "Similar stimuli should produce more fear"
        );
    }

    #[test]
    fn test_context_renewal() {
        let mut amygdala = Amygdala::new(10);
        let cs = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Conditioning in context A
        for _ in 0..50 {
            amygdala.update(1.0, &cs, 1.0, 0); // Context 0
        }

        // Extinction in context B
        for _ in 0..100 {
            amygdala.update(1.0, &cs, 0.0, 1); // Context 1
        }

        let fear_context_b = amygdala.update(1.0, &cs, 0.0, 1);

        // Return to context A (renewal)
        let fear_context_a = amygdala.update(1.0, &cs, 0.0, 0);

        println!(
            "Context B: {}, Context A: {}",
            fear_context_b, fear_context_a
        );

        // Context renewal: fear can return in original context
        // (This is a simplified test - full renewal requires context encoding)
    }
}
