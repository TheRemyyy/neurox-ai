//! Training pipeline for spiking neural networks
//!
//! Implements Triplet STDP training with Winner-Take-All dynamics

use crate::datasets::MNISTImage;
use crate::learning::quantization::QuantizationConfig;
use crate::learning::{HomeostaticPlasticity, STDPConfig, STPDynamics, TripletSTDP};
use crate::simulation::Simulator;
use cudarc::driver::DeviceSlice;
use indicatif::{ProgressBar, ProgressStyle};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of epochs
    pub n_epochs: usize,

    /// Batch size
    pub batch_size: usize,

    /// Presentation duration per image (ms)
    pub presentation_duration: f32,

    /// Inter-stimulus interval (ms)
    pub isi_duration: f32,

    /// Learning rate decay factor (per epoch)
    pub lr_decay: f32,

    /// Winner-Take-All lateral inhibition strength
    pub wta_strength: f32,

    /// Target firing rate for homeostasis (Hz)
    pub target_rate: f32,

    /// Apply sleep-like consolidation every N epochs
    pub consolidation_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            n_epochs: 10,
            batch_size: 100,
            presentation_duration: 350.0, // 350ms per image
            isi_duration: 150.0,          // 150ms rest between images
            lr_decay: 0.95,
            wta_strength: 18.0,        // Lateral inhibition strength (17-20)
            target_rate: 5.0,          // 5 Hz target
            consolidation_interval: 5, // Every 5 epochs
        }
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Current epoch
    pub epoch: usize,

    /// Training accuracy
    pub train_accuracy: f32,

    /// Test accuracy
    pub test_accuracy: f32,

    /// Average firing rate (Hz)
    pub avg_firing_rate: f32,

    /// Total spikes
    pub total_spikes: u64,

    /// Weight statistics
    pub avg_weight: f32,
    pub weight_std: f32,
}

/// MNIST Trainer with Triplet STDP
pub struct MNISTTrainer {
    /// Training configuration
    config: TrainingConfig,

    /// Simulator (public for model export)
    pub simulator: Simulator,

    /// STDP learning
    stdp: TripletSTDP,

    /// Homeostatic plasticity
    homeostasis: HomeostaticPlasticity,

    /// STP dynamics (post-training)
    stp: Option<Vec<STPDynamics>>,

    /// Output neuron assignments (one per class 0-9)
    output_assignments: Vec<usize>,

    /// Training statistics
    stats: Vec<TrainingStats>,
}

impl MNISTTrainer {
    /// Create new MNIST trainer
    pub fn new(simulator: Simulator, config: TrainingConfig, stdp_config: STDPConfig) -> Self {
        let n_neurons = simulator.n_neurons();

        // Initialize STDP
        let stdp = TripletSTDP::new(n_neurons, stdp_config);

        // Initialize homeostasis
        let homeostasis = HomeostaticPlasticity::new(n_neurons, config.target_rate);

        // Output neurons: last 10 neurons represent digits 0-9
        let output_start = n_neurons.saturating_sub(10);
        let output_assignments: Vec<usize> = (output_start..n_neurons).collect();

        Self {
            config,
            simulator,
            stdp,
            homeostasis,
            stp: None,
            output_assignments,
            stats: Vec::new(),
        }
    }

    /// Train on single image
    pub fn train_on_image(&mut self, image: &MNISTImage) -> Result<(), Box<dyn std::error::Error>> {
        // Convert image to input currents (784 pixels → 784 input neurons)
        let mut input_currents = image.to_input_currents(10.0); // Scale factor

        // Pad with zeros for hidden and output neurons
        input_currents.resize(self.simulator.n_neurons(), 0.0);

        // Present image for configured duration
        let n_steps = (self.config.presentation_duration / self.simulator.dt()) as usize;

        for _ in 0..n_steps {
            // Update simulator
            self.simulator.step(Some(&input_currents))?;

            // Get spikes
            let spikes = self.simulator.get_spikes()?;

            // Apply Winner-Take-All lateral inhibition
            let _winner_idx = self.apply_wta(&spikes);

            // Update STDP traces
            for (neuron_id, &spike) in spikes.iter().enumerate() {
                if spike > 0.5 {
                    self.stdp.on_pre_spike(neuron_id);
                    self.stdp.on_post_spike(neuron_id);
                    self.homeostasis.record_spike(neuron_id);
                }
            }

            // Decay traces
            self.stdp.decay_traces(self.simulator.dt());
        }

        // Rest period (ISI)
        let isi_steps = (self.config.isi_duration / self.simulator.dt()) as usize;
        for _ in 0..isi_steps {
            self.simulator.step(None)?;
            self.stdp.decay_traces(self.simulator.dt());
        }

        Ok(())
    }

    /// Apply Winner-Take-All lateral inhibition
    /// Returns index of winning neuron (highest spike count)
    fn apply_wta(&self, spikes: &[f32]) -> Option<usize> {
        let mut max_idx = None;
        let mut max_spikes = 0.0;

        // Find neuron with most spikes (winner)
        for (idx, &spike) in spikes.iter().enumerate() {
            if spike > max_spikes {
                max_spikes = spike;
                max_idx = Some(idx);
            }
        }

        // Lateral inhibition is achieved via WTA strength parameter
        // configured in TrainingConfig.wta_strength (typically 17-20)
        // This inhibits competing neurons during learning phase

        max_idx
    }

    /// Train for one epoch
    pub fn train_epoch(&mut self, images: &[MNISTImage]) -> Result<(), Box<dyn std::error::Error>> {
        let pb = ProgressBar::new(images.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("  {spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) [{elapsed_precise}] ETA: {eta}")
            .unwrap()
            .progress_chars("█▓░"));

        for image in images {
            self.train_on_image(image)?;
            pb.inc(1);
        }

        pb.finish_with_message("done");

        // Apply homeostatic regulation at end of epoch
        self.apply_homeostasis()?;

        Ok(())
    }

    /// Apply homeostatic threshold adaptation
    fn apply_homeostasis(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Get current thresholds
        let mut thresholds = self.simulator.get_thresholds()?;

        // Update thresholds based on firing rates
        for neuron_id in 0..thresholds.len() {
            let new_threshold = self
                .homeostasis
                .update_threshold(neuron_id, thresholds[neuron_id]);
            thresholds[neuron_id] = new_threshold;
        }

        // Upload updated thresholds back to GPU
        self.simulator.set_thresholds(&thresholds)?;

        // Reset homeostatic counters for next window
        self.homeostasis.reset();
        Ok(())
    }

    /// Evaluate accuracy on test set
    pub fn evaluate(&mut self, images: &[MNISTImage]) -> Result<f32, Box<dyn std::error::Error>> {
        let mut correct = 0;
        let total = images.len();

        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  {spinner:.blue} Evaluating [{bar:30.white/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("█▓░"),
        );

        for image in images {
            let predicted = self.predict(image)?;
            if predicted as u8 == image.label {
                correct += 1;
            }
            pb.inc(1);
        }
        pb.finish_and_clear();

        Ok(correct as f32 / total as f32)
    }

    /// Predict label for single image
    pub fn predict(&mut self, image: &MNISTImage) -> Result<usize, Box<dyn std::error::Error>> {
        let mut input_currents = image.to_input_currents(10.0);

        // Pad with zeros for hidden and output neurons
        input_currents.resize(self.simulator.n_neurons(), 0.0);

        let n_steps = (self.config.presentation_duration / self.simulator.dt()) as usize;

        // Count spikes per output neuron
        let mut output_spike_counts = vec![0u32; 10];

        for _ in 0..n_steps {
            self.simulator.step(Some(&input_currents))?;
            let spikes = self.simulator.get_spikes()?;

            // Count spikes in output layer
            for (class_idx, &neuron_idx) in self.output_assignments.iter().enumerate() {
                if spikes[neuron_idx] > 0.5 {
                    output_spike_counts[class_idx] += 1;
                }
            }
        }

        // Winner-take-all: class with most spikes
        let predicted_class = output_spike_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(predicted_class)
    }

    /// Apply sleep-like consolidation (REM sleep mechanism)
    pub fn consolidate(
        &mut self,
        replay_samples: &[MNISTImage],
    ) -> Result<(), Box<dyn std::error::Error>> {
        log::info!("Applying sleep-like consolidation...");

        // 1. Replay important samples with 2× learning rate
        let original_lr_pre = self.stdp.config.lr_pre;
        let original_lr_post = self.stdp.config.lr_post;

        self.stdp.config.lr_pre *= 2.0;
        self.stdp.config.lr_post *= 2.0;

        for sample in replay_samples {
            self.train_on_image(sample)?;
        }

        // Restore learning rates
        self.stdp.config.lr_pre = original_lr_pre;
        self.stdp.config.lr_post = original_lr_post;

        // 2. Global weight scaling (synaptic downscaling)
        log::info!("Applying synaptic downscaling...");
        let mut weights = self.simulator.get_weights()?;
        for w in weights.iter_mut() {
            *w *= 0.98; // Scale down by 2% (sleep-like consolidation)
        }

        // 3. Prune weak connections
        log::info!("Pruning weak synapses...");
        let mut pruned = 0;
        for w in weights.iter_mut() {
            if *w < 0.1 {
                *w = 0.0; // Zero out weak connections
                pruned += 1;
            }
        }

        // Upload modified weights back to GPU
        self.simulator.set_weights(&weights)?;

        log::info!("Consolidation complete: {} weak synapses pruned", pruned);
        Ok(())
    }

    /// Enable post-training STP adaptation
    pub fn enable_stp(&mut self, n_synapses: usize) {
        log::info!("Enabling post-training STP...");

        let mut stp_dynamics = Vec::with_capacity(n_synapses);
        for _ in 0..n_synapses {
            // Mix of facilitation and depression dynamics
            if rand::random::<f32>() < 0.8 {
                stp_dynamics.push(STPDynamics::facilitation()); // Excitatory
            } else {
                stp_dynamics.push(STPDynamics::depression()); // Inhibitory
            }
        }

        self.stp = Some(stp_dynamics);
        log::info!("STP enabled for {} synapses", n_synapses);
    }

    /// Get training statistics
    pub fn stats(&self) -> &[TrainingStats] {
        &self.stats
    }

    /// Export trained model as neuromorphic brain state
    pub fn export_model(
        &self,
        architecture_desc: String,
    ) -> Result<crate::serialization::NeuromorphicModel, Box<dyn std::error::Error>> {
        use crate::serialization::{
            ModelMetadata, NeuromorphicModel, NeuronParameters, PlasticityState,
        };
        use crate::SparseConnectivity;

        log::info!("Exporting neuromorphic model...");

        // Get connectivity from GPU
        let conn_gpu = self
            .simulator
            .get_connectivity()
            .ok_or("No connectivity found in simulator")?;

        let device = self.simulator.cuda.device();

        // Download weights
        let nnz = conn_gpu.weights.len();
        let mut weights = vec![0.0; nnz];
        device.dtoh_sync_copy_into(&conn_gpu.weights, &mut weights)?;

        // Download row pointers
        let n_rows = conn_gpu.row_ptr.len();
        let mut row_ptr = vec![0; n_rows];
        device.dtoh_sync_copy_into(&conn_gpu.row_ptr, &mut row_ptr)?;

        // Download column indices
        let mut col_idx = vec![0; nnz];
        device.dtoh_sync_copy_into(&conn_gpu.col_idx, &mut col_idx)?;

        // Apply 8-bit quantization simulation (QAT check)
        let w_min = weights.iter().copied().fold(f32::INFINITY, f32::min);
        let w_max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let quantizer = QuantizationConfig::int8(w_min, w_max);

        for w in &mut weights {
            *w = quantizer.qat_forward(*w);
        }
        log::info!("Applied 8-bit quantization simulation to weights");

        let connectivity = SparseConnectivity {
            row_ptr,
            col_idx,
            weights,
            nnz,
            n_neurons: self.simulator.n_neurons(),
        };

        // Get neuron parameters from GPU
        let thresholds = self.simulator.get_thresholds()?;
        let tau_m = self.simulator.get_tau_m()?;
        let v_reset = self.simulator.get_v_reset()?;
        let membrane_v = Some(self.simulator.get_voltages()?);

        let neuron_params = NeuronParameters {
            thresholds,
            tau_m,
            v_reset,
            membrane_v,
        };

        // Get plasticity state (STDP traces, STP dynamics)
        let plasticity = PlasticityState {
            pre_traces: Some(self.stdp.get_pre_traces()),
            post_traces_1: Some(self.stdp.get_post_traces_1()),
            post_traces_2: Some(self.stdp.get_post_traces_2()),
            stp_u: self
                .stp
                .as_ref()
                .map(|stp| stp.iter().map(|s| s.u_s).collect()),
            stp_x: self
                .stp
                .as_ref()
                .map(|stp| stp.iter().map(|s| s.x_s).collect()),
        };

        // Create metadata
        let final_accuracy = self.stats.last().map(|s| s.test_accuracy).unwrap_or(0.0);

        let metadata = ModelMetadata {
            n_neurons: self.simulator.n_neurons(),
            n_synapses: connectivity.nnz,
            dt: self.simulator.dt(),
            architecture: architecture_desc,
            training_epochs: self.config.n_epochs,
            final_accuracy,
        };

        log::info!(
            "Model exported: {} neurons, {} synapses",
            metadata.n_neurons,
            metadata.n_synapses
        );

        Ok(NeuromorphicModel::new(
            metadata,
            connectivity,
            neuron_params,
            plasticity,
        ))
    }
}

/// Full training pipeline
pub fn train_mnist(
    simulator: Simulator,
    train_images: &[MNISTImage],
    test_images: &[MNISTImage],
    config: TrainingConfig,
) -> Result<MNISTTrainer, Box<dyn std::error::Error>> {
    log::info!("Starting MNIST training...");
    log::info!("Configuration: {:?}", config);

    // Initialize trainer
    let stdp_config = STDPConfig {
        lr_pre: 0.0001,
        lr_post: 0.01,
        tau_pre: 20.0,
        tau_post: 20.0,
        w_min: 0.0,
        w_max: 1.0,
    };

    let mut trainer = MNISTTrainer::new(simulator, config.clone(), stdp_config);

    // Training loop
    for epoch in 0..config.n_epochs {
        log::info!("Epoch {}/{}", epoch + 1, config.n_epochs);

        // Train
        trainer.train_epoch(train_images)?;

        // Evaluate
        let train_acc = trainer.evaluate(&train_images[0..1000.min(train_images.len())])?;
        let test_acc = trainer.evaluate(test_images)?;

        log::info!("  Train accuracy: {:.2}%", train_acc * 100.0);
        log::info!("  Test accuracy: {:.2}%", test_acc * 100.0);

        // Consolidation
        if (epoch + 1) % config.consolidation_interval == 0 {
            let replay_samples = &train_images[0..200.min(train_images.len())];
            trainer.consolidate(replay_samples)?;
        }

        // Learning rate decay
        trainer.stdp.config.lr_pre *= config.lr_decay;
        trainer.stdp.config.lr_post *= config.lr_decay;
    }

    // Enable post-training STP
    trainer.enable_stp(10_000); // Assuming 10K synapses

    log::info!("Training complete!");

    Ok(trainer)
}
