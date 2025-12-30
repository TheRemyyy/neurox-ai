//! Training pipeline for spiking neural networks
//!
//! Implements Triplet STDP training with Winner-Take-All dynamics

use crate::brain::datasets::MNISTImage;
use crate::brain::learning::quantization::QuantizationConfig;
use crate::brain::learning::{HomeostaticPlasticity, STDPConfig, STPDynamics, TripletSTDP};
use crate::brain::simulation::Simulator;
use cudarc::driver::DeviceSlice;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;

/// Maps neuron ID to list of (connected_neuron_id, weight_index)
#[derive(Debug, Clone)]
struct ConnectivityMap {
    /// Outgoing connections: pre_id -> [(post_id, weight_idx)]
    outgoing: Vec<Vec<(usize, usize)>>,
    /// Incoming connections: post_id -> [(pre_id, weight_idx)]
    incoming: Vec<Vec<(usize, usize)>>,
}

impl ConnectivityMap {
    fn new(n_neurons: usize) -> Self {
        Self {
            outgoing: vec![Vec::new(); n_neurons],
            incoming: vec![Vec::new(); n_neurons],
        }
    }

    fn from_csr(row_ptr: &[i32], col_idx: &[i32], n_neurons: usize) -> Self {
        let mut map = Self::new(n_neurons);

        for row in 0..n_neurons {
            let start = row_ptr[row] as usize;
            let end = row_ptr[row + 1] as usize;

            for w_idx in start..end {
                let col = col_idx[w_idx] as usize;
                // Outgoing from row to col
                map.outgoing[row].push((col, w_idx));
                // Incoming to col from row
                map.incoming[col].push((row, w_idx));
            }
        }
        map
    }
}

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

/// MNIST Trainer with Triplet STDP + Supervision
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

    /// Hidden layer neuron range (for STDP weight updates)
    hidden_start: usize,
    hidden_end: usize,

    /// Input layer size
    input_size: usize,

    /// Cached connectivity map for efficient CPU-side STDP
    conn_map: Option<ConnectivityMap>,

    /// Accumulator for weight changes (to batch updates)
    weight_changes: Vec<f32>,
}

impl MNISTTrainer {
    /// Create new MNIST trainer
    pub fn new(simulator: Simulator, config: TrainingConfig, stdp_config: STDPConfig) -> Self {
        let n_neurons = simulator.n_neurons();

        // Architecture: 784 input -> 400 hidden -> 10 output
        let input_size = 784;
        let output_size = 10;
        let hidden_start = input_size;
        let hidden_end = n_neurons.saturating_sub(output_size);

        // Initialize STDP
        let stdp = TripletSTDP::new(n_neurons, stdp_config);

        // Initialize homeostasis
        let homeostasis = HomeostaticPlasticity::new(n_neurons, config.target_rate);

        // Output neurons: last 10 neurons represent digits 0-9
        let output_start = n_neurons.saturating_sub(output_size);
        let output_assignments: Vec<usize> = (output_start..n_neurons).collect();

        Self {
            config,
            simulator,
            stdp,
            homeostasis,
            stp: None,
            output_assignments,
            stats: Vec::new(),
            hidden_start,
            hidden_end,
            input_size,
            conn_map: None,
            weight_changes: Vec::new(),
        }
    }

    /// Initialize connectivity map from simulator (must be called before training)
    pub fn loop_init_checks(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.conn_map.is_none() {
            if let Some(conn) = self.simulator.get_connectivity() {
                let device = self.simulator.cuda.device();

                // Download topology
                let mut row_ptr = vec![0i32; conn.row_ptr.len()];
                let mut col_idx = vec![0i32; conn.col_idx.len()];

                device.dtoh_sync_copy_into(&conn.row_ptr, &mut row_ptr)?;
                device.dtoh_sync_copy_into(&conn.col_idx, &mut col_idx)?;

                self.conn_map = Some(ConnectivityMap::from_csr(
                    &row_ptr,
                    &col_idx,
                    self.simulator.n_neurons(),
                ));
                self.weight_changes = vec![0.0; conn.weights.len()];
                log::info!("Initialized connectivity map for efficient STDP");
            }
        }
        Ok(())
    }

    /// Train on single image with supervision
    pub fn train_on_image(&mut self, image: &MNISTImage) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure connectivity map is initialized
        self.loop_init_checks()?;

        let label = image.label as usize;

        // Convert image to input currents (784 pixels → 784 input neurons)
        let mut input_currents = image.to_input_currents(10.0); // Scale factor

        // Pad with zeros for hidden and output neurons
        input_currents.resize(self.simulator.n_neurons(), 0.0);

        // SUPERVISION: Inject current into correct output neuron
        let correct_output = self.output_assignments[label];
        input_currents[correct_output] = 15.0; // Strong teacher signal

        // Present image for configured duration
        let n_steps = (self.config.presentation_duration / self.simulator.dt()) as usize;

        // Track spike counts per hidden neuron for STDP
        let mut hidden_spike_counts = vec![0u32; self.hidden_end - self.hidden_start];
        let mut output_spike_counts = vec![0u32; 10];

        // Winner cache for WTA optimization
        let mut winner_cache: Option<usize> = None;

        for step in 0..n_steps {
            // Apply lateral inhibition to non-winning hidden neurons (optimized: every 100 steps)
            self.apply_wta_inhibition(&mut input_currents, step, &mut winner_cache)?;

            // Update simulator
            self.simulator.step(Some(&input_currents))?;

            // Get spikes
            let spikes = self.simulator.get_spikes()?;

            // Update STDP traces and count spikes
            for (neuron_id, &spike) in spikes.iter().enumerate() {
                if spike > 0.5 {
                    self.stdp.on_pre_spike(neuron_id);
                    self.stdp.on_post_spike(neuron_id);
                    self.homeostasis.record_spike(neuron_id);

                    // Count hidden layer spikes
                    if neuron_id >= self.hidden_start && neuron_id < self.hidden_end {
                        hidden_spike_counts[neuron_id - self.hidden_start] += 1;
                    }

                    // Count output layer spikes
                    for (class_idx, &out_idx) in self.output_assignments.iter().enumerate() {
                        if neuron_id == out_idx {
                            output_spike_counts[class_idx] += 1;
                        }
                    }

                    // ACCUMULATE STDP WEIGHT CHANGES
                    // Access cached connectivity map directly via self
                    // This avoids borrowing issues
                    if let Some(map) = &self.conn_map {
                        // 1. LTD: Update incoming connections (Post-spike)
                        for &(pre_id, w_idx) in &map.incoming[neuron_id] {
                            let dw = self.stdp.calculate_dw(pre_id, neuron_id);
                            self.weight_changes[w_idx] += dw;
                        }

                        // 2. LTP: Update outgoing connections (Pre-spike)
                        for &(post_id, w_idx) in &map.outgoing[neuron_id] {
                            let dw = self.stdp.calculate_dw(neuron_id, post_id);
                            self.weight_changes[w_idx] += dw;
                        }
                    }
                }
            }

            // Decay traces
            self.stdp.decay_traces(self.simulator.dt());
        }

        // Apply STDP weight updates ONCE at end of presentation
        self.apply_accumulated_weight_updates()?;

        // === SUPERVISION SIGNAL ===
        // Reward/punish based on whether correct output fired most
        let predicted = output_spike_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let reward = if predicted == label { 1.0 } else { -0.3 };
        self.apply_reward_modulation(reward)?;

        // Rest period (ISI)
        let isi_steps = (self.config.isi_duration / self.simulator.dt()) as usize;
        for _ in 0..isi_steps {
            self.simulator.step(None)?;
            self.stdp.decay_traces(self.simulator.dt());
        }

        Ok(())
    }

    /// Apply accumulated weight updates
    fn apply_accumulated_weight_updates(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.simulator.get_connectivity().is_none() {
            return Ok(());
        }

        let mut weights = self.simulator.get_weights()?;

        let mut changed_count = 0;
        let mut max_dw = 0.0f32;

        // Apply changes
        for (i, dw) in self.weight_changes.iter_mut().enumerate() {
            if *dw != 0.0 {
                weights[i] = self.stdp.update_weight(weights[i], *dw);
                *dw = 0.0; // Reset accumulator
                changed_count += 1;
                if dw.abs() > max_dw {
                    max_dw = dw.abs();
                }
            }
        }

        // Only upload if something changed
        if changed_count > 0 {
            self.simulator.set_weights(&weights)?;
        }

        Ok(())
    }

    /// Apply reward modulation to recent weight changes
    fn apply_reward_modulation(&mut self, reward: f32) -> Result<(), Box<dyn std::error::Error>> {
        if self.simulator.get_connectivity().is_none() {
            return Ok(());
        }

        let mut weights = self.simulator.get_weights()?;

        // Scale recent weight changes by reward
        // Positive reward: amplify changes, Negative: reverse changes
        let modulation = 1.0 + reward * 0.1;

        for w in &mut weights {
            // Soft modulation to avoid instability
            *w = (*w * modulation).clamp(self.stdp.config.w_min, self.stdp.config.w_max);
        }

        self.simulator.set_weights(&weights)?;
        Ok(())
    }

    /// Apply Winner-Take-All lateral inhibition to hidden layer
    /// Only fetches voltages every 100 steps to reduce GPU→CPU transfers
    fn apply_wta_inhibition(
        &mut self,
        currents: &mut [f32],
        step: usize,
        winner_cache: &mut Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Only recalculate winner every 100 steps to reduce GPU→CPU overhead
        if step % 100 == 0 || winner_cache.is_none() {
            let voltages = self.simulator.get_voltages()?;

            // Find winner in hidden layer (highest membrane potential)
            let mut max_v = f32::NEG_INFINITY;
            let mut best_idx = None;

            for i in self.hidden_start..self.hidden_end {
                if voltages[i] > max_v {
                    max_v = voltages[i];
                    best_idx = Some(i);
                }
            }
            *winner_cache = best_idx;
        }

        // Apply lateral inhibition to non-winners
        if let Some(winner) = *winner_cache {
            for i in self.hidden_start..self.hidden_end {
                if i != winner {
                    currents[i] -= self.config.wta_strength;
                }
            }
        }

        Ok(())
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
                .template("  {spinner:.green} Evaluating [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) [{elapsed_precise}] ETA: {eta}")
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
    ) -> Result<crate::brain::serialization::NeuromorphicModel, Box<dyn std::error::Error>> {
        use crate::brain::serialization::{
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
