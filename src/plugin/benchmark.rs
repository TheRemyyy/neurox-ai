//! Benchmark Plugin
//!
//! MNIST benchmark with quantization support.

use crate::config::BenchmarkConfig;
use crate::brain::datasets::{download_mnist, MNISTDataset};
use crate::brain::learning::quantization::QuantizedWeights;
use crate::brain::learning::STDPConfig;
use crate::brain::simulation::Simulator;
use crate::brain::training::{MNISTTrainer, TrainingConfig};
use crate::{CudaContext, ProceduralConnectivity, SparseConnectivity};
use std::sync::Arc;

// CLI Colors - white, gray, light blue only
const COLOR_RESET: &str = "\x1b[0m";
const COLOR_WHITE: &str = "\x1b[37m";
const COLOR_GRAY: &str = "\x1b[90m";
const COLOR_LIGHT_BLUE: &str = "\x1b[94m";
const BOLD: &str = "\x1b[1m";

/// Benchmark plugin for MNIST evaluation
pub struct BenchmarkPlugin {
    config: BenchmarkConfig,
}

impl BenchmarkPlugin {
    /// Create a new benchmark plugin with configuration
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run the MNIST benchmark
    pub fn run(
        &self,
        data_dir: Option<&str>,
        epochs: Option<usize>,
        bits: Option<u8>,
        neurons: Option<usize>,
        duration: Option<f32>,
        isi: Option<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Use CLI args or config defaults
        let data_dir = data_dir.unwrap_or(&self.config.data_dir);
        let epochs = epochs.unwrap_or(self.config.epochs);
        let bits = bits.unwrap_or(self.config.bits);
        let neurons = neurons.unwrap_or(self.config.neurons);
        let duration = duration.unwrap_or(self.config.presentation_duration);
        let isi = isi.unwrap_or(self.config.isi);

        println!(
            "{}╔════════════════════════════════════════════════════════════╗",
            COLOR_LIGHT_BLUE
        );
        println!("║  NeuroxAI MNIST Benchmark                                  ║");
        println!(
            "╚════════════════════════════════════════════════════════════╝{}",
            COLOR_RESET
        );
        println!();

        // Handle auto download mode
        let actual_data_dir = if data_dir == "auto" {
            println!(
                "{}Auto-downloading MNIST dataset...{}",
                COLOR_GRAY, COLOR_RESET
            );
            let download_path = "./data/mnist";
            download_mnist(download_path)?;
            println!();
            download_path.to_string()
        } else {
            data_dir.to_string()
        };

        println!("{}Configuration:{}", BOLD, COLOR_RESET);
        println!("  Data directory: {}", actual_data_dir);
        println!("  Epochs: {}", epochs);
        println!("  Quantization: {}-bit", bits);
        println!("  Hidden Neurons: {}", neurons);
        println!("  Presentation: {:.1}ms", duration);
        println!("  ISI: {:.1}ms", isi);
        println!();

        // Architecture
        let n_input = self.config.n_input;
        let n_hidden = neurons;
        let n_output = self.config.n_output;
        let n_neurons = n_input + n_hidden + n_output;

        if actual_data_dir == "synthetic" {
            self.run_synthetic_benchmark(epochs, bits, neurons, duration, isi, n_neurons)?;
        } else {
            self.run_real_benchmark(
                &actual_data_dir,
                epochs,
                bits,
                neurons,
                duration,
                isi,
                n_neurons,
            )?;
        }

        Ok(())
    }

    fn run_synthetic_benchmark(
        &self,
        epochs: usize,
        bits: u8,
        _neurons: usize,
        duration: f32,
        isi: f32,
        n_neurons: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "{}Note:{} Using synthetic data for demo.",
            COLOR_GRAY, COLOR_RESET
        );
        println!();

        println!("{}Generating synthetic data...{}", COLOR_GRAY, COLOR_RESET);
        let train_images = MNISTDataset::generate_synthetic(self.config.synthetic_train_samples);
        let test_images = MNISTDataset::generate_synthetic(self.config.synthetic_test_samples);

        println!("  Train samples: {}", train_images.len());
        println!("  Test samples: {}", test_images.len());
        println!();

        println!("{}Initializing GPU...{}", COLOR_GRAY, COLOR_RESET);
        let cuda_ctx = Arc::new(CudaContext::default()?);
        let device_info = cuda_ctx.device_info()?;
        let device_name = device_info.lines().next().unwrap_or("Unknown GPU");
        println!("  {}", device_name);
        println!();

        // Create sparse connectivity using procedural generator
        let proc_conn = ProceduralConnectivity::new(
            self.config.connectivity.seed,
            self.config.connectivity.density as f64,
            self.config.connectivity.exc_ratio,
            self.config.connectivity.inh_ratio,
        );
        let connectivity = SparseConnectivity::from_procedural(n_neurons, &proc_conn);

        // Create simulator
        let simulator = Simulator::with_connectivity(
            n_neurons,
            self.config.dt,
            cuda_ctx.clone(),
            &connectivity,
        )?;

        // Training config
        let config = TrainingConfig {
            n_epochs: epochs,
            batch_size: self.config.batch_size,
            presentation_duration: duration,
            isi_duration: isi,
            lr_decay: self.config.lr_decay,
            wta_strength: self.config.wta_strength,
            target_rate: self.config.target_rate,
            consolidation_interval: self.config.consolidation_interval,
        };

        // STDP config
        let stdp_config = STDPConfig {
            lr_pre: self.config.stdp.lr_pre,
            lr_post: self.config.stdp.lr_post,
            tau_pre: self.config.stdp.tau_pre,
            tau_post: self.config.stdp.tau_post,
            w_min: self.config.stdp.w_min,
            w_max: self.config.stdp.w_max,
        };

        let mut trainer = MNISTTrainer::new(simulator, config.clone(), stdp_config);

        println!("{}Training...{}", COLOR_GRAY, COLOR_RESET);
        let start_time = std::time::Instant::now();

        for epoch in 1..=epochs {
            trainer.train_epoch(&train_images)?;

            let train_subset =
                &train_images[..self.config.train_subset_size.min(train_images.len())];
            let train_acc = trainer.evaluate(train_subset)?;
            let test_subset = &test_images[..self.config.test_subset_size.min(test_images.len())];
            let test_acc = trainer.evaluate(test_subset)?;

            let elapsed = start_time.elapsed().as_secs_f32();
            println!(
                "  Epoch {}/{}: Train acc: {:.1}%, Test acc: {:.1}% [{:.1}s]",
                epoch,
                epochs,
                train_acc * 100.0,
                test_acc * 100.0,
                elapsed
            );

            if epoch % self.config.consolidation_interval == 0 {
                let replay_samples =
                    &train_images[..self.config.consolidation_samples.min(train_images.len())];
                trainer.consolidate(replay_samples)?;
                println!("    → Sleep consolidation applied");
            }
        }

        println!();
        let final_acc = trainer.evaluate(&test_images)?;

        println!(
            "{}Applying {}-bit quantization...{}",
            COLOR_GRAY, bits, COLOR_RESET
        );
        let weights = trainer.simulator.get_weights()?;
        let quantized = QuantizedWeights::from_float(&weights, bits);
        let compression = quantized.compression_ratio();

        let dequantized_weights = quantized.to_float();
        trainer.simulator.set_weights(&dequantized_weights)?;
        let quant_acc = trainer.evaluate(&test_images)?;

        self.print_results(final_acc, quant_acc, bits, compression);

        Ok(())
    }

    fn run_real_benchmark(
        &self,
        data_dir: &str,
        epochs: usize,
        bits: u8,
        _neurons: usize,
        duration: f32,
        isi: f32,
        n_neurons: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "{}Loading MNIST from {}...{}",
            COLOR_GRAY, data_dir, COLOR_RESET
        );

        let train_path = format!("{}/train-images-idx3-ubyte", data_dir);
        let train_labels_path = format!("{}/train-labels-idx1-ubyte", data_dir);
        let test_path = format!("{}/t10k-images-idx3-ubyte", data_dir);
        let test_labels_path = format!("{}/t10k-labels-idx1-ubyte", data_dir);

        if !std::path::Path::new(&train_path).exists() {
            println!(
                "{}Error:{} MNIST files not found at {}",
                COLOR_WHITE, COLOR_RESET, data_dir
            );
            println!();
            println!("Expected files:");
            println!("  - train-images-idx3-ubyte");
            println!("  - train-labels-idx1-ubyte");
            println!("  - t10k-images-idx3-ubyte");
            println!("  - t10k-labels-idx1-ubyte");
            println!();
            println!("Options:");
            println!("  neurox-ai benchmark --data-dir auto  (auto-download MNIST)");
            println!("  neurox-ai benchmark                  (synthetic data demo)");
            return Ok(());
        }

        let train_dataset = MNISTDataset::load(&train_path, &train_labels_path)?;
        let test_dataset = MNISTDataset::load(&test_path, &test_labels_path)?;

        println!("  Train samples: {}", train_dataset.images.len());
        println!("  Test samples: {}", test_dataset.images.len());
        println!();

        println!("{}Initializing GPU...{}", COLOR_GRAY, COLOR_RESET);
        let cuda_ctx = Arc::new(CudaContext::default()?);
        let device_info = cuda_ctx.device_info()?;
        let device_name = device_info.lines().next().unwrap_or("Unknown GPU");
        println!("  {}", device_name);
        println!();

        let proc_conn = ProceduralConnectivity::new(
            self.config.connectivity.seed,
            self.config.connectivity.density as f64,
            self.config.connectivity.exc_ratio,
            self.config.connectivity.inh_ratio,
        );
        let connectivity = SparseConnectivity::from_procedural(n_neurons, &proc_conn);

        let simulator = Simulator::with_connectivity(
            n_neurons,
            self.config.dt,
            cuda_ctx.clone(),
            &connectivity,
        )?;

        let config = TrainingConfig {
            n_epochs: epochs,
            batch_size: self.config.batch_size,
            presentation_duration: duration,
            isi_duration: isi,
            lr_decay: self.config.lr_decay,
            wta_strength: self.config.wta_strength,
            target_rate: self.config.target_rate,
            consolidation_interval: self.config.consolidation_interval,
        };

        let stdp_config = STDPConfig::default();
        let mut trainer = MNISTTrainer::new(simulator, config.clone(), stdp_config);

        println!("{}Training...{}", COLOR_GRAY, COLOR_RESET);
        let start_time = std::time::Instant::now();

        for epoch in 1..=epochs {
            trainer.train_epoch(&train_dataset.images)?;

            let train_acc =
                trainer.evaluate(&train_dataset.images[..self.config.real_train_eval_size])?;
            let test_acc = trainer.evaluate(&test_dataset.images)?;

            let elapsed = start_time.elapsed().as_secs_f32();
            println!(
                "  Epoch {}/{}: Train acc: {:.1}%, Test acc: {:.1}% [{:.1}s]",
                epoch,
                epochs,
                train_acc * 100.0,
                test_acc * 100.0,
                elapsed
            );

            if epoch % self.config.consolidation_interval == 0 {
                trainer
                    .consolidate(&train_dataset.images[..self.config.real_consolidation_samples])?;
                println!("    → Sleep consolidation applied");
            }
        }

        println!();
        let final_acc = trainer.evaluate(&test_dataset.images)?;

        println!(
            "{}Applying {}-bit quantization...{}",
            COLOR_GRAY, bits, COLOR_RESET
        );
        let weights = trainer.simulator.get_weights()?;
        let quantized = QuantizedWeights::from_float(&weights, bits);
        let compression = quantized.compression_ratio();

        let dequantized_weights = quantized.to_float();
        trainer.simulator.set_weights(&dequantized_weights)?;
        let quant_acc = trainer.evaluate(&test_dataset.images)?;

        self.print_results(final_acc, quant_acc, bits, compression);

        Ok(())
    }

    fn print_results(&self, final_acc: f32, quant_acc: f32, bits: u8, compression: f32) {
        println!();
        println!(
            "{}═══════════════════════════════════════════════════════════{}",
            COLOR_LIGHT_BLUE, COLOR_RESET
        );
        println!("{}Results:{}", BOLD, COLOR_RESET);
        println!("  FP32 Accuracy:     {:.2}%", final_acc * 100.0);
        println!(
            "  {}-bit Accuracy:    {}{:.2}%{}",
            bits,
            COLOR_LIGHT_BLUE,
            quant_acc * 100.0,
            COLOR_RESET
        );
        println!("  Compression ratio: {:.1}×", compression);
        println!(
            "  Memory saved:      {:.1}%",
            (1.0 - 1.0 / compression) * 100.0
        );
        println!(
            "{}═══════════════════════════════════════════════════════════{}",
            COLOR_LIGHT_BLUE, COLOR_RESET
        );
    }
}

impl Default for BenchmarkPlugin {
    fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }
}
