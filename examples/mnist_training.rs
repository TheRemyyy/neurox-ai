//! MNIST Training Example
//!
//! Demonstrates complete Triplet STDP training pipeline with:
//! - Rate-coded input encoding
//! - Winner-Take-All dynamics
//! - Homeostatic plasticity
//! - Sleep-like consolidation
//! - Post-training STP adaptation
//!
//! Target: 93.8% accuracy with 4-bit weights

use neurox_ai::*;
use std::sync::Arc;
use clap::Parser;

#[derive(Parser)]
#[command(name = "mnist_training")]
#[command(about = "MNIST training with Triplet STDP")]
struct Args {
    /// Number of hidden neurons
    #[arg(short = 'H', long, default_value = "500")]
    hidden: usize,

    /// Number of training epochs
    #[arg(short, long, default_value = "10")]
    epochs: usize,

    /// Batch size
    #[arg(short, long, default_value = "100")]
    batch_size: usize,

    /// Learning rate for pre-synaptic (depression)
    #[arg(long, default_value = "0.001")]
    lr_pre: f32,

    /// Learning rate for post-synaptic (potentiation)
    #[arg(long, default_value = "0.0015")]
    lr_post: f32,

    /// Winner-Take-All inhibition strength
    #[arg(short, long, default_value = "18.0")]
    wta_strength: f32,

    /// Presentation duration per image (ms)
    #[arg(short, long, default_value = "350.0")]
    presentation_duration: f32,

    /// Number of training samples (synthetic)
    #[arg(long, default_value = "6000")]
    train_samples: usize,

    /// Number of test samples (synthetic)
    #[arg(long, default_value = "1000")]
    test_samples: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();

    log::info!("=== NeuroxAI MNIST Training ===");
    log::info!("Target: 93.8% accuracy (Triplet STDP + 4-bit weights)");

    // Initialize CUDA context
    log::info!("Initializing GPU...");
    let cuda_ctx = Arc::new(CudaContext::default()?);
    log::info!("{}", cuda_ctx.device_info()?);

    // Network architecture
    const N_INPUT: usize = 784;      // 28×28 pixels
    const N_OUTPUT: usize = 10;      // 10 digits
    let n_hidden = args.hidden;
    let n_neurons = N_INPUT + n_hidden + N_OUTPUT;

    log::info!("Network architecture: {}-{}-{}", N_INPUT, n_hidden, N_OUTPUT);

    // Create sparse connectivity (input → hidden → output)
    log::info!("Creating sparse connectivity...");
    let connectivity = create_mnist_connectivity(N_INPUT, n_hidden, N_OUTPUT)?;
    log::info!("  Total synapses: {}", connectivity.nnz);
    log::info!("  Memory footprint: {:.2} MB", connectivity.memory_footprint() as f64 / 1024.0 / 1024.0);

    // Initialize simulator with connectivity
    log::info!("Creating simulator...");
    let dt = 0.1; // 0.1ms timestep
    let simulator = Simulator::with_connectivity(n_neurons, dt, cuda_ctx.clone(), &connectivity)?;

    // Load MNIST dataset
    log::info!("Loading MNIST dataset...");
    // For demo: use synthetic data (replace with real MNIST loader)
    let mnist = datasets::load_mnist_synthetic(args.train_samples, args.test_samples);
    log::info!("  Training samples: {}", mnist.train_images.len());
    log::info!("  Test samples: {}", mnist.test_images.len());

    // Training configuration
    log::info!("\nTraining configuration:");
    log::info!("  Epochs: {}", args.epochs);
    log::info!("  Batch size: {}", args.batch_size);
    log::info!("  LR pre (depression): {}", args.lr_pre);
    log::info!("  LR post (potentiation): {}", args.lr_post);
    log::info!("  WTA strength: {}", args.wta_strength);
    log::info!("  Presentation duration: {} ms", args.presentation_duration);

    let config = TrainingConfig {
        n_epochs: args.epochs,
        batch_size: args.batch_size,
        presentation_duration: args.presentation_duration,
        isi_duration: 150.0,           // 150ms rest
        lr_decay: 0.95,
        wta_strength: args.wta_strength,
        target_rate: 5.0,              // 5 Hz homeostasis
        consolidation_interval: 5,     // Every 5 epochs
    };

    // Train
    log::info!("Starting training...");
    let mut trainer = training::train_mnist(
        simulator,
        &mnist.train_images,
        &mnist.test_images,
        config,
    )?;

    // Final evaluation
    log::info!("=== Final Results ===");
    let final_test_acc = trainer.evaluate(&mnist.test_images)?;
    log::info!("Test accuracy: {:.2}%", final_test_acc * 100.0);

    if let Some(stats) = trainer.stats().last() {
        log::info!("Average firing rate: {:.2} Hz", stats.avg_firing_rate);
        log::info!("Total spikes: {}", stats.total_spikes);
        log::info!("Average weight: {:.4}", stats.avg_weight);
    }

    // Save model (TODO: implement serialization)
    log::info!("Training complete!");

    Ok(())
}

/// Create MNIST-specific connectivity (3-layer feedforward)
fn create_mnist_connectivity(
    n_input: usize,
    n_hidden: usize,
    n_output: usize,
) -> Result<SparseConnectivity, Box<dyn std::error::Error>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Layer 1: Input → Hidden (dense connections, 60% sparsity)
    let mut connections = Vec::new();

    for hidden_idx in 0..n_hidden {
        let hidden_global = n_input + hidden_idx;

        // Each hidden neuron connects to ~300 input neurons (random)
        let n_connections = 300;
        let mut inputs: Vec<usize> = (0..n_input).collect();
        use rand::seq::SliceRandom;
        inputs.shuffle(&mut rng);

        for &input_idx in inputs.iter().take(n_connections) {
            let weight = rng.gen_range(0.0..1.0); // Random initial weight
            connections.push((input_idx, hidden_global, weight));
        }
    }

    // Layer 2: Hidden → Output (dense connections)
    for output_idx in 0..n_output {
        let output_global = n_input + n_hidden + output_idx;

        // Each output neuron connects to all hidden neurons
        for hidden_idx in 0..n_hidden {
            let hidden_global = n_input + hidden_idx;
            let weight = rng.gen_range(0.0..1.0);
            connections.push((hidden_global, output_global, weight));
        }
    }

    // Convert to CSR format
    let n_neurons = n_input + n_hidden + n_output;
    let mut row_ptr = vec![0; n_neurons + 1];
    let mut col_idx = Vec::new();
    let mut weights = Vec::new();

    // Sort connections by target neuron
    connections.sort_by_key(|(_, target, _)| *target);

    let mut current_target = 0;
    for (source, target, weight) in connections {
        // Fill row_ptr for empty rows
        while current_target < target {
            row_ptr[current_target + 1] = col_idx.len() as i32;
            current_target += 1;
        }

        col_idx.push(source as i32);
        weights.push(weight);
    }

    // Fill remaining row_ptr
    for i in current_target..n_neurons {
        row_ptr[i + 1] = col_idx.len() as i32;
    }

    let nnz = col_idx.len();

    Ok(SparseConnectivity {
        row_ptr,
        col_idx,
        weights,
        nnz,
        n_neurons,
    })
}
