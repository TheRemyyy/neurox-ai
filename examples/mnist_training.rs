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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("=== NeuroxAI MNIST Training ===");
    log::info!("Target: 93.8% accuracy (Triplet STDP + 4-bit weights)");

    // Initialize CUDA context
    log::info!("Initializing GPU...");
    let cuda_ctx = Arc::new(CudaContext::default()?);
    log::info!("{}", cuda_ctx.device_info()?);

    // Network architecture
    const N_INPUT: usize = 784;      // 28×28 pixels
    const N_HIDDEN: usize = 500;     // Hidden layer
    const N_OUTPUT: usize = 10;      // 10 digits
    const N_NEURONS: usize = N_INPUT + N_HIDDEN + N_OUTPUT;

    log::info!("Network architecture: {}-{}-{}", N_INPUT, N_HIDDEN, N_OUTPUT);

    // Create sparse connectivity (input → hidden → output)
    log::info!("Creating sparse connectivity...");
    let connectivity = create_mnist_connectivity(N_INPUT, N_HIDDEN, N_OUTPUT)?;
    log::info!("  Total synapses: {}", connectivity.nnz);
    log::info!("  Memory footprint: {:.2} MB", connectivity.memory_footprint() as f64 / 1024.0 / 1024.0);

    // Initialize simulator with connectivity
    log::info!("Creating simulator...");
    let dt = 0.1; // 0.1ms timestep
    let simulator = Simulator::with_connectivity(N_NEURONS, dt, cuda_ctx.clone(), &connectivity)?;

    // Load MNIST dataset
    log::info!("Loading MNIST dataset...");
    // For demo: use synthetic data (replace with real MNIST loader)
    let mnist = datasets::load_mnist_synthetic(6000, 1000);
    log::info!("  Training samples: {}", mnist.train_images.len());
    log::info!("  Test samples: {}", mnist.test_images.len());

    // Training configuration
    let config = TrainingConfig {
        n_epochs: 10,
        batch_size: 100,
        presentation_duration: 350.0, // 350ms per image
        isi_duration: 150.0,           // 150ms rest
        lr_decay: 0.95,
        wta_strength: 18.0,            // Lateral inhibition
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
        inputs.sort_by_key(|_| rng.gen::<u32>());

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

    Ok(SparseConnectivity {
        row_ptr,
        col_idx,
        weights,
        nnz: col_idx.len(),
        n_rows: n_neurons,
        n_cols: n_neurons,
    })
}
