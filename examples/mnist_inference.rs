//! MNIST Inference with Trained Brain Model
//!
//! Load a trained .nrx model and use it for predictions

use neurox_ai::*;
use std::sync::Arc;
use clap::Parser;

#[derive(Parser)]
#[command(name = "mnist_inference")]
#[command(about = "Use trained brain model for MNIST inference")]
struct Args {
    /// Path to trained model (.nrx file)
    #[arg(short, long, default_value = "mnist_brain.nrx")]
    model: String,

    /// Number of test samples
    #[arg(short, long, default_value = "100")]
    samples: usize,

    /// Presentation duration (ms)
    #[arg(short, long, default_value = "350.0")]
    duration: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();

    log::info!("=== NeuroxAI Brain Inference ===");
    log::info!("Loading trained brain from: {}", args.model);

    // Load neuromorphic model (brain state)
    let model = serialization::NeuromorphicModel::load(&args.model)?;

    log::info!("Loaded brain model:");
    log::info!("  Architecture: {}", model.metadata.architecture);
    log::info!("  Neurons: {}", model.metadata.n_neurons);
    log::info!("  Synapses: {}", model.metadata.n_synapses);
    log::info!("  Training accuracy: {:.2}%", model.metadata.final_accuracy * 100.0);
    log::info!("  Trained for: {} epochs", model.metadata.training_epochs);

    // Initialize GPU
    log::info!("\nInitializing GPU...");
    let cuda_ctx = Arc::new(CudaContext::default()?);
    log::info!("{}", cuda_ctx.device_info()?);

    // Create simulator from saved model
    log::info!("Restoring brain state to GPU...");
    let mut simulator = create_simulator_from_model(&model, cuda_ctx)?;

    // Load test data
    log::info!("Loading test data...");
    let mnist = datasets::load_mnist_synthetic(0, args.samples);
    log::info!("  Test samples: {}", mnist.test_images.len());

    // Run inference
    log::info!("\nRunning inference...");
    let mut correct = 0;
    let total = mnist.test_images.len();

    for (idx, image) in mnist.test_images.iter().enumerate() {
        let predicted = predict(&mut simulator, image, args.duration)?;

        if predicted as u8 == image.label {
            correct += 1;
        }

        if (idx + 1) % 10 == 0 {
            log::info!("  Processed {}/{} images", idx + 1, total);
        }
    }

    // Results
    let accuracy = correct as f32 / total as f32;
    log::info!("\n=== Inference Results ===");
    log::info!("Correct: {}/{}", correct, total);
    log::info!("Accuracy: {:.2}%", accuracy * 100.0);
    log::info!("\nBrain model working! 🧠");

    Ok(())
}

/// Create simulator from loaded neuromorphic model
fn create_simulator_from_model(
    model: &serialization::NeuromorphicModel,
    cuda: Arc<CudaContext>,
) -> Result<Simulator, Box<dyn std::error::Error>> {
    // Create simulator with connectivity
    let mut simulator = Simulator::with_connectivity(
        model.metadata.n_neurons,
        model.metadata.dt,
        cuda.clone(),
        &model.connectivity,
    )?;

    // Restore neuron parameters (biological state)
    log::info!("Restoring neuron parameters...");
    simulator.set_thresholds(&model.neuron_params.thresholds)?;
    simulator.set_tau_m(&model.neuron_params.tau_m)?;
    simulator.set_v_reset(&model.neuron_params.v_reset)?;

    // Restore membrane potentials if saved
    if let Some(ref membrane_v) = model.neuron_params.membrane_v {
        log::info!("Restoring membrane potentials...");
        simulator.set_voltages(membrane_v)?;
    }

    log::info!("Brain state fully restored to GPU");
    Ok(simulator)
}

/// Predict digit using brain model
fn predict(
    simulator: &mut Simulator,
    image: &MNISTImage,
    duration: f32,
) -> Result<usize, Box<dyn std::error::Error>> {
    // Encode image as input currents
    let mut input_currents = image.to_input_currents(10.0);
    input_currents.resize(simulator.n_neurons(), 0.0);

    let n_steps = (duration / simulator.dt()) as usize;

    // Output neurons are last 10 neurons
    let n_neurons = simulator.n_neurons();
    let output_start = n_neurons - 10;
    let mut output_spike_counts = vec![0u32; 10];

    // Run simulation
    for _ in 0..n_steps {
        simulator.step(Some(&input_currents))?;
        let spikes = simulator.get_spikes()?;

        // Count output spikes
        for class_idx in 0..10 {
            let neuron_idx = output_start + class_idx;
            if spikes[neuron_idx] > 0.5 {
                output_spike_counts[class_idx] += 1;
            }
        }
    }

    // Winner-take-all
    let predicted = output_spike_counts
        .iter()
        .enumerate()
        .max_by_key(|(_, &count)| count)
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    Ok(predicted)
}
