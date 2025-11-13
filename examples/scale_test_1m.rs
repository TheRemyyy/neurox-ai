//! 1 Million Neuron Scale Test
//!
//! Validates event-driven processing at massive scale
//! Target: 20-50× realtime performance with biological sparsity

use neurox_ai::*;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("=== 1M Neuron Scale Test ===");
    log::info!("Target: 20-50× realtime @ 1% sparsity");

    // Initialize CUDA context
    log::info!("Initializing GPU...");
    let cuda_ctx = Arc::new(CudaContext::default()?);
    log::info!("{}", cuda_ctx.device_info()?);

    // Test configurations
    let test_configs = vec![
        ("100K neurons", 100_000),
        ("250K neurons", 250_000),
        ("500K neurons", 500_000),
        ("1M neurons", 1_000_000),
    ];

    for (name, n_neurons) in test_configs {
        log::info!("\n=== {} ===", name);
        run_scale_test(n_neurons, cuda_ctx.clone())?;
    }

    Ok(())
}

fn run_scale_test(
    n_neurons: usize,
    cuda_ctx: Arc<CudaContext>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create sparse connectivity (small-world, 1% connection prob)
    log::info!("Creating sparse connectivity...");
    let connectivity_gen = ProceduralConnectivity {
        seed: 42,
        connection_prob: 0.01, // 1% connection probability
        weight_mean: 0.5,
        weight_std: 0.2,
        topology: ConnectivityType::SmallWorld { k: 10, beta: 0.3 },
        exc_ratio: 0.8, // Dale's principle
    };

    let connectivity = connectivity_gen.to_sparse_matrix(n_neurons)?;
    log::info!("  Synapses: {}", connectivity.nnz);
    log::info!("  Memory: {:.2} MB", connectivity.memory_footprint() as f64 / 1024.0 / 1024.0);
    log::info!("  Sparsity: {:.2}%", connectivity.sparsity() * 100.0);

    // Initialize simulator
    log::info!("Creating simulator with event-driven processing...");
    let dt = 0.1; // 0.1ms timestep
    let mut simulator = Simulator::with_connectivity(n_neurons, dt, cuda_ctx, &connectivity)?;

    // Simulation parameters
    const SIM_DURATION_MS: f32 = 1000.0; // 1 second biological time
    const N_STEPS: usize = (SIM_DURATION_MS / 0.1) as usize; // 10,000 steps

    // Generate sparse random input (1% of neurons active)
    let n_active_input = (n_neurons as f32 * 0.01) as usize;
    let mut input = vec![0.0; n_neurons];
    for i in 0..n_active_input {
        input[i] = 10.0; // Strong input current
    }

    // Benchmark
    log::info!("Running simulation ({} ms biological time)...", SIM_DURATION_MS);
    let start = Instant::now();

    let mut total_spikes = 0u64;
    for step in 0..N_STEPS {
        // Vary input to maintain activity
        if step % 100 == 0 {
            // Shuffle active neurons every 10ms
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            input.shuffle(&mut rng);
        }

        simulator.step(Some(&input))?;

        // Count spikes
        if step % 100 == 0 {
            let spikes = simulator.get_spikes()?;
            total_spikes += spikes.iter().filter(|&&s| s > 0.5).count() as u64;
        }
    }

    simulator.synchronize()?;
    let elapsed = start.elapsed();

    // Results
    let sim_time_sec = SIM_DURATION_MS / 1000.0;
    let realtime_factor = sim_time_sec / elapsed.as_secs_f32();

    log::info!("\n=== Results ===");
    log::info!("  Wall time: {:.2}s", elapsed.as_secs_f32());
    log::info!("  Simulated time: {:.2}s biological", sim_time_sec);
    log::info!("  Realtime factor: {:.2}×", realtime_factor);
    log::info!("  Total spikes: {}", total_spikes);
    log::info!("  Avg firing rate: {:.2} Hz",
               total_spikes as f32 / n_neurons as f32 / sim_time_sec);
    log::info!("  Throughput: {:.2} Mneuron-updates/sec",
               (n_neurons * N_STEPS) as f64 / elapsed.as_secs_f64() / 1_000_000.0);

    // Check target
    if realtime_factor >= 20.0 {
        log::info!("  ✅ Target achieved (20-50× realtime)");
    } else {
        log::warn!("  ⚠️ Below target (got {:.2}×, target 20×)", realtime_factor);
    }

    Ok(())
}
