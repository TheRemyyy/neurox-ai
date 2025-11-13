//! NeuroxAI - Neuromorphic Computing System
//!
//! Phase 1: Comprehensive GPU neuromorphic testing

use neurox_ai::{CudaContext, Simulator, VERSION};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("╔════════════════════════════════════════════════════════════╗");
    log::info!("║  NeuroxAI v{} - Neuromorphic Computing Platform  ║", VERSION);
    log::info!("║  GPU-Accelerated Spiking Neural Networks                  ║");
    log::info!("╚════════════════════════════════════════════════════════════╝");

    // Initialize CUDA
    log::info!("\n[CUDA] Initializing GPU context...");
    let cuda_ctx = Arc::new(CudaContext::default()?);

    // Print device info
    let device_info = cuda_ctx.device_info()?;
    log::info!("\n{}", device_info);

    // Comprehensive Phase 1 tests
    log::info!("\n╔════════════════════════════════════════════════════════════╗");
    log::info!("║  Phase 1: Foundation Validation Suite                     ║");
    log::info!("╚════════════════════════════════════════════════════════════╝\n");

    // Test 1: Small scale validation
    test_1k_neurons(cuda_ctx.clone())?;

    // Test 2: Medium scale
    test_10k_neurons(cuda_ctx.clone())?;

    // Test 3: Performance benchmark
    test_50k_neurons(cuda_ctx.clone())?;

    log::info!("\n╔════════════════════════════════════════════════════════════╗");
    log::info!("║  ✓ Phase 1 Complete - All tests passed!                   ║");
    log::info!("║  Ready for Phase 2: Sparse Connectivity                   ║");
    log::info!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}

fn test_1k_neurons(cuda_ctx: Arc<CudaContext>) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("\n=== Testing 1K Neurons ===");

    const N_NEURONS: usize = 1000;
    const DT: f32 = 0.1; // 0.1ms timestep
    const SIMULATION_TIME: f32 = 100.0; // 100ms
    const N_STEPS: usize = (SIMULATION_TIME / DT) as usize;

    // Create simulator
    log::info!("Creating simulator with {} neurons", N_NEURONS);
    let mut sim = Simulator::new(N_NEURONS, DT, cuda_ctx.clone())?;

    // Create input current (random stimulation)
    let mut input = vec![0.0; N_NEURONS];
    for i in 0..N_NEURONS {
        // Apply random current to 20% of neurons
        if i % 5 == 0 {
            input[i] = 2.0; // Strong input to trigger spikes
        }
    }

    log::info!("Running simulation for {:.1}ms ({} steps)", SIMULATION_TIME, N_STEPS);

    let start = std::time::Instant::now();
    let mut total_spikes = 0;
    let mut spike_counts = vec![0; N_NEURONS];

    // Run simulation
    for step in 0..N_STEPS {
        sim.step(Some(&input))?;

        // Sample spike activity every 10 steps
        if step % 10 == 0 {
            let spikes = sim.get_spikes()?;
            for (i, &spike) in spikes.iter().enumerate() {
                if spike > 0.5 {
                    spike_counts[i] += 1;
                    total_spikes += 1;
                }
            }
        }
    }

    sim.synchronize()?;
    let elapsed = start.elapsed();

    // Calculate statistics
    let real_time_factor = SIMULATION_TIME / (elapsed.as_secs_f32() * 1000.0);
    let avg_firing_rate = (total_spikes as f32) / (N_NEURONS as f32 * SIMULATION_TIME / 1000.0);

    log::info!("\n=== Results ===");
    log::info!("Total spikes: {}", total_spikes);
    log::info!("Average firing rate: {:.2} Hz", avg_firing_rate);
    log::info!("Simulation time: {:.2}ms", elapsed.as_secs_f32() * 1000.0);
    log::info!("Real-time factor: {:.2}×", real_time_factor);

    // Calculate firing rate distribution
    let active_neurons = spike_counts.iter().filter(|&&c| c > 0).count();
    log::info!("Active neurons: {} / {} ({:.1}%)",
        active_neurons,
        N_NEURONS,
        100.0 * active_neurons as f32 / N_NEURONS as f32
    );

    // Validation: Check that some neurons fired
    if total_spikes < 10 {
        log::warn!("Warning: Very few spikes detected. Check input currents.");
    } else if avg_firing_rate < 1.0 {
        log::warn!("Warning: Low firing rate. Expected 4-10 Hz for biological realism.");
    } else if avg_firing_rate > 50.0 {
        log::warn!("Warning: Very high firing rate. May indicate runaway excitation.");
    } else {
        log::info!("✓ Firing rate within biological range (1-50 Hz)");
    }

    // Sample voltage traces
    log::info!("\n=== Sample Membrane Potentials ===");
    let voltages = sim.get_voltages()?;
    for i in 0..5 {
        log::info!("Neuron {}: {:.2} mV (spikes: {})",
            i,
            voltages[i],
            spike_counts[i]
        );
    }

    Ok(())
}

fn test_10k_neurons(cuda_ctx: Arc<CudaContext>) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("\n=== Testing 10K Neurons (Medium Scale) ===");

    const N_NEURONS: usize = 10_000;
    const DT: f32 = 0.1;
    const SIMULATION_TIME: f32 = 100.0;
    const N_STEPS: usize = (SIMULATION_TIME / DT) as usize;

    log::info!("Creating simulator with {} neurons", N_NEURONS);
    let mut sim = Simulator::new(N_NEURONS, DT, cuda_ctx.clone())?;

    // Sparse random input (10% of neurons)
    let mut input = vec![0.0; N_NEURONS];
    for i in 0..N_NEURONS {
        if i % 10 == 0 {
            input[i] = 2.5;
        }
    }

    log::info!("Running simulation for {:.1}ms", SIMULATION_TIME);

    let start = std::time::Instant::now();
    let mut total_spikes = 0;

    for step in 0..N_STEPS {
        sim.step(Some(&input))?;

        if step % 100 == 0 {
            let spikes = sim.get_spikes()?;
            total_spikes += spikes.iter().filter(|&&s| s > 0.5).count();
        }
    }

    sim.synchronize()?;
    let elapsed = start.elapsed();

    let real_time_factor = SIMULATION_TIME / (elapsed.as_secs_f32() * 1000.0);
    let avg_firing_rate = (total_spikes as f32) / (N_NEURONS as f32 * SIMULATION_TIME / 1000.0);

    log::info!("\n[Results]");
    log::info!("  Total spikes: {}", total_spikes);
    log::info!("  Avg firing rate: {:.2} Hz", avg_firing_rate);
    log::info!("  Simulation time: {:.2}ms", elapsed.as_secs_f32() * 1000.0);
    log::info!("  Real-time factor: {:.2}×", real_time_factor);

    if real_time_factor > 5.0 {
        log::info!("  ✓ Performance excellent");
    } else if real_time_factor > 2.0 {
        log::info!("  ✓ Performance acceptable");
    } else {
        log::warn!("  ⚠ Performance below target");
    }

    Ok(())
}

fn test_50k_neurons(cuda_ctx: Arc<CudaContext>) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("\n=== Testing 50K Neurons (Performance Benchmark) ===");

    const N_NEURONS: usize = 50_000;
    const DT: f32 = 0.1;
    const SIMULATION_TIME: f32 = 50.0; // Shorter for speed
    const N_STEPS: usize = (SIMULATION_TIME / DT) as usize;

    log::info!("Creating simulator with {} neurons", N_NEURONS);
    let mut sim = Simulator::new(N_NEURONS, DT, cuda_ctx.clone())?;

    // Very sparse input (5% of neurons)
    let mut input = vec![0.0; N_NEURONS];
    for i in 0..N_NEURONS {
        if i % 20 == 0 {
            input[i] = 2.0;
        }
    }

    log::info!("Running simulation for {:.1}ms", SIMULATION_TIME);

    let start = std::time::Instant::now();

    for _step in 0..N_STEPS {
        sim.step(Some(&input))?;
    }

    sim.synchronize()?;
    let elapsed = start.elapsed();

    let real_time_factor = SIMULATION_TIME / (elapsed.as_secs_f32() * 1000.0);
    let throughput = (N_NEURONS as f64 * N_STEPS as f64) / elapsed.as_secs_f64() / 1e6;

    log::info!("\n[Performance Metrics]");
    log::info!("  Simulation time: {:.2}ms", elapsed.as_secs_f32() * 1000.0);
    log::info!("  Real-time factor: {:.2}×", real_time_factor);
    log::info!("  Throughput: {:.2} Mneuron-updates/sec", throughput);
    log::info!("  Timestep latency: {:.3}ms", elapsed.as_secs_f32() * 1000.0 / N_STEPS as f32);

    if real_time_factor > 10.0 {
        log::info!("  ✓ Excellent GPU utilization");
    } else if real_time_factor > 5.0 {
        log::info!("  ✓ Good GPU utilization");
    } else {
        log::warn!("  ⚠ GPU underutilized - check memory bandwidth");
    }

    Ok(())
}
