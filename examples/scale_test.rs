//! Large-Scale Neuromorphic Performance Test
//!
//! Validates event-driven processing at massive scale
//! Target: 20-50× realtime performance with biological sparsity

use neurox_ai::*;
use std::sync::Arc;
use std::time::Instant;
use clap::Parser;

#[derive(Parser)]
#[command(name = "scale_test")]
#[command(about = "GPU-accelerated neuromorphic scaling benchmark")]
struct Args {
    /// Number of neurons to simulate
    #[arg(short, long, default_value = "100000")]
    neurons: usize,

    /// Simulation duration in biological milliseconds
    #[arg(short, long, default_value = "1000.0")]
    duration: f32,

    /// Connection probability (sparsity)
    #[arg(short, long, default_value = "0.01")]
    sparsity: f64,

    /// Input activity (% of neurons receiving input)
    #[arg(short, long, default_value = "0.01")]
    input_activity: f32,

    /// Input current strength (mA)
    #[arg(long, default_value = "50.0")]
    input_current: f32,

    /// Run multiple scales (ignores --neurons)
    #[arg(long)]
    multi_scale: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with colors
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .format(|buf, record| {
            use std::io::Write;
            use env_logger::fmt::Color;

            let mut style = buf.style();
            let level_color = match record.level() {
                log::Level::Error => Color::Red,
                log::Level::Warn => Color::Yellow,
                log::Level::Info => Color::Green,
                log::Level::Debug => Color::Blue,
                log::Level::Trace => Color::Cyan,
            };
            style.set_color(level_color).set_bold(true);

            writeln!(
                buf,
                "{} {}",
                style.value(record.level()),
                record.args()
            )
        })
        .init();

    let args = Args::parse();

    log::info!("=== NeuroxAI Large-Scale Performance Test ===");
    log::info!("Target: 20-50× realtime @ biological sparsity");

    // Initialize CUDA context
    log::info!("Initializing GPU...");
    let cuda_ctx = Arc::new(CudaContext::default()?);
    log::info!("{}", cuda_ctx.device_info()?);

    if args.multi_scale {
        // Test multiple scales
        let test_configs = vec![
            ("100K neurons", 100_000),
            ("250K neurons", 250_000),
            ("500K neurons", 500_000),
            ("1M neurons", 1_000_000),
        ];

        for (name, n_neurons) in test_configs {
            log::info!("\n=== {} ===", name);
            run_scale_test(
                n_neurons,
                args.duration,
                args.sparsity,
                args.input_activity,
                args.input_current,
                cuda_ctx.clone()
            )?;
        }
    } else {
        // Single configuration
        log::info!("\n=== {} neurons ===", args.neurons);
        run_scale_test(
            args.neurons,
            args.duration,
            args.sparsity,
            args.input_activity,
            args.input_current,
            cuda_ctx.clone()
        )?;
    }

    Ok(())
}

fn run_scale_test(
    n_neurons: usize,
    sim_duration_ms: f32,
    connection_prob: f64,
    input_activity: f32,
    input_current: f32,
    cuda_ctx: Arc<CudaContext>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create sparse connectivity (small-world)
    log::info!("Creating sparse connectivity...");
    log::info!("  Connection probability: {:.2}%", connection_prob * 100.0);
    let connectivity_gen = ProceduralConnectivity {
        seed: 42,
        connection_prob,
        weight_mean: 0.5,
        weight_std: 0.2,
        topology: ConnectivityType::SmallWorld { k: 10, beta: 0.3 },
        exc_ratio: 0.8, // Dale's principle
    };

    let connectivity = SparseConnectivity::from_procedural(n_neurons, &connectivity_gen);
    log::info!("  Synapses: {}", connectivity.nnz);
    log::info!("  Memory: {:.2} MB", connectivity.memory_footprint() as f64 / 1024.0 / 1024.0);
    log::info!("  Sparsity: {:.4}%", connectivity.sparsity());

    // Initialize simulator
    log::info!("Creating simulator with event-driven processing...");
    let dt = 0.1; // 0.1ms timestep
    let mut simulator = Simulator::with_connectivity(n_neurons, dt, cuda_ctx, &connectivity)?;

    // Simulation parameters
    let n_steps = (sim_duration_ms / dt) as usize;

    // Generate sparse random input
    let n_active_input = (n_neurons as f32 * input_activity) as usize;
    log::info!("  Input activity: {} neurons ({:.2}%)", n_active_input, input_activity * 100.0);
    log::info!("  Input current: {} mA", input_current);
    let mut input = vec![0.0; n_neurons];
    for i in 0..n_active_input {
        input[i] = input_current;
    }

    // Benchmark
    log::info!("Running simulation ({} ms biological time, {} steps)...", sim_duration_ms, n_steps);
    let start = Instant::now();

    let mut total_spikes = 0u64;
    for step in 0..n_steps {
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
    let sim_time_sec = sim_duration_ms / 1000.0;
    let realtime_factor = sim_time_sec / elapsed.as_secs_f32();

    log::info!("\n=== Results ===");
    log::info!("  Wall time: {:.2}s", elapsed.as_secs_f32());
    log::info!("  Simulated time: {:.2}s biological", sim_time_sec);
    log::info!("  Realtime factor: {:.2}×", realtime_factor);
    log::info!("  Total spikes: {}", total_spikes);
    log::info!("  Avg firing rate: {:.2} Hz",
               total_spikes as f32 / n_neurons as f32 / sim_time_sec);
    log::info!("  Throughput: {:.2} Mneuron-updates/sec",
               (n_neurons * n_steps) as f64 / elapsed.as_secs_f64() / 1_000_000.0);

    // Check target
    if realtime_factor >= 20.0 {
        log::info!("  ✅ Target achieved (20-50× realtime)");
    } else {
        log::warn!("  ⚠️ Below target (got {:.2}×, target 20×)", realtime_factor);
    }

    Ok(())
}
