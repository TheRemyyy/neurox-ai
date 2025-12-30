//! NeuroxAI - Neuromorphic Computing System
//!
//! GPU-accelerated spiking neural network platform targeting 1-10M neurons
//! with biological accuracy and real-time performance.

use clap::Parser;
use neurox_ai::commands::{Cli, Commands};
use neurox_ai::VERSION;

// CLI Colors - white, gray, light blue only
const COLOR_RESET: &str = "\x1b[0m";
const COLOR_LIGHT_BLUE: &str = "\x1b[94m";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Quiet logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Info) => neurox_ai::commands::info::run()?,
        Some(Commands::Chat {
            vocab,
            pattern_dim,
            neurons,
            context,
            sensitivity,
        }) => neurox_ai::commands::chat::run(vocab, pattern_dim, neurons, context, sensitivity)?,
        Some(Commands::Solve {
            problem_type,
            problem,
        }) => neurox_ai::commands::solve::run(&problem_type, &problem)?,
        Some(Commands::Benchmark {
            data_dir,
            epochs,
            bits,
            neurons,
            duration,
            isi,
        }) => neurox_ai::commands::benchmark::run(data_dir, epochs, bits, neurons, duration, isi)?,
        None => display_welcome(),
    }

    Ok(())
}

fn display_welcome() {
    println!(
        "{}╔════════════════════════════════════════════════════════════╗",
        COLOR_LIGHT_BLUE
    );
    println!(
        "║  NeuroxAI v{} - Neuromorphic Computing Platform  ║",
        VERSION
    );
    println!("║  GPU-Accelerated Spiking Neural Networks                  ║");
    println!(
        "╚════════════════════════════════════════════════════════════╝{}",
        COLOR_RESET
    );
}
