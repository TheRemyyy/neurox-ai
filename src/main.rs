//! NeuroxAI - Neuromorphic Computing System
//!
//! GPU-accelerated spiking neural network platform targeting 1-10M neurons
//! with biological accuracy and real-time performance.

use clap::{Parser, Subcommand};
use neurox_ai::{CudaContext, VERSION};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "neurox-ai")]
#[command(version = VERSION)]
#[command(about = "GPU-accelerated neuromorphic computing platform", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Display GPU device information
    Info,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Info) => {
            display_system_info()?;
        }
        None => {
            display_welcome();
        }
    }

    Ok(())
}

fn display_welcome() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  NeuroxAI v{} - Neuromorphic Computing Platform  ║", VERSION);
    println!("║  GPU-Accelerated Spiking Neural Networks                  ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("Available examples:");
    println!("  cargo run --release --example mnist_training  - Train on MNIST (configurable)");
    println!("  cargo run --release --example scale_test_1m   - Performance benchmarks (configurable)");
    println!();
    println!("Example usage:");
    println!("  cargo run --release --example scale_test_1m -- --neurons 500000 --duration 2000");
    println!("  cargo run --release --example mnist_training -- --hidden 1000 --epochs 20");
    println!();
    println!("Commands:");
    println!("  neurox-ai info    - Display GPU device information");
    println!();
    println!("For documentation, see PROJECT_STATUS.md");
}

fn display_system_info() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing GPU context...\n");

    let cuda_ctx = Arc::new(CudaContext::default()?);
    let device_info = cuda_ctx.device_info()?;

    println!("{}", device_info);
    println!();
    println!("System ready for neuromorphic computation.");

    Ok(())
}
