//! Commands Module
//!
//! CLI command definitions and handlers.

pub mod benchmark;
pub mod chat;
pub mod info;
pub mod solve;

use crate::VERSION;
use clap::{Parser, Subcommand};

/// CLI argument parser
#[derive(Parser)]
#[command(name = "neurox-ai")]
#[command(version = VERSION)]
#[command(about = "GPU-accelerated neuromorphic computing platform", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

/// Available CLI commands
#[derive(Subcommand)]
pub enum Commands {
    /// Display system and GPU information
    Info,
    /// Interactive chat with the neuromorphic brain
    Chat {
        /// Vocabulary size
        #[arg(long)]
        vocab: Option<usize>,
        /// Pattern dimension
        #[arg(long)]
        pattern_dim: Option<usize>,
        /// Total neurons in the brain
        #[arg(long)]
        neurons: Option<usize>,
        /// Context window size
        #[arg(long)]
        context: Option<usize>,
        /// Dopamine sensitivity (reward scaling)
        #[arg(long)]
        sensitivity: Option<f32>,
    },
    /// Solve problems (math, chemistry)
    Solve {
        /// Type: math, chemistry
        #[arg(long, default_value = "math")]
        problem_type: String,
        /// Problem to solve
        problem: String,
    },
    /// Benchmark MNIST accuracy with quantization
    Benchmark {
        /// Path to MNIST data directory (or 'synthetic' for generated data)
        #[arg(long)]
        data_dir: Option<String>,
        /// Number of epochs
        #[arg(long)]
        epochs: Option<usize>,
        /// Quantization bits (4 or 8)
        #[arg(long)]
        bits: Option<u8>,
        /// Number of hidden neurons
        #[arg(long)]
        neurons: Option<usize>,
        /// Presentation duration per image (ms)
        #[arg(long)]
        duration: Option<f32>,
        /// Inter-stimulus interval (ms)
        #[arg(long)]
        isi: Option<f32>,
    },
}
