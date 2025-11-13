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

    /// Interactive chat with GPU brain
    Chat {
        /// Vocabulary size
        #[arg(long, default_value_t = 5000)]
        vocab: usize,

        /// Pattern dimension
        #[arg(long, default_value_t = 256)]
        pattern_dim: usize,

        /// Working memory capacity
        #[arg(long, default_value_t = 7)]
        wm_capacity: usize,
    },
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
        Some(Commands::Chat {
            vocab,
            pattern_dim,
            wm_capacity,
        }) => {
            run_chat_interface(vocab, pattern_dim, wm_capacity)?;
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
    println!("Commands:");
    println!("  neurox-ai info    - Display GPU device information");
    println!("  neurox-ai chat    - Interactive chat with GPU brain (100% GPU!)");
    println!();
    println!("Chat with the brain:");
    println!("  cargo run --release chat");
    println!();
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

/// Run interactive chat interface with GPU brain
fn run_chat_interface(
    vocab: usize,
    pattern_dim: usize,
    wm_capacity: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{self, Write};
    use neurox_ai::GpuBrain;
    use cudarc::driver::CudaDevice;

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     NeuroxAI Neural Processor - Interactive Console       ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("Initializing neural processor...");

    // Initialize CUDA device (CudaDevice::new already returns Arc)
    let device = CudaDevice::new(0)?;
    println!("✓ CUDA device initialized: {}", device.name()?);

    // Create neural processor
    let mut brain = GpuBrain::new(device, vocab, pattern_dim, wm_capacity)?;
    println!("✓ Neural processor ready");
    println!();

    // Display stats
    let stats = brain.stats()?;
    println!("{}", stats);
    println!();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Neural processor online. Ready for interaction.          ║");
    println!("║  Commands:                                                 ║");
    println!("║    /stats   - System statistics                            ║");
    println!("║    /train   - Toggle learning mode                         ║");
    println!("║    /vocab   - Display vocabulary                           ║");
    println!("║    /gen <text> - Generate from prompt                      ║");
    println!("║    quit     - Shutdown                                     ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let mut training_mode = false;

    loop {
        // Print prompt
        if training_mode {
            print!("learn> ");
        } else {
            print!("> ");
        }
        io::stdout().flush()?;

        // Read user input
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // Handle commands
        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" {
            println!("→ Neural processor shutdown initiated");
            break;
        }

        if input == "/stats" {
            let stats = brain.stats()?;
            println!("{}", stats);
            continue;
        }

        if input == "/train" {
            training_mode = !training_mode;
            if training_mode {
                println!("→ Learning mode enabled");
            } else {
                println!("→ Learning mode disabled");
            }
            continue;
        }

        if input == "/vocab" {
            let vocab = brain.vocabulary();
            println!("Learned vocabulary ({} words):", vocab.len());
            if vocab.is_empty() {
                println!("  (no words learned yet - start training!)");
            } else {
                let mut sorted_vocab = vocab;
                sorted_vocab.sort();
                for (i, word) in sorted_vocab.iter().enumerate() {
                    if i % 10 == 0 && i > 0 {
                        println!();
                    }
                    print!("{:12}", word);
                }
                println!();
            }
            continue;
        }

        if input.starts_with("/gen ") {
            let prompt = input.strip_prefix("/gen ").unwrap();
            println!("Generating from: \"{}\"", prompt);
            match brain.generate_text_gpu(prompt, 20) {
                Ok(generated) => {
                    println!("brain> {}", generated);
                }
                Err(e) => {
                    println!("✗ Error generating: {}", e);
                }
            }
            println!();
            continue;
        }

        // Process input
        if training_mode {
            match brain.train_on_text_gpu(input) {
                Ok(_) => println!("✓ Learned"),
                Err(e) => println!("✗ Error: {}", e),
            }
        } else {
            match brain.process_text_gpu(input) {
                Ok(response) => {
                    println!("← {}", response);
                }
                Err(e) => {
                    println!("✗ Error: {}", e);
                }
            }
        }

        println!();
    }

    Ok(())
}
