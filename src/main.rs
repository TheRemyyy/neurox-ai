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
    println!("║         GPU Brain Interactive Chat Interface              ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("Initializing GPU brain...");

    // Initialize CUDA device (CudaDevice::new already returns Arc)
    let device = CudaDevice::new(0)?;
    println!("✓ CUDA device initialized: {}", device.name()?);

    // Create GPU brain
    let mut brain = GpuBrain::new(device, vocab, pattern_dim, wm_capacity)?;
    println!("✓ GPU brain created");
    println!();

    // Display stats
    let stats = brain.stats()?;
    println!("{}", stats);
    println!();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Brain is ready! Start chatting (type 'quit' to exit)    ║");
    println!("║  Commands:                                                 ║");
    println!("║    /stats   - Show brain statistics                        ║");
    println!("║    /train   - Toggle training mode                         ║");
    println!("║    /vocab   - Show learned vocabulary                      ║");
    println!("║    /gen <text> - Generate continuation from prompt         ║");
    println!("║    quit     - Exit                                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let mut training_mode = false;

    loop {
        // Print prompt
        if training_mode {
            print!("train> ");
        } else {
            print!("you> ");
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
            println!("Shutting down brain...");
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
                println!("Training mode ON - your messages will train the brain");
            } else {
                println!("Training mode OFF");
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
            println!("Training on: \"{}\"", input);
            match brain.train_on_text_gpu(input) {
                Ok(_) => println!("✓ Trained"),
                Err(e) => println!("✗ Error training: {}", e),
            }
        } else {
            println!("Processing: \"{}\"", input);
            match brain.process_text_gpu(input) {
                Ok(response) => {
                    println!("brain> {}", response);
                }
                Err(e) => {
                    println!("✗ Error: {}", e);
                }
            }
        }

        println!();
    }

    println!("Goodbye!");
    Ok(())
}
