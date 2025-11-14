//! NeuroxAI - Neuromorphic Computing System
//!
//! GPU-accelerated spiking neural network platform targeting 1-10M neurons
//! with biological accuracy and real-time performance.

use clap::{Parser, Subcommand};
use neurox_ai::{CudaContext, VERSION};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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

    /// Interactive neural processor console
    Chat {
        /// Max vocabulary size (10k = 400MB, 20k = 1.6GB)
        #[arg(long, default_value_t = 10000)]
        vocab: usize,

        /// Pattern dimension (512 = balanced, 1024 = high quality)
        #[arg(long, default_value_t = 512)]
        pattern_dim: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with custom time format (HH:MM)
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp(Some(env_logger::fmt::TimestampPrecision::Seconds))
        .format(|buf, record| {
            use std::io::Write;
            let timestamp = chrono::Local::now().format("%H:%M");
            writeln!(
                buf,
                "[{}] {}",
                timestamp,
                record.args()
            )
        })
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Info) => {
            display_system_info()?;
        }
        Some(Commands::Chat {
            vocab,
            pattern_dim,
        }) => {
            run_chat_interface(vocab, pattern_dim)?;
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
    println!("  neurox-ai chat    - Interactive neural processor console");
    println!();
    println!("Quick start:");
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

/// Run interactive neural processor console
fn run_chat_interface(
    vocab: usize,
    pattern_dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{self, Write};
    use neurox_ai::NeuralProcessor;
    use cudarc::driver::CudaDevice;

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     NeuroxAI Neural Processor - Interactive Console       ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("Initializing neural processor...");

    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    println!("✓ CUDA device initialized: {}", device.name()?);

    // Create neural processor
    let mut processor = NeuralProcessor::new(device, vocab, pattern_dim)?;
    println!("✓ Neural processor ready");
    println!();

    // Display stats
    let stats = processor.stats()?;
    println!("{}", stats);
    println!();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Neural processor online. Ready for interaction.          ║");
    println!("║  Commands:                                                 ║");
    println!("║    /stats   - System statistics                            ║");
    println!("║    /train   - Toggle learning mode                         ║");
    println!("║    /vocab   - Display vocabulary                           ║");
    println!("║    /gen <text> - Generate from prompt                      ║");
    println!("║    quit     - Shutdown (or Ctrl+C)                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Setup Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })?;

    let mut training_mode = false;

    loop {
        // Check if interrupted
        if !running.load(Ordering::SeqCst) {
            println!("\n→ Neural processor shutdown initiated (Ctrl+C)");
            break;
        }

        // Print prompt
        if training_mode {
            print!("learn> ");
        } else {
            print!("> ");
        }
        io::stdout().flush()?;

        // Read user input
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {},
            Err(_) => {
                // Likely Ctrl+C during read
                if !running.load(Ordering::SeqCst) {
                    println!("\n→ Neural processor shutdown initiated (Ctrl+C)");
                    break;
                }
                continue;
            }
        }
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
            let stats = processor.stats()?;
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
            let vocab = processor.vocabulary();
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
            match processor.generate_text(prompt, 20) {
                Ok(generated) => {
                    println!("← {}", generated);
                }
                Err(e) => {
                    println!("✗ Error: {}", e);
                }
            }
            println!();
            continue;
        }

        // Process input
        if training_mode {
            match processor.train_on_text(input) {
                Ok(_) => println!("✓ Learned"),
                Err(e) => println!("✗ Error: {}", e),
            }
        } else {
            match processor.process_text(input) {
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
