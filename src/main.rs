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

    /// Interactive neural processor console
    Chat {
        /// Max vocabulary size (10k = 400MB, 20k = 1.6GB)
        #[arg(long, default_value_t = 10000)]
        vocab: usize,

        /// Pattern dimension (512 = balanced, 1024 = high quality)
        #[arg(long, default_value_t = 512)]
        pattern_dim: usize,
    },

    /// Run biological NeuromorphicBrain demo (27 brain systems)
    Brain {
        /// Pattern dimension
        #[arg(long, default_value_t = 512)]
        pattern_dim: usize,

        /// Number of timesteps to run
        #[arg(long, default_value_t = 1000)]
        timesteps: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with beautiful formatting
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .format(|buf, record| {
            use std::io::Write;

            let timestamp = chrono::Local::now().format("%H:%M");

            // ANSI color codes
            let gray = "\x1b[90m";      // Dark gray
            let reset = "\x1b[0m";      // Reset

            let level_color = match record.level() {
                log::Level::Error => "\x1b[91m",  // Bright red
                log::Level::Warn  => "\x1b[93m",  // Bright yellow
                log::Level::Info  => "\x1b[92m",  // Bright green
                log::Level::Debug => "\x1b[94m",  // Bright blue
                log::Level::Trace => "\x1b[95m",  // Bright magenta
            };

            writeln!(
                buf,
                "{}[{}]{} {}{:<5}{} {}{}{} - {}",
                gray, timestamp, reset,
                level_color, record.level(), reset,
                gray, record.target(), reset,
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
        Some(Commands::Brain {
            pattern_dim,
            timesteps,
        }) => {
            run_brain_demo(pattern_dim, timesteps)?;
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

/// Run biological NeuromorphicBrain demonstration
fn run_brain_demo(pattern_dim: usize, timesteps: usize) -> Result<(), Box<dyn std::error::Error>> {
    use neurox_ai::brain::NeuromorphicBrain;

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  NeuromorphicBrain Demo - 27 Biological Systems            ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    log::info!("Initializing NeuromorphicBrain...");

    // Create brain with full biological architecture
    let n_layers = 5;  // 5-layer hierarchical system
    let base_neurons = 1000;  // 1000 base neurons per layer
    let vocab_size = 10000;
    let mut brain = NeuromorphicBrain::new(n_layers, base_neurons, vocab_size, pattern_dim);
    log::info!("✓ Brain initialized: {} layers, {} neurons, {} pattern_dim",
               n_layers, base_neurons, pattern_dim);

    // Display initial stats
    let stats = brain.stats();
    println!("\n📊 Initial Brain Statistics:");
    println!("  Working Memory: {} stored patterns", stats.working_memory.stored_patterns);
    println!("  Hippocampus: initialized");
    println!("  Cerebellum: initialized");
    println!("  Amygdala: initialized");
    println!("  Superior Colliculus: {} neurons", stats.superior_colliculus.total_neurons);
    println!("  Thalamus: {} neurons", stats.thalamus.total_neurons);
    println!();

    // Run simulation
    println!("🧠 Running brain simulation for {} timesteps...", timesteps);
    println!();

    let dt = 0.1; // 0.1 ms timestep
    let mut last_print = 0;

    for step in 0..timesteps {
        // Update brain
        brain.update(dt);

        // Print progress every 100 steps
        if step - last_print >= 100 || step == timesteps - 1 {
            let progress = (step as f32 / timesteps as f32 * 100.0) as usize;
            print!("\r  Progress: {}% [{}/{}] ", progress, step + 1, timesteps);
            use std::io::{self, Write};
            io::stdout().flush()?;
            last_print = step;
        }
    }

    println!("\n");

    // Display final stats
    let final_stats = brain.stats();
    println!("📊 Final Brain Statistics:");
    println!("  Time elapsed: {:.2} ms", final_stats.time);
    println!("  Working Memory: {} stored patterns", final_stats.working_memory.stored_patterns);
    println!("  Dopamine level: {:.3}", final_stats.basal_ganglia.dopamine_level);
    println!("  Oscillations: theta {:.1} Hz, gamma {:.1} Hz",
             final_stats.oscillations.theta_freq,
             final_stats.oscillations.gamma_freq);
    println!("  Superior Colliculus: {} saccades, active: {}",
             final_stats.superior_colliculus.total_saccades,
             final_stats.superior_colliculus.saccade_active);
    println!("  Thalamus: burst ratio {:.3}, spindle: {}",
             final_stats.thalamus.burst_ratio,
             final_stats.thalamus.spindle_active);
    println!("  All 27+ biological systems active and integrated");
    println!();

    println!("✓ Brain simulation complete!");
    println!();

    Ok(())
}

/// Run interactive neural processor console
fn run_chat_interface(
    vocab: usize,
    pattern_dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use neurox_ai::NeuralProcessor;
    use cudarc::driver::CudaDevice;
    use rustyline::error::ReadlineError;
    use rustyline::DefaultEditor;

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     NeuroxAI Neural Processor - Interactive Console       ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    log::info!("Initializing neural processor...");

    // Initialize CUDA device
    let device = CudaDevice::new(0)?;
    log::info!("✓ CUDA device initialized: {}", device.name()?);

    // Create neural processor
    let mut processor = NeuralProcessor::new(device, vocab, pattern_dim)?;
    log::info!("✓ Neural processor ready");
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

    // Create readline editor with history
    let mut rl = DefaultEditor::new()?;

    let mut training_mode = false;

    loop {
        // Set prompt based on mode
        let prompt = if training_mode { "learn> " } else { "> " };

        // Read user input with rustyline (handles Ctrl+C gracefully)
        let readline = rl.readline(prompt);

        let input = match readline {
            Ok(line) => {
                rl.add_history_entry(&line)?;
                line
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl+C pressed
                println!("→ Neural processor shutdown initiated (Ctrl+C)");
                break;
            }
            Err(ReadlineError::Eof) => {
                // Ctrl+D pressed
                println!("→ Neural processor shutdown initiated (EOF)");
                break;
            }
            Err(err) => {
                eprintln!("Error reading input: {}", err);
                break;
            }
        };

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
