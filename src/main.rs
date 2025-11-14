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

    /// Interactive NeuromorphicBrain console (27+ biological systems)
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

/// Run interactive NeuromorphicBrain console with all 27+ biological systems
fn run_chat_interface(
    vocab: usize,
    pattern_dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use neurox_ai::brain::NeuromorphicBrain;
    use rustyline::error::ReadlineError;
    use rustyline::DefaultEditor;

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     NeuromorphicBrain - Interactive Console (27+ Systems) ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    log::info!("Initializing biological brain systems...");

    // Create biological brain with full architecture
    let n_layers = 5;
    let base_neurons = 1000;
    let mut brain = NeuromorphicBrain::new(n_layers, base_neurons, vocab, pattern_dim);
    log::info!("✓ Brain initialized: {} layers, {} base neurons, {} vocab",
               n_layers, base_neurons, vocab);
    println!();

    // Display initial stats
    let stats = brain.stats();
    println!("🧠 Brain Systems Online:");
    println!("  • Working Memory: {} patterns", stats.working_memory.stored_patterns);
    println!("  • Dopamine: {:.3}", stats.basal_ganglia.dopamine_level);
    println!("  • Oscillations: θ={:.1}Hz γ={:.1}Hz",
             stats.oscillations.theta_freq, stats.oscillations.gamma_freq);
    println!("  • Superior Colliculus: {} neurons", stats.superior_colliculus.total_neurons);
    println!("  • Thalamus: {} neurons", stats.thalamus.total_neurons);
    println!("  • All 27+ biological systems active");
    println!();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Biological brain online. Ready for interaction.          ║");
    println!("║  Commands:                                                 ║");
    println!("║    /stats   - Brain system statistics                      ║");
    println!("║    /systems - List all 27+ active systems                  ║");
    println!("║    quit     - Shutdown (or Ctrl+C)                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Create readline editor with history
    let mut rl = DefaultEditor::new()?;

    loop {
        // Read user input with rustyline (handles Ctrl+C gracefully)
        let readline = rl.readline("🧠> ");

        let input = match readline {
            Ok(line) => {
                rl.add_history_entry(&line)?;
                line
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl+C pressed
                println!("→ Brain shutdown initiated (Ctrl+C)");
                break;
            }
            Err(ReadlineError::Eof) => {
                // Ctrl+D pressed
                println!("→ Brain shutdown initiated (EOF)");
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
            println!("→ Brain shutdown initiated");
            break;
        }

        if input == "/stats" {
            let stats = brain.stats();
            println!("\n📊 Brain Statistics:");
            println!("  Time: {:.2} ms", stats.time);
            println!("  Working Memory: {} patterns, {:.1}% capacity",
                     stats.working_memory.stored_patterns,
                     stats.working_memory.utilization * 100.0);
            println!("  Dopamine: {:.3}", stats.basal_ganglia.dopamine_level);
            println!("  ACh: {:.3}, NE: {:.3}",
                     stats.neuromodulation.ach_level,
                     stats.neuromodulation.ne_level);
            println!("  Oscillations: θ={:.1}Hz γ={:.1}Hz",
                     stats.oscillations.theta_freq,
                     stats.oscillations.gamma_freq);
            println!("  Superior Colliculus: {} saccades",
                     stats.superior_colliculus.total_saccades);
            println!("  Thalamus: burst ratio {:.3}",
                     stats.thalamus.burst_ratio);
            println!();
            continue;
        }

        if input == "/systems" {
            println!("\n🧠 Active Brain Systems (27+):");
            println!("  1. Working Memory          15. Oscillations");
            println!("  2. Hippocampus             16. Interneurons");
            println!("  3. Basal Ganglia           17. Homeostasis");
            println!("  4. Language (Dual-Stream)  18. Predictive Coding");
            println!("  5. Attention System        19. Cerebellum");
            println!("  6. Neuromodulation         20. Amygdala");
            println!("  7. Superior Colliculus     21. Spatial Navigation");
            println!("  8. Thalamus                22. Semantic System");
            println!("  9. R-STDP                  23. V1 Orientation");
            println!(" 10. ETDP                    24. Cochlea");
            println!(" 11. Memristive Network      25. MT/MST Motion");
            println!(" 12. CAdEx Neurons           26. Barrel Cortex");
            println!(" 13. Izhikevich Neurons      27. Sleep Consolidation");
            println!(" 14. Structural Plasticity   + Heterosynaptic...");
            println!();
            continue;
        }

        // Process text through full biological brain
        let response = brain.process_text(input);
        println!("🧠← {}", response);

        // Update brain continuously
        brain.update(0.1);

        println!();
    }

    Ok(())
}
