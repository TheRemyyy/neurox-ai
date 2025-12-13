//! NeuroxAI - Neuromorphic Computing System
//!
//! GPU-accelerated spiking neural network platform targeting 1-10M neurons
//! with biological accuracy and real-time performance.

use clap::{Parser, Subcommand};
use neurox_ai::{CudaContext, VERSION};
use std::sync::Arc;
use std::io::{Write, stdout};
use std::thread;
use std::time::Duration;

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
            let timestamp = chrono::Local::now().format("%H:%M");
            let gray = "\x1b[90m";
            let reset = "\x1b[0m";
            let level_color = match record.level() {
                log::Level::Error => "\x1b[91m",
                log::Level::Warn  => "\x1b[93m",
                log::Level::Info  => "\x1b[92m",
                log::Level::Debug => "\x1b[94m",
                log::Level::Trace => "\x1b[95m",
            };
            writeln!(buf, "{}[{}]{} {}{:<5}{} {}", gray, timestamp, reset, level_color, record.level(), reset, record.args())
        })
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Info) => {
            display_system_info()?;
        }
        Some(Commands::Chat { vocab, pattern_dim }) => {
            run_chat_interface(vocab, pattern_dim)?;
        }
        None => {
            display_welcome();
        }
    }

    Ok(())
}

fn display_welcome() {
    println!("\x1b[1;36m╔════════════════════════════════════════════════════════════╗");
    println!("║  NeuroxAI v{} - Neuromorphic Computing Platform  ║", VERSION);
    println!("║  GPU-Accelerated Spiking Neural Networks                  ║");
    println!("╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();
    println!("Commands:");
    println!("  neurox-ai info    - Display GPU device information");
    println!("  neurox-ai chat    - Interactive neural processor console");
    println!();
}

fn display_system_info() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing GPU context...\n");
    let cuda_ctx = Arc::new(CudaContext::default()?);
    let device_info = cuda_ctx.device_info()?;
    println!("{}", device_info);
    println!("\nSystem ready for neuromorphic computation.");
    Ok(())
}

fn run_chat_interface(vocab: usize, pattern_dim: usize) -> Result<(), Box<dyn std::error::Error>> {
    use neurox_ai::brain::NeuromorphicBrain;
    use rustyline::error::ReadlineError;
    use rustyline::DefaultEditor;

    println!("\x1b[1;36m╔════════════════════════════════════════════════════════════╗");
    println!("║     NeuromorphicBrain - Interactive Console (27+ Systems) ║");
    println!("╚════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();
    
    print!("Initializing biological brain systems... ");
    stdout().flush()?;
    
    // Simulate loading progress
    for _ in 0..5 {
        print!(".");
        stdout().flush()?;
        thread::sleep(Duration::from_millis(200));
    }
    println!();

    let n_layers = 5;
    let base_neurons = 1000;
    let mut brain = NeuromorphicBrain::new(n_layers, base_neurons, vocab, pattern_dim);
    println!("\x1b[32m✓ Brain initialized: {} layers, {} base neurons\x1b[0m", n_layers, base_neurons);
    println!();

    let mut rl = DefaultEditor::new()?;

    loop {
        // Dashboard
        let stats = brain.stats();
        print_dashboard(&stats);

        let readline = rl.readline("\x1b[1;33mUser > \x1b[0m");

        let input = match readline {
            Ok(line) => {
                rl.add_history_entry(&line)?;
                line
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(err) => {
                eprintln!("Error: {}", err);
                break;
            }
        };

        let input = input.trim();
        if input.is_empty() { continue; }
        if input == "quit" || input == "exit" { break; }

        if input.starts_with("/") {
            // Handle commands
            match input {
                "/sleep" => {
                    println!("\x1b[35m[z] Initiating Sleep Consolidation Phase...\x1b[0m");
                    for _ in 0..10 {
                        print!("zzZ ");
                        stdout().flush()?;
                        thread::sleep(Duration::from_millis(100));
                        brain.consolidate();
                    }
                    println!("\n\x1b[32m[✓] Consolidation Complete.\x1b[0m");
                },
                "/reward" => {
                    println!("\x1b[33m[$] Injecting Dopamine Reward...\x1b[0m");
                    // Mock state for reward learning
                    let state = vec![0.5; 512]; 
                    brain.learn_from_reward(&state, 0, 1.0, &state);
                },
                _ => println!("Unknown command."),
            }
            continue;
        }

        // Process text with "Thinking" visualization
        println!("\x1b[90mThinking... [Dual-Stream Lang] -> [Semantic Hub] -> [Hippocampus]\x1b[0m");
        thread::sleep(Duration::from_millis(300)); // Simulate processing latency
        
        let response = brain.process_text(input);
        
        println!("\x1b[1;36mBrain > \x1b[0m{}", response);
        
        // Post-processing simulation (Structural Plasticity, etc.)
        print!("\x1b[90mUpdating Synapses...\x1b[0m");
        for _ in 0..5 {
            brain.update(0.1);
            print!(".");
            stdout().flush()?;
            thread::sleep(Duration::from_millis(50));
        }
        println!();
        println!();
    }

    Ok(())
}

fn print_bar(val: f32, max: f32, len: usize, color_code: &str) -> String {
    let filled = ((val / max).clamp(0.0, 1.0) * len as f32) as usize;
    let bar: String = "█".repeat(filled) + &"░".repeat(len - filled);
    format!("{}{}{}\x1b[0m", color_code, bar, "\x1b[0m")
}

fn print_dashboard(stats: &neurox_ai::brain::BrainStats) {
    println!("\x1b[1m┌────────────────────── BRAIN STATE ──────────────────────┐\x1b[0m");
    println!("│ Dopamine (DA):  {} │ {:.3}", print_bar(stats.neuromodulation.dopamine_level, 1.0, 10, "\x1b[33m"), stats.neuromodulation.dopamine_level);
    println!("│ Serotonin (5HT):{} │ {:.3}", print_bar(stats.neuromodulation.sht_level, 1.0, 10, "\x1b[35m"), stats.neuromodulation.sht_level);
    println!("│ Norepin. (NE):  {} │ {:.3}", print_bar(stats.neuromodulation.ne_level, 1.0, 10, "\x1b[31m"), stats.neuromodulation.ne_level);
    println!("│ Theta Phase:    {} │ {:.2} Hz", print_bar(stats.oscillations.theta_phase, 1.0, 10, "\x1b[34m"), stats.oscillations.theta_freq);
    println!("│ Working Mem:    {} │ {} items", print_bar(stats.working_memory.utilization, 1.0, 10, "\x1b[32m"), stats.working_memory.active_count);
    println!("\x1b[1m└─────────────────────────────────────────────────────────┘\x1b[0m");
}