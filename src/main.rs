//! NeuroxAI - Neuromorphic Computing System
//!
//! GPU-accelerated spiking neural network platform targeting 1-10M neurons
//! with biological accuracy and real-time performance.

use clap::{Parser, Subcommand};
use neurox_ai::brain::NeuromorphicBrain;
use neurox_ai::language::PartOfSpeech;
use neurox_ai::{CudaContext, VERSION};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::io::{stdout, Write};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Convert JSON POS string to PartOfSpeech enum
fn pos_from_string(s: &str) -> PartOfSpeech {
    match s.to_lowercase().as_str() {
        "noun" => PartOfSpeech::Noun,
        "verb" => PartOfSpeech::Verb,
        "adjective" => PartOfSpeech::Adjective,
        "adverb" => PartOfSpeech::Adverb,
        "pronoun" => PartOfSpeech::Pronoun,
        "preposition" => PartOfSpeech::Preposition,
        "conjunction" => PartOfSpeech::Conjunction,
        "interjection" => PartOfSpeech::Interjection,
        "determiner" => PartOfSpeech::Determiner,
        "particle" => PartOfSpeech::Particle,
        _ => PartOfSpeech::Unknown,
    }
}
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
    Info,
    Chat {
        #[arg(long, default_value_t = 10000)]
        vocab: usize,
        #[arg(long, default_value_t = 512)]
        pattern_dim: usize,
    },
}

// --- TUI CONSTANTS & COLORS ---
const COLOR_RESET: &str = "\x1b[0m";
const COLOR_CYAN: &str = "\x1b[36m";
const COLOR_GREEN: &str = "\x1b[32m";
const COLOR_YELLOW: &str = "\x1b[33m";
const COLOR_RED: &str = "\x1b[31m";
const COLOR_MAGENTA: &str = "\x1b[35m";
const COLOR_BLUE: &str = "\x1b[34m";
const COLOR_GRAY: &str = "\x1b[90m";
const COLOR_WHITE_BG: &str = "\x1b[47m\x1b[30m";
const BOLD: &str = "\x1b[1m";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Quiet logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Info) => display_system_info()?,
        Some(Commands::Chat { vocab, pattern_dim }) => run_advanced_tui(vocab, pattern_dim)?,
        None => display_welcome(),
    }

    Ok(())
}

fn display_welcome() {
    println!(
        "{}╔════════════════════════════════════════════════════════════╗",
        COLOR_CYAN
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

fn display_system_info() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_ctx = Arc::new(CudaContext::default()?);
    println!("{}", cuda_ctx.device_info()?);
    Ok(())
}

struct BrainMonitor {
    history_theta: Vec<f32>,
    history_gamma: Vec<f32>,
    history_da: Vec<f32>,
    last_update: Instant,
    sim_speed: f32,
}

impl BrainMonitor {
    fn new() -> Self {
        Self {
            history_theta: vec![0.0; 20],
            history_gamma: vec![0.0; 20],
            history_da: vec![0.0; 20],
            last_update: Instant::now(),
            sim_speed: 1.0,
        }
    }

    fn update_history(&mut self, stats: &neurox_ai::brain::BrainStats) {
        push_history(
            &mut self.history_theta,
            stats.oscillations.theta_phase.sin(),
        );
        push_history(
            &mut self.history_gamma,
            stats.oscillations.gamma_phase.sin(),
        );
        push_history(&mut self.history_da, stats.neuromodulation.dopamine_level);

        let now = Instant::now();
        let duration = now.duration_since(self.last_update).as_secs_f32();
        self.sim_speed = 1.0 / (duration * 1000.0 + 0.001); // Approx speed
        self.last_update = now;
    }

    fn render_dashboard(&self, stats: &neurox_ai::brain::BrainStats, mode: &str) {
        // Clear top 8 lines (ANSI)
        print!("\x1b[s"); // Save cursor
        print!("\x1b[H"); // Go to home (top-left)

        // Header
        println!(
            "{}{}  NEUROX-AI MONITOR [STATUS: ONLINE] [MODE: {}] [SIM: {:.1}x]          {}",
            BOLD, COLOR_WHITE_BG, mode, self.sim_speed, COLOR_RESET
        );

        // Grid Layout
        // Row 1: Neuromodulation
        println!(
            " {}┌── NEUROMODULATION ────────────────────┐ ┌── OSCILLATIONS (EEG) ───────┐{}",
            COLOR_GRAY, COLOR_RESET
        );
        println!(
            " │ Dopamine:  {} {:.3} │ │ θ: {} │",
            self.bar(stats.neuromodulation.dopamine_level, COLOR_YELLOW),
            stats.neuromodulation.dopamine_level,
            self.sparkline(&self.history_theta)
        );
        println!(
            " │ Serotonin: {} {:.3} │ │ γ: {} │",
            self.bar(stats.neuromodulation.sht_level, COLOR_MAGENTA),
            stats.neuromodulation.sht_level,
            self.sparkline(&self.history_gamma)
        );
        println!(
            " │ Norepin.:  {} {:.3} │ │ Coupling: {:.2}                │",
            self.bar(stats.neuromodulation.ne_level, COLOR_RED),
            stats.neuromodulation.ne_level,
            stats.oscillations.theta_gamma_coupling
        );

        // Row 2: Regions & Memory
        println!(
            " {}├── REGION ACTIVITY ────────────────────┤ ├── MEMORY STATE ─────────────┤{}",
            COLOR_GRAY, COLOR_RESET
        );
        println!(
            " │ V1: {}  MT: {}  Amg: {} │ │ Working Mem: {}/{} │",
            self.led(stats.time % 10.0 > 5.0), // Mock activity for V1
            self.led(stats.time % 20.0 > 10.0),
            self.led(stats.amygdala.la_active_neurons > 0),
            stats.working_memory.active_count,
            stats.working_memory.capacity
        );
        println!(
            " │ Hip: {}  BG: {}  PFC: {} │ │ Hippocampus: {:<5} items   │",
            self.led(stats.hippocampus.buffer_size > 0),
            self.led(stats.basal_ganglia.dopamine_firing_rate > 2.0),
            self.led(true),
            stats.hippocampus.buffer_size
        );
        println!(
            " {}└───────────────────────────────────────┘ └─────────────────────────────┘{}",
            COLOR_GRAY, COLOR_RESET
        );

        print!("\x1b[u"); // Restore cursor
        stdout().flush().unwrap();
    }

    fn bar(&self, val: f32, color: &str) -> String {
        let width = 10;
        let filled = (val.clamp(0.0, 1.0) * width as f32) as usize;
        let empty = width - filled;
        format!(
            "{}{}{}{}",
            color,
            "█".repeat(filled),
            "░".repeat(empty),
            COLOR_RESET
        )
    }

    fn sparkline(&self, data: &[f32]) -> String {
        let symbols = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
        let mut s = String::new();
        for &v in data {
            let idx = ((v + 1.0) / 2.0 * 7.0).clamp(0.0, 7.0) as usize;
            s.push_str(symbols[idx]);
        }
        format!("{}{}{}", COLOR_CYAN, s, COLOR_RESET)
    }

    fn led(&self, active: bool) -> String {
        if active {
            format!("{}●{}", COLOR_GREEN, COLOR_RESET)
        } else {
            format!("{}○{}", COLOR_GRAY, COLOR_RESET)
        }
    }
}

// Helper function to avoid borrow checker issues with self
fn push_history(vec: &mut Vec<f32>, val: f32) {
    if !vec.is_empty() {
        vec.remove(0);
    }
    vec.push(val);
}

fn run_advanced_tui(vocab: usize, pattern_dim: usize) -> Result<(), Box<dyn std::error::Error>> {
    // Clear screen
    print!("\x1b[2J\x1b[H");

    // Initialize Brain
    println!(
        "{}Initializing Cortical Subsystems...{}",
        COLOR_GRAY, COLOR_RESET
    );
    let mut brain = NeuromorphicBrain::new(5, 1000, vocab, pattern_dim);
    let mut monitor = BrainMonitor::new();

    // Initial warmup
    for _ in 0..10 {
        brain.update(0.1);
    }

    // Reserve space for dashboard (8 lines)
    println!("\n\n\n\n\n\n\n");

    let mut rl = DefaultEditor::new()?;
    let mut mode = "INTERACTIVE".to_string();

    // Clean minimal CLI - no dashboard

    loop {
        // No dashboard refresh during chat - keeps output clean

        // Input
        let prompt = format!("{}{} > {}", BOLD, "INPUT", COLOR_RESET);
        let readline = rl.readline(&prompt);

        let input = match readline {
            Ok(line) => {
                rl.add_history_entry(&line)?;
                line
            }
            Err(_) => break,
        };

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "exit" {
            break;
        }

        // Commands
        if input.starts_with('/') {
            match input {
                "/sleep" => {
                    mode = "SLEEPING".to_string();
                    println!(
                        "{}>>> INITIATING SLEEP CONSOLIDATION SEQUENCE...{}",
                        COLOR_MAGENTA, COLOR_RESET
                    );
                    for _ in 0..20 {
                        brain.consolidate();
                        brain.update(1.0);
                        monitor.update_history(&brain.stats());
                        monitor.render_dashboard(&brain.stats(), &mode);
                        thread::sleep(Duration::from_millis(50));
                    }
                    println!(
                        "{}>>> SLEEP CYCLE COMPLETE. MEMORIES INTEGRATED.{}",
                        COLOR_GREEN, COLOR_RESET
                    );
                    mode = "INTERACTIVE".to_string();
                }
                "/shock" => {
                    println!(
                        "{}>>> INJECTING NOREPINEPHRINE (AROUSAL) ...{}",
                        COLOR_RED, COLOR_RESET
                    );
                    brain.neuromodulation.norepinephrine.update(100.0, 1.0, 1.0); // Spike NE
                    brain.update(0.1);
                }
                "/dopamine" => {
                    println!(
                        "{}>>> INJECTING DOPAMINE (REWARD) ...{}",
                        COLOR_YELLOW, COLOR_RESET
                    );
                    // Mock positive reinforcement
                    let state = vec![0.5; 512];
                    brain.learn_from_reward(&state, 0, 1.0, &state);
                }
                "/debug" => {
                    let stats = brain.stats();
                    println!("{}DEBUG DUMP:{}", COLOR_GRAY, COLOR_RESET);
                    println!(
                        "  Hippocampus sparsity: {:.4}",
                        stats.hippocampus.dg_sparsity
                    );
                    println!(
                        "  Structural plasticity: {} synapses",
                        stats.structural_plasticity.active_synapses
                    );
                    println!("  Criticality: {:.4}", stats.homeostasis.criticality_score);
                }
                "/help" => {
                    println!("{}COMMANDS:{}", COLOR_CYAN, COLOR_RESET);
                    println!(
                        "  /train <file>    - Train from file (.txt=skip-gram, .json=supervised)"
                    );
                    println!("  /learn <text>    - Immediate learning from text");
                    println!("  /vocab           - Display learned vocabulary");
                    println!("  /sleep           - Run memory consolidation");
                    println!("  /shock           - Inject norepinephrine (arousal)");
                    println!("  /dopamine        - Inject dopamine (reward)");
                    println!("  /debug           - Debug information");
                    println!("  exit             - Exit application");
                }
                "/vocab" => {
                    let vocab_size = brain.language.ventral.embeddings.idx_to_word.len();
                    println!(
                        "{}VOCABULARY ({} words):{}",
                        COLOR_CYAN, vocab_size, COLOR_RESET
                    );
                    for (i, word) in brain
                        .language
                        .ventral
                        .embeddings
                        .idx_to_word
                        .iter()
                        .take(50)
                        .enumerate()
                    {
                        print!("{} ", word);
                        if (i + 1) % 10 == 0 {
                            println!();
                        }
                    }
                    if vocab_size > 50 {
                        println!("\n  ... and {} more words", vocab_size - 50);
                    }
                    println!();
                }
                _ if input.starts_with("/train ") => {
                    let file_path = input.strip_prefix("/train ").unwrap().trim();
                    let is_json = file_path.ends_with(".json");

                    println!();
                    println!(
                        "{}{} >{} Loading: {} ({})",
                        BOLD,
                        "TRAIN",
                        COLOR_RESET,
                        file_path,
                        if is_json {
                            "JSON supervised"
                        } else {
                            "TEXT skip-gram"
                        }
                    );

                    if is_json {
                        // JSON supervised learning via BrainLoader
                        match neurox_ai::brain::loader::BrainLoader::load_json_training(
                            &mut brain, file_path,
                        ) {
                            Ok(_) => {
                                println!(
                                    "{}✓ Training complete!{} {} words in lexicon",
                                    COLOR_GREEN,
                                    COLOR_RESET,
                                    brain.lexicon.words.len()
                                );
                                mode = "INTERACTIVE".to_string();
                            }
                            Err(e) => {
                                println!(
                                    "{}ERROR: Cannot load JSON: {}{}",
                                    COLOR_RED, e, COLOR_RESET
                                );
                            }
                        }
                    } else {
                        // Text file via BrainLoader
                        match neurox_ai::brain::loader::BrainLoader::load_text_training(
                            &mut brain, file_path, 3,
                        ) {
                            Ok(_) => {
                                println!(
                                    "\n{}>>> TRAINING COMPLETE! Vocabulary: {} words{}",
                                    COLOR_GREEN,
                                    brain.language.ventral.embeddings.idx_to_word.len(),
                                    COLOR_RESET
                                );
                                mode = "INTERACTIVE".to_string();
                            }
                            Err(e) => {
                                println!(
                                    "{}ERROR: Cannot load file: {}{}",
                                    COLOR_RED, e, COLOR_RESET
                                );
                            }
                        }
                    }
                }
                _ => println!(
                    "{}Unknown command. Type /help for available commands.{}",
                    COLOR_RED, COLOR_RESET
                ),
            }
            continue;
        }

        // === CHAT PROCESSING ===
        mode = "PROCESSING".to_string();

        // Process through brain
        let response = brain.process_text(input);

        // Post-processing
        for _ in 0..3 {
            brain.update(0.1);
        }

        // Show response - simple and clean
        println!("{}{} >{} {}", BOLD, "NEUROX", COLOR_RESET, response);

        mode = "INTERACTIVE".to_string();
    }

    Ok(())
}
