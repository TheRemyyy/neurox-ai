//! Chat Plugin
//!
//! Interactive chat with the neuromorphic brain.

use crate::brain::NeuromorphicBrain;
use crate::config::ChatConfig;
use rustyline::DefaultEditor;
use std::io::{stdout, Write};
use std::thread;
use std::time::{Duration, Instant};

// CLI Colors - white, gray, light blue only
const COLOR_RESET: &str = "\x1b[0m";
const COLOR_WHITE: &str = "\x1b[37m";
const COLOR_GRAY: &str = "\x1b[90m";
const COLOR_LIGHT_BLUE: &str = "\x1b[94m";
const BOLD: &str = "\x1b[1m";

/// Brain activity monitor for dashboard display
pub struct BrainMonitor {
    history_theta: Vec<f32>,
    history_gamma: Vec<f32>,
    history_da: Vec<f32>,
    last_update: Instant,
    sim_speed: f32,
    history_size: usize,
}

impl BrainMonitor {
    /// Create a new brain monitor with configurable history size
    pub fn new(history_size: usize) -> Self {
        Self {
            history_theta: vec![0.0; history_size],
            history_gamma: vec![0.0; history_size],
            history_da: vec![0.0; history_size],
            last_update: Instant::now(),
            sim_speed: 1.0,
            history_size,
        }
    }

    /// Number of samples kept in history buffers
    pub fn history_size(&self) -> usize {
        self.history_size
    }

    /// Update history with new brain stats
    pub fn update_history(&mut self, stats: &crate::brain::BrainStats) {
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
        self.sim_speed = 1.0 / (duration * 1000.0 + 0.001);
        self.last_update = now;
    }

    /// Render the dashboard display
    pub fn render_dashboard(&self, stats: &crate::brain::BrainStats, mode: &str) {
        print!("\x1b[s");
        print!("\x1b[H");

        println!(
            "{}{}  NEUROX-AI MONITOR [STATUS: ONLINE] [MODE: {}] [SIM: {:.1}x]          {}",
            BOLD, COLOR_WHITE, mode, self.sim_speed, COLOR_RESET
        );

        println!(
            " {}┌── NEUROMODULATION ────────────────────┐ ┌── OSCILLATIONS (EEG) ───────┐{}",
            COLOR_GRAY, COLOR_RESET
        );
        println!(
            " │ Dopamine:  {} {:.3} │ │ θ: {} │",
            self.bar(stats.neuromodulation.dopamine_level, COLOR_LIGHT_BLUE),
            stats.neuromodulation.dopamine_level,
            self.sparkline(&self.history_theta)
        );
        println!(
            " │ Serotonin: {} {:.3} │ │ γ: {} │",
            self.bar(stats.neuromodulation.sht_level, COLOR_GRAY),
            stats.neuromodulation.sht_level,
            self.sparkline(&self.history_gamma)
        );
        println!(
            " │ Norepin.:  {} {:.3} │ │ Coupling: {:.2}                │",
            self.bar(stats.neuromodulation.ne_level, COLOR_WHITE),
            stats.neuromodulation.ne_level,
            stats.oscillations.theta_gamma_coupling
        );

        println!(
            " {}├── REGION ACTIVITY ────────────────────┤ ├── MEMORY STATE ─────────────┤{}",
            COLOR_GRAY, COLOR_RESET
        );
        println!(
            " │ V1: {}  MT: {}  Amg: {} │ │ Working Mem: {}/{} │",
            self.led(stats.time % 10.0 > 5.0),
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

        print!("\x1b[u");
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
        format!("{}{}{}", COLOR_LIGHT_BLUE, s, COLOR_RESET)
    }

    fn led(&self, active: bool) -> String {
        if active {
            format!("{}●{}", COLOR_LIGHT_BLUE, COLOR_RESET)
        } else {
            format!("{}○{}", COLOR_GRAY, COLOR_RESET)
        }
    }
}

/// Helper function to push value to history vector
fn push_history(vec: &mut Vec<f32>, val: f32) {
    if !vec.is_empty() {
        vec.remove(0);
    }
    vec.push(val);
}

/// Chat plugin for interactive conversation
pub struct ChatPlugin {
    config: ChatConfig,
}

impl ChatPlugin {
    /// Create a new chat plugin with configuration
    pub fn new(config: ChatConfig) -> Self {
        Self { config }
    }

    /// Run the interactive chat
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Clear screen
        print!("\x1b[2J\x1b[H");

        // Initialize Brain
        println!(
            "{}Initializing Cortical Subsystems...{}",
            COLOR_GRAY, COLOR_RESET
        );
        let mut brain = NeuromorphicBrain::new(
            5,
            self.config.neurons / 5,
            self.config.vocab,
            self.config.pattern_dim,
        );
        brain.neuromodulation.dopamine_sensitivity = self.config.sensitivity;
        brain.working_memory.capacity = self.config.context;

        let mut monitor = BrainMonitor::new(self.config.history_size);

        // Initial warmup
        for _ in 0..self.config.warmup_iterations {
            brain.update(self.config.brain_update_dt);
        }

        // Reserve space for dashboard
        println!("\n\n\n\n\n\n\n");

        let mut rl = DefaultEditor::new()?;
        let mut mode = "INTERACTIVE".to_string();

        loop {
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
                self.handle_command(input, &mut brain, &mut monitor, &mut mode)?;
                continue;
            }

            // Chat processing
            let response = brain.process_text(input);

            for _ in 0..self.config.post_processing_iterations {
                brain.update(self.config.brain_update_dt);
            }

            println!("{}NEUROX >{} {}", BOLD, COLOR_RESET, response);

            mode = "INTERACTIVE".to_string();
        }

        Ok(())
    }

    fn handle_command(
        &self,
        input: &str,
        brain: &mut NeuromorphicBrain,
        monitor: &mut BrainMonitor,
        mode: &mut String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match input {
            "/sleep" => {
                *mode = "SLEEPING".to_string();
                println!(
                    "{}>>> INITIATING SLEEP CONSOLIDATION SEQUENCE...{}",
                    COLOR_LIGHT_BLUE, COLOR_RESET
                );
                for _ in 0..self.config.sleep_consolidation_iterations {
                    brain.consolidate();
                    brain.update(self.config.sleep_update_dt);
                    monitor.update_history(&brain.stats());
                    monitor.render_dashboard(&brain.stats(), mode);
                    thread::sleep(Duration::from_millis(self.config.sleep_delay_ms));
                }
                println!(
                    "{}>>> SLEEP CYCLE COMPLETE. MEMORIES INTEGRATED.{}",
                    COLOR_LIGHT_BLUE, COLOR_RESET
                );
                *mode = "INTERACTIVE".to_string();
            }
            "/shock" => {
                println!(
                    "{}>>> INJECTING NOREPINEPHRINE (AROUSAL) ...{}",
                    COLOR_WHITE, COLOR_RESET
                );
                brain
                    .neuromodulation
                    .norepinephrine
                    .update(self.config.ne_spike_value, 1.0, 1.0);
                brain.update(self.config.brain_update_dt);
            }
            "/dopamine" => {
                println!(
                    "{}>>> INJECTING DOPAMINE (REWARD) ...{}",
                    COLOR_LIGHT_BLUE, COLOR_RESET
                );
                let state = vec![0.5; self.config.dopamine_reward_state_size];
                brain.learn_from_reward(&state, 0, self.config.dopamine_reward_value, &state);
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
                println!("{}COMMANDS:{}", COLOR_LIGHT_BLUE, COLOR_RESET);
                println!("  /train <file>    - Train from file (.txt=skip-gram, .json=supervised)");
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
                    COLOR_LIGHT_BLUE, vocab_size, COLOR_RESET
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
                    "{}TRAIN >{} Loading: {} ({})",
                    BOLD,
                    COLOR_RESET,
                    file_path,
                    if is_json {
                        "JSON supervised"
                    } else {
                        "TEXT skip-gram"
                    }
                );

                if is_json {
                    match crate::brain::loader::BrainLoader::load_json_training(brain, file_path) {
                        Ok(_) => {
                            println!(
                                "{}✓ Training complete!{} {} words in lexicon",
                                COLOR_LIGHT_BLUE,
                                COLOR_RESET,
                                brain.lexicon.words.len()
                            );
                            *mode = "INTERACTIVE".to_string();
                        }
                        Err(e) => {
                            println!(
                                "{}ERROR: Cannot load JSON: {}{}",
                                COLOR_WHITE, e, COLOR_RESET
                            );
                        }
                    }
                } else {
                    match crate::brain::loader::BrainLoader::load_text_training(brain, file_path, 3)
                    {
                        Ok(_) => {
                            println!(
                                "\n{}>>> TRAINING COMPLETE! Vocabulary: {} words{}",
                                COLOR_LIGHT_BLUE,
                                brain.language.ventral.embeddings.idx_to_word.len(),
                                COLOR_RESET
                            );
                            *mode = "INTERACTIVE".to_string();
                        }
                        Err(e) => {
                            println!(
                                "{}ERROR: Cannot load file: {}{}",
                                COLOR_WHITE, e, COLOR_RESET
                            );
                        }
                    }
                }
            }
            _ => println!(
                "{}Unknown command. Type /help for available commands.{}",
                COLOR_WHITE, COLOR_RESET
            ),
        }
        Ok(())
    }
}

impl Default for ChatPlugin {
    fn default() -> Self {
        Self::new(ChatConfig::default())
    }
}
