//! Interactive Neuromorphic Brain
//!
//! Real-time interaction with the neuromorphic brain system.
//! Features:
//! - Pattern-based memory and recall
//! - Sequence learning and prediction
//! - Attention-based processing
//! - Memory consolidation
//!
//! Usage:
//!   cargo run --release --example interactive_brain

use neurox_ai::*;
use std::collections::HashMap;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn) // Quiet mode for interaction
        .init();

    println!("=== Interactive Neuromorphic Brain ===");
    println!();
    println!("Initializing brain architecture...");

    let pattern_dim = 128;
    let n_layers = 3;
    let base_neurons = 1000;
    let vocab_size = 1000; // Larger vocab

    let mut brain = NeuromorphicBrain::new(n_layers, base_neurons, vocab_size, pattern_dim);
    let mut vocabulary = Vocabulary::new();

    println!("✓ Brain initialized ({} neurons, pattern_dim={})", base_neurons, pattern_dim);
    println!();

    // Pre-train on some basic sequences
    println!("Pre-training on basic knowledge...");
    let training_data = vec![
        "hello how are you",
        "i am fine thank you",
        "what is your name",
        "my name is neurox",
        "neurox is a brain",
        "the brain can learn",
        "learning is important",
        "memory stores patterns",
        "attention selects information",
        "consolidation transfers knowledge",
    ];

    for text in &training_data {
        let tokens = vocabulary.encode(text);
        brain.language.comprehend(&tokens, 0.1);

        // Store in hippocampus for episodic memory
        for token in &tokens {
            let pattern = encode_token(*token, pattern_dim);
            brain.hippocampus.encode(&pattern);
        }
    }

    let stats = brain.stats();
    println!("✓ Pre-training complete:");
    println!("  - Vocabulary: {} words", vocabulary.size());
    println!("  - Transitions: {}", stats.language.transition_count);
    println!("  - Hippocampal memories: {}", stats.hippocampus.buffer_size);
    println!();

    println!("Commands:");
    println!("  say <text>      - Process and respond to text");
    println!("  remember <text> - Store text in episodic memory");
    println!("  recall <text>   - Recall similar memories");
    println!("  predict <word>  - Predict next word");
    println!("  consolidate     - Run memory consolidation");
    println!("  stats           - Show brain statistics");
    println!("  help            - Show this help");
    println!("  quit            - Exit");
    println!();

    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        print!("> ");
        io::stdout().flush()?;

        input.clear();
        stdin.read_line(&mut input)?;
        let line = input.trim();

        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        let command = parts[0];
        let args = if parts.len() > 1 { parts[1] } else { "" };

        match command {
            "quit" | "exit" | "q" => {
                println!("Goodbye!");
                break;
            }

            "help" | "h" => {
                println!("Commands:");
                println!("  say <text>      - Process and respond to text");
                println!("  remember <text> - Store text in episodic memory");
                println!("  recall <text>   - Recall similar memories");
                println!("  predict <word>  - Predict next word");
                println!("  consolidate     - Run memory consolidation");
                println!("  stats           - Show brain statistics");
                println!("  help            - Show this help");
                println!("  quit            - Exit");
            }

            "say" => {
                if args.is_empty() {
                    println!("Usage: say <text>");
                    continue;
                }

                println!("Processing: \"{}\"", args);

                // Encode input
                let tokens = vocabulary.encode(args);
                println!("  Tokens: {:?}", tokens.iter()
                    .map(|t| vocabulary.decode(*t))
                    .collect::<Vec<_>>());

                // Process through brain
                brain.language.comprehend(&tokens, 0.1);

                // Generate response based on learned transitions
                if !tokens.is_empty() {
                    let start_token = *tokens.last().unwrap();
                    let response_tokens = brain.language.produce(start_token, 5);

                    let response_words: Vec<String> = response_tokens.iter()
                        .map(|t| vocabulary.decode(*t))
                        .collect();

                    if response_words.len() > 1 {
                        println!("  Response: {}", response_words.join(" "));
                    } else {
                        println!("  (No strong associations learned yet)");
                    }
                }

                // Store patterns in working memory
                for token in &tokens {
                    let pattern = encode_token(*token, pattern_dim);
                    brain.working_memory.store(&pattern, 0.8);
                }

                let wm_stats = brain.stats().working_memory;
                println!("  Working memory: {}/{} items", wm_stats.active_count, wm_stats.capacity);
            }

            "remember" => {
                if args.is_empty() {
                    println!("Usage: remember <text>");
                    continue;
                }

                println!("Storing memory: \"{}\"", args);

                let tokens = vocabulary.encode(args);

                // Store in hippocampus
                for token in &tokens {
                    let pattern = encode_token(*token, pattern_dim);
                    brain.hippocampus.encode(&pattern);
                }

                // Learn transitions
                brain.language.comprehend(&tokens, 0.1);

                let hippo_stats = brain.stats().hippocampus;
                println!("  Hippocampus: {} memories stored", hippo_stats.buffer_size);
                println!("  Transitions learned: {}", brain.stats().language.transition_count);
            }

            "recall" => {
                if args.is_empty() {
                    println!("Usage: recall <text>");
                    continue;
                }

                println!("Recalling memories similar to: \"{}\"", args);

                let tokens = vocabulary.encode(args);
                if !tokens.is_empty() {
                    let cue = encode_token(tokens[0], pattern_dim);
                    let recalled = brain.hippocampus.recall(&cue);

                    let activity = recalled.iter().sum::<f32>() / recalled.len() as f32;
                    if activity > 0.01 {
                        println!("  Memory recalled (activity={:.3})", activity);
                    } else {
                        println!("  No strong memory found");
                    }
                }
            }

            "predict" => {
                if args.is_empty() {
                    println!("Usage: predict <word>");
                    continue;
                }

                let word = args.trim();
                if let Some(token) = vocabulary.get_token(word) {
                    let next = brain.language.wernicke.predict_next(token);
                    let prob = brain.language.wernicke.get_transition_prob(token, next);

                    let next_word = vocabulary.decode(next);

                    if prob > 0.01 {
                        println!("  \"{}\" → \"{}\" (p={:.3})", word, next_word, prob);
                    } else {
                        println!("  No prediction available for \"{}\"", word);
                    }
                } else {
                    println!("  Word \"{}\" not in vocabulary", word);
                }
            }

            "consolidate" => {
                println!("Running memory consolidation...");

                let before = brain.stats().hippocampus.buffer_size;
                brain.consolidate();
                let after = brain.stats().hippocampus.buffer_size;

                println!("  Consolidated {} memories", before);
                println!("  Hippocampus: {} memories remaining", after);
                println!("  Working memory: {} items", brain.stats().working_memory.active_count);
            }

            "stats" => {
                let stats = brain.stats();

                println!("Brain Statistics:");
                println!();
                println!("Working Memory:");
                println!("  Active items: {}/{}", stats.working_memory.active_count, stats.working_memory.capacity);
                println!("  Utilization: {:.1}%", stats.working_memory.utilization * 100.0);
                println!("  Avg attention: {:.3}", stats.working_memory.avg_attention);
                println!();
                println!("Hippocampus:");
                println!("  Memories: {}/{}", stats.hippocampus.buffer_size, stats.hippocampus.max_buffer);
                println!("  DG sparsity: {:.1}%", stats.hippocampus.dg_sparsity * 100.0);
                println!("  Avg priority: {:.3}", stats.hippocampus.avg_priority);
                println!();
                println!("Attention:");
                println!("  Avg salience: {:.3}", stats.attention.avg_salience);
                println!("  Focused: {}/{}", stats.attention.focused_locations, pattern_dim);
                println!();
                println!("Language:");
                println!("  Vocabulary: {} words", vocabulary.size());
                println!("  Transitions: {}", stats.language.transition_count);
                println!("  Avg prob: {:.3}", stats.language.avg_transition_prob);
            }

            _ => {
                println!("Unknown command: {}. Type 'help' for available commands.", command);
            }
        }

        println!();
    }

    Ok(())
}

/// Simple vocabulary manager
struct Vocabulary {
    word_to_id: HashMap<String, usize>,
    id_to_word: HashMap<usize, String>,
    next_id: usize,
}

impl Vocabulary {
    fn new() -> Self {
        let mut vocab = Self {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
            next_id: 0,
        };

        // Add unknown token
        vocab.add_word("<UNK>");

        vocab
    }

    fn add_word(&mut self, word: &str) -> usize {
        if let Some(&id) = self.word_to_id.get(word) {
            return id;
        }

        let id = self.next_id;
        self.word_to_id.insert(word.to_string(), id);
        self.id_to_word.insert(id, word.to_string());
        self.next_id += 1;
        id
    }

    fn encode(&mut self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                let lower = word.to_lowercase();
                self.add_word(&lower)
            })
            .collect()
    }

    fn decode(&self, token: usize) -> String {
        self.id_to_word.get(&token)
            .cloned()
            .unwrap_or_else(|| "<UNK>".to_string())
    }

    fn get_token(&self, word: &str) -> Option<usize> {
        self.word_to_id.get(&word.to_lowercase()).copied()
    }

    fn size(&self) -> usize {
        self.word_to_id.len()
    }
}

/// Encode token as sparse pattern
fn encode_token(token: usize, dim: usize) -> Vec<f32> {
    let mut pattern = vec![0.0; dim];

    // Sparse distributed representation (20% active)
    for i in 0..dim {
        let hash = (token * 37 + i * 17) % 100;
        if hash < 20 {
            pattern[i] = 1.0;
        }
    }

    pattern
}
