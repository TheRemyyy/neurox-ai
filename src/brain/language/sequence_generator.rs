//! Sequence Generator for conversational text generation
//!
//! Implements sequence-to-sequence generation with:
//! - Beam Search for finding optimal word sequences
//! - Context Window for maintaining conversation history
//! - Temperature control for response variability

use std::collections::VecDeque;

/// Configuration for sequence generation
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Beam width for beam search (number of candidates to keep)
    pub beam_width: usize,
    /// Maximum sequence length to generate
    pub max_length: usize,
    /// Temperature for sampling (0.0 = deterministic, 1.0 = full random)
    pub temperature: f32,
    /// Context window size (number of previous messages to consider)
    pub context_size: usize,
    /// Minimum probability threshold for candidate words
    pub min_prob: f32,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            beam_width: 5,
            max_length: 50,
            temperature: 0.7,
            context_size: 5,
            min_prob: 0.01,
        }
    }
}

/// A candidate sequence during beam search
#[derive(Debug, Clone)]
pub struct BeamCandidate {
    /// Words in this candidate sequence
    pub tokens: Vec<String>,
    /// Cumulative log probability of this sequence
    pub score: f32,
    /// Whether this sequence has ended (found end token)
    pub finished: bool,
}

impl BeamCandidate {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            score: 0.0,
            finished: false,
        }
    }

    pub fn with_token(&self, token: String, log_prob: f32) -> Self {
        let mut new_tokens = self.tokens.clone();
        new_tokens.push(token);
        Self {
            tokens: new_tokens,
            score: self.score + log_prob,
            finished: false,
        }
    }

    pub fn to_string(&self) -> String {
        self.tokens.join(" ")
    }
}

/// Context window for conversation history
#[derive(Debug, Clone)]
pub struct ContextWindow {
    /// Previous messages in the conversation
    messages: VecDeque<ConversationTurn>,
    /// Maximum number of messages to keep
    max_size: usize,
}

/// A single turn in the conversation
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    /// User input
    pub user: String,
    /// AI response
    pub assistant: String,
    /// Timestamp (epoch ms)
    pub timestamp: u64,
}

impl ContextWindow {
    pub fn new(max_size: usize) -> Self {
        Self {
            messages: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Add a conversation turn to the context
    pub fn add_turn(&mut self, user: String, assistant: String) {
        let turn = ConversationTurn {
            user,
            assistant,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };

        if self.messages.len() >= self.max_size {
            self.messages.pop_front();
        }
        self.messages.push_back(turn);
    }

    /// Get all context as a formatted string
    pub fn get_context_string(&self) -> String {
        self.messages
            .iter()
            .map(|turn| format!("User: {}\nAI: {}", turn.user, turn.assistant))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Get recent user inputs for pattern matching
    pub fn get_recent_inputs(&self) -> Vec<&str> {
        self.messages.iter().map(|t| t.user.as_str()).collect()
    }

    /// Get recent AI responses for style consistency
    pub fn get_recent_responses(&self) -> Vec<&str> {
        self.messages.iter().map(|t| t.assistant.as_str()).collect()
    }

    /// Check if a topic was recently discussed
    pub fn contains_topic(&self, topic: &str) -> bool {
        let topic_lower = topic.to_lowercase();
        self.messages.iter().any(|turn| {
            turn.user.to_lowercase().contains(&topic_lower)
                || turn.assistant.to_lowercase().contains(&topic_lower)
        })
    }

    /// Get the last user message
    pub fn last_user_input(&self) -> Option<&str> {
        self.messages.back().map(|t| t.user.as_str())
    }

    /// Get the last AI response
    pub fn last_response(&self) -> Option<&str> {
        self.messages.back().map(|t| t.assistant.as_str())
    }

    /// Clear the context
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Get number of turns in context
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Check if context is empty
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

/// Sequence generator for text generation
#[derive(Debug, Clone)]
pub struct SequenceGenerator {
    /// Generator configuration
    pub config: GeneratorConfig,
    /// Conversation context
    pub context: ContextWindow,
    /// Word probabilities (vocabulary -> probability)
    word_probs: std::collections::HashMap<String, f32>,
    /// Bigram probabilities (word1 -> word2 -> probability)
    bigram_probs: std::collections::HashMap<String, std::collections::HashMap<String, f32>>,
}

impl SequenceGenerator {
    pub fn new(config: GeneratorConfig) -> Self {
        let context_size = config.context_size;
        Self {
            config,
            context: ContextWindow::new(context_size),
            word_probs: std::collections::HashMap::new(),
            bigram_probs: std::collections::HashMap::new(),
        }
    }

    /// Learn word probabilities from training responses
    pub fn learn_from_responses(&mut self, responses: &[String]) {
        let mut word_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut bigram_counts: std::collections::HashMap<
            String,
            std::collections::HashMap<String, usize>,
        > = std::collections::HashMap::new();
        let mut total_words = 0usize;

        for response in responses {
            let words: Vec<&str> = response.split_whitespace().collect();
            for (i, word) in words.iter().enumerate() {
                let word_lower = word.to_lowercase();
                *word_counts.entry(word_lower.clone()).or_insert(0) += 1;
                total_words += 1;

                // Bigram
                if i > 0 {
                    let prev_word = words[i - 1].to_lowercase();
                    *bigram_counts
                        .entry(prev_word)
                        .or_default()
                        .entry(word_lower)
                        .or_insert(0) += 1;
                }
            }
        }

        // Convert to probabilities
        for (word, count) in word_counts {
            self.word_probs
                .insert(word, count as f32 / total_words as f32);
        }

        for (prev_word, next_words) in bigram_counts {
            let total: usize = next_words.values().sum();
            let probs: std::collections::HashMap<String, f32> = next_words
                .into_iter()
                .map(|(w, c)| (w, c as f32 / total as f32))
                .collect();
            self.bigram_probs.insert(prev_word, probs);
        }
    }

    /// Apply temperature to probabilities
    fn apply_temperature(&self, probs: &mut [(String, f32)]) {
        if self.config.temperature <= 0.0 {
            // Deterministic - keep only max
            if let Some(max_idx) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap())
                .map(|(i, _)| i)
            {
                for (i, (_, p)) in probs.iter_mut().enumerate() {
                    *p = if i == max_idx { 1.0 } else { 0.0 };
                }
            }
            return;
        }

        // Apply temperature scaling
        let temp = self.config.temperature;
        let mut sum = 0.0f32;
        for (_, p) in probs.iter_mut() {
            *p = (*p).powf(1.0 / temp);
            sum += *p;
        }
        // Normalize
        if sum > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }
    }

    /// Get next word candidates based on previous word
    pub fn get_next_candidates(&self, prev_word: Option<&str>, top_k: usize) -> Vec<(String, f32)> {
        let mut candidates: Vec<(String, f32)> = if let Some(prev) = prev_word {
            if let Some(bigrams) = self.bigram_probs.get(&prev.to_lowercase()) {
                bigrams.iter().map(|(w, p)| (w.clone(), *p)).collect()
            } else {
                // Fallback to unigram
                self.word_probs
                    .iter()
                    .map(|(w, p)| (w.clone(), *p))
                    .collect()
            }
        } else {
            self.word_probs
                .iter()
                .map(|(w, p)| (w.clone(), *p))
                .collect()
        };

        self.apply_temperature(&mut candidates);

        // Filter by min probability and sort
        candidates.retain(|(_, p)| *p >= self.config.min_prob);
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(top_k);

        candidates
    }

    /// Perform beam search to generate a response
    pub fn beam_search(&self, seed_words: &[String]) -> Vec<BeamCandidate> {
        let mut beams: Vec<BeamCandidate> = vec![BeamCandidate::new()];

        // Initialize with seed words
        for word in seed_words {
            for beam in &mut beams {
                beam.tokens.push(word.clone());
            }
        }

        for _ in 0..self.config.max_length {
            let mut new_beams: Vec<BeamCandidate> = Vec::new();

            for beam in &beams {
                if beam.finished {
                    new_beams.push(beam.clone());
                    continue;
                }

                let last_word = beam.tokens.last().map(|s| s.as_str());
                let candidates = self.get_next_candidates(last_word, self.config.beam_width);

                for (word, prob) in candidates {
                    let log_prob = prob.ln();
                    let new_beam = beam.with_token(word.clone(), log_prob);
                    new_beams.push(new_beam);
                }
            }

            // Keep top beams
            new_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            new_beams.truncate(self.config.beam_width);
            beams = new_beams;

            // Check if all beams are finished
            if beams.iter().all(|b| b.finished) {
                break;
            }
        }

        beams
    }

    /// Generate a response given the input
    pub fn generate(&mut self, input: &str, matched_response: Option<&str>) -> String {
        // If we have a direct match, use it but maybe add context-aware prefix
        if let Some(response) = matched_response {
            // Store in context
            self.context
                .add_turn(input.to_string(), response.to_string());
            return response.to_string();
        }

        // Try beam search generation
        let seed: Vec<String> = input
            .split_whitespace()
            .take(2)
            .map(|s| s.to_lowercase())
            .collect();

        let beams = self.beam_search(&seed);

        if let Some(best) = beams.first() {
            let response = best.to_string();
            self.context.add_turn(input.to_string(), response.clone());
            response
        } else {
            "Hmm, na to teď nevím co říct.".to_string()
        }
    }

    /// Update configuration
    pub fn set_temperature(&mut self, temp: f32) {
        self.config.temperature = temp.clamp(0.0, 2.0);
    }

    pub fn set_beam_width(&mut self, width: usize) {
        self.config.beam_width = width.max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_window() {
        let mut ctx = ContextWindow::new(3);
        ctx.add_turn("ahoj".into(), "Ahoj!".into());
        ctx.add_turn("jak se máš".into(), "Dobře, díky!".into());

        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx.last_user_input(), Some("jak se máš"));
        assert!(ctx.contains_topic("ahoj"));
    }

    #[test]
    fn test_beam_candidate() {
        let beam = BeamCandidate::new();
        let beam2 = beam.with_token("ahoj".into(), -0.5);

        assert_eq!(beam2.tokens, vec!["ahoj"]);
        assert!((beam2.score - (-0.5)).abs() < 0.001);
    }

    #[test]
    fn test_generator_config() {
        let config = GeneratorConfig::default();
        assert_eq!(config.beam_width, 5);
        assert_eq!(config.context_size, 5);
    }
}
