//! Language and Temporal Processing
//!
//! Sequential token processing with temporal context maintenance.
//! Implements Wernicke (comprehension) and Broca (production) areas.
//!
//! # Biological Basis
//! - Wernicke's area: Temporal cortex, language comprehension
//! - Broca's area: Frontal cortex, language production
//! - Transition probability learning (n-gram-like)
//! - Temporal context via persistent activity

use crate::connectivity::SparseConnectivity;
use crate::neuron::LIFNeuron;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Temporal processor for sequential patterns
///
/// Maintains temporal context across time steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalProcessor {
    /// Context neurons (maintain temporal information)
    context: Vec<LIFNeuron>,

    /// Transition probabilities (bigram/trigram model)
    #[serde(skip)]
    transitions: HashMap<(usize, usize), f32>,

    /// Current state vector
    state: Vec<f32>,

    /// State size
    state_size: usize,

    /// Vocabulary size
    vocab_size: usize,

    /// Learning rate for transitions
    learning_rate: f32,

    /// Temporal decay (how fast context fades)
    tau_context: f32,
}

impl TemporalProcessor {
    /// Create new temporal processor
    pub fn new(state_size: usize, vocab_size: usize, learning_rate: f32, tau_context: f32) -> Self {
        Self {
            context: vec![LIFNeuron::default(); state_size],
            transitions: HashMap::new(),
            state: vec![0.0; state_size],
            state_size,
            vocab_size,
            learning_rate,
            tau_context,
        }
    }

    /// Process sequence of tokens
    ///
    /// Updates transition probabilities (learning)
    pub fn process_sequence(&mut self, tokens: &[usize], dt: f32) {
        for window in tokens.windows(2) {
            let prev = window[0];
            let curr = window[1];

            // Update transition probability
            let key = (prev, curr);
            let count = self.transitions.entry(key).or_insert(0.0);
            *count += self.learning_rate;

            // Activate context for current token
            self.activate_context(curr, dt);
        }

        // Normalize transitions
        self.normalize_transitions();
    }

    /// Predict next token given current
    ///
    /// Returns most likely next token based on learned transitions
    pub fn predict_next(&self, current: usize) -> usize {
        let mut best_next = 0;
        let mut best_prob = 0.0;

        for next in 0..self.vocab_size {
            let prob = self.transitions.get(&(current, next)).copied().unwrap_or(0.0);
            if prob > best_prob {
                best_prob = prob;
                best_next = next;
            }
        }

        best_next
    }

    /// Activate temporal context for token
    pub fn activate_context(&mut self, token: usize, dt: f32) {
        // Map token to state space (simple hash)
        let hash = token % self.state_size;

        // Activate corresponding context neurons
        for i in 0..self.state_size {
            // Distance-based activation
            let dist = ((i as isize - hash as isize).abs() as f32) / self.state_size as f32;
            let activation = (-dist * 5.0).exp(); // Gaussian-like

            self.state[i] = activation;
        }

        // Decay previous context
        let decay = (-dt / self.tau_context).exp();
        for s in &mut self.state {
            *s *= decay;
        }
    }

    /// Generate sequence starting from context
    ///
    /// Uses transition probabilities to generate likely continuation
    pub fn generate_sequence(&self, start_token: usize, max_length: usize) -> Vec<usize> {
        let mut sequence = vec![start_token];
        let mut current = start_token;

        for _ in 0..max_length {
            let next = self.predict_next(current);

            // Stop if no continuation found
            if next == 0 && current != 0 {
                break;
            }

            sequence.push(next);
            current = next;
        }

        sequence
    }

    /// Normalize transition probabilities
    fn normalize_transitions(&mut self) {
        // Normalize by source token
        for source in 0..self.vocab_size {
            let total: f32 = (0..self.vocab_size)
                .map(|target| self.transitions.get(&(source, target)).copied().unwrap_or(0.0))
                .sum();

            if total > 0.0 {
                for target in 0..self.vocab_size {
                    if let Some(prob) = self.transitions.get_mut(&(source, target)) {
                        *prob /= total;
                    }
                }
            }
        }
    }

    /// Get current state
    pub fn get_state(&self) -> &[f32] {
        &self.state
    }

    /// Get transition probability
    pub fn get_transition_prob(&self, from: usize, to: usize) -> f32 {
        self.transitions.get(&(from, to)).copied().unwrap_or(0.0)
    }
}

/// Motor sequencer for language production (Broca's area)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorSequencer {
    /// Sequence buffer
    sequence_buffer: Vec<usize>,

    /// Current position
    position: usize,

    /// Planning horizon
    horizon: usize,
}

impl MotorSequencer {
    pub fn new(horizon: usize) -> Self {
        Self {
            sequence_buffer: Vec::new(),
            position: 0,
            horizon,
        }
    }

    /// Plan sequence
    pub fn plan(&mut self, sequence: Vec<usize>) {
        self.sequence_buffer = sequence;
        self.position = 0;
    }

    /// Execute next token
    pub fn next(&mut self) -> Option<usize> {
        if self.position < self.sequence_buffer.len() {
            let token = self.sequence_buffer[self.position];
            self.position += 1;
            Some(token)
        } else {
            None
        }
    }

    /// Check if sequence complete
    pub fn is_complete(&self) -> bool {
        self.position >= self.sequence_buffer.len()
    }
}

/// Integrated language system
///
/// Combines Wernicke (comprehension) and Broca (production).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageSystem {
    /// Wernicke's area (comprehension)
    pub wernicke: TemporalProcessor,

    /// Broca's area (production)
    pub broca: MotorSequencer,

    /// Connection between comprehension and production
    connection: SparseConnectivity,
}

impl LanguageSystem {
    /// Create new language system
    pub fn new(state_size: usize, vocab_size: usize) -> Self {
        let wernicke = TemporalProcessor::new(state_size, vocab_size, 0.01, 200.0);
        let broca = MotorSequencer::new(10);

        // Create connection
        let connection = Self::create_connection(state_size, state_size);

        Self {
            wernicke,
            broca,
            connection,
        }
    }

    /// Process input text (comprehension)
    pub fn comprehend(&mut self, tokens: &[usize], dt: f32) {
        self.wernicke.process_sequence(tokens, dt);
    }

    /// Generate output text (production)
    pub fn produce(&mut self, start_token: usize, length: usize) -> Vec<usize> {
        let sequence = self.wernicke.generate_sequence(start_token, length);
        self.broca.plan(sequence.clone());
        sequence
    }

    /// Process and respond (comprehend → produce)
    pub fn process_and_respond(&mut self, input: &[usize], dt: f32, response_length: usize) -> Vec<usize> {
        // Comprehend input
        self.comprehend(input, dt);

        // Generate response starting from last input token
        let start = *input.last().unwrap_or(&0);
        self.produce(start, response_length)
    }

    /// Create connection matrix
    fn create_connection(n_source: usize, n_target: usize) -> SparseConnectivity {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut row_ptr = vec![0; n_target + 1];
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();

        for target in 0..n_target {
            for source in 0..n_source {
                if rng.gen::<f32>() < 0.1 {
                    col_idx.push(source as i32);
                    weights.push(rng.gen_range(-0.5..0.5));
                }
            }
            row_ptr[target + 1] = col_idx.len() as i32;
        }

        SparseConnectivity {
            row_ptr,
            col_idx,
            weights,
            nnz: col_idx.len(),
            n_neurons: n_target,
        }
    }

    /// Get language statistics
    pub fn stats(&self) -> LanguageStats {
        let transition_count = self.wernicke.transitions.len();
        let avg_prob = if !self.wernicke.transitions.is_empty() {
            self.wernicke.transitions.values().sum::<f32>() / transition_count as f32
        } else {
            0.0
        };

        LanguageStats {
            vocab_size: self.wernicke.vocab_size,
            transition_count,
            avg_transition_prob: avg_prob,
            state_size: self.wernicke.state_size,
        }
    }
}

/// Language statistics
#[derive(Debug, Clone)]
pub struct LanguageStats {
    pub vocab_size: usize,
    pub transition_count: usize,
    pub avg_transition_prob: f32,
    pub state_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_processor() {
        let mut processor = TemporalProcessor::new(50, 100, 0.1, 200.0);

        // Train on sequence
        let sequence = vec![1, 2, 3, 2, 4];
        processor.process_sequence(&sequence, 0.1);

        // Should learn 2→3 transition
        assert!(processor.get_transition_prob(2, 3) > 0.0);

        // Predict next after 2
        let next = processor.predict_next(2);
        assert!(next == 3 || next == 4); // Either learned transition
    }

    #[test]
    fn test_sequence_generation() {
        let mut processor = TemporalProcessor::new(50, 10, 0.5, 200.0);

        // Train simple pattern
        let pattern = vec![1, 2, 3, 1, 2, 3];
        processor.process_sequence(&pattern, 0.1);

        // Generate sequence
        let generated = processor.generate_sequence(1, 5);
        assert!(!generated.is_empty());
        assert_eq!(generated[0], 1);
    }

    #[test]
    fn test_language_system() {
        let mut lang = LanguageSystem::new(50, 20);

        // Train on simple dialogue
        let input = vec![1, 2, 3];
        lang.comprehend(&input, 0.1);

        // Generate response
        let response = lang.produce(3, 5);
        assert!(!response.is_empty());
    }

    #[test]
    fn test_motor_sequencer() {
        let mut sequencer = MotorSequencer::new(5);

        sequencer.plan(vec![1, 2, 3]);

        assert_eq!(sequencer.next(), Some(1));
        assert_eq!(sequencer.next(), Some(2));
        assert_eq!(sequencer.next(), Some(3));
        assert_eq!(sequencer.next(), None);
        assert!(sequencer.is_complete());
    }
}
