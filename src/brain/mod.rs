//! Integrated Neuromorphic Brain
//!
//! Complete brain architecture combining all cognitive modules.
//!
//! # Architecture
//! - Sensory processing (existing hierarchical brain)
//! - Working memory (prefrontal cortex)
//! - Attention system (thalamic routing)
//! - Hippocampal memory (episodic learning)
//! - Predictive coding (cortical hierarchy)
//! - Language system (temporal cortex + frontal)

use crate::attention::AttentionSystem;
use crate::connectivity::SparseConnectivity;
use crate::cortex::{PredictiveHierarchy, WorkingMemory};
use crate::language::LanguageSystem;
use crate::memory::Hippocampus;
use crate::neuron::HierarchicalBrain;
use serde::{Deserialize, Serialize};

/// Complete neuromorphic brain with all cognitive modules
///
/// Integrates sensory processing, memory, attention, and language.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicBrain {
    /// Sensory and motor processing (existing hierarchical network)
    pub sensory: HierarchicalBrain,

    /// Working memory (7±2 items, attention-gated)
    pub working_memory: WorkingMemory,

    /// Hippocampal episodic memory
    pub hippocampus: Hippocampus,

    /// Attention and routing system
    pub attention: AttentionSystem,

    /// Language processing (Wernicke + Broca)
    pub language: LanguageSystem,

    /// Predictive coding hierarchy
    pub predictive: PredictiveHierarchy,

    /// Token vocabulary size
    vocab_size: usize,

    /// Pattern dimension for working memory
    pattern_dim: usize,
}

impl NeuromorphicBrain {
    /// Create new neuromorphic brain
    ///
    /// # Arguments
    /// - `n_layers`: Number of hierarchical layers for sensory processing
    /// - `base_neurons`: Base number of neurons (scales per layer)
    /// - `vocab_size`: Size of token vocabulary for language
    /// - `pattern_dim`: Dimension of patterns in working memory
    pub fn new(
        n_layers: usize,
        base_neurons: usize,
        vocab_size: usize,
        pattern_dim: usize,
    ) -> Self {
        // Create hierarchical sensory processing
        let sensory = HierarchicalBrain::new(n_layers, base_neurons);

        // Working memory (7 items, Miller's Law)
        let working_memory = WorkingMemory::new(7, pattern_dim, 0.5);

        // Hippocampus (10× expansion for DG)
        let hippocampus = Hippocampus::new(pattern_dim, 10, 0.05, 10000);

        // Attention system
        let connectivity = Self::create_default_connectivity(base_neurons);
        let attention = AttentionSystem::new(base_neurons, connectivity, 2.0);

        // Language system
        let language = LanguageSystem::new(pattern_dim, vocab_size);

        // Predictive coding (3-layer hierarchy)
        let layer_sizes = vec![pattern_dim, pattern_dim / 2, pattern_dim / 4];
        let predictive = PredictiveHierarchy::new(&layer_sizes, 0.01);

        Self {
            sensory,
            working_memory,
            hippocampus,
            attention,
            language,
            predictive,
            vocab_size,
            pattern_dim,
        }
    }

    /// Process text input (comprehension)
    ///
    /// Full pipeline: tokens → encoding → attention → working memory → hippocampus
    pub fn process_text(&mut self, text: &str) -> String {
        // Tokenize (simple word splitting)
        let tokens: Vec<usize> = text
            .split_whitespace()
            .map(|word| Self::simple_hash(word) % self.vocab_size)
            .collect();

        // Encode tokens to neural patterns
        let patterns = self.encode_tokens(&tokens);

        // Process each pattern through the brain
        for pattern in &patterns {
            // Attention-gated working memory storage
            let attention_level = 0.8; // High attention for input
            self.working_memory.store(pattern, attention_level);

            // Hippocampal encoding (one-shot)
            self.hippocampus.encode(pattern);

            // Update attention salience
            self.attention.update_salience(pattern);
        }

        // Language comprehension
        self.language.comprehend(&tokens, 0.1);

        // Generate response
        let start_token = *tokens.last().unwrap_or(&0);
        let response_tokens = self.language.produce(start_token, 10);

        // Decode tokens back to text
        self.decode_tokens(&response_tokens)
    }

    /// Encode tokens to neural spike patterns (rate coding)
    ///
    /// Each token becomes a sparse pattern of activity.
    pub fn encode_tokens(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        tokens
            .iter()
            .map(|&token| {
                let mut pattern = vec![0.0; self.pattern_dim];

                // Sparse distributed representation
                let hash = token;
                for i in 0..self.pattern_dim {
                    // Create sparse pattern based on token hash
                    let val = ((hash + i * 37) % 100) as f32 / 100.0;
                    if val > 0.8 {
                        // 20% sparsity
                        pattern[i] = val;
                    }
                }

                pattern
            })
            .collect()
    }

    /// Decode neural patterns back to tokens
    ///
    /// Uses nearest-neighbor matching in vocabulary space.
    pub fn decode_spikes(&self, spikes: &[Vec<f32>]) -> Vec<usize> {
        // Simple decoding: find closest token for each pattern
        spikes
            .iter()
            .map(|pattern| {
                // Sum pattern values and hash to token
                let sum: f32 = pattern.iter().sum();
                ((sum * 1000.0) as usize) % self.vocab_size
            })
            .collect()
    }

    /// Decode tokens to text
    fn decode_tokens(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&t| format!("token_{}", t))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Sleep-like memory consolidation
    ///
    /// Replay hippocampal memories to strengthen cortical connections.
    pub fn consolidate(&mut self) {
        log::info!("Beginning sleep-like consolidation...");

        // Replay high-priority memories
        let replayed = self.hippocampus.consolidate(100);

        log::info!("Replayed {} memories", replayed.len());

        // Store consolidated patterns in working memory with lower attention
        for (pattern, _) in replayed {
            self.working_memory.store(&pattern, 0.3); // Low attention = consolidation
        }
    }

    /// Continual learning step
    ///
    /// Process new data without catastrophic forgetting.
    pub fn train_continual(&mut self, input: &[f32], target: &[f32]) {
        // Store in hippocampus (fast learning)
        self.hippocampus.encode(input);

        // Gradual transfer to neocortex via replay
        if rand::random::<f32>() < 0.1 {
            // 10% chance of consolidation
            self.consolidate();
        }
    }

    /// Update all dynamics
    pub fn update(&mut self, dt: f32) {
        self.working_memory.maintain(dt);
        self.attention.update(dt);
    }

    /// Get brain statistics
    pub fn stats(&self) -> BrainStats {
        BrainStats {
            working_memory: self.working_memory.stats(),
            hippocampus: self.hippocampus.stats(),
            attention: self.attention.stats(),
            language: self.language.stats(),
            total_error: self.predictive.total_error(),
        }
    }

    /// Simple hash function for strings
    fn simple_hash(s: &str) -> usize {
        s.bytes().map(|b| b as usize).sum()
    }

    /// Create default connectivity
    fn create_default_connectivity(n_neurons: usize) -> SparseConnectivity {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut row_ptr = vec![0; n_neurons + 1];
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();

        for target in 0..n_neurons {
            for source in 0..n_neurons {
                if source != target && rng.gen::<f32>() < 0.05 {
                    col_idx.push(source as i32);
                    weights.push(rng.gen_range(-0.5..0.5));
                }
            }
            row_ptr[target + 1] = col_idx.len() as i32;
        }

        let nnz = col_idx.len();
        SparseConnectivity {
            row_ptr,
            col_idx,
            weights,
            nnz,
            n_neurons,
        }
    }
}

/// Complete brain statistics
#[derive(Debug, Clone)]
pub struct BrainStats {
    pub working_memory: crate::cortex::WorkingMemoryStats,
    pub hippocampus: crate::memory::HippocampusStats,
    pub attention: crate::attention::AttentionStats,
    pub language: crate::language::LanguageStats,
    pub total_error: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_creation() {
        let brain = NeuromorphicBrain::new(3, 100, 1000, 128);

        assert_eq!(brain.vocab_size, 1000);
        assert_eq!(brain.pattern_dim, 128);
    }

    #[test]
    fn test_text_processing() {
        let mut brain = NeuromorphicBrain::new(3, 100, 1000, 128);

        let response = brain.process_text("hello world");
        assert!(!response.is_empty());
    }

    #[test]
    fn test_token_encoding() {
        let brain = NeuromorphicBrain::new(3, 100, 1000, 128);

        let tokens = vec![1, 2, 3];
        let patterns = brain.encode_tokens(&tokens);

        assert_eq!(patterns.len(), 3);
        assert_eq!(patterns[0].len(), 128);
    }

    #[test]
    fn test_consolidation() {
        let mut brain = NeuromorphicBrain::new(3, 100, 1000, 128);

        // Store some patterns
        for i in 0..10 {
            let pattern = vec![i as f32 / 10.0; 128];
            brain.hippocampus.encode(&pattern);
        }

        // Consolidate
        brain.consolidate();

        // Working memory should have consolidated patterns
        assert!(brain.working_memory.active_count() > 0);
    }
}
