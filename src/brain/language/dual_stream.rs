//! Dual-Stream Language Architecture
//!
//! Replaces oversimplified Wernicke/Broca model with modern dual-stream architecture.
//!
//! # Ventral Stream (Bilateral - Sound-to-Meaning)
//! - STG (Superior Temporal Gyrus) → MTG (Middle Temporal Gyrus) → ATL (Anterior Temporal Lobe)
//! - Processes lexical-semantic content
//! - Bilateral representation
//! - Modality-independent semantics
//!
//! # Dorsal Stream (Left-Dominant - Sound-to-Articulation)
//! - STG → Spt (Sylvian-parietal-temporal) → IFG (Inferior Frontal Gyrus) → Premotor
//! - Supports phonological working memory and production
//! - Left hemisphere dominant
//! - Sensorimotor mapping for speech
//!
//! # Multi-Timescale Processing
//! - 1 word window: Lexical/semantic (posterior temporal)
//! - 4 word window: Syntactic structures (distributed)
//! - 6+ word window: Discourse coherence (frontal/parietal)

use crate::brain::cortex::WorkingMemory;
use crate::brain::semantics::{EmbeddingLayer, SemanticHub};
use serde::{Deserialize, Serialize};

/// Superior Temporal Gyrus - primary auditory/phonological processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STG {
    /// Phonological representations
    pub phonological_buffer: Vec<Vec<f32>>,

    /// Buffer capacity (tokens)
    pub capacity: usize,

    /// Phoneme dimension
    pub phoneme_dim: usize,
}

impl STG {
    pub fn new(capacity: usize, phoneme_dim: usize) -> Self {
        Self {
            phonological_buffer: Vec::new(),
            capacity,
            phoneme_dim,
        }
    }

    /// Process phonological input
    pub fn process_phonology(&mut self, phonemes: Vec<f32>) {
        self.phonological_buffer.push(phonemes);
        if self.phonological_buffer.len() > self.capacity {
            self.phonological_buffer.remove(0);
        }
    }

    /// Get recent phonology
    pub fn get_recent(&self, n: usize) -> Vec<Vec<f32>> {
        let start = self.phonological_buffer.len().saturating_sub(n);
        self.phonological_buffer[start..].to_vec()
    }
}

/// Middle Temporal Gyrus - lexical-semantic interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MTG {
    /// Lexical representations
    pub lexical_activations: Vec<f32>,

    /// Semantic features
    pub semantic_features: Vec<f32>,

    /// Size
    pub size: usize,
}

impl MTG {
    pub fn new(size: usize) -> Self {
        Self {
            lexical_activations: vec![0.0; size],
            semantic_features: vec![0.0; size],
            size,
        }
    }

    /// Map phonology to lexical representations
    pub fn phonology_to_lexical(&mut self, phonological_input: &[f32]) {
        // Simple mapping (in reality would be learned)
        for (i, &p) in phonological_input.iter().enumerate() {
            if i < self.size {
                self.lexical_activations[i] = p;
            }
        }
    }

    /// Extract semantic features
    pub fn get_semantics(&self) -> Vec<f32> {
        self.semantic_features.clone()
    }
}

/// Anterior Temporal Lobe - semantic hub
///
/// Hub-and-spoke model binding distributed semantic features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATL {
    /// Semantic hub (concept cells)
    pub semantic_hub: SemanticHub,

    /// Multimodal integration
    pub modality_weights: Vec<f32>,
}

impl ATL {
    pub fn new(n_concept_cells: usize, embedding_dim: usize) -> Self {
        Self {
            semantic_hub: SemanticHub::new(n_concept_cells, embedding_dim, 0.02),
            modality_weights: vec![0.5; 4], // Visual, auditory, motor, emotional
        }
    }

    /// Integrate multimodal features
    pub fn integrate(&mut self, features: &[f32]) {
        self.semantic_hub.encode(features);
    }

    /// Retrieve concept from partial cue
    pub fn retrieve(&mut self, cue: &[f32]) -> Vec<f32> {
        self.semantic_hub.retrieve(cue)
    }

    /// Get active concepts
    pub fn active_concepts(&self) -> Vec<usize> {
        self.semantic_hub.active_concepts()
    }
}

/// Sylvian-parietal-temporal area - sensorimotor interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spt {
    /// Sensorimotor mappings
    pub mappings: Vec<(Vec<f32>, Vec<f32>)>, // (auditory, motor)

    /// Buffer size
    pub capacity: usize,
}

impl Spt {
    pub fn new(capacity: usize) -> Self {
        Self {
            mappings: Vec::new(),
            capacity,
        }
    }

    /// Learn sensorimotor mapping
    pub fn learn_mapping(&mut self, auditory: Vec<f32>, motor: Vec<f32>) {
        self.mappings.push((auditory, motor));
        if self.mappings.len() > self.capacity {
            self.mappings.remove(0);
        }
    }

    /// Map from auditory to motor
    pub fn auditory_to_motor(&self, auditory: &[f32]) -> Vec<f32> {
        // Find closest mapping
        let mut best_match = vec![0.0; auditory.len()];
        let mut best_similarity = 0.0;

        for (aud, mot) in &self.mappings {
            let similarity = Self::cosine_similarity(auditory, aud);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_match = mot.clone();
            }
        }

        best_match
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

/// Inferior Frontal Gyrus - articulatory planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IFG {
    /// Articulatory plans
    pub articulatory_plans: Vec<Vec<f32>>,

    /// Planning buffer
    pub plan_buffer: Vec<Vec<f32>>,

    /// Buffer capacity
    pub capacity: usize,
}

impl IFG {
    pub fn new(capacity: usize) -> Self {
        Self {
            articulatory_plans: Vec::new(),
            plan_buffer: Vec::new(),
            capacity,
        }
    }

    /// Plan articulation
    pub fn plan_articulation(&mut self, motor_sequence: Vec<f32>) {
        self.plan_buffer.push(motor_sequence);
        if self.plan_buffer.len() > self.capacity {
            self.plan_buffer.remove(0);
        }
    }

    /// Execute plan
    pub fn execute(&mut self) -> Option<Vec<f32>> {
        self.plan_buffer.pop()
    }
}

/// Ventral stream (comprehension) - bilateral
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VentralStream {
    pub stg: STG,
    pub mtg: MTG,
    pub atl: ATL,

    /// Embedding layer for lexical access
    pub embeddings: EmbeddingLayer,
}

impl VentralStream {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self {
            stg: STG::new(20, embedding_dim), // 20 token buffer, use full embedding dim
            mtg: MTG::new(embedding_dim),
            atl: ATL::new(500, embedding_dim),
            embeddings: EmbeddingLayer::new(vocab_size, embedding_dim),
        }
    }

    /// Comprehend: Sound → Meaning
    pub fn comprehend(&mut self, tokens: &[usize]) -> Vec<f32> {
        // Process through STG (phonological)
        for &token_idx in tokens {
            if let Some(emb) = self.embeddings.get_embedding_by_idx(token_idx) {
                self.stg.process_phonology(emb.clone());
            }
        }

        // MTG extracts lexical-semantic features
        if let Some(recent) = self.stg.get_recent(1).first() {
            self.mtg.phonology_to_lexical(recent);
            // Copy lexical to semantic (simplified model)
            self.mtg.semantic_features = self.mtg.lexical_activations.clone();
        }

        // ATL integrates into amodal concepts
        self.atl.integrate(&self.mtg.semantic_features);

        // Return semantic representation
        self.atl.retrieve(&self.mtg.semantic_features)
    }
}

/// Dorsal stream (production) - left dominant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DorsalStream {
    pub stg: STG,
    pub spt: Spt,
    pub ifg: IFG,

    /// Phonological working memory
    pub phonological_loop: WorkingMemory,
}

impl DorsalStream {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            stg: STG::new(10, embedding_dim),
            spt: Spt::new(100),
            ifg: IFG::new(10),
            phonological_loop: WorkingMemory::new(7, embedding_dim, 0.3),
        }
    }

    /// Produce: Meaning → Sound → Articulation
    pub fn produce(&mut self, semantic_input: &[f32]) -> Vec<Vec<f32>> {
        // Map semantics to motor plan via Spt
        let motor_plan = self.spt.auditory_to_motor(semantic_input);

        // Plan articulation in IFG
        self.ifg.plan_articulation(motor_plan);

        // Store in phonological loop
        self.phonological_loop.store(semantic_input, 0.8);

        // Execute articulatory plans
        let mut output = Vec::new();
        while let Some(plan) = self.ifg.execute() {
            output.push(plan);
        }

        output
    }

    /// Maintain phonological working memory
    pub fn maintain(&mut self, dt: f32) {
        self.phonological_loop.maintain(dt);
    }
}

impl Default for DorsalStream {
    fn default() -> Self {
        Self::new(300) // Default embedding dimension
    }
}

/// Multi-timescale processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTimescaleProcessor {
    /// 1-word window (lexical/semantic)
    pub word_buffer: Vec<Vec<f32>>,

    /// 4-word window (syntactic)
    pub phrase_buffer: Vec<Vec<f32>>,

    /// 6-word window (discourse)
    pub discourse_buffer: Vec<Vec<f32>>,

    /// Window sizes
    pub window_1: usize,
    pub window_4: usize,
    pub window_6: usize,
}

impl MultiTimescaleProcessor {
    pub fn new() -> Self {
        Self {
            word_buffer: Vec::new(),
            phrase_buffer: Vec::new(),
            discourse_buffer: Vec::new(),
            window_1: 1,
            window_4: 4,
            window_6: 6,
        }
    }

    /// Process token at multiple timescales
    pub fn process(&mut self, token_embedding: Vec<f32>) {
        // 1-word processing
        self.word_buffer.push(token_embedding.clone());
        if self.word_buffer.len() > self.window_1 {
            self.word_buffer.remove(0);
        }

        // 4-word processing
        self.phrase_buffer.push(token_embedding.clone());
        if self.phrase_buffer.len() > self.window_4 {
            self.phrase_buffer.remove(0);
        }

        // 6-word processing
        self.discourse_buffer.push(token_embedding);
        if self.discourse_buffer.len() > self.window_6 {
            self.discourse_buffer.remove(0);
        }
    }

    /// Get predictions at each timescale
    pub fn get_predictions(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let word_pred = self.aggregate(&self.word_buffer);
        let phrase_pred = self.aggregate(&self.phrase_buffer);
        let discourse_pred = self.aggregate(&self.discourse_buffer);

        (word_pred, phrase_pred, discourse_pred)
    }

    fn aggregate(&self, buffer: &[Vec<f32>]) -> Vec<f32> {
        if buffer.is_empty() {
            return Vec::new();
        }

        let dim = buffer[0].len();
        let mut result = vec![0.0; dim];

        for vec in buffer {
            for (i, &v) in vec.iter().enumerate() {
                if i < dim {
                    result[i] += v;
                }
            }
        }

        for r in &mut result {
            *r /= buffer.len() as f32;
        }

        result
    }
}

impl Default for MultiTimescaleProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Integrated dual-stream language system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualStreamLanguage {
    /// Ventral stream (comprehension)
    pub ventral: VentralStream,

    /// Dorsal stream (production)
    pub dorsal: DorsalStream,

    /// Multi-timescale processing
    pub multiscale: MultiTimescaleProcessor,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Embedding dimension
    pub embedding_dim: usize,

    /// Word associations learned from context (word_idx -> associated word indices with strength)
    #[serde(skip)]
    word_associations: std::collections::HashMap<usize, Vec<(usize, f32)>>,
}

impl DualStreamLanguage {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self {
            ventral: VentralStream::new(vocab_size, embedding_dim),
            dorsal: DorsalStream::new(embedding_dim),
            multiscale: MultiTimescaleProcessor::new(),
            vocab_size,
            embedding_dim,
            word_associations: std::collections::HashMap::new(),
        }
    }

    /// Comprehend input
    pub fn comprehend(&mut self, tokens: &[usize]) -> Vec<f32> {
        // Process through ventral stream
        let semantics = self.ventral.comprehend(tokens);

        // Multi-timescale processing
        for &token_idx in tokens {
            if let Some(emb) = self.ventral.embeddings.get_embedding_by_idx(token_idx) {
                self.multiscale.process(emb.clone());
            }
        }

        semantics
    }

    /// Produce output
    pub fn produce(&mut self, semantic_input: &[f32], length: usize) -> Vec<Vec<f32>> {
        self.dorsal.produce(semantic_input)
    }

    /// Update (maintain working memory, etc.)
    pub fn update(&mut self, dt: f32) {
        self.dorsal.maintain(dt);
    }

    /// Train on Skip-gram pairs using STDP-like Hebbian learning
    /// This teaches the network which words appear together
    pub fn train_on_pairs(&mut self, pairs: &[(usize, usize)], learning_rate: f32) {
        for &(center_idx, context_idx) in pairs {
            // 1. Update embeddings via Skip-gram
            self.ventral
                .embeddings
                .train_skipgram(center_idx, context_idx);

            // 2. Build associative memory (which words appear together)
            self.learn_association(center_idx, context_idx, learning_rate);

            // 3. Train ATL concept cells on co-occurring patterns
            if let (Some(center_emb), Some(context_emb)) = (
                self.ventral
                    .embeddings
                    .get_embedding_by_idx(center_idx)
                    .cloned(),
                self.ventral
                    .embeddings
                    .get_embedding_by_idx(context_idx)
                    .cloned(),
            ) {
                // Combined pattern for concept learning
                let mut combined: Vec<f32> = center_emb
                    .iter()
                    .zip(context_emb.iter())
                    .map(|(a, b)| (a + b) / 2.0)
                    .collect();

                self.ventral.atl.integrate(&combined);
            }
        }
    }

    /// Learn word association (Hebbian: "neurons that fire together wire together")
    pub fn learn_association(&mut self, word1: usize, word2: usize, strength: f32) {
        // Bidirectional association
        self.add_association(word1, word2, strength);
        self.add_association(word2, word1, strength);
    }

    fn add_association(&mut self, from: usize, to: usize, strength: f32) {
        let associations = self.word_associations.entry(from).or_insert_with(Vec::new);

        // Check if association already exists
        if let Some(pos) = associations.iter().position(|(idx, _)| *idx == to) {
            // Strengthen existing association (with decay toward 1.0)
            associations[pos].1 = (associations[pos].1 + strength).min(1.0);
        } else {
            // New association
            associations.push((to, strength));
        }

        // Keep only top 50 associations per word
        if associations.len() > 50 {
            associations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            associations.truncate(50);
        }
    }

    /// Find words similar to the given semantic vector
    /// Returns (word_index, similarity_score) tuples, sorted by similarity
    pub fn find_similar_words(&self, semantic: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        let mut similarities: Vec<(usize, f32)> = Vec::new();

        for idx in 0..self.ventral.embeddings.idx_to_word.len() {
            if let Some(emb) = self.ventral.embeddings.get_embedding_by_idx(idx) {
                let sim = Self::cosine_similarity(semantic, emb);
                similarities.push((idx, sim));
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(top_k);
        similarities
    }

    /// Get associated words for a given word index
    pub fn get_associations(&self, word_idx: usize, top_k: usize) -> Vec<(usize, f32)> {
        if let Some(associations) = self.word_associations.get(&word_idx) {
            let mut result = associations.clone();
            result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            result.truncate(top_k);
            result
        } else {
            Vec::new()
        }
    }

    /// Get word by index from vocabulary
    pub fn get_word(&self, idx: usize) -> Option<&str> {
        self.ventral
            .embeddings
            .idx_to_word
            .get(idx)
            .map(|s| s.as_str())
    }

    /// Get word index from vocabulary
    pub fn get_word_idx(&self, word: &str) -> Option<usize> {
        self.ventral.embeddings.word_to_idx.get(word).map(|r| *r)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Get statistics
    pub fn stats(&self) -> DualStreamStats {
        DualStreamStats {
            ventral_concepts: self.ventral.atl.active_concepts().len(),
            dorsal_plans: self.dorsal.ifg.plan_buffer.len(),
            phonological_items: self.dorsal.phonological_loop.active_count(),
            word_window: self.multiscale.word_buffer.len(),
            phrase_window: self.multiscale.phrase_buffer.len(),
            discourse_window: self.multiscale.discourse_buffer.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualStreamStats {
    pub ventral_concepts: usize,
    pub dorsal_plans: usize,
    pub phonological_items: usize,
    pub word_window: usize,
    pub phrase_window: usize,
    pub discourse_window: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ventral_stream() {
        let mut ventral = VentralStream::new(100, 50);

        ventral.embeddings.add_word("hello");
        let tokens = vec![0];

        let semantics = ventral.comprehend(&tokens);
        assert!(!semantics.is_empty());
    }

    #[test]
    fn test_dorsal_stream() {
        let embedding_dim = 300;
        let mut dorsal = DorsalStream::new(embedding_dim);

        let semantic_input = vec![0.5; embedding_dim];
        let output = dorsal.produce(&semantic_input);

        // May produce empty if no learned mappings
    }

    #[test]
    fn test_multiscale() {
        let mut ms = MultiTimescaleProcessor::new();

        for _ in 0..10 {
            ms.process(vec![0.5; 50]);
        }

        assert_eq!(ms.word_buffer.len(), 1);
        assert_eq!(ms.phrase_buffer.len(), 4);
        assert_eq!(ms.discourse_buffer.len(), 6);
    }

    #[test]
    fn test_dual_stream_system() {
        let mut system = DualStreamLanguage::new(100, 50);

        system.ventral.embeddings.add_word("test");
        let tokens = vec![0];

        let semantics = system.comprehend(&tokens);
        assert!(!semantics.is_empty());

        let stats = system.stats();
        assert_eq!(stats.word_window, 1);
    }
}
