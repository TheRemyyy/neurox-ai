//! Semantic Embeddings and Conceptual Spaces
//!
//! Learned distributed representations replacing hash tokenization.
//! Implements Word2Vec-style embeddings, semantic similarity, and compositional meaning.
//!
//! # Architecture
//! - Learned embeddings (300-dim) capturing semantic similarity
//! - ATL (Anterior Temporal Lobe) hub binding multimodal features
//! - Concept cells with high selectivity (1-3% sparsity)
//! - Grid cell semantic space for hierarchical organization
//! - Hub-and-spoke distributed network

use dashmap::DashMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Word2Vec-style embedding layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingLayer {
    /// Vocabulary size
    pub vocab_size: usize,

    /// Embedding dimension (typically 300)
    pub embedding_dim: usize,

    /// Embedding matrix: vocab_size × embedding_dim
    pub embeddings: Vec<Vec<f32>>,

    /// Word to index mapping
    #[serde(skip)]
    pub(crate) word_to_idx: Arc<DashMap<String, usize>>,

    /// Index to word mapping
    pub idx_to_word: Vec<String>,

    /// Learning rate for embedding updates
    pub learning_rate: f32,
}

impl EmbeddingLayer {
    /// Create new embedding layer
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize with small random values (Xavier initialization)
        let scale = (6.0 / (vocab_size + embedding_dim) as f32).sqrt();
        let embeddings = (0..vocab_size)
            .map(|_| {
                (0..embedding_dim)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();

        Self {
            vocab_size,
            embedding_dim,
            embeddings,
            word_to_idx: Arc::new(DashMap::new()),
            idx_to_word: Vec::new(),
            learning_rate: 0.01,
        }
    }

    /// Add word to vocabulary and grow embeddings matrix if needed
    pub fn add_word(&mut self, word: &str) -> usize {
        if let Some(idx) = self.word_to_idx.get(word) {
            return *idx;
        }

        let idx = self.idx_to_word.len();
        let word_str = word.to_string();
        self.word_to_idx.insert(word_str.clone(), idx);
        self.idx_to_word.push(word_str);

        // Grow embeddings matrix if we've exceeded initial vocab_size
        if idx >= self.embeddings.len() {
            let mut rng = rand::thread_rng();
            let scale = (6.0 / (self.embeddings.len() + 1 + self.embedding_dim) as f32).sqrt();
            let new_emb = (0..self.embedding_dim)
                .map(|_| rng.gen_range(-scale..scale))
                .collect();
            self.embeddings.push(new_emb);
            self.vocab_size = self.embeddings.len();
        }

        idx
    }

    /// Get embedding for word
    pub fn get_embedding(&self, word: &str) -> Option<&Vec<f32>> {
        self.word_to_idx
            .get(word)
            .and_then(|idx| self.embeddings.get(*idx))
    }

    /// Get embedding by index
    pub fn get_embedding_by_idx(&self, idx: usize) -> Option<&Vec<f32>> {
        self.embeddings.get(idx)
    }

    /// Compute cosine similarity between two words
    pub fn similarity(&self, word1: &str, word2: &str) -> f32 {
        match (self.get_embedding(word1), self.get_embedding(word2)) {
            (Some(emb1), Some(emb2)) => Self::cosine_similarity(emb1, emb2),
            _ => 0.0,
        }
    }

    /// Cosine similarity between two vectors
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

    /// Find nearest neighbors
    pub fn nearest_neighbors(&self, word: &str, k: usize) -> Vec<(String, f32)> {
        let query_emb = match self.get_embedding(word) {
            Some(emb) => emb,
            None => return Vec::new(),
        };

        let mut similarities: Vec<(String, f32)> = self
            .idx_to_word
            .iter()
            .filter(|w| w.as_str() != word)
            .map(|w| {
                let sim = self
                    .get_embedding(w)
                    .map(|emb| Self::cosine_similarity(query_emb, emb))
                    .unwrap_or(0.0);
                (w.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);
        similarities
    }

    /// Update embedding via gradient descent
    pub fn update_embedding(&mut self, idx: usize, gradient: &[f32]) {
        if let Some(embedding) = self.embeddings.get_mut(idx) {
            for (e, &g) in embedding.iter_mut().zip(gradient.iter()) {
                *e -= self.learning_rate * g;
            }
        }
    }

    /// Train Skip-gram (predict context from center word)
    pub fn train_skipgram(&mut self, center_idx: usize, context_idx: usize) {
        // Simplified Skip-gram update
        if let (Some(center), Some(context)) = (
            self.embeddings.get(center_idx).cloned(),
            self.embeddings.get(context_idx).cloned(),
        ) {
            // Predict context from center
            let dot: f32 = center.iter().zip(context.iter()).map(|(a, b)| a * b).sum();
            let prob = 1.0 / (1.0 + (-dot).exp()); // Sigmoid

            // Gradient: (prob - 1) * context for center word
            let gradient: Vec<f32> = context.iter().map(|&c| (prob - 1.0) * c).collect();
            self.update_embedding(center_idx, &gradient);
        }
    }
}

/// Concept cell - highly selective neuron responding to specific concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptCell {
    /// Cell ID
    pub id: usize,

    /// Concept representation (multimodal features)
    pub concept_vector: Vec<f32>,

    /// Selectivity threshold (high for sparsity)
    pub threshold: f32,

    /// Activation level
    pub activation: f32,

    /// Modalities this cell responds to
    pub modalities: Vec<Modality>,

    /// Learning rate
    pub learning_rate: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    Visual,
    Auditory,
    Textual,
    Motor,
    Emotional,
}

impl ConceptCell {
    pub fn new(id: usize, dim: usize, threshold: f32) -> Self {
        let mut rng = rand::thread_rng();
        let concept_vector = (0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect();

        Self {
            id,
            concept_vector,
            threshold,
            activation: 0.0,
            modalities: vec![Modality::Textual],
            learning_rate: 0.01,
        }
    }

    /// Update activation based on input
    pub fn update(&mut self, input: &[f32]) {
        let similarity = Self::cosine_similarity(&self.concept_vector, input);
        self.activation = if similarity > self.threshold {
            similarity
        } else {
            0.0 // Highly selective - only activate above threshold
        };
    }

    /// Hebbian learning - strengthen association with input
    pub fn learn(&mut self, input: &[f32], strength: f32) {
        for (c, &i) in self.concept_vector.iter_mut().zip(input.iter()) {
            *c += self.learning_rate * strength * i;
        }

        // Normalize
        let norm: f32 = self
            .concept_vector
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        if norm > 0.0 {
            for c in &mut self.concept_vector {
                *c /= norm;
            }
        }
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

/// ATL (Anterior Temporal Lobe) semantic hub
///
/// Hub-and-spoke architecture binding distributed features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticHub {
    /// Concept cells
    pub concept_cells: Vec<ConceptCell>,

    /// Number of concept cells
    pub n_cells: usize,

    /// Embedding dimension
    pub dim: usize,

    /// Sparsity (1-3%)
    pub sparsity: f32,
}

impl SemanticHub {
    pub fn new(n_cells: usize, dim: usize, sparsity: f32) -> Self {
        let threshold = 0.7; // High threshold for selectivity

        let concept_cells = (0..n_cells)
            .map(|i| ConceptCell::new(i, dim, threshold))
            .collect();

        Self {
            concept_cells,
            n_cells,
            dim,
            sparsity,
        }
    }

    /// Encode concept (one-shot binding)
    pub fn encode(&mut self, features: &[f32]) {
        // Update all cells
        for cell in &mut self.concept_cells {
            cell.update(features);
        }

        // Winner-take-all for sparsity
        let k = (self.n_cells as f32 * self.sparsity) as usize;
        self.apply_sparsity(k);

        // Learn in active cells
        for cell in &mut self.concept_cells {
            if cell.activation > 0.0 {
                cell.learn(features, cell.activation);
            }
        }
    }

    /// Retrieve concept from partial cue
    pub fn retrieve(&mut self, partial_cue: &[f32]) -> Vec<f32> {
        // Activate cells based on cue
        for cell in &mut self.concept_cells {
            cell.update(partial_cue);
        }

        // Reconstruct full concept
        let mut reconstruction = vec![0.0; self.dim];
        for cell in &self.concept_cells {
            if cell.activation > 0.0 {
                for (r, &c) in reconstruction.iter_mut().zip(cell.concept_vector.iter()) {
                    *r += cell.activation * c;
                }
            }
        }

        // Normalize
        let norm: f32 = reconstruction.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for r in &mut reconstruction {
                *r /= norm;
            }
        }

        reconstruction
    }

    /// Apply winner-take-all for sparsity
    fn apply_sparsity(&mut self, k: usize) {
        let mut indexed: Vec<(usize, f32)> = self
            .concept_cells
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.activation))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Zero out all but top k
        for (i, _) in indexed.iter().skip(k) {
            self.concept_cells[*i].activation = 0.0;
        }
    }

    /// Get active concepts
    pub fn active_concepts(&self) -> Vec<usize> {
        self.concept_cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.activation > 0.0)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Complete semantic system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSystem {
    /// Embedding layer
    pub embeddings: EmbeddingLayer,

    /// Semantic hub (ATL)
    pub hub: SemanticHub,

    /// Vocabulary
    vocabulary: Vec<String>,
}

impl SemanticSystem {
    pub fn new(vocab_size: usize, embedding_dim: usize, n_concept_cells: usize) -> Self {
        let embeddings = EmbeddingLayer::new(vocab_size, embedding_dim);
        let hub = SemanticHub::new(n_concept_cells, embedding_dim, 0.02);

        Self {
            embeddings,
            hub,
            vocabulary: Vec::new(),
        }
    }

    /// Process word through semantic system
    pub fn process_word(&mut self, word: &str) -> Option<Vec<f32>> {
        // Get embedding
        let embedding = self.embeddings.get_embedding(word)?.clone();

        // Encode in semantic hub
        self.hub.encode(&embedding);

        // Retrieve (pattern completion)
        Some(self.hub.retrieve(&embedding))
    }

    /// Compute semantic similarity
    pub fn similarity(&self, word1: &str, word2: &str) -> f32 {
        self.embeddings.similarity(word1, word2)
    }

    /// Train on word pair (Skip-gram)
    pub fn train(&mut self, center_word: &str, context_word: &str) {
        let center_idx = self.embeddings.word_to_idx.get(center_word).map(|x| *x);
        let context_idx = self.embeddings.word_to_idx.get(context_word).map(|x| *x);

        if let (Some(c_idx), Some(ctx_idx)) = (center_idx, context_idx) {
            self.embeddings.train_skipgram(c_idx, ctx_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_layer() {
        let mut emb = EmbeddingLayer::new(100, 50);

        emb.add_word("hello");
        emb.add_word("world");

        assert!(emb.get_embedding("hello").is_some());
        assert_eq!(emb.get_embedding("hello").unwrap().len(), 50);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((EmbeddingLayer::cosine_similarity(&a, &b) - 1.0).abs() < 0.01);
        assert!(EmbeddingLayer::cosine_similarity(&a, &c).abs() < 0.01);
    }

    #[test]
    fn test_concept_cell() {
        let mut cell = ConceptCell::new(0, 10, 0.7);

        let input = vec![0.5; 10];
        cell.update(&input);

        // May or may not activate depending on random initialization
    }

    #[test]
    fn test_semantic_hub() {
        let mut hub = SemanticHub::new(100, 20, 0.02);

        let features = vec![0.5; 20];
        hub.encode(&features);

        let active = hub.active_concepts();
        assert!(active.len() <= 2); // 2% sparsity of 100 = 2 cells
    }

    #[test]
    fn test_semantic_system() {
        let mut system = SemanticSystem::new(50, 20, 100);

        system.embeddings.add_word("cat");
        system.embeddings.add_word("dog");

        let processed = system.process_word("cat");
        assert!(processed.is_some());
    }
}

// ============================================================================
// Paraphrase Detection and Intent Clustering (Fáze 3)
// ============================================================================

/// Paraphrase detector for recognizing semantically similar inputs
#[derive(Debug, Clone)]
pub struct ParaphraseDetector {
    /// Paraphrase groups: each group contains semantically equivalent phrases
    paraphrase_groups: Vec<ParaphraseGroup>,
    /// Similarity threshold for paraphrase detection
    threshold: f32,
}

/// A group of paraphrases that mean the same thing
#[derive(Debug, Clone)]
pub struct ParaphraseGroup {
    /// Canonical form (primary phrase)
    pub canonical: String,
    /// Alternative phrasings
    pub alternatives: Vec<String>,
    /// Associated intent
    pub intent: String,
}

impl ParaphraseDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            paraphrase_groups: Vec::new(),
            threshold,
        }
    }

    /// Add a paraphrase group
    pub fn add_group(&mut self, canonical: String, alternatives: Vec<String>, intent: String) {
        self.paraphrase_groups.push(ParaphraseGroup {
            canonical,
            alternatives,
            intent,
        });
    }

    /// Load paraphrases from training data (keywords field)
    pub fn load_from_keywords(&mut self, input: &str, keywords: &[String], intent: &str) {
        if !keywords.is_empty() {
            let mut all_alternatives: Vec<String> = keywords.to_vec();
            // Remove canonical from alternatives if present
            all_alternatives.retain(|k| k != input);
            self.add_group(input.to_string(), all_alternatives, intent.to_string());
        }
    }

    /// Find the canonical form for a given input
    pub fn find_canonical(&self, input: &str) -> Option<&str> {
        let input_lower = input.to_lowercase().trim().to_string();

        for group in &self.paraphrase_groups {
            // Check canonical
            if group.canonical.to_lowercase() == input_lower {
                return Some(&group.canonical);
            }
            // Check alternatives
            for alt in &group.alternatives {
                if alt.to_lowercase() == input_lower {
                    return Some(&group.canonical);
                }
            }
        }
        None
    }

    /// Find matching group by similarity (uses embeddings if available)
    pub fn find_similar(
        &self,
        input: &str,
        embeddings: Option<&EmbeddingLayer>,
    ) -> Option<&ParaphraseGroup> {
        let input_lower = input.to_lowercase();

        // First try exact match
        for group in &self.paraphrase_groups {
            if group.canonical.to_lowercase() == input_lower {
                return Some(group);
            }
            for alt in &group.alternatives {
                if alt.to_lowercase() == input_lower {
                    return Some(group);
                }
            }
        }

        // If we have embeddings, try semantic similarity
        if let Some(emb) = embeddings {
            let mut best_group: Option<&ParaphraseGroup> = None;
            let mut best_sim = self.threshold;

            for group in &self.paraphrase_groups {
                // Check similarity with canonical
                let sim = emb.similarity(input, &group.canonical);
                if sim > best_sim {
                    best_sim = sim;
                    best_group = Some(group);
                }
                // Check alternatives
                for alt in &group.alternatives {
                    let sim = emb.similarity(input, alt);
                    if sim > best_sim {
                        best_sim = sim;
                        best_group = Some(group);
                    }
                }
            }
            return best_group;
        }

        None
    }

    /// Get intent for input
    pub fn get_intent(&self, input: &str) -> Option<&str> {
        self.find_similar(input, None).map(|g| g.intent.as_str())
    }

    /// Get number of paraphrase groups
    pub fn len(&self) -> usize {
        self.paraphrase_groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paraphrase_groups.is_empty()
    }
}

/// Intent cluster for grouping similar intents
#[derive(Debug, Clone)]
pub struct IntentCluster {
    /// Cluster name/label
    pub name: String,
    /// Keywords that trigger this intent
    pub keywords: Vec<String>,
    /// Example phrases
    pub examples: Vec<String>,
    /// Centroid embedding (average of all examples)
    pub centroid: Option<Vec<f32>>,
}

impl IntentCluster {
    pub fn new(name: String) -> Self {
        Self {
            name,
            keywords: Vec::new(),
            examples: Vec::new(),
            centroid: None,
        }
    }

    pub fn add_keyword(&mut self, keyword: String) {
        if !self.keywords.contains(&keyword) {
            self.keywords.push(keyword);
        }
    }

    pub fn add_example(&mut self, example: String) {
        if !self.examples.contains(&example) {
            self.examples.push(example);
        }
    }

    /// Check if input matches this cluster (keyword match)
    pub fn matches(&self, input: &str) -> bool {
        let input_lower = input.to_lowercase();
        self.keywords
            .iter()
            .any(|k| input_lower.contains(&k.to_lowercase()))
    }

    /// Compute centroid from examples using embeddings
    pub fn compute_centroid(&mut self, embeddings: &EmbeddingLayer) {
        if self.examples.is_empty() {
            return;
        }

        let dim = embeddings.embedding_dim;
        let mut centroid = vec![0.0f32; dim];
        let mut count = 0;

        for example in &self.examples {
            // Get average embedding for all words in example
            let words: Vec<&str> = example.split_whitespace().collect();
            for word in words {
                if let Some(emb) = embeddings.get_embedding(word) {
                    for (c, e) in centroid.iter_mut().zip(emb.iter()) {
                        *c += e;
                    }
                    count += 1;
                }
            }
        }

        if count > 0 {
            for c in &mut centroid {
                *c /= count as f32;
            }
            self.centroid = Some(centroid);
        }
    }
}

/// Intent clustering system
#[derive(Debug, Clone)]
pub struct IntentClusteringSystem {
    /// All intent clusters
    pub clusters: Vec<IntentCluster>,
}

impl IntentClusteringSystem {
    pub fn new() -> Self {
        Self {
            clusters: Vec::new(),
        }
    }

    /// Add or get cluster by name
    pub fn get_or_create_cluster(&mut self, name: &str) -> &mut IntentCluster {
        if let Some(idx) = self.clusters.iter().position(|c| c.name == name) {
            &mut self.clusters[idx]
        } else {
            self.clusters.push(IntentCluster::new(name.to_string()));
            self.clusters.last_mut().unwrap()
        }
    }

    /// Classify input to best matching cluster
    pub fn classify(&self, input: &str) -> Option<&IntentCluster> {
        self.clusters.iter().find(|c| c.matches(input))
    }

    /// Classify using embeddings (semantic similarity)
    pub fn classify_semantic(
        &self,
        input: &str,
        embeddings: &EmbeddingLayer,
    ) -> Option<&IntentCluster> {
        let mut best_cluster: Option<&IntentCluster> = None;
        let mut best_score = 0.0f32;

        // Get input embedding (average of words)
        let words: Vec<&str> = input.split_whitespace().collect();
        let mut input_emb = vec![0.0f32; embeddings.embedding_dim];
        let mut count = 0;

        for word in words {
            if let Some(emb) = embeddings.get_embedding(word) {
                for (i, e) in input_emb.iter_mut().zip(emb.iter()) {
                    *i += e;
                }
                count += 1;
            }
        }

        if count == 0 {
            return self.classify(input); // Fallback to keyword match
        }

        for i in &mut input_emb {
            *i /= count as f32;
        }

        // Compare with cluster centroids
        for cluster in &self.clusters {
            if let Some(centroid) = &cluster.centroid {
                let sim = EmbeddingLayer::cosine_similarity(&input_emb, centroid);
                if sim > best_score {
                    best_score = sim;
                    best_cluster = Some(cluster);
                }
            }
        }

        if best_score > 0.5 {
            best_cluster
        } else {
            self.classify(input) // Fallback
        }
    }

    /// Get number of clusters
    pub fn len(&self) -> usize {
        self.clusters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }
}

impl Default for IntentClusteringSystem {
    fn default() -> Self {
        Self::new()
    }
}
