//! Text Dataset for Language Learning
//!
//! Loads and processes text data for training neuromorphic language systems.
//! Supports:
//! - Skip-gram style window-based sampling
//! - JSON supervised learning format with input/output/reward

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Supervised learning pair for JSON training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisedPair {
    /// Input text/prompt
    pub input: String,

    /// Expected output/response (None = no response expected)
    pub output: Option<String>,

    /// Reward signal: positive = good, negative = bad, 0 = neutral
    #[serde(default)]
    pub reward: f32,

    /// Optional category/tag for the pair
    #[serde(default)]
    pub category: Option<String>,

    /// Keywords for flexible matching (any of these words trigger this response)
    #[serde(default)]
    pub keywords: Option<Vec<String>>,

    /// Alternative responses (randomly selected or based on context)
    #[serde(default)]
    pub alternatives: Option<Vec<String>>,

    /// Required context/mood for this response (happy, sad, angry, neutral, etc.)
    #[serde(default)]
    pub context: Option<String>,

    /// Explanation of why this is the correct response (for learning)
    #[serde(default)]
    pub explanation: Option<String>,

    /// Minimum bond level required for this response
    #[serde(default)]
    pub requires_bond: f32,

    /// Detailed neurotransmitter impact map
    /// e.g. {"dopamine": 0.2, "oxytocin": 0.5}
    #[serde(default)]
    pub neuro_impact: Option<HashMap<String, f32>>,
}

/// Vocabulary word from JSON training data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabWord {
    pub word: String,
    pub pos: String, // "noun", "verb", "adjective", etc.
    #[serde(default)]
    pub valence: f32, // -1.0 to 1.0

    // === SEMANTIC KNOWLEDGE ===
    #[serde(default)]
    pub definition: Option<String>,
    #[serde(default)]
    pub synonyms: Vec<String>,
    #[serde(default)]
    pub antonyms: Vec<String>,
    #[serde(default)]
    pub context: Vec<String>, // "informal", "formal", "greeting", etc.
    #[serde(default)]
    pub responds_to: Vec<String>, // what intents this word responds to
    #[serde(default)]
    pub triggers_mood: Option<String>, // what mood this word triggers
    #[serde(default)]
    pub requires_bond: f32, // minimum bond level to use this word (0.0-1.0)
    #[serde(default)]
    pub neuro_impact: Option<HashMap<String, f32>>, // Direct neurotransmitter impact
}

/// Sentence template from JSON training data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceTemplateJson {
    pub structure: Vec<String>, // ["pronoun", "verb", "noun"]
    pub intent: String,         // "greeting", "statement", "question"
}

/// Pragmatic rule from JSON training data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PragmaticRuleJson {
    pub input_intent: String,
    pub response_intent: String,
    #[serde(default)]
    pub mood_effect: f32,
}

/// Metacognition markers configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitionConfigJson {
    pub reasoning_markers: Vec<String>,
    pub uncertainty_markers: Vec<String>,
}

/// Intent Classification Rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentRuleJson {
    pub intent: String,
    pub keywords: Vec<String>,
}

/// Category of sentiment patterns (positive, negative, greeting, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentPatternCategory {
    /// Keywords that trigger this sentiment
    pub keywords: Vec<String>,
    /// Effect on dopamine level
    #[serde(default)]
    pub dopamine_effect: f32,
    /// Effect on serotonin level
    #[serde(default)]
    pub serotonin_effect: f32,
    /// Effect on norepinephrine level
    #[serde(default)]
    pub norepinephrine_effect: f32,
    /// Effect on bond level
    #[serde(default)]
    pub bond_effect: f32,
}

/// Sentiment patterns organized by category
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SentimentPatterns {
    #[serde(default)]
    pub positive: Option<SentimentPatternCategory>,
    #[serde(default)]
    pub negative: Option<SentimentPatternCategory>,
    #[serde(default)]
    pub greeting: Option<SentimentPatternCategory>,
    #[serde(default)]
    pub farewell: Option<SentimentPatternCategory>,
}

/// Emotion trigger rule from JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionTriggerJson {
    /// Target emotion (joy, anger, fear, etc.)
    pub emotion: String,
    /// Patterns that trigger this emotion
    pub trigger_patterns: Vec<String>,
    /// Intensity of the emotion (0.0 to 1.0)
    #[serde(default)]
    pub intensity: f32,
}

/// JSON training dataset format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonDataset {
    /// Supervised learning pairs
    pub pairs: Vec<SupervisedPair>,

    /// Vocabulary with POS tags and valence
    #[serde(default)]
    pub vocabulary: Vec<VocabWord>,

    /// Sentence templates for IFG planner
    #[serde(default)]
    pub sentence_templates: Vec<SentenceTemplateJson>,

    /// Pragmatic rules: how to respond to different intents
    #[serde(default)]
    pub pragmatic_rules: Vec<PragmaticRuleJson>,

    /// Intent detection rules (loaded from JSON)
    #[serde(default)]
    pub intent_rules: Vec<IntentRuleJson>,

    /// Optional metadata
    #[serde(default)]
    pub name: Option<String>,

    /// Optional description
    #[serde(default)]
    pub description: Option<String>,

    /// Metacognition settings (System 2)
    #[serde(default)]
    pub metacognition_markers: Option<MetacognitionConfigJson>,

    /// Sentiment patterns for emotional processing (positive/negative words)
    #[serde(default)]
    pub sentiment_patterns: Option<SentimentPatterns>,

    /// Emotion triggers for emotional state machine
    #[serde(default)]
    pub emotion_triggers: Vec<EmotionTriggerJson>,
}

impl JsonDataset {
    /// Load from JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        serde_json::from_str(&contents)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Create empty dataset
    pub fn new() -> Self {
        Self {
            pairs: Vec::new(),
            vocabulary: Vec::new(),
            sentence_templates: Vec::new(),
            pragmatic_rules: Vec::new(),
            intent_rules: Vec::new(),
            name: None,
            description: None,
            metacognition_markers: None,
            sentiment_patterns: None,
            emotion_triggers: Vec::new(),
        }
    }

    /// Add a supervised pair
    pub fn add_pair(&mut self, input: &str, output: Option<&str>, reward: f32) {
        self.pairs.push(SupervisedPair {
            input: input.to_string(),
            output: output.map(|s| s.to_string()),
            reward,
            category: None,
            keywords: None,
            alternatives: None,
            context: None,
            explanation: None,
            requires_bond: 0.0,
            neuro_impact: None,
        });
    }

    /// Get all positive pairs (reward > 0)
    pub fn positive_pairs(&self) -> Vec<&SupervisedPair> {
        self.pairs.iter().filter(|p| p.reward > 0.0).collect()
    }

    /// Get all negative pairs (reward < 0)  
    pub fn negative_pairs(&self) -> Vec<&SupervisedPair> {
        self.pairs.iter().filter(|p| p.reward < 0.0).collect()
    }

    /// Convert to TextDataset for embedding training
    pub fn to_text_dataset(&self, window_size: usize) -> TextDataset {
        let mut dataset = TextDataset::new(window_size);

        for pair in &self.pairs {
            dataset.add_sentence(&pair.input);
            if let Some(ref output) = pair.output {
                dataset.add_sentence(output);
            }
        }

        dataset
    }

    /// Get statistics
    pub fn stats(&self) -> JsonDatasetStats {
        let positive = self.pairs.iter().filter(|p| p.reward > 0.0).count();
        let negative = self.pairs.iter().filter(|p| p.reward < 0.0).count();
        let neutral = self.pairs.len() - positive - negative;

        JsonDatasetStats {
            total_pairs: self.pairs.len(),
            positive_pairs: positive,
            negative_pairs: negative,
            neutral_pairs: neutral,
        }
    }
}

impl Default for JsonDataset {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON dataset statistics
#[derive(Debug, Clone)]
pub struct JsonDatasetStats {
    pub total_pairs: usize,
    pub positive_pairs: usize,
    pub negative_pairs: usize,
    pub neutral_pairs: usize,
}

/// Text dataset for language training
#[derive(Debug, Clone)]
pub struct TextDataset {
    /// Raw sentences/lines
    pub sentences: Vec<Vec<String>>,

    /// Vocabulary: word -> index
    pub word_to_idx: HashMap<String, usize>,

    /// Reverse vocabulary: index -> word
    pub idx_to_word: Vec<String>,

    /// Window size for Skip-gram sampling
    pub window_size: usize,

    /// Total word count
    pub total_words: usize,
}

impl TextDataset {
    /// Create empty dataset
    pub fn new(window_size: usize) -> Self {
        Self {
            sentences: Vec::new(),
            word_to_idx: HashMap::new(),
            idx_to_word: Vec::new(),
            window_size,
            total_words: 0,
        }
    }

    /// Load dataset from text file (one sentence per line)
    pub fn from_file<P: AsRef<Path>>(path: P, window_size: usize) -> std::io::Result<Self> {
        let path_ref = path.as_ref();

        // Check if it's a JSON file
        if path_ref.extension().map(|e| e == "json").unwrap_or(false) {
            let json_dataset = JsonDataset::from_file(path_ref)?;
            return Ok(json_dataset.to_text_dataset(window_size));
        }

        // Otherwise treat as plain text
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut dataset = Self::new(window_size);

        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                dataset.add_sentence(&line);
            }
        }

        Ok(dataset)
    }

    /// Add a sentence to the dataset
    pub fn add_sentence(&mut self, text: &str) {
        let words: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|w| !w.is_empty())
            .collect();

        for word in &words {
            if !self.word_to_idx.contains_key(word) {
                let idx = self.idx_to_word.len();
                self.word_to_idx.insert(word.clone(), idx);
                self.idx_to_word.push(word.clone());
            }
        }

        self.total_words += words.len();
        self.sentences.push(words);
    }

    /// Add raw text (multiple sentences, split by punctuation)
    pub fn add_text(&mut self, text: &str) {
        // Split by sentence-ending punctuation
        for sentence in text.split(['.', '!', '?']) {
            let trimmed = sentence.trim();
            if !trimmed.is_empty() {
                self.add_sentence(trimmed);
            }
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.idx_to_word.len()
    }

    /// Get word index (or None if unknown)
    pub fn get_word_idx(&self, word: &str) -> Option<usize> {
        self.word_to_idx.get(&word.to_lowercase()).copied()
    }

    /// Get word by index
    pub fn get_word(&self, idx: usize) -> Option<&str> {
        self.idx_to_word.get(idx).map(|s| s.as_str())
    }

    /// Generate Skip-gram training pairs: (center_word, context_word)
    /// Returns indices for embedding training
    pub fn generate_skipgram_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();

        for sentence in &self.sentences {
            for (i, center_word) in sentence.iter().enumerate() {
                let center_idx = match self.word_to_idx.get(center_word) {
                    Some(&idx) => idx,
                    None => continue,
                };

                // Get context words within window
                let start = i.saturating_sub(self.window_size);
                let end = (i + self.window_size + 1).min(sentence.len());

                for (j, word) in sentence
                    .iter()
                    .enumerate()
                    .skip(start)
                    .take(end.saturating_sub(start))
                {
                    if i != j {
                        if let Some(&context_idx) = self.word_to_idx.get(word) {
                            pairs.push((center_idx, context_idx));
                        }
                    }
                }
            }
        }

        pairs
    }

    /// Generate training sequences (for sequential learning)
    /// Returns sequences of word indices
    pub fn generate_sequences(&self, seq_len: usize) -> Vec<Vec<usize>> {
        let mut sequences = Vec::new();

        for sentence in &self.sentences {
            if sentence.len() >= seq_len {
                for window in sentence.windows(seq_len) {
                    let seq: Vec<usize> = window
                        .iter()
                        .filter_map(|w| self.word_to_idx.get(w).copied())
                        .collect();

                    if seq.len() == seq_len {
                        sequences.push(seq);
                    }
                }
            }
        }

        sequences
    }

    /// Get statistics
    pub fn stats(&self) -> TextDatasetStats {
        TextDatasetStats {
            vocab_size: self.vocab_size(),
            total_words: self.total_words,
            sentence_count: self.sentences.len(),
            avg_sentence_len: if self.sentences.is_empty() {
                0.0
            } else {
                self.total_words as f32 / self.sentences.len() as f32
            },
        }
    }
}

/// Dataset statistics
#[derive(Debug, Clone)]
pub struct TextDatasetStats {
    pub vocab_size: usize,
    pub total_words: usize,
    pub sentence_count: usize,
    pub avg_sentence_len: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_sentence() {
        let mut dataset = TextDataset::new(2);
        dataset.add_sentence("Ahoj světe jak se máš");

        assert_eq!(dataset.vocab_size(), 5);
        assert_eq!(dataset.total_words, 5);
        assert_eq!(dataset.sentences.len(), 1);
    }

    #[test]
    fn test_skipgram_pairs() {
        let mut dataset = TextDataset::new(1);
        dataset.add_sentence("a b c");

        let pairs = dataset.generate_skipgram_pairs();

        // With window=1: (a,b), (b,a), (b,c), (c,b)
        assert_eq!(pairs.len(), 4);
    }

    #[test]
    fn test_sequences() {
        let mut dataset = TextDataset::new(2);
        dataset.add_sentence("jedna dva tři čtyři pět");

        let sequences = dataset.generate_sequences(3);

        // 5 words, window 3: [1,2,3], [2,3,4], [3,4,5]
        assert_eq!(sequences.len(), 3);
    }

    #[test]
    fn test_czech_text() {
        let mut dataset = TextDataset::new(2);
        dataset.add_text("Dobrý den. Jak se máš? Všechno je v pořádku!");

        assert_eq!(dataset.sentences.len(), 3);
        assert!(dataset.get_word_idx("dobrý").is_some());
        assert!(dataset.get_word_idx("pořádku").is_some());
    }
}
