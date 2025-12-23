//! IFG (Inferior Frontal Gyrus) - Syntactic Processing
//!
//! Based on 2024-2025 neuroscience research:
//! - IFG encodes grammatical roles (subject, object, predicate)
//! - Plans sentence structure BEFORE word selection
//! - Generates words sequentially with prediction
//!
//! Reference: Morgan, Flinker et al. (2025) Communications Psychology

use serde::{Deserialize, Serialize};
use rand::Rng;

/// Part of Speech - grammatical category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartOfSpeech {
    Noun,       // podstatné jméno (pes, dům, člověk)
    Verb,       // sloveso (jít, dělat, být)
    Adjective,  // přídavné jméno (velký, malý, dobrý)
    Adverb,     // příslovce (rychle, pomalu)
    Pronoun,    // zájmeno (já, ty, on)
    Preposition,// předložka (v, na, do)
    Conjunction,// spojka (a, nebo, ale)
    Interjection,// citoslovce (ahoj, čau)
    Determiner, // člen/ukazatel (ten, ta, to)
    Particle,   // částice (ať, kéž)
    Punctuation,// interpunkce
    Unknown,
}

/// Grammatical role in sentence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrammaticalRole {
    Subject,    // podmět (kdo? co?)
    Predicate,  // přísudek (co dělá?)
    Object,     // předmět (koho? co? komu? čemu?)
    Adverbial,  // příslovečné určení (kde? kdy? jak?)
    Attribute,  // přívlastek (jaký?)
}

/// Word with linguistic annotations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedWord {
    pub text: String,
    pub pos: PartOfSpeech,
    pub role: Option<GrammaticalRole>,
    pub embedding_idx: Option<usize>,
    /// Emotional valence: -1.0 (negative) to +1.0 (positive)
    pub emotional_valence: f32,
    
    // === SEMANTIC KNOWLEDGE ===
    /// What the word MEANS (definition)
    #[serde(default)]
    pub definition: Option<String>,
    
    /// Words with similar meaning
    #[serde(default)]
    pub synonyms: Vec<String>,
    
    /// Words with opposite meaning
    #[serde(default)]
    pub antonyms: Vec<String>,
    
    /// Context tags: when to use this word (informal, formal, greeting, insult, etc.)
    #[serde(default)]
    pub context_tags: Vec<String>,
    
    /// What intents this word responds to (greeting responds to greeting)
    #[serde(default)]
    pub responds_to: Vec<String>,
    
    /// What mood this word triggers when heard/said
    #[serde(default)]
    pub triggers_mood: Option<String>,
    
    /// Minimum bond level required to use this word (0.0-1.0)
    /// e.g. "miluju tě" requires high bond (0.7+)
    #[serde(default)]
    pub requires_bond: f32,

    /// Direct neurotransmitter impact
    /// e.g. {"dopamine": 0.2, "oxytocin": 0.5}
    #[serde(default)]
    pub neuro_impact: Option<std::collections::HashMap<String, f32>>,
}

impl AnnotatedWord {
    pub fn new(text: &str, pos: PartOfSpeech) -> Self {
        Self {
            text: text.to_string(),
            pos,
            role: None,
            embedding_idx: None,
            emotional_valence: 0.0,
            definition: None,
            synonyms: Vec::new(),
            antonyms: Vec::new(),
            context_tags: Vec::new(),
            responds_to: Vec::new(),
            triggers_mood: None,
            requires_bond: 0.0,  // Default: no bond required
            neuro_impact: None,
        }
    }
    
    pub fn with_valence(mut self, valence: f32) -> Self {
        self.emotional_valence = valence.clamp(-1.0, 1.0);
        self
    }
    
    pub fn with_definition(mut self, def: &str) -> Self {
        self.definition = Some(def.to_string());
        self
    }
    
    pub fn with_context(mut self, ctx: Vec<String>) -> Self {
        self.context_tags = ctx;
        self
    }
}

/// Sentence template for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceTemplate {
    pub structure: Vec<PartOfSpeech>,
    pub intent_type: IntentType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentType {
    Greeting,       // pozdrav
    Statement,      // tvrzení
    Question,       // otázka
    Response,       // odpověď
    Exclamation,    // zvolání
    Emotional,      // emoční výraz
    Insult,         // urážka
    Thanks,         // poděkování
    Farewell,       // rozloučení
    Explanation,    // vysvětlení (protože...)
    Humor,          // humor a vtipy
    Philosophy,     // filosofické otázky
    Clarification,  // upřesnění
}

/// Pragmatic rule - what response is appropriate for what input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PragmaticRule {
    /// What intent the input has
    pub input_intent: String,
    /// What intent the response should have
    pub response_intent: String,
    /// How this affects mood (-1.0 to +1.0)
    #[serde(default)]
    pub mood_effect: f32,
}

/// IFG Syntactic Planner - plans sentence structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IFGSyntacticPlanner {
    /// Templates for different sentence types
    pub templates: Vec<SentenceTemplate>,
    
    /// Pragmatic rules: how to respond to different intents
    #[serde(default)]
    pub pragmatic_rules: Vec<PragmaticRule>,
    
    /// Current sentence being generated
    pub current_structure: Vec<PartOfSpeech>,
    pub current_position: usize,
    
    /// Words generated so far
    pub generated_words: Vec<AnnotatedWord>,
    
    /// Working memory for syntactic context
    pub working_memory_capacity: usize,
    
    /// Current input context (for selecting appropriate words)
    #[serde(default)]
    pub input_context: Vec<String>,
    
    /// Learned complete responses (trained from supervised pairs)
    /// Maps intent -> list of grammatically correct responses
    #[serde(default)]
    pub learned_responses: std::collections::HashMap<String, Vec<String>>,

    /// Direct associative memory (hippocampal-like)
    /// Input Text -> List of (Response Text, Bond Requirement)
    #[serde(default)]
    pub direct_memory: std::collections::HashMap<String, Vec<(String, f32)>>,
}

impl Default for IFGSyntacticPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl IFGSyntacticPlanner {
    pub fn new() -> Self {
        // Czech sentence templates
        let templates = vec![
            // Greeting: "Ahoj!" / "Čau!"
            SentenceTemplate {
                structure: vec![PartOfSpeech::Interjection],
                intent_type: IntentType::Greeting,
            },
            // Simple statement: "Já jsem Neurox" (Pronoun Verb Noun)
            SentenceTemplate {
                structure: vec![PartOfSpeech::Pronoun, PartOfSpeech::Verb, PartOfSpeech::Noun],
                intent_type: IntentType::Statement,
            },
            // Response: "Dobře, díky" (Adverb Interjection)
            SentenceTemplate {
                structure: vec![PartOfSpeech::Adverb, PartOfSpeech::Interjection],
                intent_type: IntentType::Response,
            },
            // Question: "Jak se máš?" (Adverb Pronoun Verb)
            SentenceTemplate {
                structure: vec![PartOfSpeech::Adverb, PartOfSpeech::Pronoun, PartOfSpeech::Verb],
                intent_type: IntentType::Question,
            },
            // Simple verb response: "Mám se dobře" (Verb Pronoun Adverb)
            SentenceTemplate {
                structure: vec![PartOfSpeech::Verb, PartOfSpeech::Pronoun, PartOfSpeech::Adverb],
                intent_type: IntentType::Response,
            },
            // Emotional: "Super!" / "Skvělé!"
            SentenceTemplate {
                structure: vec![PartOfSpeech::Adjective],
                intent_type: IntentType::Emotional,
            },
            // Statement with object: "Mám rád X" (Verb Adjective Noun)
            SentenceTemplate {
                structure: vec![PartOfSpeech::Verb, PartOfSpeech::Adjective, PartOfSpeech::Noun],
                intent_type: IntentType::Statement,
            },
            
            // === COMPLEX SENTENCES (System 2 / Phase 6) ===
            // Explanation: "X je Y, protože Z" (Noun Verb Noun Conjunction Verb Noun)
            SentenceTemplate {
                structure: vec![
                    PartOfSpeech::Noun, PartOfSpeech::Verb, PartOfSpeech::Noun, 
                    PartOfSpeech::Conjunction, 
                    PartOfSpeech::Verb, PartOfSpeech::Noun
                ],
                intent_type: IntentType::Explanation,
            },
            // Self-description with contrast: "Jsem X, ale Y" (Verb Noun Conjunction Verb Adjective)
            SentenceTemplate {
                structure: vec![
                    PartOfSpeech::Verb, PartOfSpeech::Noun, 
                    PartOfSpeech::Conjunction, 
                    PartOfSpeech::Verb, PartOfSpeech::Adjective
                ],
                intent_type: IntentType::Statement,
            },
            // Condition/Reasoning: "Pokud X, tak Y" (Conjunction Noun Verb, Adverb Verb)
            SentenceTemplate {
                structure: vec![
                    PartOfSpeech::Conjunction, PartOfSpeech::Noun, PartOfSpeech::Verb, 
                    PartOfSpeech::Adverb, PartOfSpeech::Verb
                ],
                intent_type: IntentType::Explanation, // e.g. "Když data proudí, tak žiju"
            },
            // Rich description: "Velký X a malý Y" (Adjective Noun Conjunction Adjective Noun)
            SentenceTemplate {
                structure: vec![
                    PartOfSpeech::Adjective, PartOfSpeech::Noun, 
                    PartOfSpeech::Conjunction, 
                    PartOfSpeech::Adjective, PartOfSpeech::Noun
                ],
                intent_type: IntentType::Statement,
            },
        ];
        
        Self {
            templates,
            pragmatic_rules: Vec::new(),  // Loaded from JSON
            current_structure: Vec::new(),
            current_position: 0,
            generated_words: Vec::new(),
            working_memory_capacity: 7, // 7±2 items
            input_context: Vec::new(),
            learned_responses: std::collections::HashMap::new(),
            direct_memory: std::collections::HashMap::new(),
        }
    }
    
    /// Select template based on intent type
    pub fn plan_sentence(&mut self, intent: IntentType) {
        let matching: Vec<_> = self.templates.iter()
            .filter(|t| t.intent_type == intent)
            .collect();
        
        if !matching.is_empty() {
            let mut rng = rand::thread_rng();
            let idx = rng.gen_range(0..matching.len());
            self.current_structure = matching[idx].structure.clone();
        } else {
            // Default to simple statement
            self.current_structure = vec![PartOfSpeech::Pronoun, PartOfSpeech::Verb];
        }
        
        self.current_position = 0;
        self.generated_words.clear();
    }
    
    /// Get what POS should be next
    pub fn next_required_pos(&self) -> Option<PartOfSpeech> {
        self.current_structure.get(self.current_position).copied()
    }
    
    /// Add generated word and advance position
    pub fn add_word(&mut self, word: AnnotatedWord) {
        self.generated_words.push(word);
        self.current_position += 1;
        
        // Maintain working memory capacity
        if self.generated_words.len() > self.working_memory_capacity {
            self.generated_words.remove(0);
        }
    }
    
    /// Check if sentence is complete
    pub fn is_complete(&self) -> bool {
        self.current_position >= self.current_structure.len()
    }
    
    /// Get generated sentence as string
    pub fn get_sentence(&self) -> String {
        self.generated_words.iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    /// Reset for new sentence
    pub fn reset(&mut self) {
        self.current_structure.clear();
        self.current_position = 0;
        self.generated_words.clear();
    }
}

/// Lexicon with POS-tagged words
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lexicon {
    pub words: Vec<AnnotatedWord>,
}

impl Default for Lexicon {
    fn default() -> Self {
        Self::empty()  // Default is empty - learns from training
    }
}

impl Lexicon {
    /// Create empty lexicon - brain learns words from training data
    pub fn empty() -> Self {
        Self { words: Vec::new() }
    }
    
    /// Create lexicon with basic words (only for testing)
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::empty()
    }
    
    /// Find words by POS
    pub fn get_by_pos(&self, pos: PartOfSpeech) -> Vec<&AnnotatedWord> {
        self.words.iter().filter(|w| w.pos == pos).collect()
    }
    
    /// Find word by text
    pub fn get_by_text(&self, text: &str) -> Option<&AnnotatedWord> {
        let lower = text.to_lowercase();
        self.words.iter().find(|w| w.text == lower)
    }
    
    /// Add word to lexicon
    pub fn add_word(&mut self, word: AnnotatedWord) {
        // Check if already exists
        if self.get_by_text(&word.text).is_none() {
            self.words.push(word);
        }
    }
    
    /// Get words with positive valence (for happy mood)
    pub fn get_positive_words(&self, pos: PartOfSpeech) -> Vec<&AnnotatedWord> {
        self.words.iter()
            .filter(|w| w.pos == pos && w.emotional_valence > 0.2)
            .collect()
    }
    
    /// Get words with negative valence (for sad/angry mood)
    pub fn get_negative_words(&self, pos: PartOfSpeech) -> Vec<&AnnotatedWord> {
        self.words.iter()
            .filter(|w| w.pos == pos && w.emotional_valence < -0.2)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lexicon() {
        let lex = Lexicon::new();
        let verbs = lex.get_by_pos(PartOfSpeech::Verb);
        assert!(!verbs.is_empty());
        
        let ahoj = lex.get_by_text("ahoj");
        assert!(ahoj.is_some());
        assert_eq!(ahoj.unwrap().pos, PartOfSpeech::Interjection);
    }
    
    #[test]
    fn test_planner() {
        let mut planner = IFGSyntacticPlanner::new();
        planner.plan_sentence(IntentType::Greeting);
        
        assert!(!planner.current_structure.is_empty());
        assert_eq!(planner.next_required_pos(), Some(PartOfSpeech::Interjection));
    }
}
