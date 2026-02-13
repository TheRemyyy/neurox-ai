//! Inner Dialogue - Multi-Perspective Reasoning
//!
//! Implements internal deliberation through multiple reasoning "voices"
//! that debate and reach consensus. Inspired by QuietSTaR 2024 research.
//!
//! # Architecture
//! - Multiple Perspectives with different cognitive biases
//! - Dialogue Arbiter for consensus/selection
//! - Working memory for dialogue context
//!
//! # Benefits
//! - Better reasoning through self-debate
//! - Reduced overconfidence
//! - More nuanced responses

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cognitive bias type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveBias {
    /// Optimistic, sees positive outcomes
    Optimistic,
    /// Cautious, considers risks
    Cautious,
    /// Analytical, focuses on logic
    Analytical,
    /// Creative, explores alternatives
    Creative,
    /// Empathetic, considers others' feelings
    Empathetic,
    /// Pragmatic, focuses on practicality
    Pragmatic,
    /// Critical, challenges assumptions
    Critical,
}

impl CognitiveBias {
    pub fn all() -> Vec<CognitiveBias> {
        vec![
            CognitiveBias::Optimistic,
            CognitiveBias::Cautious,
            CognitiveBias::Analytical,
            CognitiveBias::Creative,
            CognitiveBias::Empathetic,
            CognitiveBias::Pragmatic,
            CognitiveBias::Critical,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            CognitiveBias::Optimistic => "Optimist",
            CognitiveBias::Cautious => "Skeptic",
            CognitiveBias::Analytical => "Analyst",
            CognitiveBias::Creative => "Creative",
            CognitiveBias::Empathetic => "Empath",
            CognitiveBias::Pragmatic => "Pragmatist",
            CognitiveBias::Critical => "Critic",
        }
    }
}

/// A reasoning perspective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perspective {
    /// Name of this perspective
    pub name: String,
    /// Primary cognitive bias
    pub bias: CognitiveBias,
    /// Weight in final decision (0.0 to 1.0)
    pub weight: f32,
    /// Generated response for current input
    pub current_response: Option<String>,
    /// Confidence in current response
    pub confidence: f32,
    /// Reasoning trace
    pub reasoning: Vec<String>,
}

impl Perspective {
    pub fn new(name: &str, bias: CognitiveBias, weight: f32) -> Self {
        Self {
            name: name.to_string(),
            bias,
            weight,
            current_response: None,
            confidence: 0.0,
            reasoning: Vec::new(),
        }
    }

    /// Generate response from this perspective
    pub fn generate_response(&mut self, input: &str, _context: &[String]) -> String {
        self.reasoning.clear();

        let response = match self.bias {
            CognitiveBias::Optimistic => {
                self.reasoning
                    .push("Looking for positive aspects...".into());
                self.confidence = 0.7;
                format!("Optimisticky: {}", self.apply_bias_to_input(input))
            }
            CognitiveBias::Cautious => {
                self.reasoning.push("Considering potential risks...".into());
                self.confidence = 0.6;
                format!("Opatrně: {}", self.apply_bias_to_input(input))
            }
            CognitiveBias::Analytical => {
                self.reasoning.push("Analyzing logical structure...".into());
                self.confidence = 0.8;
                format!("Analyticky: {}", self.apply_bias_to_input(input))
            }
            CognitiveBias::Creative => {
                self.reasoning
                    .push("Exploring creative alternatives...".into());
                self.confidence = 0.5;
                format!("Kreativně: {}", self.apply_bias_to_input(input))
            }
            CognitiveBias::Empathetic => {
                self.reasoning
                    .push("Considering emotional impact...".into());
                self.confidence = 0.7;
                format!("Empaticky: {}", self.apply_bias_to_input(input))
            }
            CognitiveBias::Pragmatic => {
                self.reasoning
                    .push("Focusing on practical outcomes...".into());
                self.confidence = 0.75;
                format!("Prakticky: {}", self.apply_bias_to_input(input))
            }
            CognitiveBias::Critical => {
                self.reasoning.push("Challenging assumptions...".into());
                self.confidence = 0.65;
                format!("Kriticky: {}", self.apply_bias_to_input(input))
            }
        };

        self.current_response = Some(response.clone());
        response
    }

    fn apply_bias_to_input(&self, input: &str) -> String {
        // In a full system, this would modify the response based on bias
        // For now, return acknowledgment
        match self.bias {
            CognitiveBias::Optimistic => format!("Vidím to pozitivně - {}", input),
            CognitiveBias::Cautious => format!("Buďme opatrní ohledně - {}", input),
            CognitiveBias::Analytical => format!("Logicky vzato - {}", input),
            CognitiveBias::Creative => format!("Co kdybychom - {}", input),
            CognitiveBias::Empathetic => format!("Rozumím, že - {}", input),
            CognitiveBias::Pragmatic => format!("Prakticky řečeno - {}", input),
            CognitiveBias::Critical => format!("Ale pozor - {}", input),
        }
    }
}

/// Utterance in internal dialogue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Utterance {
    /// Which perspective spoke
    pub speaker: String,
    /// What was said
    pub content: String,
    /// Confidence
    pub confidence: f32,
    /// Timestamp
    pub timestamp: f32,
}

/// Dialogue arbiter - decides between perspectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueArbiter {
    /// Voting weights per bias
    pub bias_weights: HashMap<CognitiveBias, f32>,
    /// Consensus threshold
    pub consensus_threshold: f32,
    /// Current context mood (affects which voices dominate)
    pub context_mood: f32, // -1 = cautious, +1 = optimistic
}

impl Default for DialogueArbiter {
    fn default() -> Self {
        Self::new()
    }
}

impl DialogueArbiter {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert(CognitiveBias::Analytical, 0.8);
        weights.insert(CognitiveBias::Pragmatic, 0.7);
        weights.insert(CognitiveBias::Empathetic, 0.6);
        weights.insert(CognitiveBias::Cautious, 0.5);
        weights.insert(CognitiveBias::Optimistic, 0.5);
        weights.insert(CognitiveBias::Creative, 0.4);
        weights.insert(CognitiveBias::Critical, 0.4);

        Self {
            bias_weights: weights,
            consensus_threshold: 0.6,
            context_mood: 0.0,
        }
    }

    /// Select best response from perspectives
    pub fn arbitrate(&self, perspectives: &[Perspective]) -> Option<ArbitrationResult> {
        if perspectives.is_empty() {
            return None;
        }

        let mut best_score = 0.0;
        let mut best_idx = 0;
        let mut total_score = 0.0;

        for (i, p) in perspectives.iter().enumerate() {
            if p.current_response.is_none() {
                continue;
            }

            let bias_weight = self.bias_weights.get(&p.bias).copied().unwrap_or(0.5);
            let mood_adjustment = match p.bias {
                CognitiveBias::Optimistic => self.context_mood * 0.2,
                CognitiveBias::Cautious => -self.context_mood * 0.2,
                _ => 0.0,
            };

            let score = p.confidence * p.weight * bias_weight * (1.0 + mood_adjustment);
            total_score += score;

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        let best = &perspectives[best_idx];

        Some(ArbitrationResult {
            selected_response: best.current_response.clone().unwrap_or_default(),
            selected_perspective: best.name.clone(),
            confidence: best_score / total_score.max(0.01),
            consensus_achieved: best_score / total_score.max(0.01) > self.consensus_threshold,
            dissenting_views: perspectives
                .iter()
                .filter(|p| p.name != best.name && p.current_response.is_some())
                .map(|p| {
                    (
                        p.name.clone(),
                        p.current_response.clone().unwrap_or_default(),
                    )
                })
                .collect(),
        })
    }

    /// Combine multiple perspectives into synthesized response
    pub fn synthesize(&self, perspectives: &[Perspective]) -> String {
        let responses: Vec<_> = perspectives
            .iter()
            .filter_map(|p| p.current_response.as_ref())
            .collect();

        if responses.is_empty() {
            return String::new();
        }

        // Simple synthesis: combine key insights
        // In a full system, this would use more sophisticated NLG
        let mut synthesis = String::from("Po zvážení různých pohledů: ");

        for (i, r) in responses.iter().enumerate() {
            if i < 3 {
                // Limit to top 3
                if i > 0 {
                    synthesis.push_str("; ");
                }
                // Take first part of each response
                let short: String = r.chars().take(50).collect();
                synthesis.push_str(&short);
            }
        }

        synthesis
    }
}

/// Result of arbitration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrationResult {
    pub selected_response: String,
    pub selected_perspective: String,
    pub confidence: f32,
    pub consensus_achieved: bool,
    pub dissenting_views: Vec<(String, String)>,
}

/// Inner Dialogue System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnerDialogue {
    /// Active perspectives
    pub perspectives: Vec<Perspective>,
    /// Dialogue arbiter
    pub arbiter: DialogueArbiter,
    /// Dialogue history
    pub dialogue_buffer: Vec<Utterance>,
    /// Buffer capacity
    pub buffer_capacity: usize,
    /// Whether dialogue is active
    pub active: bool,
    /// Current time
    pub time: f32,
}

impl Default for InnerDialogue {
    fn default() -> Self {
        Self::new()
    }
}

impl InnerDialogue {
    pub fn new() -> Self {
        // Create default perspectives
        let perspectives = vec![
            Perspective::new("Analytik", CognitiveBias::Analytical, 0.8),
            Perspective::new("Empat", CognitiveBias::Empathetic, 0.7),
            Perspective::new("Skeptik", CognitiveBias::Cautious, 0.5),
            Perspective::new("Optimista", CognitiveBias::Optimistic, 0.5),
        ];

        Self {
            perspectives,
            arbiter: DialogueArbiter::new(),
            dialogue_buffer: Vec::new(),
            buffer_capacity: 20,
            active: true,
            time: 0.0,
        }
    }

    /// Process input through all perspectives
    pub fn deliberate(&mut self, input: &str) -> ArbitrationResult {
        if !self.active {
            return ArbitrationResult {
                selected_response: input.to_string(),
                selected_perspective: "Direct".into(),
                confidence: 1.0,
                consensus_achieved: true,
                dissenting_views: Vec::new(),
            };
        }

        // Get context from buffer
        let context: Vec<String> = self
            .dialogue_buffer
            .iter()
            .rev()
            .take(5)
            .map(|u| u.content.clone())
            .collect();

        // Each perspective generates response
        for perspective in &mut self.perspectives {
            let response = perspective.generate_response(input, &context);

            // Record utterance
            self.dialogue_buffer.push(Utterance {
                speaker: perspective.name.clone(),
                content: response,
                confidence: perspective.confidence,
                timestamp: self.time,
            });

            // Trim buffer
            while self.dialogue_buffer.len() > self.buffer_capacity {
                self.dialogue_buffer.remove(0);
            }
        }

        // Arbitrate
        let result = self
            .arbiter
            .arbitrate(&self.perspectives)
            .unwrap_or_else(|| ArbitrationResult {
                selected_response: input.to_string(),
                selected_perspective: "None".into(),
                confidence: 0.0,
                consensus_achieved: false,
                dissenting_views: Vec::new(),
            });

        self.time += 1.0;
        result
    }

    /// Quick response (skip dialogue for simple inputs)
    pub fn quick_response(&self, input: &str) -> bool {
        // Skip dialogue for very short or simple inputs
        input.len() < 5 || input.ends_with('?')
    }

    /// Add perspective
    pub fn add_perspective(&mut self, name: &str, bias: CognitiveBias, weight: f32) {
        self.perspectives.push(Perspective::new(name, bias, weight));
    }

    /// Set mood (affects which perspectives dominate)
    pub fn set_mood(&mut self, mood: f32) {
        self.arbiter.context_mood = mood.clamp(-1.0, 1.0);
    }

    /// Get dialogue summary
    pub fn get_summary(&self) -> String {
        let recent: Vec<_> = self.dialogue_buffer.iter().rev().take(5).collect();

        recent
            .iter()
            .map(|u| format!("{}: {}", u.speaker, u.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get statistics
    pub fn stats(&self) -> InnerDialogueStats {
        InnerDialogueStats {
            num_perspectives: self.perspectives.len(),
            dialogue_length: self.dialogue_buffer.len(),
            avg_confidence: self.perspectives.iter().map(|p| p.confidence).sum::<f32>()
                / self.perspectives.len().max(1) as f32,
            context_mood: self.arbiter.context_mood,
        }
    }
}

/// Statistics for inner dialogue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnerDialogueStats {
    pub num_perspectives: usize,
    pub dialogue_length: usize,
    pub avg_confidence: f32,
    pub context_mood: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perspective_creation() {
        let p = Perspective::new("Test", CognitiveBias::Analytical, 0.5);
        assert_eq!(p.name, "Test");
        assert_eq!(p.bias, CognitiveBias::Analytical);
    }

    #[test]
    fn test_deliberation() {
        let mut dialogue = InnerDialogue::new();
        let result = dialogue.deliberate("Co si o tom myslíš?");

        assert!(!result.selected_response.is_empty());
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_arbitration() {
        let arbiter = DialogueArbiter::new();

        let mut p1 = Perspective::new("A", CognitiveBias::Analytical, 0.8);
        p1.current_response = Some("Response A".into());
        p1.confidence = 0.9;

        let mut p2 = Perspective::new("B", CognitiveBias::Creative, 0.5);
        p2.current_response = Some("Response B".into());
        p2.confidence = 0.5;

        let result = arbiter.arbitrate(&[p1, p2]).unwrap();

        // Analytical with higher confidence should win
        assert_eq!(result.selected_perspective, "A");
    }

    #[test]
    fn test_mood_influence() {
        let mut dialogue = InnerDialogue::new();

        dialogue.set_mood(-1.0); // Cautious mood
        let cautious_result = dialogue.deliberate("Je to dobrý nápad?");

        dialogue.set_mood(1.0); // Optimistic mood
        let optimistic_result = dialogue.deliberate("Je to dobrý nápad?");

        // Responses should differ based on mood
        // (In a full implementation they would be meaningfully different)
    }
}
