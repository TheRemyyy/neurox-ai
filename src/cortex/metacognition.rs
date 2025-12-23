use serde::{Deserialize, Serialize};

/// Represents the brain's internal assessment of its own thoughts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveState {
    /// How confident the brain is in the generated response (0.0 - 1.0)
    pub confidence: f32,
    /// Depth of analysis applied (simulates "thinking hard")
    pub introspection_depth: f32,
    /// Flag for uncertainty (triggers "System 2" refinement)
    pub is_uncertain: bool,
    /// Detected logical coherence
    pub coherence: f32,
}

impl Default for MetacognitiveState {
    fn default() -> Self {
        Self {
            confidence: 0.5,
            introspection_depth: 0.0,
            is_uncertain: false,
            coherence: 0.5,
        }
    }
}

/// Metacognition System (The "Observer")
/// Analyzes generated thoughts for quality, coherence, and safety.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metacognition {
    pub state: MetacognitiveState,
    pub history: Vec<MetacognitiveState>,
    
    // Configurable markers loaded from JSON
    pub reasoning_markers: Vec<String>,
    pub uncertainty_markers: Vec<String>,
}

impl Metacognition {
    pub fn new() -> Self {
        Self {
            state: MetacognitiveState::default(),
            history: Vec::new(),
            reasoning_markers: Vec::new(), // Empty by default, loaded later
            uncertainty_markers: Vec::new(),
        }
    }
    
    /// Load markers from configuration
    pub fn load_markers(&mut self, reasoning: Vec<String>, uncertainty: Vec<String>) {
        self.reasoning_markers = reasoning;
        self.uncertainty_markers = uncertainty;
    }

    /// Analyze a potential response before vocalization.
    /// Returns a state assessing the quality of the thought.
    pub fn evaluate_thought(&mut self, thought: &str, context_complexity: f32) -> MetacognitiveState {
        // Heuristic 1: Length & Detail (Correlation with "intelligence" in simple models)
        let length_score = (thought.len() as f32 / 100.0).clamp(0.0, 1.0);
        
        // Heuristic 2: Logical Connectors (simulate reasoning)
        // Uses loaded markers instead of hardcoded strings
        let reasoning_score = if !self.reasoning_markers.is_empty() && self.reasoning_markers.iter().any(|m| thought.to_lowercase().contains(m)) {
            0.9
        } else {
            0.3
        };

        // Heuristic 3: Intellectual Honesty (detection of "I don't know")
        let uncertainty_score = if !self.uncertainty_markers.is_empty() && self.uncertainty_markers.iter().any(|m| thought.to_lowercase().contains(m)) {
            0.2 // Low confidence in content, but high honesty
        } else {
            0.8
        };

        // Combine scores logic
        let raw_confidence = (length_score * 0.3) + (reasoning_score * 0.4) + (uncertainty_score * 0.3) + (context_complexity * 0.2);
        let final_confidence = raw_confidence.clamp(0.0, 1.0);

        self.state = MetacognitiveState {
            confidence: final_confidence,
            introspection_depth: if final_confidence < 0.6 { 0.9 } else { 0.2 }, // Think harder if unsure
            is_uncertain: final_confidence < 0.5,
            coherence: reasoning_score,
        };

        self.history.push(self.state.clone());
        if self.history.len() > 10 {
            self.history.remove(0);
        }

        self.state.clone()
    }

    /// Suggests whether to refine/regenerate the thought
    pub fn should_refine(&self) -> bool {
        self.state.is_uncertain || self.state.coherence < 0.4
    }
}
