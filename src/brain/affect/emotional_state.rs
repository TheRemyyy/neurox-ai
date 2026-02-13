//! Emotional State Machine - Advanced Affective Computing
//!
//! Implements Plutchik's wheel of emotions with compound emotions,
//! mood dynamics, and cognitive modulation based on 2024-2025 research.
//!
//! # Features
//! - 8 primary emotions + compound emotions
//! - Smooth state transitions with hysteresis
//! - Mood as long-term emotional bias
//! - Bidirectional emotion ↔ cognition influence
//! - Integration with neuromodulation system
//!
//! # References
//! - Plutchik's Psychoevolutionary Theory of Emotion
//! - 2025 Affective Computing with LLMs survey
//! - Emotional Neural Networks (EmNNs) research

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Primary emotions based on Plutchik's wheel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Emotion {
    // === PRIMARY EMOTIONS ===
    Joy,
    Sadness,
    Fear,
    Anger,
    Surprise,
    Disgust,
    Trust,
    Anticipation,

    // === COMPOUND EMOTIONS (dyads) ===
    /// Joy + Trust
    Love,
    /// Sadness + Disgust
    Remorse,
    /// Fear + Surprise
    Awe,
    /// Anger + Disgust
    Contempt,
    /// Joy + Anticipation
    Optimism,
    /// Sadness + Fear
    Despair,
    /// Trust + Fear
    Submission,
    /// Anger + Anticipation  
    Aggressiveness,

    /// Neutral/baseline state
    #[default]
    Neutral,
}

impl Emotion {
    /// Get the valence of this emotion (-1.0 to 1.0)
    pub fn valence(&self) -> f32 {
        match self {
            Emotion::Joy => 0.9,
            Emotion::Trust => 0.7,
            Emotion::Anticipation => 0.5,
            Emotion::Surprise => 0.2,
            Emotion::Love => 1.0,
            Emotion::Optimism => 0.8,
            Emotion::Neutral => 0.0,
            Emotion::Fear => -0.6,
            Emotion::Sadness => -0.8,
            Emotion::Disgust => -0.5,
            Emotion::Anger => -0.7,
            Emotion::Contempt => -0.6,
            Emotion::Remorse => -0.7,
            Emotion::Despair => -1.0,
            Emotion::Awe => 0.3,
            Emotion::Submission => -0.2,
            Emotion::Aggressiveness => -0.4,
        }
    }

    /// Get the arousal level of this emotion (0.0 to 1.0)
    pub fn arousal(&self) -> f32 {
        match self {
            Emotion::Joy => 0.7,
            Emotion::Anger => 0.9,
            Emotion::Fear => 0.8,
            Emotion::Surprise => 0.9,
            Emotion::Anticipation => 0.6,
            Emotion::Aggressiveness => 0.95,
            Emotion::Awe => 0.7,
            Emotion::Sadness => 0.3,
            Emotion::Disgust => 0.5,
            Emotion::Trust => 0.4,
            Emotion::Neutral => 0.3,
            Emotion::Love => 0.6,
            Emotion::Optimism => 0.5,
            Emotion::Contempt => 0.4,
            Emotion::Remorse => 0.4,
            Emotion::Despair => 0.2,
            Emotion::Submission => 0.3,
        }
    }

    /// Get opposite emotion
    pub fn opposite(&self) -> Emotion {
        match self {
            Emotion::Joy => Emotion::Sadness,
            Emotion::Sadness => Emotion::Joy,
            Emotion::Fear => Emotion::Anger,
            Emotion::Anger => Emotion::Fear,
            Emotion::Surprise => Emotion::Anticipation,
            Emotion::Anticipation => Emotion::Surprise,
            Emotion::Disgust => Emotion::Trust,
            Emotion::Trust => Emotion::Disgust,
            Emotion::Love => Emotion::Remorse,
            Emotion::Remorse => Emotion::Love,
            Emotion::Optimism => Emotion::Despair,
            Emotion::Despair => Emotion::Optimism,
            _ => Emotion::Neutral,
        }
    }

    /// Get all primary emotions
    pub fn primary_emotions() -> Vec<Emotion> {
        vec![
            Emotion::Joy,
            Emotion::Sadness,
            Emotion::Fear,
            Emotion::Anger,
            Emotion::Surprise,
            Emotion::Disgust,
            Emotion::Trust,
            Emotion::Anticipation,
        ]
    }
}

/// Emotional state with intensity for each emotion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    /// Intensity of each emotion (0.0 to 1.0)
    pub intensities: HashMap<Emotion, f32>,
    /// Dominant emotion (highest intensity)
    pub dominant: Emotion,
    /// Overall valence (-1.0 to 1.0)
    pub valence: f32,
    /// Overall arousal (0.0 to 1.0)
    pub arousal: f32,
}

impl Default for EmotionalState {
    fn default() -> Self {
        let mut intensities = HashMap::new();
        intensities.insert(Emotion::Neutral, 0.5);
        Self {
            intensities,
            dominant: Emotion::Neutral,
            valence: 0.0,
            arousal: 0.3,
        }
    }
}

impl EmotionalState {
    /// Create new emotional state with given emotion
    pub fn new(emotion: Emotion, intensity: f32) -> Self {
        let mut intensities = HashMap::new();
        intensities.insert(emotion, intensity.clamp(0.0, 1.0));
        Self {
            intensities,
            dominant: emotion,
            valence: emotion.valence() * intensity,
            arousal: emotion.arousal() * intensity,
        }
    }

    /// Add emotion with intensity
    pub fn add_emotion(&mut self, emotion: Emotion, intensity: f32) {
        let clamped = intensity.clamp(0.0, 1.0);
        *self.intensities.entry(emotion).or_insert(0.0) += clamped;
        self.intensities
            .insert(emotion, self.intensities[&emotion].min(1.0));
        self.recalculate();
    }

    /// Set emotion intensity directly
    pub fn set_emotion(&mut self, emotion: Emotion, intensity: f32) {
        self.intensities.insert(emotion, intensity.clamp(0.0, 1.0));
        self.recalculate();
    }

    /// Get intensity of specific emotion
    pub fn get_intensity(&self, emotion: Emotion) -> f32 {
        *self.intensities.get(&emotion).unwrap_or(&0.0)
    }

    /// Decay all emotions toward baseline
    pub fn decay(&mut self, dt: f32, decay_rate: f32) {
        for (_, intensity) in self.intensities.iter_mut() {
            *intensity *= (-dt * decay_rate).exp();
            if *intensity < 0.01 {
                *intensity = 0.0;
            }
        }
        // Ensure neutral doesn't decay
        self.intensities.insert(Emotion::Neutral, 0.3);
        self.recalculate();
    }

    /// Blend with another emotional state
    pub fn blend(&mut self, other: &EmotionalState, weight: f32) {
        for (emotion, intensity) in &other.intensities {
            let current = self.get_intensity(*emotion);
            let blended = current * (1.0 - weight) + intensity * weight;
            self.intensities.insert(*emotion, blended);
        }
        self.recalculate();
    }

    /// Recalculate dominant emotion, valence, and arousal
    fn recalculate(&mut self) {
        // Find dominant emotion
        let mut max_intensity = 0.0;
        let mut dominant = Emotion::Neutral;

        for (emotion, intensity) in &self.intensities {
            if *intensity > max_intensity {
                max_intensity = *intensity;
                dominant = *emotion;
            }
        }
        self.dominant = dominant;

        // Calculate weighted valence and arousal
        let mut total_weight = 0.0;
        let mut weighted_valence = 0.0;
        let mut weighted_arousal = 0.0;

        for (emotion, intensity) in &self.intensities {
            if *intensity > 0.01 {
                weighted_valence += emotion.valence() * intensity;
                weighted_arousal += emotion.arousal() * intensity;
                total_weight += intensity;
            }
        }

        if total_weight > 0.0 {
            self.valence = (weighted_valence / total_weight).clamp(-1.0, 1.0);
            self.arousal = (weighted_arousal / total_weight).clamp(0.0, 1.0);
        }
    }

    /// Check if compound emotion should emerge
    pub fn check_compounds(&mut self) {
        let joy = self.get_intensity(Emotion::Joy);
        let trust = self.get_intensity(Emotion::Trust);
        let sadness = self.get_intensity(Emotion::Sadness);
        let fear = self.get_intensity(Emotion::Fear);
        let anger = self.get_intensity(Emotion::Anger);
        let anticipation = self.get_intensity(Emotion::Anticipation);
        let disgust = self.get_intensity(Emotion::Disgust);
        let surprise = self.get_intensity(Emotion::Surprise);

        // Joy + Trust = Love
        if joy > 0.3 && trust > 0.3 {
            self.add_emotion(Emotion::Love, (joy + trust) / 2.0 * 0.5);
        }

        // Sadness + Disgust = Remorse
        if sadness > 0.3 && disgust > 0.3 {
            self.add_emotion(Emotion::Remorse, (sadness + disgust) / 2.0 * 0.5);
        }

        // Fear + Surprise = Awe
        if fear > 0.2 && surprise > 0.4 {
            self.add_emotion(Emotion::Awe, (fear + surprise) / 2.0 * 0.5);
        }

        // Joy + Anticipation = Optimism
        if joy > 0.3 && anticipation > 0.3 {
            self.add_emotion(Emotion::Optimism, (joy + anticipation) / 2.0 * 0.5);
        }

        // Sadness + Fear = Despair
        if sadness > 0.4 && fear > 0.4 {
            self.add_emotion(Emotion::Despair, (sadness + fear) / 2.0 * 0.5);
        }

        // Anger + Anticipation = Aggressiveness
        if anger > 0.4 && anticipation > 0.3 {
            self.add_emotion(Emotion::Aggressiveness, (anger + anticipation) / 2.0 * 0.5);
        }

        // Anger + Disgust = Contempt
        if anger > 0.3 && disgust > 0.3 {
            self.add_emotion(Emotion::Contempt, (anger + disgust) / 2.0 * 0.5);
        }
    }
}

/// Mood state - long-term emotional bias
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodState {
    /// Long-term valence bias
    pub valence_bias: f32,
    /// Long-term arousal bias
    pub arousal_bias: f32,
    /// Mood stability (resistance to change)
    pub stability: f32,
    /// Time constant for mood changes (ms)
    pub tau: f32,
}

impl Default for MoodState {
    fn default() -> Self {
        Self {
            valence_bias: 0.1, // Slightly positive baseline
            arousal_bias: 0.3, // Moderate baseline arousal
            stability: 0.8,    // Fairly stable
            tau: 60000.0,      // 1 minute time constant
        }
    }
}

impl MoodState {
    /// Update mood based on emotional state
    pub fn update(&mut self, emotional_state: &EmotionalState, dt: f32) {
        let alpha = 1.0 - (-dt / self.tau).exp();
        let change_rate = alpha * (1.0 - self.stability);

        // Mood slowly tracks emotional state
        self.valence_bias += change_rate * (emotional_state.valence - self.valence_bias);
        self.arousal_bias += change_rate * (emotional_state.arousal - self.arousal_bias);

        // Clamp
        self.valence_bias = self.valence_bias.clamp(-1.0, 1.0);
        self.arousal_bias = self.arousal_bias.clamp(0.0, 1.0);
    }

    /// Get mood category
    pub fn category(&self) -> &'static str {
        match (self.valence_bias > 0.2, self.arousal_bias > 0.5) {
            (true, true) => "excited",
            (true, false) => "content",
            (false, true) => "stressed",
            (false, false) => "melancholic",
        }
    }
}

/// Emotion transition rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionTransition {
    /// Trigger condition (input pattern match score threshold)
    pub trigger_threshold: f32,
    /// Source emotion (or None for any)
    pub from_emotion: Option<Emotion>,
    /// Target emotion
    pub to_emotion: Emotion,
    /// Intensity of transition
    pub intensity: f32,
    /// Associated input patterns (keywords, etc.)
    pub trigger_patterns: Vec<String>,
}

/// How emotions modulate cognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveModulation {
    /// Attention focus modifier
    pub attention_modifier: f32,
    /// Memory encoding strength modifier
    pub memory_modifier: f32,
    /// Risk tolerance modifier
    pub risk_tolerance: f32,
    /// Processing speed modifier (arousal-based)
    pub processing_speed: f32,
    /// Creativity/exploration modifier
    pub creativity_modifier: f32,
}

impl CognitiveModulation {
    /// Calculate modulation from emotional state
    pub fn from_emotional_state(state: &EmotionalState, mood: &MoodState) -> Self {
        let valence = state.valence;
        let arousal = state.arousal;

        Self {
            // High arousal → narrow attention, low arousal → broad attention
            attention_modifier: 0.5 + arousal * 0.5,
            // Emotional events are better remembered
            memory_modifier: 1.0 + arousal * 0.5,
            // Positive mood → more risk tolerant
            risk_tolerance: 0.5 + valence * 0.3 + mood.valence_bias * 0.2,
            // Arousal speeds up processing
            processing_speed: 0.8 + arousal * 0.4,
            // Positive mood → more creative exploration
            creativity_modifier: 0.7 + valence * 0.2 + (1.0 - arousal) * 0.1,
        }
    }
}

/// Main Emotional State Machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalStateMachine {
    /// Current emotional state
    pub current_state: EmotionalState,
    /// Long-term mood
    pub mood: MoodState,
    /// Transition rules
    pub transitions: Vec<EmotionTransition>,
    /// Decay rates per emotion
    pub decay_rates: HashMap<Emotion, f32>,
    /// Cognitive modulation output
    pub cognitive_modulation: CognitiveModulation,
    /// History of emotional states (for patterns)
    pub history: Vec<(f32, EmotionalState)>,
    /// History capacity
    pub history_capacity: usize,
}

impl Default for EmotionalStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl EmotionalStateMachine {
    /// Create new emotional state machine
    pub fn new() -> Self {
        let mut decay_rates = HashMap::new();
        // Different emotions decay at different rates
        decay_rates.insert(Emotion::Surprise, 0.1); // Fast decay
        decay_rates.insert(Emotion::Anger, 0.02); // Slow decay
        decay_rates.insert(Emotion::Fear, 0.03);
        decay_rates.insert(Emotion::Joy, 0.04);
        decay_rates.insert(Emotion::Sadness, 0.01); // Very slow decay
        decay_rates.insert(Emotion::Trust, 0.005); // Trust builds slowly, decays slowly
        decay_rates.insert(Emotion::Love, 0.002); // Love is persistent
        decay_rates.insert(Emotion::Anticipation, 0.05);
        decay_rates.insert(Emotion::Disgust, 0.03);

        Self {
            current_state: EmotionalState::default(),
            mood: MoodState::default(),
            transitions: Self::default_transitions(),
            decay_rates,
            cognitive_modulation: CognitiveModulation::from_emotional_state(
                &EmotionalState::default(),
                &MoodState::default(),
            ),
            history: Vec::new(),
            history_capacity: 100,
        }
    }

    /// Default emotion transition rules - empty, loaded from training data
    /// Use `load_transitions()` to populate from JSON training data
    fn default_transitions() -> Vec<EmotionTransition> {
        // No hardcoded patterns - all loaded from training data (emotion_triggers in JSON)
        Vec::new()
    }

    /// Load emotion transitions from training data
    pub fn load_transitions(&mut self, transitions: Vec<EmotionTransition>) {
        self.transitions = transitions;
    }

    /// Add a single emotion transition
    pub fn add_transition(&mut self, transition: EmotionTransition) {
        self.transitions.push(transition);
    }

    /// Process input text and update emotional state
    pub fn process_input(&mut self, text: &str, dt: f32) {
        let text_lower = text.to_lowercase();

        // Check transition rules
        for transition in &self.transitions {
            let mut match_score = 0.0;
            for pattern in &transition.trigger_patterns {
                if text_lower.contains(pattern) {
                    match_score += 1.0;
                }
            }

            // Normalize by number of patterns
            if !transition.trigger_patterns.is_empty() {
                match_score /= transition.trigger_patterns.len() as f32;
            }

            if match_score >= transition.trigger_threshold {
                // Check source emotion constraint
                let source_ok = transition
                    .from_emotion
                    .is_none_or(|from| self.current_state.dominant == from);

                if source_ok {
                    self.current_state
                        .add_emotion(transition.to_emotion, transition.intensity * match_score);
                }
            }
        }

        // Check for compound emotions
        self.current_state.check_compounds();

        // Update mood
        self.mood.update(&self.current_state, dt);

        // Update cognitive modulation
        self.cognitive_modulation =
            CognitiveModulation::from_emotional_state(&self.current_state, &self.mood);

        // Record history
        if self.history.len() >= self.history_capacity {
            self.history.remove(0);
        }
        self.history.push((dt, self.current_state.clone()));
    }

    /// Decay emotional state over time
    pub fn update(&mut self, dt: f32) {
        // Apply emotion-specific decay rates
        for (emotion, rate) in &self.decay_rates {
            let current = self.current_state.get_intensity(*emotion);
            if current > 0.01 {
                let decayed = current * (-dt * rate).exp();
                self.current_state.set_emotion(*emotion, decayed);
            }
        }

        // Default decay for emotions without specific rate
        self.current_state.decay(dt, 0.02);

        // Update mood
        self.mood.update(&self.current_state, dt);

        // Update cognitive modulation
        self.cognitive_modulation =
            CognitiveModulation::from_emotional_state(&self.current_state, &self.mood);
    }

    /// Set emotion directly (e.g., from neuromodulation)
    pub fn set_emotion_from_neuromodulation(
        &mut self,
        dopamine: f32,
        serotonin: f32,
        norepinephrine: f32,
        oxytocin: f32,
    ) {
        // Dopamine → Joy/Anticipation
        if dopamine > 0.6 {
            self.current_state
                .add_emotion(Emotion::Joy, (dopamine - 0.5) * 0.5);
            self.current_state
                .add_emotion(Emotion::Anticipation, (dopamine - 0.5) * 0.3);
        }

        // Low serotonin → Sadness
        if serotonin < 0.3 {
            self.current_state
                .add_emotion(Emotion::Sadness, (0.4 - serotonin) * 0.5);
        }

        // High norepinephrine → Fear/Anger (arousal)
        if norepinephrine > 0.7 {
            self.current_state
                .add_emotion(Emotion::Fear, (norepinephrine - 0.6) * 0.3);
            // Context determines if fear or anger
        }

        // Oxytocin → Trust/Love
        if oxytocin > 0.5 {
            self.current_state
                .add_emotion(Emotion::Trust, (oxytocin - 0.4) * 0.5);
        }

        self.current_state.check_compounds();
    }

    /// Get dominant emotion name
    pub fn dominant_emotion_name(&self) -> &'static str {
        match self.current_state.dominant {
            Emotion::Joy => "joy",
            Emotion::Sadness => "sadness",
            Emotion::Fear => "fear",
            Emotion::Anger => "anger",
            Emotion::Surprise => "surprise",
            Emotion::Disgust => "disgust",
            Emotion::Trust => "trust",
            Emotion::Anticipation => "anticipation",
            Emotion::Love => "love",
            Emotion::Remorse => "remorse",
            Emotion::Awe => "awe",
            Emotion::Contempt => "contempt",
            Emotion::Optimism => "optimism",
            Emotion::Despair => "despair",
            Emotion::Submission => "submission",
            Emotion::Aggressiveness => "aggressiveness",
            Emotion::Neutral => "neutral",
        }
    }

    /// Get statistics
    pub fn stats(&self) -> EmotionalStats {
        EmotionalStats {
            dominant_emotion: self.dominant_emotion_name().to_string(),
            dominant_intensity: self
                .current_state
                .get_intensity(self.current_state.dominant),
            valence: self.current_state.valence,
            arousal: self.current_state.arousal,
            mood_valence: self.mood.valence_bias,
            mood_category: self.mood.category().to_string(),
            attention_mod: self.cognitive_modulation.attention_modifier,
            memory_mod: self.cognitive_modulation.memory_modifier,
        }
    }
}

/// Statistics for emotional state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalStats {
    pub dominant_emotion: String,
    pub dominant_intensity: f32,
    pub valence: f32,
    pub arousal: f32,
    pub mood_valence: f32,
    pub mood_category: String,
    pub attention_mod: f32,
    pub memory_mod: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_valence() {
        assert!(Emotion::Joy.valence() > 0.0);
        assert!(Emotion::Sadness.valence() < 0.0);
        assert_eq!(Emotion::Neutral.valence(), 0.0);
    }

    #[test]
    fn test_emotional_state_creation() {
        let state = EmotionalState::new(Emotion::Joy, 0.8);
        assert_eq!(state.dominant, Emotion::Joy);
        assert!(state.valence > 0.5);
    }

    #[test]
    fn test_compound_emotions() {
        let mut state = EmotionalState::default();
        state.add_emotion(Emotion::Joy, 0.5);
        state.add_emotion(Emotion::Trust, 0.5);
        state.check_compounds();

        // Should generate Love
        assert!(state.get_intensity(Emotion::Love) > 0.0);
    }

    #[test]
    fn test_emotional_transitions() {
        let mut esm = EmotionalStateMachine::new();

        // Process greeting
        esm.process_input("Ahoj!", 0.1);

        assert!(esm.current_state.get_intensity(Emotion::Joy) > 0.0);
    }

    #[test]
    fn test_mood_tracking() {
        let mut esm = EmotionalStateMachine::new();

        // Repeated positive input should shift mood
        for _ in 0..10 {
            esm.process_input("super skvělý výborně", 100.0);
        }

        assert!(esm.mood.valence_bias > 0.0);
    }

    #[test]
    fn test_cognitive_modulation() {
        let state = EmotionalState::new(Emotion::Fear, 0.8);
        let mood = MoodState::default();
        let modulation = CognitiveModulation::from_emotional_state(&state, &mood);

        // High arousal emotion should increase attention
        assert!(modulation.attention_modifier > 0.7);
        // Negative valence should decrease risk tolerance
        assert!(modulation.risk_tolerance < 0.5);
    }
}
