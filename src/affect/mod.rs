//! Affective Computing Module
//!
//! Implements emotional state machines and mood dynamics for
//! neuromorphic AI with human-like emotional processing.
//!
//! Based on 2024-2025 research in affective computing and
//! computational models of emotion.

pub mod emotional_state;

pub use emotional_state::{
    CognitiveModulation, Emotion, EmotionTransition, EmotionalState, EmotionalStateMachine,
    EmotionalStats, MoodState,
};
