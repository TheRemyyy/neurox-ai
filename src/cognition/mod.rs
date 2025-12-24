//! Cognition Module
//!
//! Higher-order cognitive functions for human-like intelligence:
//! - Theory of Mind: Understanding other agents' mental states
//! - Inner Dialogue: Multi-perspective reasoning
//! - Self Model: Predictive model of own behavior
//! - Metacognition: Thinking about thinking
//!
//! Based on 2024-2025 research in cognitive architectures,
//! ToM-SNN, and metacognitive AI.

pub mod inner_dialogue;
pub mod metacognition;
pub mod self_model;
pub mod theory_of_mind;

pub use inner_dialogue::{DialogueArbiter, InnerDialogue, Perspective};
pub use metacognition::{
    CognitiveStrategy, ConfidenceLevel, Metacognition, MetacognitionStats, MetacognitiveAssessment,
};
pub use self_model::{BehavioralProfile, CapabilityModel, SelfModel};
pub use theory_of_mind::{AgentModel, BDIModel, BeliefState, TheoryOfMind};
