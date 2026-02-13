//! Theory of Mind - Understanding Other Agents' Mental States
//!
//! Implements inference of beliefs, desires, and intentions of other agents.
//! Enables the AI to predict others' behavior and understand perspectives.
//!
//! # Architecture
//! - BDI (Belief-Desire-Intention) models for each tracked agent
//! - Perspective-taking via simulation
//! - False-belief reasoning (Sally-Anne paradigm)
//!
//! # References
//! - ToM-SNN (2022) Brain-Inspired Theory of Mind
//! - 2024-2025 research on cognitive architectures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for an agent
pub type AgentId = u64;

/// Belief about the world (proposition + confidence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    /// What the agent believes (semantic representation)
    pub content: Vec<f32>,
    /// Confidence in this belief (0.0 to 1.0)
    pub confidence: f32,
    /// When this belief was formed/updated
    pub timestamp: f32,
    /// Source of belief (perception, inference, communication)
    pub source: BeliefSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefSource {
    Perception,
    Inference,
    Communication,
    Memory,
}

/// Desire/Goal representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Desire {
    /// Goal state representation
    pub goal_state: Vec<f32>,
    /// Priority/importance (0.0 to 1.0)
    pub priority: f32,
    /// Is this desire currently active?
    pub active: bool,
}

/// Intention - commitment to an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intention {
    /// Planned action representation
    pub action: Vec<f32>,
    /// Associated desire
    pub for_desire_idx: usize,
    /// Commitment strength
    pub commitment: f32,
}

/// Belief state (collection of beliefs)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BeliefState {
    pub beliefs: Vec<Belief>,
}

impl BeliefState {
    pub fn add_belief(
        &mut self,
        content: Vec<f32>,
        confidence: f32,
        timestamp: f32,
        source: BeliefSource,
    ) {
        self.beliefs.push(Belief {
            content,
            confidence,
            timestamp,
            source,
        });
    }

    /// Find most similar belief
    pub fn find_similar(&self, query: &[f32], threshold: f32) -> Option<&Belief> {
        self.beliefs
            .iter()
            .filter(|b| Self::cosine_similarity(&b.content, query) > threshold)
            .max_by(|a, b| {
                Self::cosine_similarity(&a.content, query)
                    .partial_cmp(&Self::cosine_similarity(&b.content, query))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a * norm_b > 1e-10 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// BDI Model for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BDIModel {
    /// Agent's beliefs about the world
    pub beliefs: BeliefState,
    /// Agent's desires/goals
    pub desires: Vec<Desire>,
    /// Agent's current intentions
    pub intentions: Vec<Intention>,
    /// Emotional state estimate
    pub emotional_valence: f32,
    /// Trust level (how predictable is this agent)
    pub predictability: f32,
    /// Last observed action
    pub last_action: Option<Vec<f32>>,
    /// Observation count
    pub observation_count: u32,
}

impl Default for BDIModel {
    fn default() -> Self {
        Self {
            beliefs: BeliefState::default(),
            desires: Vec::new(),
            intentions: Vec::new(),
            emotional_valence: 0.0,
            predictability: 0.5,
            last_action: None,
            observation_count: 0,
        }
    }
}

/// Agent model with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentModel {
    pub id: AgentId,
    pub name: Option<String>,
    pub bdi: BDIModel,
    /// Relationship with this agent
    pub relationship: f32, // -1.0 (hostile) to 1.0 (friendly)
    /// Is this agent currently present/active
    pub active: bool,
}

/// Action prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPrediction {
    /// Predicted action representation
    pub action: Vec<f32>,
    /// Confidence in prediction
    pub confidence: f32,
    /// Reasoning for prediction
    pub reasoning: String,
}

/// Theory of Mind System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoryOfMind {
    /// Models of other agents
    pub agent_models: HashMap<AgentId, AgentModel>,

    /// Self ID (to distinguish from others)
    pub self_id: AgentId,

    /// Perspective-taking buffer (for simulation)
    pub perspective_buffer: Vec<f32>,

    /// Inference confidence threshold
    pub inference_threshold: f32,

    /// Maximum number of tracked agents
    pub max_agents: usize,

    /// Total interactions
    pub total_interactions: u64,
}

impl Default for TheoryOfMind {
    fn default() -> Self {
        Self::new(0)
    }
}

impl TheoryOfMind {
    pub fn new(self_id: AgentId) -> Self {
        Self {
            agent_models: HashMap::new(),
            self_id,
            perspective_buffer: Vec::new(),
            inference_threshold: 0.3,
            max_agents: 50,
            total_interactions: 0,
        }
    }

    /// Register a new agent
    pub fn register_agent(&mut self, id: AgentId, name: Option<String>) {
        if self.agent_models.len() >= self.max_agents {
            // Remove least recently interacted agent
            let oldest = self
                .agent_models
                .iter()
                .min_by_key(|(_, m)| m.bdi.observation_count)
                .map(|(id, _)| *id);
            if let Some(oldest_id) = oldest {
                self.agent_models.remove(&oldest_id);
            }
        }

        self.agent_models.insert(
            id,
            AgentModel {
                id,
                name,
                bdi: BDIModel::default(),
                relationship: 0.0,
                active: true,
            },
        );
    }

    /// Update agent model from observation
    pub fn observe_action(
        &mut self,
        agent_id: AgentId,
        action: &[f32],
        context: &[f32],
        timestamp: f32,
    ) {
        // Compute inferred belief before mutable borrow
        let inferred_belief = Self::infer_belief_from_action_static(action, context);

        if let Some(model) = self.agent_models.get_mut(&agent_id) {
            model.bdi.observation_count += 1;
            model.bdi.last_action = Some(action.to_vec());

            // Add the inferred belief
            model
                .bdi
                .beliefs
                .add_belief(inferred_belief, 0.5, timestamp, BeliefSource::Inference);

            // Update predictability based on consistency
            if let Some(ref last) = model.bdi.last_action {
                let consistency = BeliefState::cosine_similarity(last, action);
                model.bdi.predictability = model.bdi.predictability * 0.9 + consistency * 0.1;
            }

            self.total_interactions += 1;
        } else {
            // New agent, register and observe
            self.register_agent(agent_id, None);
            self.observe_action(agent_id, action, context, timestamp);
        }
    }

    /// Infer what belief led to an action (static version)
    fn infer_belief_from_action_static(action: &[f32], context: &[f32]) -> Vec<f32> {
        // Simple inference: belief ≈ action weighted by context
        // In a full system, this would use a learned inverse model
        let mut belief = Vec::with_capacity(action.len().max(context.len()));

        for i in 0..action.len().max(context.len()) {
            let a = action.get(i).copied().unwrap_or(0.0);
            let c = context.get(i).copied().unwrap_or(0.0);
            belief.push((a + c) / 2.0);
        }

        belief
    }

    /// Predict what an agent will do
    pub fn predict_action(&self, agent_id: AgentId, _context: &[f32]) -> Option<ActionPrediction> {
        let model = self.agent_models.get(&agent_id)?;

        // Simple prediction based on desires and beliefs
        // In a full system, this would use a forward model

        if model.bdi.observation_count < 2 {
            return None; // Not enough data
        }

        // Use last action as baseline prediction (persistence model)
        let predicted = model.bdi.last_action.clone()?;

        Some(ActionPrediction {
            action: predicted,
            confidence: model.bdi.predictability,
            reasoning: format!(
                "Based on {} observations, agent {} has predictability {:.2}",
                model.bdi.observation_count, agent_id, model.bdi.predictability
            ),
        })
    }

    /// Infer beliefs of an agent
    pub fn infer_beliefs(&self, agent_id: AgentId) -> Option<&BeliefState> {
        self.agent_models.get(&agent_id).map(|m| &m.bdi.beliefs)
    }

    /// Take perspective of another agent
    pub fn take_perspective(&mut self, agent_id: AgentId, world_state: &[f32]) -> Vec<f32> {
        // Simulate what the world looks like from agent's perspective
        // Filter world state through agent's believed visibility

        let model = match self.agent_models.get(&agent_id) {
            Some(m) => m,
            None => return world_state.to_vec(), // Unknown agent, return as-is
        };

        // Simple perspective: weight world state by agent's beliefs
        let mut perspective = world_state.to_vec();

        for belief in &model.bdi.beliefs.beliefs {
            for (i, &b) in belief.content.iter().enumerate() {
                if i < perspective.len() {
                    // Blend belief into perspective
                    perspective[i] = perspective[i] * 0.7 + b * 0.3 * belief.confidence;
                }
            }
        }

        self.perspective_buffer = perspective.clone();
        perspective
    }

    /// False belief reasoning (Sally-Anne test)
    /// "Agent A believes X, even though X is now false"
    pub fn check_false_belief(&self, agent_id: AgentId, current_truth: &[f32]) -> Option<Vec<f32>> {
        let model = self.agent_models.get(&agent_id)?;

        // Find beliefs that contradict current truth
        for belief in &model.bdi.beliefs.beliefs {
            let similarity = BeliefState::cosine_similarity(&belief.content, current_truth);
            if similarity < 0.5 && belief.confidence > 0.5 {
                // This is a false belief
                return Some(belief.content.clone());
            }
        }

        None
    }

    /// Update relationship with agent
    pub fn update_relationship(&mut self, agent_id: AgentId, delta: f32) {
        if let Some(model) = self.agent_models.get_mut(&agent_id) {
            model.relationship = (model.relationship + delta).clamp(-1.0, 1.0);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> ToMStats {
        ToMStats {
            num_tracked_agents: self.agent_models.len(),
            total_interactions: self.total_interactions,
            avg_predictability: self
                .agent_models
                .values()
                .map(|m| m.bdi.predictability)
                .sum::<f32>()
                / self.agent_models.len().max(1) as f32,
        }
    }
}

/// Statistics for Theory of Mind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToMStats {
    pub num_tracked_agents: usize,
    pub total_interactions: u64,
    pub avg_predictability: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_registration() {
        let mut tom = TheoryOfMind::new(0);
        tom.register_agent(1, Some("Alice".into()));

        assert!(tom.agent_models.contains_key(&1));
        assert_eq!(tom.agent_models.get(&1).unwrap().name, Some("Alice".into()));
    }

    #[test]
    fn test_observation() {
        let mut tom = TheoryOfMind::new(0);
        tom.register_agent(1, None);

        let action = vec![1.0, 0.0, 0.0];
        let context = vec![0.5, 0.5, 0.0];

        tom.observe_action(1, &action, &context, 0.0);

        let model = tom.agent_models.get(&1).unwrap();
        assert_eq!(model.bdi.observation_count, 1);
        assert!(model.bdi.last_action.is_some());
    }

    #[test]
    fn test_prediction() {
        let mut tom = TheoryOfMind::new(0);
        tom.register_agent(1, None);

        // Need multiple observations
        let action = vec![1.0, 0.0, 0.0];
        let context = vec![0.5, 0.5, 0.0];

        tom.observe_action(1, &action, &context, 0.0);
        tom.observe_action(1, &action, &context, 1.0);
        tom.observe_action(1, &action, &context, 2.0);

        let prediction = tom.predict_action(1, &context);
        assert!(prediction.is_some());
    }

    #[test]
    fn test_perspective_taking() {
        let mut tom = TheoryOfMind::new(0);
        tom.register_agent(1, None);

        // Give agent a belief
        tom.observe_action(1, &[1.0, 0.0], &[1.0, 0.0], 0.0);

        let world = vec![0.5, 0.5];
        let perspective = tom.take_perspective(1, &world);

        // Perspective should be influenced by belief
        assert_ne!(perspective, world);
    }
}
