//! Curiosity-Driven Learning System
//!
//! Implements intrinsic motivation based on information gain and prediction
//! error. Drives exploration in the absence of external rewards.
//!
//! # Theory
//! Curiosity is modeled as the desire to reduce uncertainty about the world.
//! The agent is rewarded for learning (reducing prediction error) rather than
//! just achieving external goals.
//!
//! # Key Components
//! 1. **Prediction Error**: Novelty signal from failed predictions
//! 2. **Information Gain**: Expected reduction in uncertainty from exploration
//! 3. **Competence Progress**: Rate of learning improvement
//!
//! # References
//! - Oudeyer & Kaplan (2007) "Intrinsic Motivation Systems"
//! - Schmidhuber (2010) "Formal Theory of Creativity"
//! - 2024-2025 Curiosity-Based SNN research

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Circular buffer for tracking history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularBuffer<T> {
    data: VecDeque<T>,
    capacity: usize,
}

impl<T: Clone> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(item);
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn last(&self) -> Option<&T> {
        self.data.back()
    }

    pub fn mean(&self) -> f32
    where
        T: Into<f32> + Copy,
    {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.data.iter().map(|x| (*x).into()).sum();
        sum / self.data.len() as f32
    }
}

impl CircularBuffer<f32> {
    pub fn mean_f32(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.data.iter().sum();
        sum / self.data.len() as f32
    }

    pub fn variance(&self) -> f32 {
        if self.data.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_f32();
        let sum_sq: f32 = self.data.iter().map(|x| (x - mean).powi(2)).sum();
        sum_sq / (self.data.len() - 1) as f32
    }

    pub fn std(&self) -> f32 {
        self.variance().sqrt()
    }
}

/// Information gain estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationGain {
    /// Prior entropy estimate
    pub prior_entropy: f32,
    /// Posterior entropy after observation
    pub posterior_entropy: f32,
    /// Computed gain
    pub gain: f32,
}

impl InformationGain {
    /// Compute information gain from prediction distribution change
    pub fn compute(prior: &[f32], posterior: &[f32]) -> Self {
        let prior_entropy = Self::entropy(prior);
        let posterior_entropy = Self::entropy(posterior);

        Self {
            prior_entropy,
            posterior_entropy,
            gain: (prior_entropy - posterior_entropy).max(0.0),
        }
    }

    /// Shannon entropy
    fn entropy(probs: &[f32]) -> f32 {
        let mut h = 0.0;
        for p in probs {
            if *p > 1e-10 {
                h -= p * p.ln();
            }
        }
        h
    }
}

/// Competence Progress - tracks learning rate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetenceProgress {
    /// Historical error on different tasks/regions
    pub error_history: CircularBuffer<f32>,
    /// Recent error
    pub recent_error: f32,
    /// Progress rate (negative error derivative)
    pub progress_rate: f32,
}

impl CompetenceProgress {
    pub fn new(capacity: usize) -> Self {
        Self {
            error_history: CircularBuffer::new(capacity),
            recent_error: 0.5,
            progress_rate: 0.0,
        }
    }

    /// Update with new error observation
    pub fn update(&mut self, error: f32) {
        let old_mean = self.error_history.mean_f32();
        self.error_history.push(error);
        let new_mean = self.error_history.mean_f32();

        // Progress = reduction in error over time
        self.progress_rate = old_mean - new_mean;
        self.recent_error = error;
    }

    /// Get intrinsic reward from learning progress
    pub fn learning_reward(&self) -> f32 {
        // Reward proportional to progress (learning = good)
        self.progress_rate.max(0.0) * 10.0
    }
}

/// Curiosity-Driven Learning System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriosityDrive {
    /// Prediction error history (novelty signal)
    pub prediction_errors: CircularBuffer<f32>,

    /// Competence progress tracker
    pub competence: CompetenceProgress,

    /// Exploration vs exploitation balance (0 = exploit, 1 = explore)
    pub exploration_rate: f32,

    /// Baseline exploration rate
    pub base_exploration: f32,

    /// Novelty threshold for triggering exploration
    pub novelty_threshold: f32,

    /// Curiosity level (running average of normalized prediction error)
    pub curiosity_level: f32,

    /// Intrinsic reward scale
    pub intrinsic_reward_scale: f32,

    /// Habituation rate (repeated stimuli become less interesting)
    pub habituation_rate: f32,

    /// Visited state signatures (for novelty detection)
    pub visited_states: Vec<Vec<f32>>,

    /// Maximum stored states
    pub max_stored_states: usize,

    /// Similarity threshold for "already seen"
    pub similarity_threshold: f32,
}

impl Default for CuriosityDrive {
    fn default() -> Self {
        Self::new()
    }
}

impl CuriosityDrive {
    /// Create new curiosity drive system
    pub fn new() -> Self {
        Self {
            prediction_errors: CircularBuffer::new(100),
            competence: CompetenceProgress::new(50),
            exploration_rate: 0.3,
            base_exploration: 0.2,
            novelty_threshold: 0.5,
            curiosity_level: 0.5,
            intrinsic_reward_scale: 1.0,
            habituation_rate: 0.05,
            visited_states: Vec::new(),
            max_stored_states: 1000,
            similarity_threshold: 0.9,
        }
    }

    /// Process prediction error and compute curiosity reward
    ///
    /// # Arguments
    /// - `prediction_error`: Error from world model prediction (0.0 to 1.0+)
    /// - `state`: Current state representation for novelty detection
    ///
    /// # Returns
    /// Intrinsic reward value
    pub fn process(&mut self, prediction_error: f32, state: &[f32]) -> f32 {
        // 1. Record prediction error
        self.prediction_errors.push(prediction_error);

        // 2. Update competence tracker
        self.competence.update(prediction_error);

        // 3. Compute novelty (is this state new?)
        let novelty = self.compute_novelty(state);

        // 4. Update curiosity level (smoothed novelty)
        let alpha = 0.1;
        self.curiosity_level = self.curiosity_level * (1.0 - alpha) + novelty * alpha;

        // 5. Adjust exploration rate based on curiosity
        self.update_exploration_rate();

        // 6. Compute intrinsic reward
        let intrinsic_reward = self.compute_intrinsic_reward(prediction_error, novelty);

        // 7. Maybe store this state
        if novelty > self.novelty_threshold {
            self.store_state(state);
        }

        intrinsic_reward
    }

    /// Compute novelty of current state
    fn compute_novelty(&self, state: &[f32]) -> f32 {
        if self.visited_states.is_empty() {
            return 1.0; // First state is maximally novel
        }

        // Find most similar stored state
        let mut max_similarity = 0.0;
        for stored in &self.visited_states {
            let sim = self.cosine_similarity(state, stored);
            if sim > max_similarity {
                max_similarity = sim;
            }
        }

        // Novelty = 1 - similarity to nearest neighbor
        (1.0 - max_similarity).max(0.0)
    }

    /// Cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
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

    /// Store state signature for future novelty detection
    fn store_state(&mut self, state: &[f32]) {
        // Don't store if too similar to existing
        for stored in &self.visited_states {
            if self.cosine_similarity(state, stored) > self.similarity_threshold {
                return;
            }
        }

        // Add new state
        if self.visited_states.len() >= self.max_stored_states {
            // Remove oldest (FIFO)
            self.visited_states.remove(0);
        }
        self.visited_states.push(state.to_vec());
    }

    /// Update exploration rate based on curiosity
    fn update_exploration_rate(&mut self) {
        // High curiosity → more exploration
        // But also consider competence progress
        let curiosity_bonus = self.curiosity_level * 0.3;
        let progress_bonus = self.competence.progress_rate.max(0.0) * 5.0;

        self.exploration_rate =
            (self.base_exploration + curiosity_bonus + progress_bonus).clamp(0.1, 0.9);
    }

    /// Compute intrinsic reward from curiosity
    fn compute_intrinsic_reward(&self, prediction_error: f32, novelty: f32) -> f32 {
        // Combination of:
        // 1. Learning progress (improvement = reward)
        let progress_reward = self.competence.learning_reward();

        // 2. Prediction error (novelty = reward, but not too high)
        // Use "Goldilocks" zone - not too easy, not too hard
        let optimal_difficulty = 0.5;
        let difficulty_match = 1.0 - (prediction_error - optimal_difficulty).abs();
        let difficulty_reward = difficulty_match * 0.5;

        // 3. Novelty bonus
        let novelty_reward = novelty * 0.3;

        (progress_reward + difficulty_reward + novelty_reward) * self.intrinsic_reward_scale
    }

    /// Get exploration probability for action selection
    pub fn should_explore(&self) -> bool {
        rand::random::<f32>() < self.exploration_rate
    }

    /// Get curiosity bonus for a potential action
    pub fn action_curiosity_bonus(&self, predicted_next_state: &[f32]) -> f32 {
        // How novel would the next state be?
        self.compute_novelty(predicted_next_state)
    }

    /// Reset habituation (e.g., after sleep)
    pub fn reset_habituation(&mut self) {
        // Clear some stored states to allow re-exploration
        let keep = self.visited_states.len() / 2;
        self.visited_states.truncate(keep);
    }

    /// Get statistics
    pub fn stats(&self) -> CuriosityStats {
        CuriosityStats {
            curiosity_level: self.curiosity_level,
            exploration_rate: self.exploration_rate,
            recent_prediction_error: *self.prediction_errors.last().unwrap_or(&0.0),
            learning_progress: self.competence.progress_rate,
            num_stored_states: self.visited_states.len(),
            avg_prediction_error: self.prediction_errors.mean_f32(),
        }
    }
}

/// Statistics for curiosity system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriosityStats {
    pub curiosity_level: f32,
    pub exploration_rate: f32,
    pub recent_prediction_error: f32,
    pub learning_progress: f32,
    pub num_stored_states: usize,
    pub avg_prediction_error: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_buffer() {
        let mut buf: CircularBuffer<f32> = CircularBuffer::new(3);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.push(4.0); // Should push out 1.0

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.mean_f32(), 3.0);
    }

    #[test]
    fn test_novelty_detection() {
        let mut curiosity = CuriosityDrive::new();

        // First state is novel
        let state1 = vec![1.0, 0.0, 0.0];
        let reward1 = curiosity.process(0.5, &state1);
        assert!(reward1 > 0.0);

        // Same state again - less novel
        let reward2 = curiosity.process(0.5, &state1);
        assert!(reward2 < reward1 || (reward1 - reward2).abs() < 0.1);

        // New different state - novel again
        let state2 = vec![0.0, 1.0, 0.0];
        let reward3 = curiosity.process(0.5, &state2);
        assert!(reward3 > 0.0);
    }

    #[test]
    fn test_competence_progress() {
        let mut progress = CompetenceProgress::new(10);

        // Simulate improving (decreasing error)
        progress.update(0.8);
        progress.update(0.7);
        progress.update(0.6);
        progress.update(0.5);

        // Should show positive progress
        assert!(progress.progress_rate > 0.0);
        assert!(progress.learning_reward() > 0.0);
    }

    #[test]
    fn test_exploration_rate() {
        let mut curiosity = CuriosityDrive::new();
        curiosity.base_exploration = 0.2;

        // High novelty should increase exploration
        for _ in 0..10 {
            let novel_state = vec![rand::random(), rand::random(), rand::random()];
            curiosity.process(0.8, &novel_state);
        }

        assert!(curiosity.exploration_rate > curiosity.base_exploration);
    }

    #[test]
    fn test_information_gain() {
        let prior = vec![0.25, 0.25, 0.25, 0.25]; // Uniform
        let posterior = vec![0.7, 0.1, 0.1, 0.1]; // More certain

        let gain = InformationGain::compute(&prior, &posterior);
        assert!(gain.gain > 0.0);
        assert!(gain.prior_entropy > gain.posterior_entropy);
    }
}
