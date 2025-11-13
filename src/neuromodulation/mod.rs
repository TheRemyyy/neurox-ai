//! Neuromodulatory Systems - ACh, NE, 5-HT
//!
//! Global modulatory signals adjusting learning rates, state-dependent plasticity,
//! exploration-exploitation balance, and temporal discounting.
//!
//! # Systems
//! - **Acetylcholine (ACh)**: Attention and memory encoding
//!   - High ACh → enhanced learning (encoding mode)
//!   - Low ACh → consolidation mode
//!   - Modulates learning rate: lr_effective = lr_base × (1 + ACh)
//!
//! - **Norepinephrine (NE)**: Arousal, novelty, uncertainty
//!   - Promotes exploration when high
//!   - Signals unexpected uncertainty
//!   - Exploration bonus: ε = 0.1 × NE
//!
//! - **Serotonin (5-HT)**: Patience and temporal discounting
//!   - High 5-HT → patient (γ → 0.99)
//!   - Low 5-HT → impulsive (γ → 0.95)
//!   - Opponent to dopamine for nuanced value computation

use serde::{Deserialize, Serialize};

/// Acetylcholine (ACh) system
///
/// Regulates attention and memory encoding/consolidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcetylcholineSystem {
    /// Current ACh level (0.0-1.0)
    pub level: f32,

    /// Baseline ACh level
    pub baseline: f32,

    /// Time constant for ACh dynamics (ms)
    pub tau: f32,

    /// Attention-driven ACh release
    pub attention_gain: f32,

    /// Encoding vs consolidation mode
    pub encoding_mode: bool,
}

impl Default for AcetylcholineSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl AcetylcholineSystem {
    pub fn new() -> Self {
        Self {
            level: 0.3,
            baseline: 0.3,
            tau: 1000.0,  // 1 second
            attention_gain: 2.0,
            encoding_mode: true,
        }
    }

    /// Update ACh level based on attention
    pub fn update(&mut self, dt: f32, attention_level: f32) {
        // Decay toward baseline
        let decay = (-dt / self.tau).exp();
        self.level = self.level * decay + self.baseline * (1.0 - decay);

        // Attention-driven phasic release
        if attention_level > 0.7 {
            self.level += (attention_level - 0.7) * self.attention_gain * dt / 1000.0;
            self.level = self.level.min(1.0);
        }

        // Update encoding mode
        self.encoding_mode = self.level > 0.5;
    }

    /// Modulate learning rate
    ///
    /// High ACh → enhanced learning (encoding)
    /// Low ACh → reduced learning (consolidation)
    pub fn modulate_learning_rate(&self, base_lr: f32) -> f32 {
        base_lr * (1.0 + self.level)
    }

    /// Set behavioral state
    pub fn set_encoding_mode(&mut self, encoding: bool) {
        self.encoding_mode = encoding;
        self.level = if encoding { 0.8 } else { 0.2 };
    }

    /// Get encoding strength (0.0-1.0)
    pub fn encoding_strength(&self) -> f32 {
        self.level
    }

    /// Get consolidation strength (inverse of encoding)
    pub fn consolidation_strength(&self) -> f32 {
        1.0 - self.level
    }
}

/// Norepinephrine (NE) system
///
/// Signals arousal, novelty, and uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NorepinephrineSystem {
    /// Current NE level (0.0-1.0)
    pub level: f32,

    /// Baseline NE level
    pub baseline: f32,

    /// Time constant for NE dynamics (ms)
    pub tau: f32,

    /// Novelty detection gain
    pub novelty_gain: f32,

    /// Uncertainty gain
    pub uncertainty_gain: f32,

    /// Recent prediction errors (for novelty)
    prediction_errors: Vec<f32>,
}

impl Default for NorepinephrineSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl NorepinephrineSystem {
    pub fn new() -> Self {
        Self {
            level: 0.3,
            baseline: 0.3,
            tau: 500.0,  // 500ms
            novelty_gain: 3.0,
            uncertainty_gain: 2.0,
            prediction_errors: Vec::new(),
        }
    }

    /// Update NE level based on novelty and uncertainty
    pub fn update(&mut self, dt: f32, prediction_error: f32, uncertainty: f32) {
        // Decay toward baseline
        let decay = (-dt / self.tau).exp();
        self.level = self.level * decay + self.baseline * (1.0 - decay);

        // Novelty detection (large prediction errors)
        if prediction_error.abs() > 0.3 {
            let novelty = prediction_error.abs();
            self.level += novelty * self.novelty_gain * dt / 1000.0;
        }

        // Uncertainty signals
        if uncertainty > 0.5 {
            self.level += (uncertainty - 0.5) * self.uncertainty_gain * dt / 1000.0;
        }

        self.level = self.level.clamp(0.0, 1.0);

        // Store prediction error
        self.prediction_errors.push(prediction_error);
        if self.prediction_errors.len() > 100 {
            self.prediction_errors.remove(0);
        }
    }

    /// Get exploration bonus
    ///
    /// High NE → more exploration
    pub fn exploration_bonus(&self) -> f32 {
        0.1 * self.level
    }

    /// Modulate learning rate (inverted-U: moderate NE optimal)
    pub fn modulate_learning_rate(&self, base_lr: f32) -> f32 {
        // Inverted-U relationship (Yerkes-Dodson)
        let optimal_level = 0.5;
        let deviation = (self.level - optimal_level).abs();
        let modulation = 1.0 + (1.0 - 2.0 * deviation);
        base_lr * modulation.max(0.5)
    }

    /// Get arousal level
    pub fn arousal(&self) -> f32 {
        self.level
    }

    /// Compute novelty from recent prediction errors
    pub fn novelty(&self) -> f32 {
        if self.prediction_errors.is_empty() {
            return 0.0;
        }

        let variance: f32 = {
            let mean: f32 = self.prediction_errors.iter().sum::<f32>()
                / self.prediction_errors.len() as f32;
            self.prediction_errors
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / self.prediction_errors.len() as f32
        };

        variance.sqrt().min(1.0)
    }
}

/// Serotonin (5-HT) system
///
/// Regulates patience and temporal discounting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerotoninSystem {
    /// Current 5-HT level (0.0-1.0)
    pub level: f32,

    /// Baseline 5-HT level
    pub baseline: f32,

    /// Time constant for 5-HT dynamics (ms)
    pub tau: f32,

    /// Minimum discount factor (impulsive)
    pub gamma_min: f32,

    /// Maximum discount factor (patient)
    pub gamma_max: f32,
}

impl Default for SerotoninSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl SerotoninSystem {
    pub fn new() -> Self {
        Self {
            level: 0.5,
            baseline: 0.5,
            tau: 2000.0,  // 2 seconds (slow dynamics)
            gamma_min: 0.90,
            gamma_max: 0.99,
        }
    }

    /// Update 5-HT level
    pub fn update(&mut self, dt: f32, positive_outcomes: bool) {
        // Decay toward baseline
        let decay = (-dt / self.tau).exp();
        self.level = self.level * decay + self.baseline * (1.0 - decay);

        // Positive outcomes increase 5-HT (slowly)
        if positive_outcomes {
            self.level += 0.01 * dt / 1000.0;
        }

        self.level = self.level.clamp(0.0, 1.0);
    }

    /// Get temporal discount factor
    ///
    /// High 5-HT → patient (high γ)
    /// Low 5-HT → impulsive (low γ)
    pub fn discount_factor(&self) -> f32 {
        self.gamma_min + (self.gamma_max - self.gamma_min) * self.level
    }

    /// Modulate reward sensitivity
    pub fn modulate_reward(&self, reward: f32) -> f32 {
        // Low 5-HT → increased sensitivity to immediate rewards
        let sensitivity = 1.0 + (1.0 - self.level);
        reward * sensitivity
    }

    /// Get patience level
    pub fn patience(&self) -> f32 {
        self.level
    }

    /// Set mood (affects 5-HT baseline)
    pub fn set_mood(&mut self, mood: f32) {
        self.baseline = mood.clamp(0.0, 1.0);
    }
}

/// Integrated neuromodulatory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodulationSystem {
    pub acetylcholine: AcetylcholineSystem,
    pub norepinephrine: NorepinephrineSystem,
    pub serotonin: SerotoninSystem,

    /// Dopamine level (from basal ganglia, stored here for opponent processing)
    pub dopamine_level: f32,
}

impl Default for NeuromodulationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuromodulationSystem {
    pub fn new() -> Self {
        Self {
            acetylcholine: AcetylcholineSystem::new(),
            norepinephrine: NorepinephrineSystem::new(),
            serotonin: SerotoninSystem::new(),
            dopamine_level: 0.3,
        }
    }

    /// Update all neuromodulators
    pub fn update(
        &mut self,
        dt: f32,
        attention: f32,
        prediction_error: f32,
        uncertainty: f32,
        positive_outcome: bool,
    ) {
        self.acetylcholine.update(dt, attention);
        self.norepinephrine.update(dt, prediction_error, uncertainty);
        self.serotonin.update(dt, positive_outcome);
    }

    /// Get effective learning rate (modulated by all systems)
    pub fn effective_learning_rate(&self, base_lr: f32) -> f32 {
        let mut lr = base_lr;

        // ACh: encoding vs consolidation
        lr = self.acetylcholine.modulate_learning_rate(lr);

        // NE: inverted-U arousal
        lr = self.norepinephrine.modulate_learning_rate(lr);

        lr
    }

    /// Get exploration parameter
    pub fn exploration_epsilon(&self) -> f32 {
        self.norepinephrine.exploration_bonus()
    }

    /// Get temporal discount factor
    pub fn discount_factor(&self) -> f32 {
        self.serotonin.discount_factor()
    }

    /// Opponent DA-5HT processing for value computation
    ///
    /// DA: Positive valence, approach, reward
    /// 5-HT: Negative valence, avoidance, punishment
    pub fn compute_value(&self, reward: f32, punishment: f32) -> f32 {
        let da_contribution = self.dopamine_level * reward;
        let sht_contribution = self.serotonin.level * punishment;

        da_contribution - sht_contribution
    }

    /// Set dopamine level (from basal ganglia)
    pub fn set_dopamine(&mut self, level: f32) {
        self.dopamine_level = level.clamp(0.0, 1.0);
    }

    /// Get comprehensive statistics
    pub fn stats(&self) -> NeuromodulationStats {
        NeuromodulationStats {
            ach_level: self.acetylcholine.level,
            ach_encoding_mode: self.acetylcholine.encoding_mode,
            ne_level: self.norepinephrine.level,
            ne_arousal: self.norepinephrine.arousal(),
            ne_novelty: self.norepinephrine.novelty(),
            sht_level: self.serotonin.level,
            sht_gamma: self.serotonin.discount_factor(),
            dopamine_level: self.dopamine_level,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeuromodulationStats {
    pub ach_level: f32,
    pub ach_encoding_mode: bool,
    pub ne_level: f32,
    pub ne_arousal: f32,
    pub ne_novelty: f32,
    pub sht_level: f32,
    pub sht_gamma: f32,
    pub dopamine_level: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acetylcholine_encoding() {
        let mut ach = AcetylcholineSystem::new();

        // High attention → high ACh → encoding mode
        ach.update(100.0, 0.9);
        assert!(ach.encoding_mode);
        assert!(ach.level > 0.5);

        // Low attention → low ACh → consolidation mode
        for _ in 0..100 {
            ach.update(100.0, 0.1);
        }
        assert!(!ach.encoding_mode);
    }

    #[test]
    fn test_norepinephrine_novelty() {
        let mut ne = NorepinephrineSystem::new();

        let baseline = ne.level;

        // Large prediction error → novelty → NE release
        ne.update(100.0, 0.8, 0.2);

        assert!(ne.level > baseline);
        assert!(ne.exploration_bonus() > 0.0);
    }

    #[test]
    fn test_serotonin_patience() {
        let mut sht = SerotoninSystem::new();

        // High 5-HT → patient → high gamma
        sht.level = 0.9;
        let patient_gamma = sht.discount_factor();

        // Low 5-HT → impulsive → low gamma
        sht.level = 0.1;
        let impulsive_gamma = sht.discount_factor();

        assert!(patient_gamma > impulsive_gamma);
        assert!(patient_gamma > 0.95);
        assert!(impulsive_gamma < 0.95);
    }

    #[test]
    fn test_opponent_da_sht() {
        let mut system = NeuromodulationSystem::new();

        system.set_dopamine(0.8);  // High DA
        system.serotonin.level = 0.3;  // Low 5-HT

        // Should favor approach (DA > 5-HT)
        let value = system.compute_value(1.0, 1.0);
        assert!(value > 0.0);
    }

    #[test]
    fn test_integrated_learning_rate() {
        let system = NeuromodulationSystem::new();

        let base_lr = 0.01;
        let effective_lr = system.effective_learning_rate(base_lr);

        // Should be modulated
        assert!(effective_lr != base_lr);
    }
}
