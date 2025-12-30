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
            tau: 1000.0, // 1 second
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
            tau: 500.0, // 500ms
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
            let mean: f32 =
                self.prediction_errors.iter().sum::<f32>() / self.prediction_errors.len() as f32;
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
            tau: 2000.0, // 2 seconds (slow dynamics)
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

    pub fn set_mood(&mut self, mood: f32) {
        self.baseline = mood.clamp(0.0, 1.0);
    }
}

/// Oxytocin (OXT) system
///
/// Regulates social bonding, trust, and stress reduction
///
/// - High OXT → High trust, lowered stress response
/// - Low OXT → Low trust, social wariness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OxytocinSystem {
    /// Current OXT level (0.0-1.0)
    pub level: f32,

    /// Baseline OXT level (depends on relationship history)
    pub baseline: f32,

    /// Time constant for OXT dynamics (ms)
    /// Oxytocin acts slowly and persists longer
    pub tau: f32,
}

impl Default for OxytocinSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl OxytocinSystem {
    pub fn new() -> Self {
        Self {
            level: 0.1,
            baseline: 0.1,
            tau: 5000.0, // 5 seconds (slower dynamics)
        }
    }

    /// Update OXT level
    pub fn update(&mut self, dt: f32, social_interaction: f32) {
        // Decay toward baseline
        let decay = (-dt / self.tau).exp();
        self.level = self.level * decay + self.baseline * (1.0 - decay);

        // Social interaction increases OXT
        if social_interaction > 0.0 {
            // Gain is dampened at high levels (saturation)
            let gain = 1.0 - self.level;
            self.level += social_interaction * gain * 0.05 * dt / 1000.0;
        }

        self.level = self.level.clamp(0.0, 1.0);
    }

    /// Modulation of stress (Norepinephrine)
    /// High OXT dampens stress/fear response
    pub fn modulate_stress(&self, stress_input: f32) -> f32 {
        let damping = 1.0 - (self.level * 0.5); // Can reduce stress by up to 50%
        stress_input * damping
    }

    /// Modulation of trust
    pub fn trust_level(&self) -> f32 {
        self.level
    }
}

/// Integrated neuromodulatory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodulationSystem {
    pub acetylcholine: AcetylcholineSystem,
    pub norepinephrine: NorepinephrineSystem,
    pub serotonin: SerotoninSystem,
    pub oxytocin: OxytocinSystem,

    /// Dopamine level (from basal ganglia, stored here for opponent processing)
    pub dopamine_level: f32,

    /// Dopamine sensitivity (scales impact of dopamine)
    pub dopamine_sensitivity: f32,
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
            oxytocin: OxytocinSystem::new(),
            dopamine_level: 0.5,
            dopamine_sensitivity: 1.0,
        }
    }

    /// Update all systems
    pub fn update(
        &mut self,
        dt: f32,
        attention: f32,
        prediction_error: f32,
        uncertainty: f32,
        positive_outcomes: bool,
    ) {
        self.acetylcholine.update(dt, attention);

        // Oxytocin dampens NE response (stress buffering)
        let buffered_error = self.oxytocin.modulate_stress(prediction_error);
        let buffered_uncertainty = self.oxytocin.modulate_stress(uncertainty);

        self.norepinephrine
            .update(dt, buffered_error, buffered_uncertainty);
        self.serotonin.update(dt, positive_outcomes);

        // Assume some social interaction if positive outcomes are present (simplification)
        let social_signal = if positive_outcomes { 0.5 } else { 0.0 };
        self.oxytocin.update(dt, social_signal);
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
        let da_contribution = self.dopamine_level * self.dopamine_sensitivity * reward;
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
            oxytocin_level: self.oxytocin.level,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodulationStats {
    pub ach_level: f32,
    pub ach_encoding_mode: bool,
    pub ne_level: f32,
    pub ne_arousal: f32,
    pub ne_novelty: f32,
    pub sht_level: f32,
    pub sht_gamma: f32,
    pub dopamine_level: f32,
    pub oxytocin_level: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acetylcholine_encoding() {
        let mut ach = AcetylcholineSystem::new();

        // High attention → high ACh → encoding mode
        // Need multiple updates for ACh to accumulate
        for _ in 0..20 {
            ach.update(100.0, 0.9);
        }
        assert!(
            ach.encoding_mode,
            "ACh encoding mode should be true with high attention"
        );
        assert!(
            ach.level > 0.4,
            "ACh level ({}) should be > 0.4 with sustained high attention",
            ach.level
        );

        // Low attention → low ACh → consolidation mode
        for _ in 0..100 {
            ach.update(100.0, 0.1);
        }
        assert!(
            !ach.encoding_mode,
            "ACh encoding mode should be false after low attention"
        );
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

        system.set_dopamine(0.8); // High DA
        system.serotonin.level = 0.3; // Low 5-HT

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

    #[test]
    fn test_ach_learning_rate_modulation() {
        let mut system = NeuromodulationSystem::new();

        let base_lr = 0.01;

        // High ACh -> higher learning rate
        system.acetylcholine.level = 0.9;
        let high_ach_lr = system.effective_learning_rate(base_lr);

        // Low ACh -> lower learning rate
        system.acetylcholine.level = 0.1;
        let low_ach_lr = system.effective_learning_rate(base_lr);

        assert!(
            high_ach_lr > low_ach_lr,
            "High ACh should increase learning rate"
        );
    }

    #[test]
    fn test_ne_arousal_learning_rate() {
        let mut system = NeuromodulationSystem::new();

        let base_lr = 0.01;

        // Moderate NE -> optimal learning
        system.norepinephrine.level = 0.5;
        let optimal_lr = system.effective_learning_rate(base_lr);

        // Very low or very high NE -> reduced learning (Yerkes-Dodson)
        system.norepinephrine.level = 0.0;
        let low_ne_lr = system.effective_learning_rate(base_lr);

        system.norepinephrine.level = 1.0;
        let high_ne_lr = system.effective_learning_rate(base_lr);

        assert!(
            optimal_lr > low_ne_lr,
            "Moderate NE should be better than low"
        );
        assert!(
            optimal_lr > high_ne_lr,
            "Moderate NE should be better than high"
        );
    }

    #[test]
    fn test_neuromodulation_stats() {
        let mut system = NeuromodulationSystem::new();

        system.acetylcholine.level = 0.6;
        system.norepinephrine.level = 0.4;
        system.serotonin.level = 0.7;
        system.dopamine_level = 0.5;

        let stats = system.stats();

        assert_eq!(stats.ach_level, 0.6);
        assert_eq!(stats.ne_level, 0.4);
        assert_eq!(stats.sht_level, 0.7);
        assert_eq!(stats.dopamine_level, 0.5);
    }

    #[test]
    fn test_integrated_update() {
        let mut system = NeuromodulationSystem::new();

        system.update(100.0, 0.8, 0.3, 0.2, true);

        // All systems should have updated
        assert!(system.acetylcholine.level > 0.0);
        assert!(system.norepinephrine.level > 0.0);
        assert!(system.serotonin.level > 0.0);
    }

    #[test]
    fn test_biological_realism_ach_range() {
        let system = AcetylcholineSystem::new();

        assert!(
            system.level >= 0.0 && system.level <= 1.0,
            "ACh level should be in 0-1 range"
        );
        assert!(
            system.baseline >= 0.0 && system.baseline <= 1.0,
            "ACh baseline should be in 0-1 range"
        );
        assert!(system.tau > 0.0, "ACh tau should be positive");
    }

    #[test]
    fn test_biological_realism_ne_range() {
        let system = NorepinephrineSystem::new();

        assert!(
            system.level >= 0.0 && system.level <= 1.0,
            "NE level should be in 0-1 range"
        );
        assert!(
            system.baseline >= 0.0 && system.baseline <= 1.0,
            "NE baseline should be in 0-1 range"
        );
        assert!(system.tau > 0.0, "NE tau should be positive");
    }

    #[test]
    fn test_biological_realism_serotonin_range() {
        let system = SerotoninSystem::new();

        assert!(
            system.level >= 0.0 && system.level <= 1.0,
            "Serotonin level should be in 0-1 range"
        );
        assert!(
            system.baseline >= 0.0 && system.baseline <= 1.0,
            "Serotonin baseline should be in 0-1 range"
        );
        assert!(system.tau > 0.0, "Serotonin tau should be positive");
        assert!(
            system.gamma_min >= 0.0 && system.gamma_min <= 1.0,
            "Gamma min should be in discount factor range"
        );
        assert!(
            system.gamma_max >= 0.0 && system.gamma_max <= 1.0,
            "Gamma max should be in discount factor range"
        );
        assert!(
            system.gamma_max > system.gamma_min,
            "Gamma max should be greater than gamma min"
        );
    }

    #[test]
    fn test_ach_consolidation_strength() {
        let mut ach = AcetylcholineSystem::new();

        ach.level = 0.2; // Low ACh
        let consolidation = ach.consolidation_strength();
        let encoding = ach.encoding_strength();

        assert!(
            consolidation > encoding,
            "Low ACh should favor consolidation"
        );
        assert_eq!(
            consolidation + encoding,
            1.0,
            "Consolidation + encoding should sum to 1"
        );
    }

    #[test]
    fn test_ne_prediction_error_tracking() {
        let mut ne = NorepinephrineSystem::new();

        // Add several prediction errors
        for i in 0..10 {
            ne.update(100.0, (i as f32) * 0.1, 0.3);
        }

        // Should have tracked prediction errors
        assert!(ne.prediction_errors.len() > 0);
        assert!(ne.prediction_errors.len() <= 100); // Should cap at 100
    }

    #[test]
    fn test_ne_novelty_computation() {
        let mut ne = NorepinephrineSystem::new();

        // Add constant predictions (low novelty)
        for _ in 0..10 {
            ne.update(100.0, 0.5, 0.3);
        }
        let low_novelty = ne.novelty();

        // Add variable predictions (high novelty)
        ne = NorepinephrineSystem::new();
        for i in 0..10 {
            ne.update(100.0, (i % 2) as f32, 0.3);
        }
        let high_novelty = ne.novelty();

        assert!(
            high_novelty > low_novelty,
            "Variable prediction errors should produce higher novelty"
        );
    }

    #[test]
    fn test_serotonin_mood_setting() {
        let mut sht = SerotoninSystem::new();

        sht.set_mood(0.8);
        assert_eq!(sht.baseline, 0.8);

        // Should clamp to valid range
        sht.set_mood(1.5);
        assert_eq!(sht.baseline, 1.0);

        sht.set_mood(-0.5);
        assert_eq!(sht.baseline, 0.0);
    }

    #[test]
    fn test_serotonin_reward_modulation() {
        let mut sht = SerotoninSystem::new();

        sht.level = 0.2; // Low serotonin
        let impulsive_reward = sht.modulate_reward(1.0);

        sht.level = 0.8; // High serotonin
        let patient_reward = sht.modulate_reward(1.0);

        assert!(
            impulsive_reward > patient_reward,
            "Low serotonin should increase immediate reward sensitivity"
        );
    }

    #[test]
    fn test_dopamine_serotonin_opponent_processing() {
        let mut system = NeuromodulationSystem::new();

        // High DA, low 5-HT -> approach
        system.dopamine_level = 0.8;
        system.serotonin.level = 0.2;
        let approach_value = system.compute_value(1.0, 1.0);

        // Low DA, high 5-HT -> avoidance
        system.dopamine_level = 0.2;
        system.serotonin.level = 0.8;
        let avoidance_value = system.compute_value(1.0, 1.0);

        assert!(
            approach_value > 0.0,
            "High DA/low 5-HT should favor approach"
        );
        assert!(
            avoidance_value < 0.0,
            "Low DA/high 5-HT should favor avoidance"
        );
    }

    #[test]
    fn test_exploration_exploitation_balance() {
        let mut system = NeuromodulationSystem::new();

        // Low NE -> exploitation
        system.norepinephrine.level = 0.1;
        let exploit_epsilon = system.exploration_epsilon();

        // High NE -> exploration
        system.norepinephrine.level = 0.9;
        let explore_epsilon = system.exploration_epsilon();

        assert!(
            explore_epsilon > exploit_epsilon,
            "High NE should increase exploration"
        );
    }

    #[test]
    fn test_ach_attention_dynamics() {
        let mut ach = AcetylcholineSystem::new();

        let initial_level = ach.level;

        // Low attention should not change ACh much
        for _ in 0..10 {
            ach.update(100.0, 0.3);
        }
        let low_attention_level = ach.level;

        // High attention should increase ACh
        ach.level = initial_level;
        for _ in 0..10 {
            ach.update(100.0, 0.9);
        }
        let high_attention_level = ach.level;

        assert!(
            high_attention_level > low_attention_level,
            "High attention should increase ACh"
        );
    }

    #[test]
    fn test_ne_uncertainty_response() {
        let mut ne = NorepinephrineSystem::new();

        let baseline = ne.level;

        // High uncertainty should increase NE
        for _ in 0..10 {
            ne.update(100.0, 0.1, 0.9); // Low PE, high uncertainty
        }

        assert!(ne.level > baseline, "High uncertainty should increase NE");
    }

    #[test]
    fn test_serotonin_positive_outcomes() {
        let mut sht = SerotoninSystem::new();

        let initial_level = sht.level;

        // Positive outcomes should increase serotonin
        for _ in 0..100 {
            sht.update(100.0, true);
        }

        assert!(
            sht.level > initial_level,
            "Positive outcomes should increase serotonin"
        );
    }

    #[test]
    fn test_temporal_dynamics_decay() {
        let mut system = NeuromodulationSystem::new();

        // Boost all systems
        system.acetylcholine.level = 0.9;
        system.norepinephrine.level = 0.9;
        system.serotonin.level = 0.9;

        // Let them decay toward baseline
        for _ in 0..100 {
            system.update(100.0, 0.0, 0.0, 0.0, false);
        }

        // All should decay toward baseline
        assert!(
            system.acetylcholine.level < 0.9,
            "ACh should decay toward baseline"
        );
        assert!(
            system.norepinephrine.level < 0.9,
            "NE should decay toward baseline"
        );
        assert!(
            system.serotonin.level < 0.9,
            "Serotonin should decay toward baseline"
        );
    }

    #[test]
    fn test_edge_case_zero_attention() {
        let mut ach = AcetylcholineSystem::new();

        ach.update(100.0, 0.0);

        // Should not crash with zero attention
        assert!(ach.level >= 0.0);
    }

    #[test]
    fn test_edge_case_extreme_prediction_error() {
        let mut ne = NorepinephrineSystem::new();

        ne.update(100.0, 100.0, 0.5); // Extreme prediction error

        // Should clamp to valid range
        assert!(ne.level >= 0.0 && ne.level <= 1.0);
    }

    #[test]
    fn test_performance_many_updates() {
        let mut system = NeuromodulationSystem::new();

        // Should handle many updates efficiently
        for i in 0..1000 {
            let attention = (i % 100) as f32 / 100.0;
            let pe = ((i * 7) % 100) as f32 / 100.0;
            system.update(1.0, attention, pe, 0.5, i % 2 == 0);
        }

        // Should still be in valid range
        let stats = system.stats();
        assert!(stats.ach_level >= 0.0 && stats.ach_level <= 1.0);
        assert!(stats.ne_level >= 0.0 && stats.ne_level <= 1.0);
        assert!(stats.sht_level >= 0.0 && stats.sht_level <= 1.0);
    }
}
