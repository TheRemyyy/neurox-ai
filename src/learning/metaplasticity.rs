//! Metaplasticity and Homeostatic Mechanisms
//!
//! Prevents runaway dynamics and maintains optimal computational regime.
//!
//! # Mechanisms
//! - **BCM Theory**: Sliding threshold θ_m = <y²> determines LTP/LTD
//! - **Synaptic Scaling**: 24-48h multiplicative adjustment preserving relative strengths
//! - **Criticality Homeostasis**: Tune toward edge of phase transition (2024 discovery)
//! - **Intrinsic Plasticity**: Adjust neuron excitability to maintain target rates
//!
//! # Time Scales
//! - Fast metaplasticity: 10s-1h
//! - Slow synaptic scaling: 24-48h
//! - Criticality monitoring: continuous
//! - Target firing rate: 1-10Hz

use serde::{Deserialize, Serialize};

/// BCM (Bienenstock-Cooper-Munro) metaplasticity
///
/// Sliding threshold separating LTP and LTD based on recent activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BCMMetaplasticity {
    /// Sliding threshold θ_m
    pub threshold: f32,

    /// Target for threshold (based on <y²>)
    target_activity_squared: f32,

    /// Time constant for threshold update (ms)
    pub tau_threshold: f32,

    /// Activity history for computing <y²>
    activity_history: Vec<f32>,

    /// History window size
    history_size: usize,
}

impl BCMMetaplasticity {
    pub fn new(tau_threshold: f32, history_size: usize) -> Self {
        Self {
            threshold: 1.0,
            target_activity_squared: 0.0,
            tau_threshold,
            activity_history: Vec::new(),
            history_size,
        }
    }

    /// Update threshold based on recent activity
    pub fn update(&mut self, dt: f32, current_activity: f32) {
        // Store activity
        self.activity_history.push(current_activity);
        if self.activity_history.len() > self.history_size {
            self.activity_history.remove(0);
        }

        // Compute <y²>
        if !self.activity_history.is_empty() {
            self.target_activity_squared = self.activity_history
                .iter()
                .map(|&y| y * y)
                .sum::<f32>() / self.activity_history.len() as f32;

            // Slide threshold toward <y²>
            let decay = (-dt / self.tau_threshold).exp();
            self.threshold = self.threshold * decay
                + self.target_activity_squared * (1.0 - decay);
        }
    }

    /// Determine plasticity rule based on BCM
    ///
    /// post < θ_m: LTD
    /// post > θ_m: LTP
    pub fn plasticity_direction(&self, pre_activity: f32, post_activity: f32) -> f32 {
        let post_squared = post_activity * post_activity;

        // φ(y) = y(y - θ_m)
        let phi = post_activity * (post_squared - self.threshold);

        // Δw = η * x * φ(y)
        pre_activity * phi
    }

    /// Get current threshold
    pub fn get_threshold(&self) -> f32 {
        self.threshold
    }
}

/// Synaptic scaling for global homeostasis
///
/// Multiplicatively scales all weights to maintain target firing rate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticScaling {
    /// Target firing rate (Hz)
    pub target_rate: f32,

    /// Time constant for scaling (ms, typically 24-48h)
    pub tau_scaling: f32,

    /// Current scaling factor
    pub scaling_factor: f32,

    /// Recent firing rates
    rate_history: Vec<f32>,

    /// History window
    history_size: usize,
}

impl SynapticScaling {
    pub fn new(target_rate: f32) -> Self {
        Self {
            target_rate,
            tau_scaling: 172800000.0,  // 48 hours in ms
            scaling_factor: 1.0,
            rate_history: Vec::new(),
            history_size: 1000,
        }
    }

    /// Update scaling factor based on recent activity
    pub fn update(&mut self, dt: f32, current_rate: f32) {
        self.rate_history.push(current_rate);
        if self.rate_history.len() > self.history_size {
            self.rate_history.remove(0);
        }

        if !self.rate_history.is_empty() {
            let avg_rate = self.rate_history.iter().sum::<f32>()
                / self.rate_history.len() as f32;

            // Multiplicative scaling: w_new = w_old * (target / actual)
            let target_scaling = self.target_rate / avg_rate.max(0.1);

            // Slow update
            let decay = (-dt / self.tau_scaling).exp();
            self.scaling_factor = self.scaling_factor * decay
                + target_scaling * (1.0 - decay);

            // Clamp to reasonable range
            self.scaling_factor = self.scaling_factor.clamp(0.1, 10.0);
        }
    }

    /// Apply scaling to weight
    pub fn scale_weight(&self, weight: f32) -> f32 {
        weight * self.scaling_factor
    }

    /// Apply scaling to all weights (preserves relative strengths)
    pub fn scale_weights(&self, weights: &mut [f32]) {
        for w in weights {
            *w *= self.scaling_factor;
        }
    }
}

/// Criticality homeostasis (2024 discovery)
///
/// Maintains network at edge of phase transition (optimal computation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalityHomeostasis {
    /// Branching ratio (activity propagation)
    pub branching_ratio: f32,

    /// Target branching ratio (1.0 = criticality)
    pub target_branching: f32,

    /// Avalanche size distribution
    avalanche_sizes: Vec<usize>,

    /// Deviation from criticality
    pub criticality_deviation: f32,

    /// Adjustment rate
    pub adjustment_rate: f32,
}

impl CriticalityHomeostasis {
    pub fn new() -> Self {
        Self {
            branching_ratio: 1.0,
            target_branching: 1.0,
            avalanche_sizes: Vec::new(),
            criticality_deviation: 0.0,
            adjustment_rate: 0.001,
        }
    }

    /// Record neural avalanche
    pub fn record_avalanche(&mut self, size: usize) {
        self.avalanche_sizes.push(size);
        if self.avalanche_sizes.len() > 1000 {
            self.avalanche_sizes.remove(0);
        }
    }

    /// Update criticality metrics
    pub fn update(&mut self) {
        if self.avalanche_sizes.len() < 100 {
            return;
        }

        // Compute branching ratio (activity_t+1 / activity_t)
        let mut ratios = Vec::new();
        for window in self.avalanche_sizes.windows(2) {
            if window[0] > 0 {
                ratios.push(window[1] as f32 / window[0] as f32);
            }
        }

        if !ratios.is_empty() {
            self.branching_ratio = ratios.iter().sum::<f32>() / ratios.len() as f32;
        }

        // Measure deviation from criticality
        self.criticality_deviation = (self.branching_ratio - self.target_branching).abs();
    }

    /// Adjust excitability to move toward criticality
    ///
    /// Returns adjustment to synaptic strength
    pub fn adjust_toward_criticality(&self) -> f32 {
        if self.branching_ratio < self.target_branching {
            // Too subcritical → increase excitability
            self.adjustment_rate
        } else if self.branching_ratio > self.target_branching {
            // Too supercritical → decrease excitability
            -self.adjustment_rate
        } else {
            0.0
        }
    }

    /// Check if network is at criticality
    pub fn is_critical(&self) -> bool {
        self.criticality_deviation < 0.05
    }

    /// Get criticality score (0-1, 1 = optimal)
    pub fn criticality_score(&self) -> f32 {
        (1.0 - self.criticality_deviation.min(1.0)).max(0.0)
    }
}

impl Default for CriticalityHomeostasis {
    fn default() -> Self {
        Self::new()
    }
}

/// Intrinsic plasticity - adjust neuron excitability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrinsicPlasticity {
    /// Target firing rate (Hz)
    pub target_rate: f32,

    /// Current threshold
    pub threshold: f32,

    /// Threshold adjustment rate
    pub eta: f32,

    /// Recent firing rates
    rate_buffer: Vec<f32>,

    /// Buffer size
    buffer_size: usize,
}

impl IntrinsicPlasticity {
    pub fn new(target_rate: f32, initial_threshold: f32) -> Self {
        Self {
            target_rate,
            threshold: initial_threshold,
            eta: 0.01,
            rate_buffer: Vec::new(),
            buffer_size: 100,
        }
    }

    /// Update threshold based on recent firing rate
    pub fn update(&mut self, current_rate: f32) {
        self.rate_buffer.push(current_rate);
        if self.rate_buffer.len() > self.buffer_size {
            self.rate_buffer.remove(0);
        }

        let avg_rate = self.rate_buffer.iter().sum::<f32>()
            / self.rate_buffer.len().max(1) as f32;

        // Adjust threshold to maintain target rate
        // High firing → increase threshold (reduce excitability)
        // Low firing → decrease threshold (increase excitability)
        let delta_theta = self.eta * (avg_rate - self.target_rate);
        self.threshold += delta_theta;

        // Keep threshold in reasonable range
        self.threshold = self.threshold.clamp(-60.0, -40.0);
    }

    /// Get adjusted threshold
    pub fn get_threshold(&self) -> f32 {
        self.threshold
    }
}

/// Integrated homeostatic system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeostaticSystem {
    /// BCM metaplasticity
    pub bcm: BCMMetaplasticity,

    /// Synaptic scaling
    pub scaling: SynapticScaling,

    /// Criticality homeostasis
    pub criticality: CriticalityHomeostasis,

    /// Intrinsic plasticity
    pub intrinsic: IntrinsicPlasticity,

    /// Global time (ms)
    time: f32,
}

impl HomeostaticSystem {
    pub fn new(target_rate: f32, initial_threshold: f32) -> Self {
        Self {
            bcm: BCMMetaplasticity::new(10000.0, 1000),  // 10s window
            scaling: SynapticScaling::new(target_rate),
            criticality: CriticalityHomeostasis::new(),
            intrinsic: IntrinsicPlasticity::new(target_rate, initial_threshold),
            time: 0.0,
        }
    }

    /// Update all homeostatic mechanisms
    pub fn update(
        &mut self,
        dt: f32,
        firing_rate: f32,
        activity: f32,
        avalanche_size: usize,
    ) {
        self.time += dt;

        // Update BCM threshold (fast, 10s-1h)
        self.bcm.update(dt, activity);

        // Update synaptic scaling (slow, 24-48h)
        self.scaling.update(dt, firing_rate);

        // Update criticality (continuous)
        self.criticality.record_avalanche(avalanche_size);
        if (self.time as usize) % 1000 == 0 {  // Every second
            self.criticality.update();
        }

        // Update intrinsic plasticity (intermediate, ~hours)
        if (self.time as usize) % 100 == 0 {  // Every 100ms
            self.intrinsic.update(firing_rate);
        }
    }

    /// Get modulated learning rate
    pub fn modulated_learning_rate(
        &self,
        base_lr: f32,
        pre_activity: f32,
        post_activity: f32,
    ) -> f32 {
        // BCM determines sign and magnitude
        let bcm_modulation = self.bcm.plasticity_direction(pre_activity, post_activity);

        // Criticality adjusts overall strength
        let criticality_adjustment = 1.0 + self.criticality.adjust_toward_criticality();

        base_lr * bcm_modulation * criticality_adjustment
    }

    /// Apply all homeostatic adjustments to weights
    pub fn apply_homeostasis(&self, weights: &mut [f32]) {
        // Synaptic scaling
        self.scaling.scale_weights(weights);

        // Criticality adjustment
        let crit_adj = 1.0 + self.criticality.adjust_toward_criticality();
        for w in weights {
            *w *= crit_adj;
        }
    }

    /// Get comprehensive stats
    pub fn stats(&self) -> HomeostaticStats {
        HomeostaticStats {
            bcm_threshold: self.bcm.get_threshold(),
            scaling_factor: self.scaling.scaling_factor,
            branching_ratio: self.criticality.branching_ratio,
            criticality_score: self.criticality.criticality_score(),
            intrinsic_threshold: self.intrinsic.get_threshold(),
            is_critical: self.criticality.is_critical(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HomeostaticStats {
    pub bcm_threshold: f32,
    pub scaling_factor: f32,
    pub branching_ratio: f32,
    pub criticality_score: f32,
    pub intrinsic_threshold: f32,
    pub is_critical: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bcm() {
        let mut bcm = BCMMetaplasticity::new(1000.0, 100);

        // Low activity → LTD for moderate post
        bcm.update(10.0, 0.2);
        let delta1 = bcm.plasticity_direction(1.0, 0.5);

        // High activity → shift threshold → change rule
        for _ in 0..100 {
            bcm.update(10.0, 2.0);
        }
        let delta2 = bcm.plasticity_direction(1.0, 0.5);

        // Direction should change with threshold
        assert_ne!(delta1.signum(), delta2.signum());
    }

    #[test]
    fn test_synaptic_scaling() {
        let mut scaling = SynapticScaling::new(5.0);

        // Low firing → scale up
        for _ in 0..100 {
            scaling.update(1000.0, 1.0);  // 1Hz << 5Hz target
        }

        assert!(scaling.scaling_factor > 1.0);

        // High firing → scale down
        let mut scaling2 = SynapticScaling::new(5.0);
        for _ in 0..100 {
            scaling2.update(1000.0, 20.0);  // 20Hz >> 5Hz target
        }

        assert!(scaling2.scaling_factor < 1.0);
    }

    #[test]
    fn test_criticality() {
        let mut crit = CriticalityHomeostasis::new();

        // Subcritical pattern
        for i in 10..1 {
            crit.record_avalanche(i);
        }
        crit.update();

        assert!(crit.branching_ratio < 1.0);
        assert!(crit.adjust_toward_criticality() > 0.0);  // Should increase
    }

    #[test]
    fn test_intrinsic_plasticity() {
        let mut ip = IntrinsicPlasticity::new(5.0, -55.0);

        let initial_threshold = ip.threshold;

        // High firing → increase threshold
        for _ in 0..50 {
            ip.update(15.0);
        }

        assert!(ip.threshold > initial_threshold);
    }

    #[test]
    fn test_homeostatic_system() {
        let mut system = HomeostaticSystem::new(5.0, -55.0);

        // Update over time
        for i in 0..100 {
            system.update(10.0, 3.0, 0.5, i % 10);
        }

        let stats = system.stats();
        assert!(stats.bcm_threshold > 0.0);
    }
}
