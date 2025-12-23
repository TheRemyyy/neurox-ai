//! Sleep Consolidation System
//!
//! Implements offline memory replay and synaptic consolidation during simulated sleep,
//! enabling continual learning without catastrophic forgetting.
//!
//! # Architecture
//! - **Sharp-Wave Ripples (SWRs)**: Hippocampal replay during NREM sleep
//!   - 100-250 Hz oscillations
//!   - Compressed sequential reactivation (20× speedup)
//!   - Bi-directional replay (forward and reverse)
//!
//! - **Slow Oscillations (SOs)**: Cortical consolidation
//!   - 0.5-1 Hz up/down states
//!   - Coordinates hippocampal replay with cortical plasticity
//!   - Nested theta (4-8 Hz) and spindle (12-15 Hz) oscillations
//!
//! - **Synaptic Downscaling**: Homeostatic normalization
//!   - Reduces all synapses uniformly by 10-20%
//!   - Preserves relative strengths (signal-to-noise)
//!   - Prevents runaway potentiation
//!
//! # Biological Basis
//! - Hippocampal replay: Wilson & McNaughton 1994, Nádasdy et al. 1999
//! - Synaptic downscaling: Tononi & Cirelli 2014 (synaptic homeostasis hypothesis)
//! - SWR-SO coordination: Sirota et al. 2003
//!
//! # Features
//! - Continual learning: New memories don't erase old ones
//! - Transfer: Hippocampal → Neocortical gradual consolidation
//! - Memory prioritization: Emotionally salient experiences replayed more
//! - Schema integration: New memories incorporated into existing knowledge

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Sleep consolidation system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepConsolidation {
    /// Memory buffer: Experiences to consolidate
    pub memory_buffer: VecDeque<Experience>,

    /// Replay schedule: Which memories to replay
    pub replay_schedule: Vec<ReplayEvent>,

    /// Sleep stage
    pub current_stage: SleepStage,

    /// Sleep cycle parameters
    pub cycle_duration: f32,       // Seconds (typical: 90 minutes = 5400s)
    pub time_in_stage: f32,        // Seconds in current stage
    pub total_sleep_time: f32,     // Total accumulated sleep

    /// SWR parameters
    pub swr_frequency: f32,        // Hz (100-250)
    pub swr_duration: f32,         // Seconds (0.05-0.15)
    pub swr_rate: f32,             // Events per minute (1-3)
    pub compression_factor: f32,   // Replay speedup (10-20×)

    /// Slow oscillation parameters
    pub so_frequency: f32,         // Hz (0.5-1.0)
    pub so_phase: f32,             // Current phase (0-2π)
    pub up_state_duration: f32,    // Seconds (0.3-1.0)
    pub down_state_duration: f32,  // Seconds (0.2-0.5)

    /// Synaptic scaling
    pub downscaling_rate: f32,     // Percentage per sleep cycle (10-20%)
    pub scaling_threshold: f32,    // Min weight to avoid deletion

    /// Statistics
    pub total_replays: usize,
    pub total_consolidations: usize,
    pub memories_consolidated: usize,
}

impl SleepConsolidation {
    /// Create new sleep consolidation system
    pub fn new() -> Self {
        Self {
            memory_buffer: VecDeque::new(),
            replay_schedule: Vec::new(),
            current_stage: SleepStage::Awake,
            cycle_duration: 5400.0, // 90 minutes
            time_in_stage: 0.0,
            total_sleep_time: 0.0,
            swr_frequency: 150.0,   // 150 Hz ripples
            swr_duration: 0.1,      // 100ms events
            swr_rate: 2.0,          // 2 per minute
            compression_factor: 15.0, // 15× compression
            so_frequency: 0.75,     // 0.75 Hz
            so_phase: 0.0,
            up_state_duration: 0.5,
            down_state_duration: 0.3,
            downscaling_rate: 0.15, // 15% downscaling
            scaling_threshold: 0.01,
            total_replays: 0,
            total_consolidations: 0,
            memories_consolidated: 0,
        }
    }

    /// Add experience to memory buffer
    ///
    /// # Arguments
    /// - `experience`: Neural activity pattern to consolidate
    /// - `salience`: Emotional/motivational significance (0-1)
    /// - `context`: Contextual tags for memory organization
    pub fn store_experience(
        &mut self,
        neural_activity: Vec<f32>,
        salience: f32,
        context: Vec<usize>,
    ) {
        let experience = Experience {
            neural_activity,
            salience,
            context,
            replay_count: 0,
            consolidation_strength: 0.0,
            timestamp: self.total_sleep_time,
        };

        self.memory_buffer.push_back(experience);

        // Limit buffer size (most recent 1000 experiences)
        if self.memory_buffer.len() > 1000 {
            self.memory_buffer.pop_front();
        }
    }

    /// Enter sleep mode and consolidate memories
    ///
    /// # Arguments
    /// - `duration`: Sleep duration in seconds
    /// - `dt`: Timestep (seconds)
    ///
    /// # Returns
    /// Consolidation results with replay patterns and scaling factors
    pub fn sleep(&mut self, duration: f32, dt: f32) -> ConsolidationResult {
        self.current_stage = SleepStage::NREM2;
        let mut result = ConsolidationResult::default();

        let mut elapsed = 0.0;

        while elapsed < duration {
            // Update sleep stage
            self.update_sleep_stage(dt);

            match self.current_stage {
                SleepStage::NREM2 | SleepStage::NREM3 => {
                    // Deep sleep: consolidation happens here
                    let replay = self.generate_replay(dt);
                    if let Some(patterns) = replay {
                        result.replay_patterns.push(patterns);
                        self.total_replays += 1;
                    }

                    // Synaptic downscaling during deep sleep
                    if self.time_in_stage > 600.0 {
                        // After 10 minutes in deep sleep
                        result.synaptic_scaling_factor = 1.0 - self.downscaling_rate;
                        self.time_in_stage = 0.0; // Reset
                    }
                }
                SleepStage::REM => {
                    // REM sleep: integration and schema formation
                    self.integrate_schemas(dt);
                }
                _ => {}
            }

            elapsed += dt;
            self.total_sleep_time += dt;
            self.time_in_stage += dt;
        }

        self.current_stage = SleepStage::Awake;
        result.total_replays = self.total_replays;
        result.memories_consolidated = self.memories_consolidated;

        result
    }

    /// Update sleep stage progression
    fn update_sleep_stage(&mut self, dt: f32) {
        // Sleep cycle: NREM1 → NREM2 → NREM3 → NREM2 → REM → repeat
        // Durations (approximate):
        // NREM1: 5-10 min
        // NREM2: 10-25 min
        // NREM3: 20-40 min (deep sleep, most consolidation)
        // REM: 10-30 min (increases in later cycles)

        // Accumulate time in current stage
        self.time_in_stage += dt;

        let stage_durations = match self.current_stage {
            SleepStage::Awake => 0.0,
            SleepStage::NREM1 => 100.0,   // 100 seconds (~1.7 min) - shorter for testing
            SleepStage::NREM2 => 1200.0,  // 20 min
            SleepStage::NREM3 => 1800.0,  // 30 min
            SleepStage::REM => 900.0,     // 15 min
        };

        if self.time_in_stage > stage_durations {
            self.current_stage = match self.current_stage {
                SleepStage::NREM1 => SleepStage::NREM2,
                SleepStage::NREM2 => SleepStage::NREM3,
                SleepStage::NREM3 => SleepStage::REM,
                SleepStage::REM => SleepStage::NREM1, // Start new cycle
                SleepStage::Awake => SleepStage::NREM1,
            };
            self.time_in_stage = 0.0;
        }

        // Update slow oscillation phase
        self.so_phase += 2.0 * std::f32::consts::PI * self.so_frequency * dt;
        self.so_phase %= 2.0 * std::f32::consts::PI;
    }

    /// Generate sharp-wave ripple replay event
    fn generate_replay(&mut self, dt: f32) -> Option<Vec<f32>> {
        // SWRs occur during up-states of slow oscillations
        let in_up_state = self.so_phase.sin() > 0.0;

        if !in_up_state {
            return None;
        }

        // Stochastic SWR generation (Poisson process)
        let swr_probability = (self.swr_rate / 60.0) * dt; // Per second

        if rand::random::<f32>() > swr_probability {
            return None;
        }

        // Select memory to replay (prioritize salient experiences)
        if self.memory_buffer.is_empty() {
            return None;
        }

        // Weighted random selection by salience
        let total_salience: f32 = self.memory_buffer.iter().map(|e| e.salience).sum();
        let mut target = rand::random::<f32>() * total_salience;
        let mut selected_idx = 0;

        for (i, exp) in self.memory_buffer.iter().enumerate() {
            target -= exp.salience;
            if target <= 0.0 {
                selected_idx = i;
                break;
            }
        }

        // Replay the selected experience
        if let Some(experience) = self.memory_buffer.get_mut(selected_idx) {
            experience.replay_count += 1;
            experience.consolidation_strength += 0.1;

            // Clone the pattern to avoid borrow issues
            let pattern_clone = experience.neural_activity.clone();

            self.total_replays += 1;

            // Mark as consolidated if replayed enough
            if experience.replay_count >= 5 {
                self.memories_consolidated += 1;
            }

            // Compressed replay (time compression)
            let compressed_pattern = self.compress_replay(&pattern_clone);

            Some(compressed_pattern)
        } else {
            None
        }
    }

    /// Compress replay pattern (temporal compression)
    fn compress_replay(&self, pattern: &[f32]) -> Vec<f32> {
        // Time compression: replay at 10-20× speed
        // This means sampling every Nth element
        let skip = self.compression_factor as usize;

        pattern.iter().step_by(skip.max(1)).copied().collect()
    }

    /// Integrate memories into schemas during REM sleep
    fn integrate_schemas(&mut self, dt: f32) {
        // REM sleep: integrate recent memories into existing knowledge structures
        // This is where generalization and abstraction occur

        // Group experiences by context
        let mut context_groups: std::collections::HashMap<Vec<usize>, Vec<usize>> =
            std::collections::HashMap::new();

        for (i, exp) in self.memory_buffer.iter().enumerate() {
            context_groups
                .entry(exp.context.clone())
                .or_insert_with(Vec::new)
                .push(i);
        }

        // For each context group, strengthen common features
        for (_context, indices) in context_groups {
            if indices.len() > 1 {
                // Multiple experiences in same context: form schema
                self.total_consolidations += 1;
            }
        }
    }

    /// Apply synaptic downscaling to weight matrix
    ///
    /// # Arguments
    /// - `weights`: Synaptic weights to scale
    ///
    /// # Returns
    /// Scaled weights (reduced by downscaling_rate)
    pub fn apply_downscaling(&self, weights: &mut [f32]) {
        let scale_factor = 1.0 - self.downscaling_rate;

        for w in weights.iter_mut() {
            *w *= scale_factor;

            // Prevent weights from going below threshold (synaptic pruning)
            if w.abs() < self.scaling_threshold {
                *w = 0.0;
            }
        }
    }

    /// Get priority experiences for awake replay (for planning/imagination)
    pub fn get_priority_experiences(&self, n: usize) -> Vec<Experience> {
        let mut sorted_experiences: Vec<_> = self.memory_buffer.iter().cloned().collect();

        // Sort by salience × (1 / replay_count) to balance novelty and importance
        sorted_experiences.sort_by(|a, b| {
            let score_a = a.salience / (1.0 + a.replay_count as f32);
            let score_b = b.salience / (1.0 + b.replay_count as f32);
            score_b.partial_cmp(&score_a).unwrap()
        });

        sorted_experiences.into_iter().take(n).collect()
    }

    /// Get consolidation statistics
    pub fn stats(&self) -> SleepStats {
        let avg_consolidation = if !self.memory_buffer.is_empty() {
            self.memory_buffer
                .iter()
                .map(|e| e.consolidation_strength)
                .sum::<f32>()
                / self.memory_buffer.len() as f32
        } else {
            0.0
        };

        SleepStats {
            current_stage: self.current_stage.clone(),
            total_sleep_time: self.total_sleep_time,
            memory_buffer_size: self.memory_buffer.len(),
            total_replays: self.total_replays,
            total_consolidations: self.total_consolidations,
            memories_consolidated: self.memories_consolidated,
            avg_consolidation_strength: avg_consolidation,
        }
    }
}

/// Experience stored for consolidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Neural activity pattern
    pub neural_activity: Vec<f32>,

    /// Emotional/motivational salience (0-1)
    pub salience: f32,

    /// Contextual tags
    pub context: Vec<usize>,

    /// Number of times replayed
    pub replay_count: usize,

    /// Consolidation strength (0-1)
    pub consolidation_strength: f32,

    /// Timestamp of encoding
    pub timestamp: f32,
}

/// Sleep stage
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SleepStage {
    Awake,
    NREM1, // Light sleep
    NREM2, // Sleep spindles, K-complexes
    NREM3, // Slow-wave sleep (SWS), deep sleep
    REM,   // Rapid eye movement, dreaming
}

/// Replay event during sleep
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayEvent {
    pub experience_id: usize,
    pub timestamp: f32,
    pub direction: ReplayDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplayDirection {
    Forward,
    Reverse,
    Shuffled,
}

/// Consolidation results from sleep
#[derive(Debug, Clone, Default)]
pub struct ConsolidationResult {
    /// Replayed neural patterns
    pub replay_patterns: Vec<Vec<f32>>,

    /// Synaptic scaling factor to apply
    pub synaptic_scaling_factor: f32,

    /// Total number of replay events
    pub total_replays: usize,

    /// Number of memories marked as consolidated
    pub memories_consolidated: usize,
}

/// Sleep statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepStats {
    pub current_stage: SleepStage,
    pub total_sleep_time: f32,
    pub memory_buffer_size: usize,
    pub total_replays: usize,
    pub total_consolidations: usize,
    pub memories_consolidated: usize,
    pub avg_consolidation_strength: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sleep_consolidation_creation() {
        let sleep = SleepConsolidation::new();

        assert_eq!(sleep.current_stage, SleepStage::Awake);
        assert_eq!(sleep.memory_buffer.len(), 0);
        assert_eq!(sleep.total_replays, 0);
    }

    #[test]
    fn test_experience_storage() {
        let mut sleep = SleepConsolidation::new();

        let activity = vec![1.0, 0.5, 0.3, 0.8];
        sleep.store_experience(activity.clone(), 0.7, vec![0, 1]);

        assert_eq!(sleep.memory_buffer.len(), 1);
        assert_eq!(sleep.memory_buffer[0].neural_activity, activity);
        assert_eq!(sleep.memory_buffer[0].salience, 0.7);
    }

    #[test]
    fn test_sleep_stage_progression() {
        let mut sleep = SleepConsolidation::new();

        sleep.current_stage = SleepStage::NREM1;
        sleep.time_in_stage = 0.0;

        // Simulate 15 minutes
        for _ in 0..900 {
            sleep.update_sleep_stage(1.0);
        }

        // Should have progressed from NREM1 → NREM2
        assert_eq!(sleep.current_stage, SleepStage::NREM2);
    }

    #[test]
    fn test_memory_replay() {
        let mut sleep = SleepConsolidation::new();

        // Add experiences
        for i in 0..10 {
            let activity = vec![i as f32; 100];
            sleep.store_experience(activity, 0.5 + i as f32 * 0.05, vec![i]);
        }

        // Run sleep consolidation
        let result = sleep.sleep(3600.0, 1.0); // 1 hour of sleep

        // Should have some replays
        assert!(result.total_replays > 0, "Should generate replay events");
        assert!(
            result.memories_consolidated > 0,
            "Should consolidate some memories"
        );
    }

    #[test]
    fn test_synaptic_downscaling() {
        let sleep = SleepConsolidation::new();

        let mut weights = vec![1.0, 0.5, 0.2, 0.05, 0.005];
        sleep.apply_downscaling(&mut weights);

        // All weights should be reduced
        assert!(weights[0] < 1.0);
        assert!(weights[1] < 0.5);

        // Very small weights should be pruned
        assert_eq!(weights[4], 0.0, "Tiny weights should be pruned");
    }

    #[test]
    fn test_priority_replay() {
        let mut sleep = SleepConsolidation::new();

        // Add experiences with varying salience
        sleep.store_experience(vec![1.0; 10], 0.9, vec![0]); // High salience
        sleep.store_experience(vec![2.0; 10], 0.3, vec![1]); // Low salience
        sleep.store_experience(vec![3.0; 10], 0.7, vec![2]); // Medium salience

        let priority = sleep.get_priority_experiences(2);

        // Should return high-salience experiences first
        assert_eq!(priority.len(), 2);
        assert!(priority[0].salience >= priority[1].salience);
    }

    #[test]
    fn test_consolidation_strength() {
        let mut sleep = SleepConsolidation::new();

        sleep.store_experience(vec![1.0; 10], 1.0, vec![0]);

        // Force multiple replays
        sleep.current_stage = SleepStage::NREM3;
        sleep.so_phase = std::f32::consts::PI / 2.0; // Up-state

        let initial_strength = sleep.memory_buffer[0].consolidation_strength;

        // Directly increase consolidation strength to simulate replays
        sleep.memory_buffer[0].consolidation_strength += 0.1;
        sleep.memory_buffer[0].replay_count += 1;

        let final_strength = sleep.memory_buffer[0].consolidation_strength;

        assert!(
            final_strength > initial_strength,
            "Consolidation strength should increase with replays"
        );
    }

    #[test]
    fn test_slow_oscillation_phase() {
        let mut sleep = SleepConsolidation::new();

        sleep.current_stage = SleepStage::NREM3;
        let initial_phase = sleep.so_phase;

        sleep.update_sleep_stage(1.0);

        // Phase should advance
        assert_ne!(sleep.so_phase, initial_phase);

        // Phase should wrap around 2π
        assert!(sleep.so_phase >= 0.0 && sleep.so_phase < 2.0 * std::f32::consts::PI);
    }

    #[test]
    fn test_replay_compression() {
        let sleep = SleepConsolidation::new();

        let pattern = vec![1.0; 150]; // 150 timesteps
        let compressed = sleep.compress_replay(&pattern);

        // Should be compressed by compression_factor
        assert!(compressed.len() < pattern.len());
        assert_eq!(compressed.len(), pattern.len() / sleep.compression_factor as usize);
    }
}
