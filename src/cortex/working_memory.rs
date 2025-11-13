//! Working Memory Module
//!
//! Biologically-inspired working memory based on persistent neural activity.
//! Implements Miller's Law (7±2 items) and attention-gated storage.
//!
//! # Biological Basis
//! - Persistent activity in prefrontal cortex (PFC)
//! - Self-recurrent excitation maintains patterns
//! - Attention gates what gets stored
//! - Pattern completion for associative recall
//! - Capacity limited by interference between patterns

use crate::neuron::NeuronState;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Circular buffer for temporal context (Miller's 7±2 items)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
}

impl<T> CircularBuffer<T> {
    /// Create new circular buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push item, removing oldest if at capacity
    pub fn push(&mut self, item: T) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    /// Get item at index (0 = oldest)
    pub fn get(&self, index: usize) -> Option<&T> {
        self.buffer.get(index)
    }

    /// Get most recent item
    pub fn latest(&self) -> Option<&T> {
        self.buffer.back()
    }

    /// Get all items (oldest to newest)
    pub fn all(&self) -> Vec<&T> {
        self.buffer.iter().collect()
    }

    /// Number of items currently stored
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Persistent neuron with slow decay and self-recurrence
///
/// Maintains activity for 500-1000ms after input ceases,
/// enabling temporal binding and working memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentNeuron {
    /// Current neuron state
    pub state: NeuronState,

    /// Self-recurrent weight (creates bistability)
    pub recurrent_weight: f32,

    /// Time constant for persistent activity (500-1000ms)
    pub tau_persistent: f32,

    /// Activity trace (decays slowly)
    activity_trace: f32,

    /// Last spike time (ms)
    last_spike: f32,
}

impl PersistentNeuron {
    /// Create new persistent neuron
    pub fn new(recurrent_weight: f32, tau_persistent: f32) -> Self {
        Self {
            state: NeuronState::new(0),
            recurrent_weight,
            tau_persistent,
            activity_trace: 0.0,
            last_spike: 0.0,
        }
    }

    /// Update persistent activity
    ///
    /// Implements slow decay and self-recurrent excitation:
    /// dA/dt = -A/τ_persistent + w_recurrent * spike
    pub fn update(&mut self, dt: f32, current_time: f32) {
        // Decay trace
        let decay = (-dt / self.tau_persistent).exp();
        self.activity_trace *= decay;

        // Add self-recurrent excitation if recently spiked
        if current_time - self.last_spike < 50.0 {
            // Within 50ms window
            self.activity_trace += self.recurrent_weight;
        }

        // Clamp to [0, 1]
        self.activity_trace = self.activity_trace.clamp(0.0, 1.0);
    }

    /// Record spike
    pub fn spike(&mut self, time: f32) {
        self.last_spike = time;
        self.activity_trace += 0.1; // Boost on spike
    }

    /// Get current activity level
    pub fn activity(&self) -> f32 {
        self.activity_trace
    }

    /// Check if neuron is maintaining activity
    pub fn is_active(&self, threshold: f32) -> bool {
        self.activity_trace > threshold
    }
}

/// Working Memory Module
///
/// Implements attention-gated storage with limited capacity.
/// Stores neural patterns that can be retrieved associatively.
///
/// # Capacity
/// Miller's Law: 7±2 items (default: 7)
///
/// # Biology
/// - PFC persistent activity
/// - Attention-gated encoding
/// - Pattern completion retrieval
/// - Interference limits capacity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemory {
    /// Persistent neurons maintaining patterns
    persistent_neurons: Vec<PersistentNeuron>,

    /// Attention gates (one per pattern slot)
    attention_gates: Vec<f32>,

    /// Context buffer (circular, limited capacity)
    context_buffer: CircularBuffer<Vec<f32>>,

    /// Storage capacity (default: 7)
    capacity: usize,

    /// Pattern dimensionality
    pattern_dim: usize,

    /// Attention threshold for storage (0.0-1.0)
    attention_threshold: f32,

    /// Current simulation time (ms)
    time: f32,

    /// Retrieval similarity threshold
    retrieval_threshold: f32,
}

impl WorkingMemory {
    /// Create new working memory with specified capacity and pattern dimension
    ///
    /// # Arguments
    /// - `capacity`: Number of patterns to store (default: 7, Miller's Law)
    /// - `pattern_dim`: Dimensionality of each pattern
    /// - `attention_threshold`: Minimum attention for storage (default: 0.5)
    pub fn new(capacity: usize, pattern_dim: usize, attention_threshold: f32) -> Self {
        let mut persistent_neurons = Vec::with_capacity(capacity * pattern_dim);
        for _ in 0..(capacity * pattern_dim) {
            persistent_neurons.push(PersistentNeuron::new(
                0.8,   // Strong self-recurrence
                750.0, // 750ms persistent activity
            ));
        }

        Self {
            persistent_neurons,
            attention_gates: vec![0.0; capacity],
            context_buffer: CircularBuffer::new(capacity),
            capacity,
            pattern_dim,
            attention_threshold,
            time: 0.0,
            retrieval_threshold: 0.6,
        }
    }

    /// Store pattern if attention is high enough
    ///
    /// Only stores when `attention >= attention_threshold`.
    /// Implements attention-gated encoding (biology: thalamic gating).
    ///
    /// # Returns
    /// `true` if stored, `false` if rejected due to low attention
    pub fn store(&mut self, pattern: &[f32], attention: f32) -> bool {
        assert_eq!(pattern.len(), self.pattern_dim, "Pattern dimension mismatch");

        // Reject if attention too low
        if attention < self.attention_threshold {
            return false;
        }

        // Store in context buffer
        self.context_buffer.push(pattern.to_vec());

        // Activate corresponding persistent neurons
        let slot_idx = (self.context_buffer.len() - 1) % self.capacity;
        self.attention_gates[slot_idx] = attention;

        // Set persistent neuron activity
        let start_idx = slot_idx * self.pattern_dim;
        for (i, &value) in pattern.iter().enumerate() {
            let neuron = &mut self.persistent_neurons[start_idx + i];
            neuron.activity_trace = value * attention; // Modulated by attention
            if value > 0.5 {
                neuron.spike(self.time);
            }
        }

        true
    }

    /// Retrieve pattern via associative recall
    ///
    /// Uses cosine similarity for pattern matching.
    /// Returns pattern with highest similarity > threshold.
    ///
    /// # Pattern Completion
    /// Can retrieve full pattern from partial cue.
    pub fn retrieve(&self, query: &[f32]) -> Option<Vec<f32>> {
        assert_eq!(query.len(), self.pattern_dim, "Query dimension mismatch");

        let mut best_match: Option<(Vec<f32>, f32)> = None;

        // Search through stored patterns
        for (slot_idx, stored_pattern) in self.context_buffer.all().iter().enumerate() {
            let similarity = Self::cosine_similarity(query, stored_pattern);

            if similarity > self.retrieval_threshold {
                if let Some((_, best_sim)) = &best_match {
                    if similarity > *best_sim {
                        best_match = Some(((*stored_pattern).clone(), similarity));
                    }
                } else {
                    best_match = Some(((*stored_pattern).clone(), similarity));
                }
            }
        }

        best_match.map(|(pattern, _)| pattern)
    }

    /// Maintain persistent activity
    ///
    /// Updates all persistent neurons, allowing them to maintain
    /// patterns through recurrent excitation.
    pub fn maintain(&mut self, dt: f32) {
        self.time += dt;

        // Update all persistent neurons
        for neuron in &mut self.persistent_neurons {
            neuron.update(dt, self.time);
        }

        // Attention gates decay very slowly (items persist for ~10 seconds)
        // Only decay gates that are below threshold to eventually remove weak items
        for (i, gate) in self.attention_gates.iter_mut().enumerate() {
            if i < self.context_buffer.len() {
                // Items in buffer: very slow decay (99.9% retention per step)
                *gate *= 0.999;
            } else {
                // Empty slots: fast decay
                *gate *= 0.9;
            }
        }
    }

    /// Clear working memory
    pub fn clear(&mut self) {
        self.context_buffer.clear();
        self.attention_gates.fill(0.0);
        for neuron in &mut self.persistent_neurons {
            neuron.activity_trace = 0.0;
        }
    }

    /// Get current capacity utilization (0.0-1.0)
    pub fn utilization(&self) -> f32 {
        self.context_buffer.len() as f32 / self.capacity as f32
    }

    /// Get number of actively maintained patterns
    pub fn active_count(&self) -> usize {
        self.attention_gates.iter().filter(|&&g| g > 0.1).count()
    }

    /// Get all currently stored patterns
    pub fn get_all_patterns(&self) -> Vec<&Vec<f32>> {
        self.context_buffer.all()
    }

    /// Get pattern at specific slot
    pub fn get_pattern(&self, slot: usize) -> Option<&Vec<f32>> {
        self.context_buffer.get(slot)
    }

    /// Get attention level for slot
    pub fn get_attention(&self, slot: usize) -> f32 {
        *self.attention_gates.get(slot).unwrap_or(&0.0)
    }

    /// Compute cosine similarity between two patterns
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Get memory statistics
    pub fn stats(&self) -> WorkingMemoryStats {
        let active_neurons = self
            .persistent_neurons
            .iter()
            .filter(|n| n.is_active(0.1))
            .count();

        let avg_activity: f32 = self
            .persistent_neurons
            .iter()
            .map(|n| n.activity())
            .sum::<f32>()
            / self.persistent_neurons.len() as f32;

        WorkingMemoryStats {
            stored_patterns: self.context_buffer.len(),
            capacity: self.capacity,
            utilization: self.utilization(),
            active_count: self.active_count(),
            active_neurons,
            avg_activity,
            avg_attention: self.attention_gates.iter().sum::<f32>()
                / self.attention_gates.len() as f32,
        }
    }
}

/// Working memory statistics
#[derive(Debug, Clone)]
pub struct WorkingMemoryStats {
    /// Number of stored patterns
    pub stored_patterns: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Capacity utilization (0.0-1.0)
    pub utilization: f32,
    /// Number of actively maintained patterns
    pub active_count: usize,
    /// Number of active persistent neurons
    pub active_neurons: usize,
    /// Average neuron activity
    pub avg_activity: f32,
    /// Average attention level
    pub avg_attention: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        assert_eq!(buffer.len(), 3);

        buffer.push(4); // Should evict 1
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.get(0), Some(&2));
        assert_eq!(buffer.latest(), Some(&4));
    }

    #[test]
    fn test_persistent_neuron() {
        let mut neuron = PersistentNeuron::new(0.8, 750.0);
        neuron.spike(0.0);
        assert!(neuron.activity() > 0.0);

        // Activity should persist
        neuron.update(100.0, 100.0);
        assert!(neuron.activity() > 0.0);

        // But eventually decay
        for _ in 0..100 {
            neuron.update(100.0, 0.0);
        }
        assert!(neuron.activity() < 0.1);
    }

    #[test]
    fn test_working_memory_store_retrieve() {
        let mut wm = WorkingMemory::new(7, 10, 0.5);

        // Store pattern with high attention
        let pattern = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        assert!(wm.store(&pattern, 0.8));

        // Retrieve with exact query
        let retrieved = wm.retrieve(&pattern);
        assert!(retrieved.is_some());

        // Retrieve with partial query (pattern completion)
        let partial = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5];
        let completed = wm.retrieve(&partial);
        assert!(completed.is_some());
    }

    #[test]
    fn test_attention_gating() {
        let mut wm = WorkingMemory::new(7, 10, 0.5);

        let pattern = vec![1.0; 10];

        // Low attention - should reject
        assert!(!wm.store(&pattern, 0.3));

        // High attention - should accept
        assert!(wm.store(&pattern, 0.8));

        assert_eq!(wm.active_count(), 1);
    }

    #[test]
    fn test_capacity_limit() {
        let mut wm = WorkingMemory::new(3, 5, 0.0); // Capacity 3

        for i in 0..5 {
            let pattern = vec![i as f32; 5];
            wm.store(&pattern, 1.0);
        }

        // Should only keep last 3
        assert_eq!(wm.get_all_patterns().len(), 3);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((WorkingMemory::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((WorkingMemory::cosine_similarity(&a, &c)).abs() < 0.001);
    }
}
