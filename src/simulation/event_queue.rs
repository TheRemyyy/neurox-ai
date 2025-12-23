//! Event-driven spike processing for sparse neural networks
//!
//! Exploits biological sparsity (~1% active neurons per timestep)

use serde::{Deserialize, Serialize};

/// Spike event with timing
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct SpikeEvent {
    /// Neuron ID that spiked
    pub neuron_id: u32,

    /// Timestep when spike occurred
    pub timestamp: u16,

    /// Region ID (for hierarchical networks)
    pub region_id: u8,

    /// Padding for alignment
    _padding: u8,
}

impl SpikeEvent {
    pub fn new(neuron_id: u32, timestamp: u16, region_id: u8) -> Self {
        Self {
            neuron_id,
            timestamp,
            region_id,
            _padding: 0,
        }
    }
}

/// Circular buffer for spike events
pub struct EventQueue {
    /// Event buffer (circular)
    events: Vec<SpikeEvent>,

    /// Maximum capacity
    capacity: usize,

    /// Current head position
    head: usize,

    /// Current tail position
    tail: usize,

    /// Number of events in queue
    count: usize,
}

impl EventQueue {
    /// Create new event queue with capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            events: vec![SpikeEvent::new(0, 0, 0); capacity],
            capacity,
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    /// Push event to queue
    pub fn push(&mut self, event: SpikeEvent) -> bool {
        if self.count >= self.capacity {
            return false; // Queue full
        }

        self.events[self.tail] = event;
        self.tail = (self.tail + 1) % self.capacity;
        self.count += 1;
        true
    }

    /// Pop event from queue
    pub fn pop(&mut self) -> Option<SpikeEvent> {
        if self.count == 0 {
            return None;
        }

        let event = self.events[self.head];
        self.head = (self.head + 1) % self.capacity;
        self.count -= 1;
        Some(event)
    }

    /// Get number of events
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clear queue
    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.count = 0;
    }
}

/// Ring buffer for synaptic delays
pub struct DelayBuffer {
    /// Circular buffer of spike times
    buffer: Vec<Vec<u32>>,

    /// Maximum delay (timesteps)
    max_delay: usize,

    /// Current position
    head: usize,
}

impl DelayBuffer {
    /// Create new delay buffer
    pub fn new(max_delay: usize) -> Self {
        Self {
            buffer: vec![Vec::new(); max_delay],
            max_delay,
            head: 0,
        }
    }

    /// Add spike with delay
    pub fn add_spike(&mut self, neuron_id: u32, delay: usize) {
        let idx = (self.head + delay) % self.max_delay;
        self.buffer[idx].push(neuron_id);
    }

    /// Get spikes for current timestep
    pub fn get_current_spikes(&mut self) -> &[u32] {
        &self.buffer[self.head]
    }

    /// Advance to next timestep
    pub fn advance(&mut self) {
        self.buffer[self.head].clear();
        self.head = (self.head + 1) % self.max_delay;
    }
}
