//! Hierarchical multi-scale brain architecture
//!
//! 3-level abstraction: Detailed (8.6M) + Medium (1B→10K) + Abstract (85B→75K)

use super::LIFNeuron;
use serde::{Deserialize, Serialize};

/// Neuron detail level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuronLevel {
    /// Full LIF simulation (8 bytes/neuron)
    Detailed(LIFNeuron),

    /// Regional group average (~1KB/region)
    Medium(RegionGroup),

    /// Mean-field approximation (~100 bytes/region)
    Abstract(MeanFieldRegion),
}

/// Regional group (Medium level)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionGroup {
    /// Region ID
    pub id: u32,

    /// Number of neurons represented
    pub n_neurons: usize,

    /// Average membrane potential
    pub avg_v: f32,

    /// Average firing rate (Hz)
    pub avg_rate: f32,

    /// Regional activity level (0-1)
    pub activity: f32,
}

impl RegionGroup {
    pub fn new(id: u32, n_neurons: usize) -> Self {
        Self {
            id,
            n_neurons,
            avg_v: -70.0,
            avg_rate: 5.0,
            activity: 0.0,
        }
    }

    /// Update from detailed neurons
    pub fn update_from_neurons(&mut self, neurons: &[LIFNeuron]) {
        let sum_v: f32 = neurons.iter().map(|n| n.state.v).sum();
        self.avg_v = sum_v / neurons.len() as f32;
    }
}

/// Mean-field region (Abstract level)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanFieldRegion {
    /// Region ID
    pub id: u32,

    /// Population size
    pub population: usize,

    /// Mean firing rate
    pub mean_rate: f32,

    /// Population variance
    pub variance: f32,
}

impl MeanFieldRegion {
    pub fn new(id: u32, population: usize) -> Self {
        Self {
            id,
            population,
            mean_rate: 5.0,
            variance: 1.0,
        }
    }
}

/// Hierarchical brain with 3 levels
pub struct HierarchicalBrain {
    /// Detailed neurons (8.6M @ 20 bytes = 172 MB)
    pub detailed_neurons: Vec<LIFNeuron>,

    /// Medium regions (10K @ 1KB = 10 MB)
    pub medium_regions: Vec<RegionGroup>,

    /// Abstract regions (75K @ 100 bytes = 7.5 MB)
    pub abstract_regions: Vec<MeanFieldRegion>,

    /// Total represented neurons
    pub total_neurons: usize,
}

impl HierarchicalBrain {
    /// Create hierarchical brain for 86B neurons
    pub fn for_full_brain() -> Self {
        const DETAILED: usize = 8_600_000; // 0.01% at full detail
        const MEDIUM_REGIONS: usize = 10_000; // 1% represented
        const ABSTRACT_REGIONS: usize = 75_000; // 98.99% abstracted

        log::info!("Creating hierarchical brain architecture:");
        log::info!("  Detailed: {} neurons", DETAILED);
        log::info!("  Medium: {} regions (~1B neurons)", MEDIUM_REGIONS);
        log::info!("  Abstract: {} regions (~85B neurons)", ABSTRACT_REGIONS);

        let detailed_neurons = (0..DETAILED)
            .map(|i| LIFNeuron::new(i as u32))
            .collect();

        let medium_regions = (0..MEDIUM_REGIONS)
            .map(|i| RegionGroup::new(i as u32, 100_000))
            .collect();

        let abstract_regions = (0..ABSTRACT_REGIONS)
            .map(|i| MeanFieldRegion::new(i as u32, 1_000_000))
            .collect();

        Self {
            detailed_neurons,
            medium_regions,
            abstract_regions,
            total_neurons: 86_000_000_000,
        }
    }

    /// Memory footprint
    pub fn memory_footprint(&self) -> usize {
        let detailed = self.detailed_neurons.len() * 20; // 20 bytes per LIF
        let medium = self.medium_regions.len() * 1024; // 1KB per region
        let abstract_mem = self.abstract_regions.len() * 100; // 100 bytes per region
        detailed + medium + abstract_mem
    }

    /// Update medium regions from detailed neurons (aggregation)
    pub fn aggregate_detailed_to_medium(&mut self) {
        // Map detailed neurons to medium regions (assuming 860 neurons per region)
        let neurons_per_region = 860;

        for (region_idx, region) in self.medium_regions.iter_mut().enumerate() {
            let start_idx = region_idx * neurons_per_region;
            let end_idx = (start_idx + neurons_per_region).min(self.detailed_neurons.len());

            if start_idx < self.detailed_neurons.len() {
                let region_neurons = &self.detailed_neurons[start_idx..end_idx];
                region.update_from_neurons(region_neurons);
            }
        }
    }

    /// Update abstract regions from medium regions (further aggregation)
    pub fn aggregate_medium_to_abstract(&mut self) {
        // Map medium regions to abstract regions (assuming ~0.13 medium per abstract)
        let _regions_per_abstract = 1; // Simplified: 1:1 mapping for now

        for (abs_idx, abs_region) in self.abstract_regions.iter_mut().enumerate() {
            if abs_idx < self.medium_regions.len() {
                let medium = &self.medium_regions[abs_idx];
                abs_region.mean_rate = medium.avg_rate;
                abs_region.variance = (medium.activity * 10.0).powi(2); // Simplified variance estimate
            }
        }
    }

    /// Broadcast abstract-level signals to medium level (top-down modulation)
    pub fn broadcast_abstract_to_medium(&mut self) {
        // Top-down attention/neuromodulation signals
        for (med_idx, medium) in self.medium_regions.iter_mut().enumerate() {
            if med_idx < self.abstract_regions.len() {
                let abstract_rate = self.abstract_regions[med_idx].mean_rate;
                // Modulate medium region activity based on abstract-level context
                medium.activity = (medium.activity * 0.9) + (abstract_rate / 50.0);
            }
        }
    }

    /// Full hierarchical update cycle
    pub fn update_hierarchy(&mut self) {
        // Bottom-up aggregation
        self.aggregate_detailed_to_medium();
        self.aggregate_medium_to_abstract();

        // Top-down modulation
        self.broadcast_abstract_to_medium();
    }
}

/// Streaming buffer for SSD-based region swapping
pub struct StreamingBuffer {
    /// Currently loaded region IDs
    loaded_regions: Vec<usize>,

    /// Maximum loaded regions
    max_loaded: usize,

    /// LRU cache tracking
    access_counts: Vec<u32>,
}

impl StreamingBuffer {
    /// Create new streaming buffer
    pub fn new(max_loaded: usize, total_regions: usize) -> Self {
        Self {
            loaded_regions: Vec::with_capacity(max_loaded),
            max_loaded,
            access_counts: vec![0; total_regions],
        }
    }

    /// Request region (may trigger SSD load)
    pub fn request_region(&mut self, region_id: usize) -> bool {
        self.access_counts[region_id] += 1;

        if self.loaded_regions.contains(&region_id) {
            return true; // Already loaded
        }

        if self.loaded_regions.len() < self.max_loaded {
            self.loaded_regions.push(region_id);
            return true;
        }

        // Evict LRU region
        let lru_idx = self.find_lru_region();
        if let Some(idx) = lru_idx {
            self.loaded_regions[idx] = region_id;
            log::info!("Swapped region {} from SSD", region_id);
            return true;
        }

        false
    }

    /// Find least recently used region
    fn find_lru_region(&self) -> Option<usize> {
        let mut min_access = u32::MAX;
        let mut min_idx = None;

        for (idx, &region_id) in self.loaded_regions.iter().enumerate() {
            if self.access_counts[region_id] < min_access {
                min_access = self.access_counts[region_id];
                min_idx = Some(idx);
            }
        }

        min_idx
    }

    /// Reset access counts (periodic decay)
    pub fn decay_access_counts(&mut self) {
        for count in &mut self.access_counts {
            *count = count.saturating_sub(*count / 2);
        }
    }
}
