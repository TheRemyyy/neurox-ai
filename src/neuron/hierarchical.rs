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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Create hierarchical brain with custom layer configuration
    ///
    /// # Arguments
    /// - `n_layers`: Number of hierarchical layers (typically 3-5)
    /// - `base_neurons`: Base number of neurons at detailed layer
    pub fn new(n_layers: usize, base_neurons: usize) -> Self {
        let detailed_count = base_neurons;
        let medium_count = (detailed_count as f32 * 0.1) as usize;
        let abstract_count = (detailed_count as f32 * 0.01) as usize;

        log::info!("Creating hierarchical brain ({} layers):", n_layers);
        log::info!("  Detailed: {} neurons", detailed_count);
        log::info!("  Medium: {} regions", medium_count);
        log::info!("  Abstract: {} regions", abstract_count);

        let detailed_neurons = (0..detailed_count)
            .map(|i| LIFNeuron::new(i as u32))
            .collect();

        let medium_regions = (0..medium_count)
            .map(|i| RegionGroup::new(i as u32, base_neurons / medium_count))
            .collect();

        let abstract_regions = (0..abstract_count)
            .map(|i| MeanFieldRegion::new(i as u32, base_neurons * 10))
            .collect();

        Self {
            detailed_neurons,
            medium_regions,
            abstract_regions,
            total_neurons: base_neurons * n_layers,
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_group_creation() {
        let region = RegionGroup::new(0, 1000);

        assert_eq!(region.id, 0);
        assert_eq!(region.n_neurons, 1000);
        assert_eq!(region.avg_v, -70.0);
        assert_eq!(region.avg_rate, 5.0);
    }

    #[test]
    fn test_region_group_update() {
        let mut region = RegionGroup::new(0, 3);

        let neurons = vec![
            LIFNeuron::new(0),
            LIFNeuron::new(1),
            LIFNeuron::new(2),
        ];

        region.update_from_neurons(&neurons);

        // Average should be around -70.0 (resting potential)
        assert!(region.avg_v < -60.0 && region.avg_v > -80.0);
    }

    #[test]
    fn test_mean_field_region_creation() {
        let region = MeanFieldRegion::new(5, 1000000);

        assert_eq!(region.id, 5);
        assert_eq!(region.population, 1000000);
        assert_eq!(region.mean_rate, 5.0);
        assert_eq!(region.variance, 1.0);
    }

    #[test]
    fn test_hierarchical_brain_creation() {
        let brain = HierarchicalBrain::new(3, 1000);

        assert_eq!(brain.detailed_neurons.len(), 1000);
        assert_eq!(brain.medium_regions.len(), 100); // 10% of detailed
        assert_eq!(brain.abstract_regions.len(), 10);  // 1% of detailed
        assert_eq!(brain.total_neurons, 3000); // 3 layers * 1000
    }

    #[test]
    fn test_full_brain_architecture() {
        let brain = HierarchicalBrain::for_full_brain();

        assert_eq!(brain.detailed_neurons.len(), 8_600_000);
        assert_eq!(brain.medium_regions.len(), 10_000);
        assert_eq!(brain.abstract_regions.len(), 75_000);
        assert_eq!(brain.total_neurons, 86_000_000_000);
    }

    #[test]
    fn test_memory_footprint() {
        let brain = HierarchicalBrain::new(3, 1000);
        let footprint = brain.memory_footprint();

        // Should be reasonable for 1000 detailed neurons
        // 1000 * 20 + 100 * 1024 + 10 * 100 = 20000 + 102400 + 1000 = 123400
        assert!(footprint > 100_000);
        assert!(footprint < 200_000);
    }

    #[test]
    fn test_aggregate_detailed_to_medium() {
        let mut brain = HierarchicalBrain::new(3, 8600);

        // First aggregate to see what the initial state is
        brain.aggregate_detailed_to_medium();

        // Set some specific values in detailed neurons AFTER first aggregation
        for neuron in &mut brain.detailed_neurons {
            neuron.state.v = -65.0;
        }

        // Now aggregate again
        brain.aggregate_detailed_to_medium();

        // Medium regions should reflect averaged detailed neurons
        // Check that at least some regions have updated values
        let updated_regions = brain.medium_regions.iter()
            .filter(|r| r.avg_v > -68.0 && r.avg_v < -62.0)
            .count();

        assert!(updated_regions > 0,
            "At least some regions should have updated values near -65.0");
    }

    #[test]
    fn test_aggregate_medium_to_abstract() {
        let mut brain = HierarchicalBrain::new(3, 1000);

        // Set medium region activity
        for region in &mut brain.medium_regions {
            region.avg_rate = 10.0;
            region.activity = 0.5;
        }

        brain.aggregate_medium_to_abstract();

        // Abstract regions should reflect medium regions
        for (i, region) in brain.abstract_regions.iter().enumerate() {
            if i < brain.medium_regions.len() {
                assert_eq!(region.mean_rate, 10.0);
            }
        }
    }

    #[test]
    fn test_broadcast_abstract_to_medium() {
        let mut brain = HierarchicalBrain::new(3, 1000);

        // Set abstract region rates
        for region in &mut brain.abstract_regions {
            region.mean_rate = 25.0;
        }

        brain.broadcast_abstract_to_medium();

        // Medium regions should be modulated
        for (i, region) in brain.medium_regions.iter().enumerate() {
            if i < brain.abstract_regions.len() {
                assert!(region.activity > 0.0);
            }
        }
    }

    #[test]
    fn test_hierarchical_update_cycle() {
        let mut brain = HierarchicalBrain::new(3, 1000);

        // Run full update cycle
        brain.update_hierarchy();

        // Should complete without panic
        assert!(brain.detailed_neurons.len() > 0);
    }

    #[test]
    fn test_streaming_buffer_creation() {
        let buffer = StreamingBuffer::new(100, 1000);

        assert_eq!(buffer.max_loaded, 100);
        assert_eq!(buffer.access_counts.len(), 1000);
        assert_eq!(buffer.loaded_regions.len(), 0);
    }

    #[test]
    fn test_streaming_buffer_request() {
        let mut buffer = StreamingBuffer::new(10, 100);

        // First request should load
        assert!(buffer.request_region(5));
        assert!(buffer.loaded_regions.contains(&5));
        assert_eq!(buffer.access_counts[5], 1);

        // Second request to same region should be cached
        assert!(buffer.request_region(5));
        assert_eq!(buffer.access_counts[5], 2);
    }

    #[test]
    fn test_streaming_buffer_eviction() {
        let mut buffer = StreamingBuffer::new(3, 100);

        // Load 3 regions
        buffer.request_region(0);
        buffer.request_region(1);
        buffer.request_region(2);

        assert_eq!(buffer.loaded_regions.len(), 3);

        // Request 4th region should trigger eviction
        buffer.request_region(3);

        assert_eq!(buffer.loaded_regions.len(), 3);
        assert!(buffer.loaded_regions.contains(&3));
    }

    #[test]
    fn test_streaming_buffer_lru() {
        let mut buffer = StreamingBuffer::new(2, 100);

        // Access pattern: 0, 1, 0 (0 accessed twice, 1 once)
        buffer.request_region(0);
        buffer.request_region(1);
        buffer.request_region(0);

        // Request new region should evict region 1 (LRU)
        buffer.request_region(2);

        assert!(buffer.loaded_regions.contains(&0));
        assert!(buffer.loaded_regions.contains(&2));
    }

    #[test]
    fn test_streaming_buffer_decay() {
        let mut buffer = StreamingBuffer::new(10, 100);

        buffer.request_region(5);
        buffer.request_region(5);
        buffer.request_region(5);

        assert_eq!(buffer.access_counts[5], 3);

        buffer.decay_access_counts();

        // Decay uses saturating_sub(count / 2), so 3 - (3/2) = 3 - 1 = 2
        assert_eq!(buffer.access_counts[5], 2);
    }

    #[test]
    fn test_neuron_level_enum() {
        let detailed = NeuronLevel::Detailed(LIFNeuron::new(0));
        let medium = NeuronLevel::Medium(RegionGroup::new(0, 1000));
        let abstract_level = NeuronLevel::Abstract(MeanFieldRegion::new(0, 1000000));

        // Should be able to create all three levels
        match detailed {
            NeuronLevel::Detailed(_) => {},
            _ => panic!("Expected Detailed variant"),
        }

        match medium {
            NeuronLevel::Medium(_) => {},
            _ => panic!("Expected Medium variant"),
        }

        match abstract_level {
            NeuronLevel::Abstract(_) => {},
            _ => panic!("Expected Abstract variant"),
        }
    }

    #[test]
    fn test_biological_realism_membrane_potential() {
        let brain = HierarchicalBrain::new(3, 100);

        // All detailed neurons should start near biological resting potential
        for neuron in &brain.detailed_neurons {
            assert!(neuron.state.v >= -80.0 && neuron.state.v <= -60.0,
                "Membrane potential should be in biological range");
        }
    }

    #[test]
    fn test_biological_realism_firing_rates() {
        let brain = HierarchicalBrain::new(3, 100);

        // Medium regions should have biologically plausible firing rates
        for region in &brain.medium_regions {
            assert!(region.avg_rate >= 0.0 && region.avg_rate <= 100.0,
                "Firing rate should be in biological range (0-100 Hz)");
        }

        // Abstract regions should have biologically plausible firing rates
        for region in &brain.abstract_regions {
            assert!(region.mean_rate >= 0.0 && region.mean_rate <= 100.0,
                "Mean rate should be in biological range (0-100 Hz)");
        }
    }

    #[test]
    fn test_integration_hierarchical_levels() {
        let mut brain = HierarchicalBrain::new(3, 1000);

        // Simulate activity in detailed neurons
        for (i, neuron) in brain.detailed_neurons.iter_mut().enumerate() {
            if i % 10 == 0 {
                neuron.state.v = -55.0; // Depolarized
            }
        }

        // Propagate through hierarchy
        brain.aggregate_detailed_to_medium();
        brain.aggregate_medium_to_abstract();
        brain.broadcast_abstract_to_medium();

        // Medium regions should show increased activity
        let avg_medium_v: f32 = brain.medium_regions.iter().map(|r| r.avg_v).sum::<f32>()
            / brain.medium_regions.len() as f32;
        assert!(avg_medium_v > -70.0, "Medium regions should reflect detailed activity");
    }
}
