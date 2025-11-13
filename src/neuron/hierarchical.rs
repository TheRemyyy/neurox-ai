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
}
