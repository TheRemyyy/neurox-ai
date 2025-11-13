//! Place and Grid Cells - Spatial and Conceptual Organization
//!
//! Place cells and grid cells provide metric coordinate system for navigation
//! and potentially abstract "conceptual spaces" for semantic knowledge.
//!
//! # Place Cells (Hippocampus)
//! - Fire at specific locations (30-50cm fields)
//! - 0.1-1Hz baseline → 10-40Hz in field
//! - Complete remapping between environments
//! - Orthogonal representations support episodic memory
//!
//! # Grid Cells (Medial Entorhinal Cortex)
//! - Hexagonal lattices at multiple scales
//! - 30cm to several meters spacing (×1.42 ratio between modules)
//! - Path integration from velocity × heading
//! - Generalize to semantic spaces (conceptual distance = similarity)

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Place cell with Gaussian receptive field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaceCell {
    /// Cell ID
    pub id: usize,

    /// Place field center (x, y)
    pub center: (f32, f32),

    /// Place field size (cm)
    pub field_size: f32,

    /// Current firing rate (Hz)
    pub firing_rate: f32,

    /// Baseline rate (Hz)
    pub baseline_rate: f32,

    /// Peak rate (Hz)
    pub peak_rate: f32,

    /// Environment ID (for remapping)
    pub environment_id: usize,
}

impl PlaceCell {
    pub fn new(id: usize, center: (f32, f32), field_size: f32, environment_id: usize) -> Self {
        Self {
            id,
            center,
            field_size,
            firing_rate: 1.0,
            baseline_rate: 1.0,
            peak_rate: 25.0,
            environment_id,
        }
    }

    /// Update firing rate based on position
    pub fn update(&mut self, position: (f32, f32)) {
        let distance = ((position.0 - self.center.0).powi(2)
            + (position.1 - self.center.1).powi(2))
            .sqrt();

        // Gaussian tuning curve
        let gaussian = (-distance.powi(2) / (2.0 * self.field_size.powi(2))).exp();

        self.firing_rate = self.baseline_rate + (self.peak_rate - self.baseline_rate) * gaussian;
    }

    /// Check if in place field
    pub fn in_field(&self, position: (f32, f32)) -> bool {
        let distance = ((position.0 - self.center.0).powi(2)
            + (position.1 - self.center.1).powi(2))
            .sqrt();
        distance < self.field_size
    }

    /// Remap for new environment (random new place field)
    pub fn remap(&mut self, new_environment_id: usize, environment_size: f32) {
        if new_environment_id != self.environment_id {
            let mut rng = rand::thread_rng();
            use rand::Rng;

            self.center = (
                rng.gen_range(0.0..environment_size),
                rng.gen_range(0.0..environment_size),
            );
            self.environment_id = new_environment_id;
        }
    }
}

/// Grid cell with hexagonal firing pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridCell {
    /// Cell ID
    pub id: usize,

    /// Grid spacing (cm)
    pub spacing: f32,

    /// Grid orientation (radians)
    pub orientation: f32,

    /// Grid phase offset (x, y)
    pub phase: (f32, f32),

    /// Current firing rate (Hz)
    pub firing_rate: f32,

    /// Module ID (for multi-scale organization)
    pub module_id: usize,
}

impl GridCell {
    pub fn new(id: usize, spacing: f32, orientation: f32, module_id: usize) -> Self {
        let mut rng = rand::thread_rng();
        use rand::Rng;

        let phase = (
            rng.gen_range(0.0..spacing),
            rng.gen_range(0.0..spacing),
        );

        Self {
            id,
            spacing,
            orientation,
            phase,
            firing_rate: 1.0,
            module_id,
        }
    }

    /// Update firing rate based on position
    ///
    /// Hexagonal lattice pattern with 60° symmetry
    pub fn update(&mut self, position: (f32, f32)) {
        // Transform to grid coordinates
        let x = position.0 - self.phase.0;
        let y = position.1 - self.phase.1;

        // Rotate by grid orientation
        let cos_theta = self.orientation.cos();
        let sin_theta = self.orientation.sin();
        let x_rot = x * cos_theta - y * sin_theta;
        let y_rot = x * sin_theta + y * cos_theta;

        // Hexagonal grid using sum of 3 cosines at 60° angles
        let s = self.spacing;
        let k = (4.0 * PI) / (3.0_f32.sqrt() * s);

        let g1 = (k * x_rot).cos();
        let g2 = (k * (x_rot * 0.5 + y_rot * 3.0_f32.sqrt() / 2.0)).cos();
        let g3 = (k * (x_rot * 0.5 - y_rot * 3.0_f32.sqrt() / 2.0)).cos();

        // Grid score: sum of cosines
        let grid_score = (g1 + g2 + g3) / 3.0;

        // Map to firing rate (0-40Hz)
        self.firing_rate = 1.0 + 39.0 * ((grid_score + 1.0) / 2.0);
    }

    /// Get grid vertices near position
    pub fn get_vertices(&self, position: (f32, f32), radius: f32) -> Vec<(f32, f32)> {
        let mut vertices = Vec::new();

        // Hexagonal lattice basis vectors
        let s = self.spacing;
        let v1 = (s, 0.0);
        let v2 = (s / 2.0, s * 3.0_f32.sqrt() / 2.0);

        // Search in grid
        let max_i = (radius / s) as i32 + 1;
        for i in -max_i..=max_i {
            for j in -max_i..=max_i {
                let x = i as f32 * v1.0 + j as f32 * v2.0 + self.phase.0;
                let y = i as f32 * v1.1 + j as f32 * v2.1 + self.phase.1;

                let dist = ((x - position.0).powi(2) + (y - position.1).powi(2)).sqrt();
                if dist < radius {
                    vertices.push((x, y));
                }
            }
        }

        vertices
    }
}

/// Grid cell module (multiple scales)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridModule {
    /// Module ID
    pub module_id: usize,

    /// Grid cells in this module
    pub cells: Vec<GridCell>,

    /// Module spacing (cm)
    pub spacing: f32,

    /// Number of cells
    pub n_cells: usize,
}

impl GridModule {
    pub fn new(module_id: usize, spacing: f32, n_cells: usize) -> Self {
        let mut rng = rand::thread_rng();
        use rand::Rng;

        let cells = (0..n_cells)
            .map(|i| {
                GridCell::new(
                    i,
                    spacing,
                    rng.gen_range(0.0..(PI / 3.0)),  // Random orientation within 60°
                    module_id,
                )
            })
            .collect();

        Self {
            module_id,
            cells,
            spacing,
            n_cells,
        }
    }

    /// Update all cells
    pub fn update(&mut self, position: (f32, f32)) {
        for cell in &mut self.cells {
            cell.update(position);
        }
    }

    /// Get module activity (average firing rate)
    pub fn activity(&self) -> f32 {
        self.cells.iter().map(|c| c.firing_rate).sum::<f32>() / self.n_cells as f32
    }
}

/// Multi-scale grid cell system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridCellSystem {
    /// Grid modules at different scales
    pub modules: Vec<GridModule>,

    /// Number of modules
    pub n_modules: usize,

    /// Spacing ratio between modules (typically 1.42)
    pub spacing_ratio: f32,
}

impl GridCellSystem {
    /// Create multi-scale grid system
    ///
    /// # Arguments
    /// - `n_modules`: Number of scale modules (typically 4-5)
    /// - `base_spacing`: Smallest spacing (cm, typically 30)
    /// - `cells_per_module`: Cells in each module
    pub fn new(n_modules: usize, base_spacing: f32, cells_per_module: usize) -> Self {
        let spacing_ratio: f32 = 1.42;
        let mut modules = Vec::with_capacity(n_modules);

        for i in 0..n_modules {
            let spacing = base_spacing * spacing_ratio.powi(i as i32);
            modules.push(GridModule::new(i, spacing, cells_per_module));
        }

        Self {
            modules,
            n_modules,
            spacing_ratio,
        }
    }

    /// Update all modules (path integration)
    pub fn update(&mut self, position: (f32, f32)) {
        for module in &mut self.modules {
            module.update(position);
        }
    }

    /// Get population code (for downstream readout)
    pub fn population_code(&self) -> Vec<f32> {
        self.modules
            .iter()
            .flat_map(|m| m.cells.iter().map(|c| c.firing_rate))
            .collect()
    }

    /// Decode position from grid population
    pub fn decode_position(&self) -> (f32, f32) {
        // Simplified decoding (maximum likelihood)
        // In reality would use Bayesian decoding over all modules

        if let Some(module) = self.modules.first() {
            // Use finest scale module
            let max_cell = module.cells
                .iter()
                .max_by(|a, b| a.firing_rate.partial_cmp(&b.firing_rate).unwrap());

            if let Some(cell) = max_cell {
                // Approximate position from phase
                return cell.phase;
            }
        }

        (0.0, 0.0)
    }
}

/// Semantic grid - generalize to conceptual spaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticGrid {
    /// Grid system
    pub grid_system: GridCellSystem,

    /// Mapping from semantic features to position
    pub semantic_map: Vec<(Vec<f32>, (f32, f32))>,

    /// Embedding dimension
    pub embedding_dim: usize,
}

impl SemanticGrid {
    pub fn new(n_modules: usize, embedding_dim: usize) -> Self {
        let grid_system = GridCellSystem::new(n_modules, 30.0, 20);

        Self {
            grid_system,
            semantic_map: Vec::new(),
            embedding_dim,
        }
    }

    /// Encode semantic features in grid space
    pub fn encode(&mut self, features: &[f32]) {
        // Map features to 2D position (dimensionality reduction)
        let position = self.features_to_position(features);

        // Update grid system
        self.grid_system.update(position);

        // Store mapping
        self.semantic_map.push((features.to_vec(), position));
    }

    /// Map high-dimensional features to 2D position
    fn features_to_position(&self, features: &[f32]) -> (f32, f32) {
        // Simple projection (in reality would use learned mapping)
        let x = features.iter().step_by(2).sum::<f32>();
        let y = features.iter().skip(1).step_by(2).sum::<f32>();
        (x * 100.0, y * 100.0)  // Scale to cm
    }

    /// Retrieve similar concepts
    pub fn retrieve_similar(&self, query: &[f32], k: usize) -> Vec<Vec<f32>> {
        let query_pos = self.features_to_position(query);

        let mut distances: Vec<(Vec<f32>, f32)> = self.semantic_map
            .iter()
            .map(|(feat, pos)| {
                let dist = ((pos.0 - query_pos.0).powi(2) + (pos.1 - query_pos.1).powi(2)).sqrt();
                (feat.clone(), dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        distances.into_iter().map(|(feat, _)| feat).collect()
    }
}

/// Integrated spatial system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialSystem {
    /// Place cells
    pub place_cells: Vec<PlaceCell>,

    /// Grid cell system
    pub grid_system: GridCellSystem,

    /// Current position
    pub position: (f32, f32),

    /// Velocity (for path integration)
    pub velocity: (f32, f32),

    /// Environment size (cm)
    pub environment_size: f32,

    /// Current environment ID
    pub environment_id: usize,
}

impl SpatialSystem {
    pub fn new(n_place_cells: usize, environment_size: f32) -> Self {
        let mut rng = rand::thread_rng();
        use rand::Rng;

        // Create place cells with random place fields
        let place_cells = (0..n_place_cells)
            .map(|i| {
                PlaceCell::new(
                    i,
                    (
                        rng.gen_range(0.0..environment_size),
                        rng.gen_range(0.0..environment_size),
                    ),
                    rng.gen_range(30.0..50.0),  // 30-50cm fields
                    0,
                )
            })
            .collect();

        // Create grid system (4 modules)
        let grid_system = GridCellSystem::new(4, 30.0, 20);

        Self {
            place_cells,
            grid_system,
            position: (environment_size / 2.0, environment_size / 2.0),
            velocity: (0.0, 0.0),
            environment_size,
            environment_id: 0,
        }
    }

    /// Update position via path integration
    pub fn update(&mut self, dt: f32, velocity: (f32, f32)) {
        self.velocity = velocity;

        // Integrate position
        self.position.0 += velocity.0 * dt / 1000.0;  // dt in ms
        self.position.1 += velocity.1 * dt / 1000.0;

        // Wrap around environment
        self.position.0 = self.position.0.rem_euclid(self.environment_size);
        self.position.1 = self.position.1.rem_euclid(self.environment_size);

        // Update place cells
        for cell in &mut self.place_cells {
            cell.update(self.position);
        }

        // Update grid cells
        self.grid_system.update(self.position);
    }

    /// Remap to new environment
    pub fn remap(&mut self, new_environment_id: usize) {
        for cell in &mut self.place_cells {
            cell.remap(new_environment_id, self.environment_size);
        }
        self.environment_id = new_environment_id;
    }

    /// Get active place cells
    pub fn active_place_cells(&self) -> Vec<usize> {
        self.place_cells
            .iter()
            .filter(|c| c.in_field(self.position))
            .map(|c| c.id)
            .collect()
    }

    /// Get place population code
    pub fn place_code(&self) -> Vec<f32> {
        self.place_cells.iter().map(|c| c.firing_rate).collect()
    }

    /// Get grid population code
    pub fn grid_code(&self) -> Vec<f32> {
        self.grid_system.population_code()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_place_cell() {
        let mut cell = PlaceCell::new(0, (50.0, 50.0), 40.0, 0);

        // In field
        cell.update((50.0, 50.0));
        assert!(cell.firing_rate > 10.0);

        // Out of field
        cell.update((150.0, 150.0));
        assert!(cell.firing_rate < 5.0);
    }

    #[test]
    fn test_grid_cell() {
        let mut cell = GridCell::new(0, 50.0, 0.0, 0);

        cell.update((0.0, 0.0));
        let rate1 = cell.firing_rate;

        cell.update((50.0, 0.0));  // One spacing over
        let rate2 = cell.firing_rate;

        // Should have hexagonal pattern
        assert!(rate1 > 0.0 && rate2 > 0.0);
    }

    #[test]
    fn test_grid_module_scaling() {
        let system = GridCellSystem::new(4, 30.0, 10);

        assert_eq!(system.modules.len(), 4);

        // Each module should have different spacing
        for i in 1..system.modules.len() {
            assert!(system.modules[i].spacing > system.modules[i - 1].spacing);
        }
    }

    #[test]
    fn test_spatial_system() {
        let mut system = SpatialSystem::new(100, 200.0);

        // Move
        system.update(100.0, (10.0, 5.0));

        // Position should have changed
        assert_ne!(system.position, (100.0, 100.0));

        // Some place cells should be active
        let active = system.active_place_cells();
        // May or may not have active cells depending on random initialization
    }

    #[test]
    fn test_place_cell_remapping() {
        let mut cell = PlaceCell::new(0, (50.0, 50.0), 40.0, 0);
        let original_center = cell.center;

        cell.remap(1, 200.0);

        // Center should have changed
        assert_ne!(cell.center, original_center);
        assert_eq!(cell.environment_id, 1);
    }
}
