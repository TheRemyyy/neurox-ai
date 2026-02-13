//! Integration test: high-level brain creation and one step (no GPU required when built without default features).

use neurox_ai::{NeuromorphicBrain, DEFAULT_TIMESTEP};

#[test]
fn brain_creation_and_single_update() {
    let mut brain = NeuromorphicBrain::new(
        3,    // n_layers
        100,  // base_neurons
        1000, // vocab_size
        128,  // pattern_dim
    );

    brain.update(DEFAULT_TIMESTEP);

    let stats = brain.stats();
    assert!(stats.time >= 0.0);
    assert!(stats.total_error >= 0.0);
}

#[test]
fn brain_process_text() {
    let mut brain = NeuromorphicBrain::new(3, 100, 500, 128);
    let response = brain.process_text("hello");
    assert!(!response.is_empty());
}
