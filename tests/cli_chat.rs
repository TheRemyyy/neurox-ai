//! Integration test: chat command with minimal config (library API, no interactive loop).

use neurox_ai::NeuromorphicBrain;

#[test]
fn chat_flow_single_turn() {
    let mut brain = NeuromorphicBrain::new(3, 50, 500, 64);
    let out = brain.process_text("hi");
    assert!(!out.is_empty());
}
