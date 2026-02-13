//! Integration test: `info` command (system/GPU info).

use neurox_ai::commands::info;

#[test]
fn info_command_runs() {
    let result = info::run();
    // With CUDA: may succeed and print GPU info. Without CUDA: prints message and returns Ok.
    assert!(result.is_ok());
}
