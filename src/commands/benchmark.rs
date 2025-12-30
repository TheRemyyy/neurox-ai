//! Benchmark Command Handler

use crate::config::ConfigLoader;
use crate::plugin::BenchmarkPlugin;

/// Run the benchmark command with optional CLI overrides
pub fn run(
    data_dir: Option<String>,
    epochs: Option<usize>,
    bits: Option<u8>,
    neurons: Option<usize>,
    duration: Option<f32>,
    isi: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load config from file or use defaults
    let config = ConfigLoader::load_benchmark_config().unwrap_or_default();

    let plugin = BenchmarkPlugin::new(config);
    plugin.run(data_dir.as_deref(), epochs, bits, neurons, duration, isi)
}
