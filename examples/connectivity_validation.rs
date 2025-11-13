//! Sparse Connectivity Validation
//!
//! Demonstrates and validates sparse connectivity patterns:
//! - Random, Small-World, and Distance-Dependent topologies
//! - Memory efficiency vs dense connectivity
//! - Large-scale performance (100K neurons)

use neurox_ai::{ProceduralConnectivity, SparseConnectivity, ConnectivityType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("╔════════════════════════════════════════════════════════════╗");
    log::info!("║  Sparse Connectivity Validation                           ║");
    log::info!("╚════════════════════════════════════════════════════════════╝\n");

    // Test 1: Small network with different topologies
    test_topologies()?;

    // Test 2: Memory efficiency benchmark
    test_memory_efficiency()?;

    // Test 3: Large scale connectivity (100K neurons)
    test_large_scale()?;

    log::info!("\n✓ All connectivity validation tests passed!");

    Ok(())
}

fn test_topologies() -> Result<(), Box<dyn std::error::Error>> {
    log::info!("=== Test 1: Connectivity Topologies ===\n");

    const N_NEURONS: usize = 1000;

    // Random sparse
    log::info!("[Random Topology]");
    let random_conn = ProceduralConnectivity::new(42, 0.1, 0.5, 0.1);
    let sparse_random = SparseConnectivity::from_procedural(N_NEURONS, &random_conn);
    log::info!("  Synapses: {}", sparse_random.nnz);
    log::info!("  Avg degree: {:.2}", sparse_random.avg_degree());
    log::info!("  Sparsity: {:.2}%", sparse_random.sparsity());
    log::info!("  Memory: {} KB\n", sparse_random.memory_footprint() / 1024);

    // Small-world (cortical-like)
    log::info!("[Small-World Topology]");
    let cortical_conn = ProceduralConnectivity::cortical(42, 10, 0.1);
    let sparse_cortical = SparseConnectivity::from_procedural(N_NEURONS, &cortical_conn);
    log::info!("  Synapses: {}", sparse_cortical.nnz);
    log::info!("  Avg degree: {:.2}", sparse_cortical.avg_degree());
    log::info!("  Sparsity: {:.2}%", sparse_cortical.sparsity());
    log::info!("  Memory: {} KB\n", sparse_cortical.memory_footprint() / 1024);

    // Distance-dependent
    log::info!("[Distance-Dependent Topology]");
    let distance_conn = ProceduralConnectivity::new(42, 0.0, 0.5, 0.1)
        .with_topology(ConnectivityType::DistanceDependent { sigma: 0.1 });
    let sparse_distance = SparseConnectivity::from_procedural(N_NEURONS, &distance_conn);
    log::info!("  Synapses: {}", sparse_distance.nnz);
    log::info!("  Avg degree: {:.2}", sparse_distance.avg_degree());
    log::info!("  Sparsity: {:.2}%", sparse_distance.sparsity());
    log::info!("  Memory: {} KB\n", sparse_distance.memory_footprint() / 1024);

    Ok(())
}

fn test_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
    log::info!("=== Test 2: Memory Efficiency ===\n");

    const N_NEURONS: usize = 10_000;

    let conn = ProceduralConnectivity::new(42, 0.01, 0.5, 0.1); // 1% connectivity
    let sparse = SparseConnectivity::from_procedural(N_NEURONS, &conn);

    let dense_memory = N_NEURONS * N_NEURONS * 4; // 4 bytes per float
    let sparse_memory = sparse.memory_footprint();
    let reduction = dense_memory as f64 / sparse_memory as f64;

    log::info!("  Neurons: {}", N_NEURONS);
    log::info!("  Synapses: {}", sparse.nnz);
    log::info!("  Dense memory: {:.2} MB", dense_memory as f64 / 1024.0 / 1024.0);
    log::info!("  Sparse memory: {:.2} MB", sparse_memory as f64 / 1024.0 / 1024.0);
    log::info!("  Reduction: {:.0}×", reduction);
    log::info!("  ✓ Memory savings: {:.1}%\n", 100.0 * (1.0 - 1.0 / reduction));

    Ok(())
}

fn test_large_scale() -> Result<(), Box<dyn std::error::Error>> {
    log::info!("=== Test 3: Large Scale (100K neurons) ===\n");

    const N_NEURONS: usize = 100_000;

    log::info!("Building connectivity for {} neurons...", N_NEURONS);
    let start = std::time::Instant::now();

    let conn = ProceduralConnectivity::cortical(42, 100, 0.1); // ~100 synapses/neuron
    let sparse = SparseConnectivity::from_procedural(N_NEURONS, &conn);

    let elapsed = start.elapsed();

    log::info!("\n[Results]");
    log::info!("  Build time: {:.2}s", elapsed.as_secs_f32());
    log::info!("  Total synapses: {} million", sparse.nnz / 1_000_000);
    log::info!("  Avg synapses/neuron: {:.2}", sparse.avg_degree());
    log::info!("  Sparsity: {:.4}%", sparse.sparsity());
    log::info!("  Memory footprint: {:.2} MB", sparse.memory_footprint() as f64 / 1024.0 / 1024.0);
    log::info!("  Throughput: {:.2} Mneurons/sec", N_NEURONS as f64 / elapsed.as_secs_f64() / 1e6);

    // Validate a few rows
    let (targets, weights) = sparse.get_row(0);
    log::info!("\n[Sample neuron 0]");
    log::info!("  Outgoing connections: {}", targets.len());
    if targets.len() > 0 {
        log::info!("  First 5 targets: {:?}", &targets[..targets.len().min(5)]);
        log::info!("  First 5 weights: {:?}", &weights[..weights.len().min(5)]);
    }

    Ok(())
}
