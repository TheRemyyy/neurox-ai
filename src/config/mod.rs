//! Configuration Module
//!
//! Loads and provides configuration from cfg/ JSON files.

use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Main application configuration
#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub version: String,
    pub default_plugin: String,
}

/// Chat plugin configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ChatConfig {
    pub vocab: usize,
    pub pattern_dim: usize,
    pub neurons: usize,
    pub context: usize,
    pub sensitivity: f32,
    pub brain_update_dt: f32,
    pub warmup_iterations: usize,
    pub post_processing_iterations: usize,
    pub history_size: usize,
    pub sleep_consolidation_iterations: usize,
    pub sleep_update_dt: f32,
    pub sleep_delay_ms: u64,
    pub ne_spike_value: f32,
    pub dopamine_reward_state_size: usize,
    pub dopamine_reward_value: f32,
}

/// Solve plugin configuration
#[derive(Debug, Clone, Deserialize)]
pub struct SolveConfig {
    pub default_type: String,
}

/// STDP configuration
#[derive(Debug, Clone, Deserialize)]
pub struct StdpConfig {
    pub lr_pre: f32,
    pub lr_post: f32,
    pub tau_pre: f32,
    pub tau_post: f32,
    pub w_min: f32,
    pub w_max: f32,
}

/// Connectivity configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ConnectivityConfig {
    pub seed: u64,
    pub density: f32,
    pub exc_ratio: f32,
    pub inh_ratio: f32,
}

/// Benchmark plugin configuration
#[derive(Debug, Clone, Deserialize)]
pub struct BenchmarkConfig {
    pub data_dir: String,
    pub epochs: usize,
    pub bits: u8,
    pub neurons: usize,
    pub presentation_duration: f32,
    pub isi: f32,
    pub batch_size: usize,
    pub lr_decay: f32,
    pub wta_strength: f32,
    pub target_rate: f32,
    pub consolidation_interval: usize,
    pub n_input: usize,
    pub n_output: usize,
    pub dt: f32,
    pub synthetic_train_samples: usize,
    pub synthetic_test_samples: usize,
    pub train_subset_size: usize,
    pub test_subset_size: usize,
    pub consolidation_samples: usize,
    pub real_train_eval_size: usize,
    pub real_consolidation_samples: usize,
    pub stdp: StdpConfig,
    pub connectivity: ConnectivityConfig,
}

/// Info plugin configuration (currently empty)
#[derive(Debug, Clone, Deserialize, Default)]
pub struct InfoConfig {}

/// Configuration loader
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load main application config
    pub fn load_app_config() -> Result<AppConfig, Box<dyn std::error::Error>> {
        Self::load_config("cfg/cfg.json")
    }

    /// Load chat plugin config
    pub fn load_chat_config() -> Result<ChatConfig, Box<dyn std::error::Error>> {
        Self::load_config("cfg/chat.json")
    }

    /// Load solve plugin config
    pub fn load_solve_config() -> Result<SolveConfig, Box<dyn std::error::Error>> {
        Self::load_config("cfg/solve.json")
    }

    /// Load benchmark plugin config
    pub fn load_benchmark_config() -> Result<BenchmarkConfig, Box<dyn std::error::Error>> {
        Self::load_config("cfg/benchmark.json")
    }

    /// Load info plugin config
    pub fn load_info_config() -> Result<InfoConfig, Box<dyn std::error::Error>> {
        let path = Path::new("cfg/info.json");
        if path.exists() {
            let content = fs::read_to_string(path)?;
            if content.trim() == "{}" {
                return Ok(InfoConfig::default());
            }
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(InfoConfig::default())
        }
    }

    /// Generic config loader
    fn load_config<T: for<'de> Deserialize<'de>>(
        path: &str,
    ) -> Result<T, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            vocab: 10000,
            pattern_dim: 512,
            neurons: 10000,
            context: 128,
            sensitivity: 1.0,
            brain_update_dt: 0.1,
            warmup_iterations: 10,
            post_processing_iterations: 3,
            history_size: 20,
            sleep_consolidation_iterations: 20,
            sleep_update_dt: 1.0,
            sleep_delay_ms: 50,
            ne_spike_value: 100.0,
            dopamine_reward_state_size: 512,
            dopamine_reward_value: 1.0,
        }
    }
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            default_type: "math".to_string(),
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            data_dir: "synthetic".to_string(),
            epochs: 10,
            bits: 4,
            neurons: 400,
            presentation_duration: 50.0,
            isi: 20.0,
            batch_size: 100,
            lr_decay: 0.95,
            wta_strength: 18.0,
            target_rate: 5.0,
            consolidation_interval: 5,
            n_input: 784,
            n_output: 10,
            dt: 0.1,
            synthetic_train_samples: 1000,
            synthetic_test_samples: 200,
            train_subset_size: 100,
            test_subset_size: 1000,
            consolidation_samples: 50,
            real_train_eval_size: 1000,
            real_consolidation_samples: 200,
            stdp: StdpConfig::default(),
            connectivity: ConnectivityConfig::default(),
        }
    }
}

impl Default for StdpConfig {
    fn default() -> Self {
        Self {
            lr_pre: 0.0001,
            lr_post: 0.01,
            tau_pre: 20.0,
            tau_post: 20.0,
            w_min: 0.0,
            w_max: 1.0,
        }
    }
}

impl Default for ConnectivityConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            density: 0.1,
            exc_ratio: 0.5,
            inh_ratio: 0.1,
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            default_plugin: "chat".to_string(),
        }
    }
}
