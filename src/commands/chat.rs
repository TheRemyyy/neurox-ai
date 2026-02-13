//! Chat Command Handler

use crate::config::{ChatConfig, ConfigLoader};
use crate::plugin::ChatPlugin;

/// Run the chat command with optional CLI overrides
pub fn run(
    vocab: Option<usize>,
    pattern_dim: Option<usize>,
    neurons: Option<usize>,
    context: Option<usize>,
    sensitivity: Option<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load config from file or use defaults (propagate parse errors, use default only for missing file)
    let mut config = match ConfigLoader::load_chat_config() {
        Ok(c) => c,
        Err(e) => {
            if std::path::Path::new("cfg/chat.json").exists() {
                return Err(e);
            }
            log::info!("Using default chat config (cfg/chat.json not found)");
            ChatConfig::default()
        }
    };

    // Apply CLI overrides
    if let Some(v) = vocab {
        config.vocab = v;
    }
    if let Some(p) = pattern_dim {
        config.pattern_dim = p;
    }
    if let Some(n) = neurons {
        config.neurons = n;
    }
    if let Some(c) = context {
        config.context = c;
    }
    if let Some(s) = sensitivity {
        config.sensitivity = s;
    }

    let plugin = ChatPlugin::new(config);
    plugin.run()
}
