//! Info Command Handler

use crate::plugin::InfoPlugin;

/// Run the info command
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let plugin = InfoPlugin::new();
    plugin.display_info()
}
