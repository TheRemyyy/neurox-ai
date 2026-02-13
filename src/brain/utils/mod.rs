//! Utility functions and helpers

use std::time::Instant;

/// Performance timer
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            name: name.into(),
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1000.0
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        log::debug!("{}: {:.2}ms", self.name, self.elapsed_ms());
    }
}

/// Calculate firing rate statistics
pub fn firing_rate_stats(spikes: &[f32]) -> (f32, f32, f32) {
    let n_spikes: f32 = spikes.iter().sum();
    let mean = n_spikes / spikes.len() as f32;

    let variance: f32 =
        spikes.iter().map(|&s| (s - mean).powi(2)).sum::<f32>() / spikes.len() as f32;

    let std = variance.sqrt();

    (mean, std, n_spikes)
}

/// Format bytes as human-readable
pub fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_idx])
}
