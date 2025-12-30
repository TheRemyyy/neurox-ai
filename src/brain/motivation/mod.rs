//! Motivation Module
//!
//! Implements intrinsic motivation systems including curiosity-driven
//! exploration and reward prediction for neuromorphic AI.
//!
//! Based on 2024-2025 research on curiosity-driven learning and
//! information-theoretic approaches to exploration.

pub mod curiosity;

pub use curiosity::{CuriosityDrive, CuriosityStats, InformationGain};
