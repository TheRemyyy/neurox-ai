//! Language and Temporal Processing Systems
//!
//! Sequential processing, language comprehension and production.
//! Includes modern dual-stream architecture for biological accuracy.

pub mod temporal_processing;
pub mod dual_stream;

pub use temporal_processing::{
    LanguageStats, LanguageSystem, MotorSequencer, TemporalProcessor,
};
pub use dual_stream::{
    DualStreamLanguage, DualStreamStats, VentralStream, DorsalStream,
    MultiTimescaleProcessor, STG, MTG, ATL, Spt, IFG,
};
