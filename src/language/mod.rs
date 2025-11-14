//! Language and Temporal Processing Systems
//!
//! Dual-stream language architecture for biological accuracy.
//! - Ventral stream: Sound-to-meaning comprehension (STG→MTG→ATL)
//! - Dorsal stream: Sound-to-articulation production (STG→Spt→IFG)

pub mod dual_stream;

pub use dual_stream::{
    DualStreamLanguage, DualStreamStats, VentralStream, DorsalStream,
    MultiTimescaleProcessor, STG, MTG, ATL, Spt, IFG,
};
