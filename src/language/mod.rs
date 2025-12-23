//! Language and Temporal Processing Systems
//!
//! Dual-stream language architecture for biological accuracy.
//! - Ventral stream: Sound-to-meaning comprehension (STG→MTG→ATL)
//! - Dorsal stream: Sound-to-articulation production (STG→Spt→IFG)
//! - IFG syntactic planner: Sentence structure and word-by-word generation

pub mod dual_stream;
pub mod ifg;

pub use dual_stream::{
    DualStreamLanguage, DualStreamStats, VentralStream, DorsalStream,
    MultiTimescaleProcessor, STG, MTG, ATL, Spt, IFG,
};

pub use ifg::{
    PartOfSpeech, GrammaticalRole, AnnotatedWord, 
    SentenceTemplate, IntentType, IFGSyntacticPlanner, Lexicon, PragmaticRule,
};
