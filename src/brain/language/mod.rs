//! Language and Temporal Processing Systems
//!
//! Dual-stream language architecture for biological accuracy.
//! - Ventral stream: Sound-to-meaning comprehension (STG→MTG→ATL)
//! - Dorsal stream: Sound-to-articulation production (STG→Spt→IFG)
//! - IFG syntactic planner: Sentence structure and word-by-word generation
//! - Sequence generator: Beam search and context-aware text generation

pub mod dual_stream;
pub mod ifg;
pub mod sequence_generator;

pub use dual_stream::{
    DorsalStream, DualStreamLanguage, DualStreamStats, MultiTimescaleProcessor, Spt, VentralStream,
    ATL, IFG, MTG, STG,
};

pub use ifg::{
    AnnotatedWord, GrammaticalRole, IFGSyntacticPlanner, IntentType, Lexicon, PartOfSpeech,
    PragmaticRule, SentenceTemplate,
};

pub use sequence_generator::{
    BeamCandidate, ContextWindow, ConversationTurn, GeneratorConfig, SequenceGenerator,
};
