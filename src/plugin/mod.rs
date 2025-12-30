//! Plugin Module
//!
//! Contains all NeuroxAI plugins: chat, solve, benchmark, info.

pub mod benchmark;
pub mod chat;
pub mod info;
pub mod solve;

pub use benchmark::BenchmarkPlugin;
pub use chat::ChatPlugin;
pub use info::InfoPlugin;
pub use solve::{ChemicalAnalysis, ChemistrySolver, MathAnalysis, MathResult, MathSolver};
