//! Problem Solving Module
//!
//! Symbolic and neural-assisted problem solving for math and chemistry.

pub mod chemistry;
pub mod math;

pub use chemistry::{BalancedEquation, ChemistrySolver, ChemicalAnalysis};
pub use math::{MathResult, MathSolver, MathAnalysis};
