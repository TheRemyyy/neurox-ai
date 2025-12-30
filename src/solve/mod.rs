//! Problem Solving Module
//!
//! Symbolic and neural-assisted problem solving for math and chemistry.

pub mod chemistry;
pub mod math;

pub use chemistry::{BalancedEquation, ChemistrySolver};
pub use math::{MathResult, MathSolver};
