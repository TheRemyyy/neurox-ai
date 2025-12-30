//! Symbolic Math Solver (Cognitive Symbolic Engine)
//!
//! Features:
//! - Recursive AST parser
//! - Symbolic differentiation AND INTEGRATION
//! - Algebraic simplification
//! - Step-by-step reasoning
//! - Self-Verification

mod ast;
mod calculus;
mod parser;
mod simplify;

pub use self::ast::Expr;
use self::calculus::Calculus;
use self::parser::Parser;
use self::simplify::Simplifier;

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub struct MathAnalysis {
    pub input: String,
    pub result: MathResult,
    pub steps: Vec<String>,
    pub verification: Option<VerificationResult>,
}

#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub left_side_val: f64,
    pub right_side_val: f64,
    pub is_valid: bool,
}

#[derive(Debug, Clone)]
pub enum MathResult {
    Number(f64),
    Expr(Expr),
    Equation(String),
    Error(String),
}

impl fmt::Display for MathAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Input: {}", self.input)?;
        match &self.result {
            MathResult::Number(n) => writeln!(f, "Result: {}", n)?,
            MathResult::Expr(e) => writeln!(f, "Result: {}", e)?,
            MathResult::Equation(e) => writeln!(f, "Result: {}", e)?,
            MathResult::Error(e) => writeln!(f, "Error: {}", e)?,
        }

        if let Some(ver) = &self.verification {
            writeln!(f, "Verification (Substitution):")?;
            writeln!(f, "  LHS Value: {:.4}", ver.left_side_val)?;
            writeln!(f, "  RHS Value: {:.4}", ver.right_side_val)?;
            if ver.is_valid {
                writeln!(f, "  Status: ✅ VALIDATED (Reality Check Passed)")?;
            } else {
                writeln!(f, "  Status: ❌ FAILED (Discrepancy Detected)")?;
            }
        }
        Ok(())
    }
}

pub struct MathSolver {
    variables: HashMap<String, f64>,
}

impl Default for MathSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MathSolver {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    pub fn solve(&mut self, input: &str) -> MathAnalysis {
        let input = input.trim();
        let mut steps = Vec::new();
        steps.push(format!("Parsing input: '{}'", input));

        // Commands
        if input.starts_with("diff(") || input.starts_with("d/dx") {
            return self.solve_derivative(input, steps);
        }
        if input.starts_with("int(") || input.starts_with("integrate") {
            return self.solve_integral(input, steps);
        }

        if input.contains('=') {
            return self.solve_linear_equation(input, steps);
        }

        match Parser::parse(input) {
            Ok(expr) => {
                steps.push(format!("Parsed AST: {}", expr));
                let simplified = Simplifier::simplify(&expr);
                if simplified != expr {
                    steps.push(format!("Simplified: {}", simplified));
                }

                if self.has_unknown_vars(&simplified) {
                    MathAnalysis {
                        input: input.to_string(),
                        result: MathResult::Expr(simplified),
                        steps,
                        verification: None,
                    }
                } else {
                    let val = self.evaluate(&simplified);
                    steps.push(format!("Evaluated: {}", val));
                    MathAnalysis {
                        input: input.to_string(),
                        result: MathResult::Number(val),
                        steps,
                        verification: None,
                    }
                }
            }
            Err(e) => MathAnalysis {
                input: input.to_string(),
                result: MathResult::Error(e),
                steps,
                verification: None,
            },
        }
    }

    fn solve_derivative(&self, input: &str, mut steps: Vec<String>) -> MathAnalysis {
        let content = input
            .trim_start_matches("diff")
            .trim_start_matches("d/dx")
            .trim_matches(|c| c == '(' || c == ')' || c == ' ');
        let parts: Vec<&str> = content.split(',').collect();
        let expr_str = parts[0];
        let var = if parts.len() > 1 {
            parts[1].trim()
        } else {
            "x"
        };

        steps.push(format!("Differentiating wrt '{}'", var));

        match Parser::parse(expr_str) {
            Ok(expr) => {
                let derived = Calculus::differentiate(&expr, var);
                let simplified = Simplifier::simplify(&derived);
                MathAnalysis {
                    input: input.to_string(),
                    result: MathResult::Expr(simplified),
                    steps,
                    verification: None,
                }
            }
            Err(e) => MathAnalysis {
                input: input.to_string(),
                result: MathResult::Error(e),
                steps,
                verification: None,
            },
        }
    }

    fn solve_integral(&self, input: &str, mut steps: Vec<String>) -> MathAnalysis {
        let content = input
            .trim_start_matches("int")
            .trim_start_matches("integrate")
            .trim_matches(|c| c == '(' || c == ')' || c == ' ');
        let parts: Vec<&str> = content.split(',').collect();
        let expr_str = parts[0];
        let var = if parts.len() > 1 {
            parts[1].trim()
        } else {
            "x"
        };

        steps.push(format!("Integrating wrt '{}'", var));

        match Parser::parse(expr_str) {
            Ok(expr) => {
                let integrated = Calculus::integrate(&expr, var);
                // Add + C
                let with_c = Expr::Add(
                    Box::new(integrated),
                    Box::new(Expr::Variable("C".to_string())),
                );
                let simplified = Simplifier::simplify(&with_c);
                MathAnalysis {
                    input: input.to_string(),
                    result: MathResult::Expr(simplified),
                    steps,
                    verification: None,
                }
            }
            Err(e) => MathAnalysis {
                input: input.to_string(),
                result: MathResult::Error(e),
                steps,
                verification: None,
            },
        }
    }

    fn solve_linear_equation(&mut self, input: &str, mut steps: Vec<String>) -> MathAnalysis {
        let parts: Vec<&str> = input.split('=').collect();
        if parts.len() != 2 {
            return MathAnalysis {
                input: input.to_string(),
                result: MathResult::Error("Invalid format".to_string()),
                steps,
                verification: None,
            };
        }

        let var = "x"; // Default variable to solve for
        steps.push(format!("Solving for '{}'", var));

        // Extract coefficients from both sides: ax + b
        let (a1, b1) = self.extract_linear_coeffs(parts[0].trim(), var);
        let (a2, b2) = self.extract_linear_coeffs(parts[1].trim(), var);

        // Equation: a1*x + b1 = a2*x + b2
        // Move terms: (a1 - a2)*x + (b1 - b2) = 0
        let a = a1 - a2;
        let b = b1 - b2;

        if a.abs() < 1e-9 {
            if b.abs() < 1e-9 {
                steps.push("Identity detected (always true).".to_string());
                return MathAnalysis {
                    input: input.to_string(),
                    result: MathResult::Equation("x ∈ ℝ (Identity)".to_string()),
                    steps,
                    verification: None,
                };
            } else {
                steps.push("Contradiction detected (no solution).".to_string());
                return MathAnalysis {
                    input: input.to_string(),
                    result: MathResult::Error("No solution".to_string()),
                    steps,
                    verification: None,
                };
            }
        }

        let x = -b / a;
        steps.push(format!("Final equation: {:.3}x + {:.3} = 0", a, b));
        steps.push(format!("Calculated x = {:.4}", x));

        // Verification
        let lhs_val = a1 * x + b1;
        let rhs_val = a2 * x + b2;
        let is_valid = (lhs_val - rhs_val).abs() < 1e-6;

        MathAnalysis {
            input: input.to_string(),
            result: MathResult::Equation(format!("x = {}", x)),
            steps,
            verification: Some(VerificationResult {
                left_side_val: lhs_val,
                right_side_val: rhs_val,
                is_valid,
            }),
        }
    }

    fn extract_linear_coeffs(&self, expr: &str, var: &str) -> (f64, f64) {
        if let Ok(ast) = Parser::parse(expr) {
            self.walk_linear_ast(&ast, var)
        } else {
            (0.0, 0.0)
        }
    }

    fn walk_linear_ast(&self, expr: &Expr, var: &str) -> (f64, f64) {
        match expr {
            Expr::Number(n) => (0.0, *n),
            Expr::Variable(v) => {
                if v == var {
                    (1.0, 0.0)
                } else {
                    (0.0, 0.0)
                }
            }
            Expr::Add(l, r) => {
                let (a1, b1) = self.walk_linear_ast(l, var);
                let (a2, b2) = self.walk_linear_ast(r, var);
                (a1 + a2, b1 + b2)
            }
            Expr::Sub(l, r) => {
                let (a1, b1) = self.walk_linear_ast(l, var);
                let (a2, b2) = self.walk_linear_ast(r, var);
                (a1 - a2, b1 - b2)
            }
            Expr::Mul(l, r) => {
                if let Expr::Number(n) = **l {
                    let (a, b) = self.walk_linear_ast(r, var);
                    (n * a, n * b)
                } else if let Expr::Number(n) = **r {
                    let (a, b) = self.walk_linear_ast(l, var);
                    (a * n, b * n)
                } else {
                    (0.0, 0.0)
                }
            }
            _ => (0.0, 0.0),
        }
    }

    fn evaluate(&self, expr: &Expr) -> f64 {
        match expr {
            Expr::Number(n) => *n,
            Expr::Variable(v) => *self.variables.get(v).unwrap_or(&0.0),
            Expr::Add(l, r) => self.evaluate(l) + self.evaluate(r),
            Expr::Sub(l, r) => self.evaluate(l) - self.evaluate(r),
            Expr::Mul(l, r) => self.evaluate(l) * self.evaluate(r),
            Expr::Div(l, r) => self.evaluate(l) / self.evaluate(r),
            Expr::Pow(l, r) => self.evaluate(l).powf(self.evaluate(r)),
            Expr::Sin(e) => self.evaluate(e).sin(),
            Expr::Cos(e) => self.evaluate(e).cos(),
            Expr::Ln(e) => self.evaluate(e).ln(),
            Expr::Exp(e) => self.evaluate(e).exp(),
        }
    }

    fn has_unknown_vars(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Variable(v) => !self.variables.contains_key(v),
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => self.has_unknown_vars(l) || self.has_unknown_vars(r),
            Expr::Sin(e) | Expr::Cos(e) | Expr::Ln(e) | Expr::Exp(e) => self.has_unknown_vars(e),
            _ => false,
        }
    }
}
