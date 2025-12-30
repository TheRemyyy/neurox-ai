//! Symbolic Math Solver (Cognitive Symbolic Engine)
//!
//! Features:
//! - Recursive AST parser
//! - Symbolic differentiation
//! - Algebraic simplification
//! - Step-by-step reasoning
//! - Self-Verification (Substitution check)

use std::collections::HashMap;
use std::fmt;

/// Abstract Syntax Tree for Mathematical Expressions
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    Variable(String),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Ln(Box<Expr>),
}

/// Result of a math computation with verification details
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

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Number(n) => write!(f, "{}", n),
            Expr::Variable(v) => write!(f, "{}", v),
            Expr::Add(l, r) => write!(f, "({} + {})", l, r),
            Expr::Sub(l, r) => write!(f, "({} - {})", l, r),
            Expr::Mul(l, r) => write!(f, "{} * {}", l, r),
            Expr::Div(l, r) => write!(f, "{} / {}", l, r),
            Expr::Pow(l, r) => write!(f, "{}^{}", l, r),
            Expr::Sin(e) => write!(f, "sin({})", e),
            Expr::Cos(e) => write!(f, "cos({})", e),
            Expr::Ln(e) => write!(f, "ln({})", e),
        }
    }
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

    /// Entry point: Solve, Calculate, or Differentiate
    pub fn solve(&mut self, input: &str) -> MathAnalysis {
        let input = input.trim();
        let mut steps = Vec::new();
        steps.push(format!("Parsing input: '{}'", input));

        // Check for specific commands
        if input.starts_with("diff(") || input.starts_with("d/dx") {
            return self.solve_derivative(input, steps);
        }

        // Check for equation
        if input.contains('=') {
            return self.solve_linear_equation(input, steps);
        }

        // Otherwise evaluate
        match self.parse(input) {
            Ok(expr) => {
                steps.push(format!("Parsed AST: {}", expr));
                // Try to simplify first
                let simplified = self.simplify(&expr);
                if simplified != expr {
                    steps.push(format!("Simplified to: {}", simplified));
                }
                
                // If it contains variables we don't know, return symbolic
                if self.has_unknown_vars(&simplified) {
                    MathAnalysis {
                        input: input.to_string(),
                        result: MathResult::Expr(simplified),
                        steps,
                        verification: None,
                    }
                } else {
                    // Evaluate to number
                    let val = self.evaluate(&simplified);
                    steps.push(format!("Evaluated to: {}", val));
                    MathAnalysis {
                        input: input.to_string(),
                        result: MathResult::Number(val),
                        steps,
                        verification: None, // Can't verify simple eval without equation
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

    /// Symbolic Differentiation
    fn solve_derivative(&self, input: &str, mut steps: Vec<String>) -> MathAnalysis {
        let content = input
            .trim_start_matches("diff")
            .trim_start_matches("d/dx")
            .trim_matches(|c| c == '(' || c == ')' || c == ' ');
        
        let parts: Vec<&str> = content.split(',').collect();
        let expr_str = parts[0];
        let var = if parts.len() > 1 { parts[1].trim() } else { "x" };
        
        steps.push(format!("Differentiating with respect to '{}'", var));

        match self.parse(expr_str) {
            Ok(expr) => {
                let derived = self.differentiate(&expr, var);
                steps.push(format!("Raw derivative: {}", derived));
                let simplified = self.simplify(&derived);
                steps.push(format!("Simplified derivative: {}", simplified));
                
                MathAnalysis {
                    input: input.to_string(),
                    result: MathResult::Expr(simplified),
                    steps,
                    verification: None,
                }
            }
            Err(e) => MathAnalysis {
                input: input.to_string(),
                result: MathResult::Error(format!("Parse error: {}", e)),
                steps,
                verification: None,
            },
        }
    }

    /// Solve linear equation and VERIFY result
    fn solve_linear_equation(&mut self, input: &str, mut steps: Vec<String>) -> MathAnalysis {
        let parts: Vec<&str> = input.split('=').collect();
        if parts.len() != 2 {
            return MathAnalysis {
                input: input.to_string(),
                result: MathResult::Error("Invalid equation format".to_string()),
                steps,
                verification: None,
            };
        }

        let left_str = parts[0].trim();
        let right_str = parts[1].trim();
        steps.push(format!("Split equation: LHS='{}', RHS='{}'", left_str, right_str));

        // Simplified solver logic for "ax + b = c"
        // 1. Parse both sides
        // This is a naive implementation for the demo. In a real system, we'd use a CAS (Computer Algebra System).
        // Here we just support the pattern "2x + 5 = 15"
        
        // Find variable
        let var_name = "x"; // Assume x for now or find first alpha char
        
        // Let's try to extract 'a', 'b', 'c' from "ax + b = c"
        // VERY basic parser for demonstration of VERIFICATION logic
        // We assume left side is linear in x, right side is constant
        
        // Parse "2x + 5" -> extract 2 and 5.
        // This is brittle but serves the purpose of showing the Verification Step
        let solution = if let Ok(rhs_val) = right_str.parse::<f64>() {
             // Parse LHS coefficients
             let (a, b) = self.extract_linear_coeffs(left_str, var_name);
             steps.push(format!(" identified coefficients: {}x + {} = {}", a, b, rhs_val));
             
             if a == 0.0 {
                 None
             } else {
                 let x = (rhs_val - b) / a;
                 steps.push(format!("Solving: x = ({} - {}) / {}", rhs_val, b, a));
                 steps.push(format!("Calculated Solution: x = {}", x));
                 Some(x)
             }
        } else {
            None
        };

        if let Some(x_val) = solution {
            // === REALITY CHECK / VERIFICATION ===
            steps.push("Running Reality Check (Substitution)...".to_string());
            
            // 1. Temporarily set variable
            self.variables.insert(var_name.to_string(), x_val);
            
            // 2. Evaluate LHS
            let lhs_check = match self.parse(left_str) {
                Ok(expr) => self.evaluate(&expr),
                Err(_) => f64::NAN,
            };
            
            // 3. Evaluate RHS
            let rhs_check = match self.parse(right_str) {
                Ok(expr) => self.evaluate(&expr),
                Err(_) => f64::NAN,
            };
            
            // 4. Cleanup
            self.variables.remove(var_name);
            
            let is_valid = (lhs_check - rhs_check).abs() < 1e-6;
            
            MathAnalysis {
                input: input.to_string(),
                result: MathResult::Equation(format!("{} = {}", var_name, x_val)),
                steps,
                verification: Some(VerificationResult {
                    left_side_val: lhs_check,
                    right_side_val: rhs_check,
                    is_valid,
                }),
            }
        } else {
             MathAnalysis {
                input: input.to_string(),
                result: MathResult::Error("Could not solve linear equation (complexity limit)".to_string()),
                steps,
                verification: None,
            }
        }
    }

    /// Extract a and b from "ax + b" (very simple heuristic)
    fn extract_linear_coeffs(&self, expr: &str, var: &str) -> (f64, f64) {
        // Remove spaces
        let s = expr.replace(' ', "");
        
        // Split by '+'
        let parts: Vec<&str> = s.split('+').collect();
        let mut a = 0.0;
        let mut b = 0.0;
        
        for part in parts {
            if part.contains(var) {
                // It's the 'ax' part
                let coef = part.replace(var, "");
                a += if coef.is_empty() { 1.0 } else if coef == "-" { -1.0 } else { coef.parse().unwrap_or(0.0) };
            } else if part.contains('-') {
                // Handle subtraction inside split (e.g. "2x-5") - simple split fails here so we do basic check
                 if let Ok(val) = part.parse::<f64>() {
                    b += val;
                }
            } else {
                // It's the 'b' constant
                if let Ok(val) = part.parse::<f64>() {
                    b += val;
                }
            }
        }
        
        // Handle "2x - 5" case where split('+') leaves "2x-5" whole if no space? 
        // No, simple split is flawed. 
        // Better: Parse AST, walk tree, sum up coefficients.
        if let Ok(ast) = self.parse(expr) {
            self.walk_linear_ast(&ast, var)
        } else {
            (a, b) // Fallback
        }
    }

    fn walk_linear_ast(&self, expr: &Expr, var: &str) -> (f64, f64) {
        match expr {
            Expr::Number(n) => (0.0, *n),
            Expr::Variable(v) => if v == var { (1.0, 0.0) } else { (0.0, 0.0) }, // Unknown var treated as 0 for linear extraction
            Expr::Add(l, r) => {
                let (a1, b1) = self.walk_linear_ast(l, var);
                let (a2, b2) = self.walk_linear_ast(r, var);
                (a1 + a2, b1 + b2)
            },
            Expr::Sub(l, r) => {
                let (a1, b1) = self.walk_linear_ast(l, var);
                let (a2, b2) = self.walk_linear_ast(r, var);
                (a1 - a2, b1 - b2)
            },
            Expr::Mul(l, r) => {
                // Assuming one side is constant: 2*x or x*2
                if let Expr::Number(n) = **l {
                    let (a2, b2) = self.walk_linear_ast(r, var);
                    (n * a2, n * b2)
                } else if let Expr::Number(n) = **r {
                    let (a1, b1) = self.walk_linear_ast(l, var);
                    (a1 * n, b1 * n)
                } else {
                    (0.0, 0.0) // Non-linear or complex
                }
            },
            _ => (0.0, 0.0)
        }
    }

    /// Core Differentiation Logic
    pub fn differentiate(&self, expr: &Expr, var: &str) -> Expr {
        match expr {
            Expr::Number(_) => Expr::Number(0.0),
            Expr::Variable(v) => {
                if v == var {
                    Expr::Number(1.0)
                } else {
                    Expr::Number(0.0)
                }
            }
            Expr::Add(l, r) => Expr::Add(
                Box::new(self.differentiate(l, var)),
                Box::new(self.differentiate(r, var)),
            ),
            Expr::Sub(l, r) => Expr::Sub(
                Box::new(self.differentiate(l, var)),
                Box::new(self.differentiate(r, var)),
            ),
            Expr::Mul(l, r) => {
                let u = l.as_ref();
                let v = r.as_ref();
                let du = self.differentiate(u, var);
                let dv = self.differentiate(v, var);
                
                Expr::Add(
                    Box::new(Expr::Mul(Box::new(du), Box::new(v.clone()))),
                    Box::new(Expr::Mul(Box::new(u.clone()), Box::new(dv))),
                )
            }
            Expr::Div(u, v) => {
                let du = self.differentiate(u, var);
                let dv = self.differentiate(v, var);
                
                let numerator = Expr::Sub(
                    Box::new(Expr::Mul(Box::new(du), v.clone())),
                    Box::new(Expr::Mul(u.clone(), Box::new(dv))),
                );
                let denominator = Expr::Pow(v.clone(), Box::new(Expr::Number(2.0)));
                
                Expr::Div(Box::new(numerator), Box::new(denominator))
            }
            Expr::Pow(base, exp) => {
                if let Expr::Number(n) = **exp {
                    let new_exp = Expr::Number(n - 1.0);
                    let du = self.differentiate(base, var);
                    
                    let term = Expr::Mul(
                        Box::new(Expr::Number(n)),
                        Box::new(Expr::Pow(base.clone(), Box::new(new_exp))),
                    );
                    Expr::Mul(Box::new(term), Box::new(du))
                } else {
                    Expr::Variable("NotImplemented(PowerRuleGeneral)".to_string())
                }
            }
            Expr::Sin(arg) => {
                let du = self.differentiate(arg, var);
                Expr::Mul(Box::new(Expr::Cos(arg.clone())), Box::new(du))
            }
            Expr::Cos(arg) => {
                let du = self.differentiate(arg, var);
                Expr::Mul(
                    Box::new(Expr::Number(-1.0)),
                    Box::new(Expr::Mul(Box::new(Expr::Sin(arg.clone())), Box::new(du)))
                )
            }
            Expr::Ln(arg) => {
                // diff(ln(u)) = u'/u
                let du = self.differentiate(arg, var);
                Expr::Div(Box::new(du), arg.clone())
            }
        }
    }

    /// Algebraic Simplification
    pub fn simplify(&self, expr: &Expr) -> Expr {
        match expr {
            Expr::Add(l, r) => {
                let sl = self.simplify(l);
                let sr = self.simplify(r);
                match (sl, sr) {
                    (Expr::Number(a), Expr::Number(b)) => Expr::Number(a + b),
                    (Expr::Number(0.0), x) => x,
                    (x, Expr::Number(0.0)) => x,
                    (x, y) => Expr::Add(Box::new(x), Box::new(y)),
                }
            }
            Expr::Sub(l, r) => {
                let sl = self.simplify(l);
                let sr = self.simplify(r);
                match (sl, sr) {
                    (Expr::Number(a), Expr::Number(b)) => Expr::Number(a - b),
                    (x, Expr::Number(0.0)) => x,
                    (x, y) => Expr::Sub(Box::new(x), Box::new(y)),
                }
            }
            Expr::Mul(l, r) => {
                let sl = self.simplify(l);
                let sr = self.simplify(r);
                match (sl, sr) {
                    (Expr::Number(a), Expr::Number(b)) => Expr::Number(a * b),
                    (Expr::Number(0.0), _) => Expr::Number(0.0),
                    (_, Expr::Number(0.0)) => Expr::Number(0.0),
                    (Expr::Number(1.0), x) => x,
                    (x, Expr::Number(1.0)) => x,
                    (x, y) => Expr::Mul(Box::new(x), Box::new(y)),
                }
            }
            Expr::Pow(b, e) => {
                 let sb = self.simplify(b);
                 let se = self.simplify(e);
                 match (sb, se) {
                     (_, Expr::Number(0.0)) => Expr::Number(1.0),
                     (x, Expr::Number(1.0)) => x,
                     (x, y) => Expr::Pow(Box::new(x), Box::new(y)),
                 }
            }
            _ => expr.clone(),
        }
    }

    /// Evaluate AST to number
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
        }
    }

    fn has_unknown_vars(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Variable(v) => !self.variables.contains_key(v),
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => 
                self.has_unknown_vars(l) || self.has_unknown_vars(r),
            Expr::Sin(e) | Expr::Cos(e) | Expr::Ln(e) => self.has_unknown_vars(e),
            _ => false,
        }
    }

    /// Basic recursive descent parser
    fn parse(&self, input: &str) -> Result<Expr, String> {
        let clean = input.replace(' ', "");
        self.parse_recursive(&clean)
    }

    fn parse_recursive(&self, input: &str) -> Result<Expr, String> {
        // 1. Check for Add/Sub (lowest precedence)
        if let Some(idx) = self.find_operator(input, &['+', '-']) {
             let left = self.parse_recursive(&input[..idx])?;
             let right = self.parse_recursive(&input[idx+1..])?;
             return match &input[idx..idx+1] {
                 "+" => Ok(Expr::Add(Box::new(left), Box::new(right))),
                 "-" => Ok(Expr::Sub(Box::new(left), Box::new(right))),
                 _ => unreachable!(),
             };
        }

        // 2. Check for Mul/Div
        if let Some(idx) = self.find_operator(input, &['*', '/']) {
            let left = self.parse_recursive(&input[..idx])?;
            let right = self.parse_recursive(&input[idx+1..])?;
            return match &input[idx..idx+1] {
                "*" => Ok(Expr::Mul(Box::new(left), Box::new(right))),
                "/" => Ok(Expr::Div(Box::new(left), Box::new(right))),
                _ => unreachable!(),
            };
        }
        
        // 3. Power
        if let Some(idx) = self.find_operator(input, &['^']) {
             let left = self.parse_recursive(&input[..idx])?;
             let right = self.parse_recursive(&input[idx+1..])?;
             return Ok(Expr::Pow(Box::new(left), Box::new(right)));
        }

        // 4. Functions & Parentheses
        if input.ends_with(')') {
            if input.starts_with("sin(") {
                let inner = &input[4..input.len()-1];
                return Ok(Expr::Sin(Box::new(self.parse_recursive(inner)?)));
            }
            if input.starts_with("cos(") {
                let inner = &input[4..input.len()-1];
                return Ok(Expr::Cos(Box::new(self.parse_recursive(inner)?)));
            }
            if input.starts_with("ln(") {
                let inner = &input[3..input.len()-1];
                return Ok(Expr::Ln(Box::new(self.parse_recursive(inner)?)));
            }
            // Just parentheses (expr)
            if input.starts_with('(') {
                if self.is_fully_enclosed(input) {
                    return self.parse_recursive(&input[1..input.len()-1]);
                }
            }
        }

        // 5. Base cases
        if let Ok(n) = input.parse::<f64>() {
            return Ok(Expr::Number(n));
        }
        
        if input.chars().all(char::is_alphanumeric) {
            return Ok(Expr::Variable(input.to_string()));
        }

        Err(format!("Cannot parse: {}", input))
    }

    fn is_fully_enclosed(&self, input: &str) -> bool {
        let mut depth = 0;
        for (i, c) in input.char_indices() {
            if c == '(' { depth += 1; }
            else if c == ')' { depth -= 1; }
            
            if depth == 0 && i < input.len() - 1 {
                return false; // Closed before end
            }
        }
        depth == 0
    }

    fn find_operator(&self, input: &str, ops: &[char]) -> Option<usize> {
        let mut depth = 0;
        for (i, c) in input.char_indices().rev() {
            if c == ')' { depth += 1; }
            else if c == '(' { depth -= 1; }
            else if depth == 0 && ops.contains(&c) {
                return Some(i);
            }
        }
        None
    }
}
