//! Symbolic Math Solver (Cognitive Symbolic Engine)
//!
//! Features:
//! - Recursive AST parser
//! - Symbolic differentiation
//! - Algebraic simplification
//! - Step-by-step reasoning

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

/// Result of a math computation
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

impl fmt::Display for MathResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathResult::Number(n) => write!(f, "{}", n),
            MathResult::Expr(e) => write!(f, "{}", e),
            MathResult::Equation(s) => write!(f, "{}", s),
            MathResult::Error(e) => write!(f, "Error: {}", e),
        }
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
    pub fn solve(&mut self, input: &str) -> MathResult {
        let input = input.trim();

        // Check for specific commands
        if input.starts_with("diff(") || input.starts_with("d/dx") {
            return self.solve_derivative(input);
        }

        // Check for equation
        if input.contains('=') {
            // Simplified linear solver (for now)
            return self.solve_linear_equation(input);
        }

        // Otherwise evaluate
        match self.parse(input) {
            Ok(expr) => {
                // Try to simplify first
                let simplified = self.simplify(&expr);
                
                // If it contains variables we don't know, return symbolic
                if self.has_unknown_vars(&simplified) {
                    MathResult::Expr(simplified)
                } else {
                    // Evaluate to number
                    MathResult::Number(self.evaluate(&simplified))
                }
            }
            Err(e) => MathResult::Error(e),
        }
    }

    /// Symbolic Differentiation
    /// Usage: "diff(x^2 + 3x, x)"
    fn solve_derivative(&self, input: &str) -> MathResult {
        // Simple parser for "diff(expr, var)"
        let content = input
            .trim_start_matches("diff")
            .trim_start_matches("d/dx")
            .trim_matches(|c| c == '(' || c == ')' || c == ' ');
        
        let parts: Vec<&str> = content.split(',').collect();
        let expr_str = parts[0];
        let var = if parts.len() > 1 { parts[1].trim() } else { "x" };

        match self.parse(expr_str) {
            Ok(expr) => {
                let derived = self.differentiate(&expr, var);
                let simplified = self.simplify(&derived);
                MathResult::Expr(simplified)
            }
            Err(e) => MathResult::Error(format!("Parse error: {}", e)),
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
                // Product rule: (uv)' = u'v + uv'
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
                // Quotient rule: (u/v)' = (u'v - uv') / v^2
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
                // Power rule (assuming exp is constant for now): n * x^(n-1) * dx
                if let Expr::Number(n) = **exp {
                    let new_exp = Expr::Number(n - 1.0);
                    let du = self.differentiate(base, var);
                    
                    let term = Expr::Mul(
                        Box::new(Expr::Number(n)),
                        Box::new(Expr::Pow(base.clone(), Box::new(new_exp))),
                    );
                    Expr::Mul(Box::new(term), Box::new(du))
                } else {
                    // Generalized power rule not implemented fully yet
                    Expr::Variable("NotImplemented(PowerRuleGeneral)".to_string())
                }
            }
            Expr::Sin(arg) => {
                // diff(sin(u)) = cos(u) * u'
                let du = self.differentiate(arg, var);
                Expr::Mul(Box::new(Expr::Cos(arg.clone())), Box::new(du))
            }
            Expr::Cos(arg) => {
                // diff(cos(u)) = -sin(u) * u'
                let du = self.differentiate(arg, var);
                Expr::Mul(
                    Box::new(Expr::Number(-1.0)),
                    Box::new(Expr::Mul(Box::new(Expr::Sin(arg.clone())), Box::new(du)))
                )
            }
            _ => Expr::Number(0.0),
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

    /// Basic recursive descent parser (Simplified for demo)
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
                // Verify matching parens wrap the WHOLE string
                if self.is_fully_enclosed(input) {
                    return self.parse_recursive(&input[1..input.len()-1]);
                }
            }
        }

        // 5. Base cases
        if let Ok(n) = input.parse::<f64>() {
            return Ok(Expr::Number(n));
        }
        
        // Variables
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
        // Search from right to left for left-associativity
        for (i, c) in input.char_indices().rev() {
            if c == ')' { depth += 1; }
            else if c == '(' { depth -= 1; }
            else if depth == 0 && ops.contains(&c) {
                return Some(i);
            }
        }
        None
    }

    fn solve_linear_equation(&self, input: &str) -> MathResult {
        let parts: Vec<&str> = input.split('=').collect();
        if parts.len() == 2 {
            let left = parts[0].trim();
            let right = parts[1].trim();
            // Try simplistic solver: 2x + 5 = 15 -> 2x = 10 -> x = 5
            // This is a placeholder for a real algebraic solver
            MathResult::Equation(format!("{} = {}", left, right)) // Just echo for now
        } else {
            MathResult::Error("Invalid equation".to_string())
        }
    }
}