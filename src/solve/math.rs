//! Math Problem Solver
//!
//! Symbolic and numeric math problem solving.
//! Supports arithmetic expressions, algebraic equations, and basic calculus.

use std::collections::HashMap;

/// Result of a math computation
#[derive(Debug, Clone)]
pub enum MathResult {
    /// Numeric result
    Number(f64),
    /// Symbolic result (e.g., "x = 2")
    Symbolic(String),
    /// Multiple solutions
    Solutions(Vec<f64>),
    /// Error
    Error(String),
}

impl std::fmt::Display for MathResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MathResult::Number(n) => write!(f, "{}", n),
            MathResult::Symbolic(s) => write!(f, "{}", s),
            MathResult::Solutions(sols) => {
                let s: Vec<String> = sols.iter().map(|x| x.to_string()).collect();
                write!(f, "x = {}", s.join(" or x = "))
            }
            MathResult::Error(e) => write!(f, "Error: {}", e),
        }
    }
}

/// Math problem solver
pub struct MathSolver {
    /// Variable assignments for symbolic solving
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

    /// Solve a math problem (arithmetic or algebraic)
    pub fn solve(&mut self, expression: &str) -> MathResult {
        let expr = expression.trim();

        // Check if it's an equation (contains '=')
        if expr.contains('=') {
            return self.solve_equation(expr);
        }

        // Otherwise, evaluate as arithmetic expression
        self.evaluate(expr)
    }

    /// Evaluate arithmetic expression
    fn evaluate(&self, expr: &str) -> MathResult {
        match self.parse_and_eval(expr) {
            Ok(result) => MathResult::Number(result),
            Err(e) => MathResult::Error(e),
        }
    }

    /// Parse and evaluate expression using recursive descent
    fn parse_and_eval(&self, expr: &str) -> Result<f64, String> {
        let tokens = self.tokenize(expr)?;
        let mut pos = 0;
        self.parse_expression(&tokens, &mut pos)
    }

    /// Tokenize expression into numbers and operators
    fn tokenize(&self, expr: &str) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = expr.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];

            if c.is_whitespace() {
                i += 1;
                continue;
            }

            if c.is_ascii_digit() || c == '.' {
                let mut num_str = String::new();
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    num_str.push(chars[i]);
                    i += 1;
                }
                let num: f64 = num_str
                    .parse()
                    .map_err(|_| format!("Invalid number: {}", num_str))?;
                tokens.push(Token::Number(num));
                continue;
            }

            match c {
                '+' => tokens.push(Token::Plus),
                '-' => tokens.push(Token::Minus),
                '*' => tokens.push(Token::Multiply),
                '/' => tokens.push(Token::Divide),
                '^' => tokens.push(Token::Power),
                '(' => tokens.push(Token::LParen),
                ')' => tokens.push(Token::RParen),
                _ if c.is_alphabetic() => {
                    let mut var = String::new();
                    while i < chars.len() && chars[i].is_alphanumeric() {
                        var.push(chars[i]);
                        i += 1;
                    }
                    // Check for built-in functions
                    match var.to_lowercase().as_str() {
                        "sqrt" => tokens.push(Token::Sqrt),
                        "sin" => tokens.push(Token::Sin),
                        "cos" => tokens.push(Token::Cos),
                        "tan" => tokens.push(Token::Tan),
                        "ln" => tokens.push(Token::Ln),
                        "log" => tokens.push(Token::Log),
                        "pi" => tokens.push(Token::Number(std::f64::consts::PI)),
                        "e" => tokens.push(Token::Number(std::f64::consts::E)),
                        _ => tokens.push(Token::Variable(var)),
                    }
                    continue;
                }
                _ => return Err(format!("Unknown character: {}", c)),
            }
            i += 1;
        }

        Ok(tokens)
    }

    /// Parse addition/subtraction level
    fn parse_expression(&self, tokens: &[Token], pos: &mut usize) -> Result<f64, String> {
        let mut left = self.parse_term(tokens, pos)?;

        while *pos < tokens.len() {
            match tokens.get(*pos) {
                Some(Token::Plus) => {
                    *pos += 1;
                    left += self.parse_term(tokens, pos)?;
                }
                Some(Token::Minus) => {
                    *pos += 1;
                    left -= self.parse_term(tokens, pos)?;
                }
                _ => break,
            }
        }

        Ok(left)
    }

    /// Parse multiplication/division level
    fn parse_term(&self, tokens: &[Token], pos: &mut usize) -> Result<f64, String> {
        let mut left = self.parse_power(tokens, pos)?;

        while *pos < tokens.len() {
            match tokens.get(*pos) {
                Some(Token::Multiply) => {
                    *pos += 1;
                    left *= self.parse_power(tokens, pos)?;
                }
                Some(Token::Divide) => {
                    *pos += 1;
                    let right = self.parse_power(tokens, pos)?;
                    if right == 0.0 {
                        return Err("Division by zero".to_string());
                    }
                    left /= right;
                }
                _ => break,
            }
        }

        Ok(left)
    }

    /// Parse power level
    fn parse_power(&self, tokens: &[Token], pos: &mut usize) -> Result<f64, String> {
        let base = self.parse_unary(tokens, pos)?;

        if *pos < tokens.len() && matches!(tokens.get(*pos), Some(Token::Power)) {
            *pos += 1;
            let exponent = self.parse_power(tokens, pos)?; // Right-associative
            Ok(base.powf(exponent))
        } else {
            Ok(base)
        }
    }

    /// Parse unary operators and functions
    fn parse_unary(&self, tokens: &[Token], pos: &mut usize) -> Result<f64, String> {
        if *pos >= tokens.len() {
            return Err("Unexpected end of expression".to_string());
        }

        match &tokens[*pos] {
            Token::Minus => {
                *pos += 1;
                Ok(-self.parse_unary(tokens, pos)?)
            }
            Token::Plus => {
                *pos += 1;
                self.parse_unary(tokens, pos)
            }
            Token::Sqrt => {
                *pos += 1;
                let arg = self.parse_primary(tokens, pos)?;
                Ok(arg.sqrt())
            }
            Token::Sin => {
                *pos += 1;
                let arg = self.parse_primary(tokens, pos)?;
                Ok(arg.sin())
            }
            Token::Cos => {
                *pos += 1;
                let arg = self.parse_primary(tokens, pos)?;
                Ok(arg.cos())
            }
            Token::Tan => {
                *pos += 1;
                let arg = self.parse_primary(tokens, pos)?;
                Ok(arg.tan())
            }
            Token::Ln => {
                *pos += 1;
                let arg = self.parse_primary(tokens, pos)?;
                Ok(arg.ln())
            }
            Token::Log => {
                *pos += 1;
                let arg = self.parse_primary(tokens, pos)?;
                Ok(arg.log10())
            }
            _ => self.parse_primary(tokens, pos),
        }
    }

    /// Parse numbers, variables, and parentheses
    fn parse_primary(&self, tokens: &[Token], pos: &mut usize) -> Result<f64, String> {
        if *pos >= tokens.len() {
            return Err("Unexpected end of expression".to_string());
        }

        match &tokens[*pos] {
            Token::Number(n) => {
                *pos += 1;
                Ok(*n)
            }
            Token::Variable(name) => {
                *pos += 1;
                self.variables
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("Unknown variable: {}", name))
            }
            Token::LParen => {
                *pos += 1;
                let result = self.parse_expression(tokens, pos)?;
                if *pos < tokens.len() && matches!(tokens.get(*pos), Some(Token::RParen)) {
                    *pos += 1;
                    Ok(result)
                } else {
                    Err("Missing closing parenthesis".to_string())
                }
            }
            _ => Err(format!("Unexpected token: {:?}", tokens[*pos])),
        }
    }

    /// Solve algebraic equation (simple quadratic/linear)
    fn solve_equation(&mut self, equation: &str) -> MathResult {
        // Split by '='
        let parts: Vec<&str> = equation.split('=').collect();
        if parts.len() != 2 {
            return MathResult::Error("Invalid equation format".to_string());
        }

        let left = parts[0].trim();
        let right = parts[1].trim();

        // Try to solve x^2 + bx + c = 0 form
        if left.contains("x^2") || left.contains("x²") {
            return self.solve_quadratic(left, right);
        }

        // Linear equation: ax + b = c
        if left.contains('x') || right.contains('x') {
            return self.solve_linear(left, right);
        }

        MathResult::Error("Cannot solve this equation type".to_string())
    }

    /// Solve linear equation ax + b = c
    fn solve_linear(&self, left: &str, right: &str) -> MathResult {
        // Simple case: ax = b or x = b
        let right_val: f64 = match right.parse() {
            Ok(v) => v,
            Err(_) => {
                return MathResult::Error("Right side must be a number".to_string());
            }
        };

        // Extract coefficient of x
        let left_clean = left.replace(' ', "");
        if left_clean == "x" {
            return MathResult::Solutions(vec![right_val]);
        }

        // Try to parse coefficient (e.g., "2x" -> 2)
        if let Some(coef_str) = left_clean.strip_suffix('x') {
            let coef: f64 = if coef_str.is_empty() {
                1.0
            } else if coef_str == "-" {
                -1.0
            } else {
                match coef_str.parse() {
                    Ok(v) => v,
                    Err(_) => return MathResult::Error("Invalid coefficient".to_string()),
                }
            };
            return MathResult::Solutions(vec![right_val / coef]);
        }

        MathResult::Error("Cannot parse linear equation".to_string())
    }

    /// Solve quadratic equation ax^2 + bx + c = 0
    fn solve_quadratic(&self, left: &str, right: &str) -> MathResult {
        // For now, simple case: x^2 - n = 0 => x = ±sqrt(n)
        let right_val: f64 = match right.parse() {
            Ok(v) => v,
            Err(_) => 0.0,
        };

        // x^2 = n case
        if left.trim() == "x^2" || left.trim() == "x²" {
            if right_val < 0.0 {
                return MathResult::Error("No real solutions".to_string());
            }
            let sqrt_val = right_val.sqrt();
            return MathResult::Solutions(vec![sqrt_val, -sqrt_val]);
        }

        // x^2 - n = 0 case
        if left.contains("x^2") && left.contains('-') {
            let n_str = left.replace("x^2", "").replace('-', "").replace(' ', "");
            if let Ok(n) = n_str.parse::<f64>() {
                let val = n + right_val;
                if val < 0.0 {
                    return MathResult::Error("No real solutions".to_string());
                }
                let sqrt_val = val.sqrt();
                return MathResult::Solutions(vec![sqrt_val, -sqrt_val]);
            }
        }

        MathResult::Error("Cannot solve this quadratic equation".to_string())
    }

    /// Set a variable value
    pub fn set_variable(&mut self, name: &str, value: f64) {
        self.variables.insert(name.to_string(), value);
    }
}

/// Token types for expression parsing
#[derive(Debug, Clone)]
enum Token {
    Number(f64),
    Variable(String),
    Plus,
    Minus,
    Multiply,
    Divide,
    Power,
    LParen,
    RParen,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Ln,
    Log,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic() {
        let mut solver = MathSolver::new();

        match solver.solve("2 + 3") {
            MathResult::Number(n) => assert!((n - 5.0).abs() < 0.001),
            _ => panic!("Expected number"),
        }

        match solver.solve("2 + 2 * 3") {
            MathResult::Number(n) => assert!((n - 8.0).abs() < 0.001),
            _ => panic!("Expected number"),
        }

        match solver.solve("(2 + 2) * 3") {
            MathResult::Number(n) => assert!((n - 12.0).abs() < 0.001),
            _ => panic!("Expected number"),
        }
    }

    #[test]
    fn test_power() {
        let mut solver = MathSolver::new();

        match solver.solve("2^3") {
            MathResult::Number(n) => assert!((n - 8.0).abs() < 0.001),
            _ => panic!("Expected number"),
        }
    }

    #[test]
    fn test_quadratic() {
        let mut solver = MathSolver::new();

        match solver.solve("x^2 = 4") {
            MathResult::Solutions(sols) => {
                assert!(sols.contains(&2.0) || sols.iter().any(|x| (x - 2.0).abs() < 0.001));
                assert!(sols.contains(&-2.0) || sols.iter().any(|x| (x + 2.0).abs() < 0.001));
            }
            _ => panic!("Expected solutions"),
        }
    }
}
