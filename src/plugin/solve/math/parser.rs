use super::ast::Expr;

pub struct Parser;

impl Parser {
    pub fn parse(input: &str) -> Result<Expr, String> {
        let clean = input.replace(' ', "");
        Self::parse_recursive(&clean)
    }

    fn parse_recursive(input: &str) -> Result<Expr, String> {
        // 1. Check for Add/Sub (lowest precedence)
        if let Some(idx) = Self::find_operator(input, &['+', '-']) {
             let left = Self::parse_recursive(&input[..idx])?;
             let right = Self::parse_recursive(&input[idx+1..])?;
             return match &input[idx..idx+1] {
                 "+" => Ok(Expr::Add(Box::new(left), Box::new(right))),
                 "-" => Ok(Expr::Sub(Box::new(left), Box::new(right))),
                 _ => unreachable!(),
             };
        }

        // 2. Check for Mul/Div
        if let Some(idx) = Self::find_operator(input, &['*', '/']) {
            let left = Self::parse_recursive(&input[..idx])?;
            let right = Self::parse_recursive(&input[idx+1..])?;
            return match &input[idx..idx+1] {
                "*" => Ok(Expr::Mul(Box::new(left), Box::new(right))),
                "/" => Ok(Expr::Div(Box::new(left), Box::new(right))),
                _ => unreachable!(),
            };
        }
        
        // 3. Power
        if let Some(idx) = Self::find_operator(input, &['^']) {
             let left = Self::parse_recursive(&input[..idx])?;
             let right = Self::parse_recursive(&input[idx+1..])?;
             return Ok(Expr::Pow(Box::new(left), Box::new(right)));
         }

        // 4. Functions & Parentheses
        if input.ends_with(')') {
            if input.starts_with("sin(") {
                let inner = &input[4..input.len()-1];
                return Ok(Expr::Sin(Box::new(Self::parse_recursive(inner)?)));
            }
            if input.starts_with("cos(") {
                let inner = &input[4..input.len()-1];
                return Ok(Expr::Cos(Box::new(Self::parse_recursive(inner)?)));
            }
            if input.starts_with("ln(") {
                let inner = &input[3..input.len()-1];
                return Ok(Expr::Ln(Box::new(Self::parse_recursive(inner)?)));
            }
            if input.starts_with("exp(") {
                let inner = &input[4..input.len()-1];
                return Ok(Expr::Exp(Box::new(Self::parse_recursive(inner)?)));
            }
            // Just parentheses (expr)
            if input.starts_with('(') {
                if Self::is_fully_enclosed(input) {
                    return Self::parse_recursive(&input[1..input.len()-1]);
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

    fn is_fully_enclosed(input: &str) -> bool {
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

    fn find_operator(input: &str, ops: &[char]) -> Option<usize> {
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
