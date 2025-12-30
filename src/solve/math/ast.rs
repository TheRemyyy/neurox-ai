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
    Exp(Box<Expr>), // e^x
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
            Expr::Exp(e) => write!(f, "exp({})", e),
        }
    }
}
