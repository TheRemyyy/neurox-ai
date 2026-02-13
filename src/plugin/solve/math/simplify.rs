use super::ast::Expr;

pub struct Simplifier;

impl Simplifier {
    pub fn simplify(expr: &Expr) -> Expr {
        match expr {
            Expr::Add(l, r) => {
                let sl = Self::simplify(l);
                let sr = Self::simplify(r);
                match (sl, sr) {
                    (Expr::Number(a), Expr::Number(b)) => Expr::Number(a + b),
                    (Expr::Number(0.0), x) => x,
                    (x, Expr::Number(0.0)) => x,
                    (x, y) => Expr::Add(Box::new(x), Box::new(y)),
                }
            }
            Expr::Sub(l, r) => {
                let sl = Self::simplify(l);
                let sr = Self::simplify(r);
                match (sl, sr) {
                    (Expr::Number(a), Expr::Number(b)) => Expr::Number(a - b),
                    (x, Expr::Number(0.0)) => x,
                    (x, y) => Expr::Sub(Box::new(x), Box::new(y)),
                }
            }
            Expr::Mul(l, r) => {
                let sl = Self::simplify(l);
                let sr = Self::simplify(r);
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
                let sb = Self::simplify(b);
                let se = Self::simplify(e);
                match (sb, se) {
                    (_, Expr::Number(0.0)) => Expr::Number(1.0),
                    (x, Expr::Number(1.0)) => x,
                    (x, y) => Expr::Pow(Box::new(x), Box::new(y)),
                }
            }
            _ => expr.clone(),
        }
    }
}
