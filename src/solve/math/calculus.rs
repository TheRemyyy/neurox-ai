use super::ast::Expr;

pub struct Calculus;

impl Calculus {
    pub fn differentiate(expr: &Expr, var: &str) -> Expr {
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
                Box::new(Self::differentiate(l, var)),
                Box::new(Self::differentiate(r, var)),
            ),
            Expr::Sub(l, r) => Expr::Sub(
                Box::new(Self::differentiate(l, var)),
                Box::new(Self::differentiate(r, var)),
            ),
            Expr::Mul(l, r) => {
                // Product rule: (uv)' = u'v + uv'
                let u = l.as_ref();
                let v = r.as_ref();
                let du = Self::differentiate(u, var);
                let dv = Self::differentiate(v, var);
                
                Expr::Add(
                    Box::new(Expr::Mul(Box::new(du), Box::new(v.clone()))),
                    Box::new(Expr::Mul(Box::new(u.clone()), Box::new(dv))),
                )
            }
            Expr::Div(u, v) => {
                // Quotient rule: (u/v)' = (u'v - uv') / v^2
                let du = Self::differentiate(u, var);
                let dv = Self::differentiate(v, var);
                
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
                    let du = Self::differentiate(base, var);
                    
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
                let du = Self::differentiate(arg, var);
                Expr::Mul(Box::new(Expr::Cos(arg.clone())), Box::new(du))
            }
            Expr::Cos(arg) => {
                let du = Self::differentiate(arg, var);
                Expr::Mul(
                    Box::new(Expr::Number(-1.0)),
                    Box::new(Expr::Mul(Box::new(Expr::Sin(arg.clone())), Box::new(du)))
                )
            }
            Expr::Ln(arg) => {
                let du = Self::differentiate(arg, var);
                Expr::Div(Box::new(du), arg.clone())
            }
            Expr::Exp(arg) => {
                let du = Self::differentiate(arg, var);
                Expr::Mul(Box::new(Expr::Exp(arg.clone())), Box::new(du))
            }
        }
    }

    pub fn integrate(expr: &Expr, var: &str) -> Expr {
        match expr {
            Expr::Number(n) => Expr::Mul(Box::new(Expr::Number(*n)), Box::new(Expr::Variable(var.to_string()))), // int c dx = cx
            Expr::Variable(v) => {
                if v == var {
                    // int x dx = x^2/2
                    Expr::Div(
                        Box::new(Expr::Pow(Box::new(Expr::Variable(v.clone())), Box::new(Expr::Number(2.0)))),
                        Box::new(Expr::Number(2.0))
                    )
                } else {
                    // int y dx = yx
                    Expr::Mul(Box::new(Expr::Variable(v.clone())), Box::new(Expr::Variable(var.to_string())))
                }
            },
            Expr::Add(l, r) => Expr::Add(Box::new(Self::integrate(l, var)), Box::new(Self::integrate(r, var))),
            Expr::Sub(l, r) => Expr::Sub(Box::new(Self::integrate(l, var)), Box::new(Self::integrate(r, var))),
            Expr::Mul(l, r) => {
                // Constant multiple rule: int c*f(x) = c * int f(x)
                if let Expr::Number(c) = **l {
                    return Expr::Mul(Box::new(Expr::Number(c)), Box::new(Self::integrate(r, var)));
                }
                if let Expr::Number(c) = **r {
                    return Expr::Mul(Box::new(Expr::Number(c)), Box::new(Self::integrate(l, var)));
                }
                Expr::Variable("Integral(Complex)".to_string()) 
            },
            Expr::Pow(b, e) => {
                // Power rule: int x^n = x^(n+1)/(n+1)
                if let Expr::Variable(v) = &**b {
                    if v == var {
                        if let Expr::Number(n) = **e {
                            if n == -1.0 {
                                return Expr::Ln(Box::new(Expr::Variable(var.to_string())));
                            }
                            let new_exp = n + 1.0;
                            return Expr::Div(
                                Box::new(Expr::Pow(Box::new(Expr::Variable(v.clone())), Box::new(Expr::Number(new_exp)))),
                                Box::new(Expr::Number(new_exp))
                            );
                        }
                    }
                }
                Expr::Variable("Integral(ComplexPower)".to_string())
            },
            Expr::Sin(arg) => {
                 // int sin(x) = -cos(x)
                 if let Expr::Variable(v) = &**arg {
                     if v == var {
                         return Expr::Mul(Box::new(Expr::Number(-1.0)), Box::new(Expr::Cos(Box::new(Expr::Variable(var.to_string())))));
                     }
                 }
                 Expr::Variable("Integral(SinComplex)".to_string())
            },
            Expr::Cos(arg) => {
                 // int cos(x) = sin(x)
                 if let Expr::Variable(v) = &**arg {
                     if v == var {
                         return Expr::Sin(Box::new(Expr::Variable(var.to_string())));
                     }
                 }
                 Expr::Variable("Integral(CosComplex)".to_string())
            },
            Expr::Exp(arg) => {
                if let Expr::Variable(v) = &**arg {
                    if v == var {
                        return Expr::Exp(arg.clone());
                    }
                }
                 Expr::Variable("Integral(ExpComplex)".to_string())
            }
            _ => Expr::Variable("Integral(Unknown)".to_string()),
        }
    }
}
