//! Chemistry Equation Solver
//!
//! Balances chemical equations using linear algebra approach.

use std::collections::HashMap;

/// A balanced chemical equation
#[derive(Debug, Clone)]
pub struct BalancedEquation {
    /// Original equation string
    pub original: String,
    /// Balanced equation string
    pub balanced: String,
    /// Coefficients for reactants
    pub reactant_coefficients: Vec<u32>,
    /// Coefficients for products
    pub product_coefficients: Vec<u32>,
}

impl std::fmt::Display for BalancedEquation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.balanced)
    }
}

/// Chemistry problem solver
pub struct ChemistrySolver;

impl Default for ChemistrySolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ChemistrySolver {
    pub fn new() -> Self {
        Self
    }

    /// Balance a chemical equation
    pub fn balance(&self, equation: &str) -> Result<BalancedEquation, String> {
        // Parse equation: "H2 + O2 -> H2O" or "H2 + O2 = H2O"
        let separator = if equation.contains("->") {
            "->"
        } else if equation.contains('=') {
            "="
        } else {
            return Err(
                "Invalid equation format. Use '->' or '=' to separate reactants and products."
                    .to_string(),
            );
        };

        let parts: Vec<&str> = equation.split(separator).collect();
        if parts.len() != 2 {
            return Err("Equation must have exactly one arrow/equals sign".to_string());
        }

        let reactants: Vec<&str> = parts[0].split('+').map(|s| s.trim()).collect();
        let products: Vec<&str> = parts[1].split('+').map(|s| s.trim()).collect();

        // Parse molecules to get element counts
        let reactant_molecules: Vec<HashMap<String, i32>> = reactants
            .iter()
            .map(|m| self.parse_molecule(m))
            .collect::<Result<Vec<_>, _>>()?;

        let product_molecules: Vec<HashMap<String, i32>> = products
            .iter()
            .map(|m| self.parse_molecule(m))
            .collect::<Result<Vec<_>, _>>()?;

        // Get all elements
        let mut elements: Vec<String> = Vec::new();
        for mol in reactant_molecules.iter().chain(product_molecules.iter()) {
            for elem in mol.keys() {
                if !elements.contains(elem) {
                    elements.push(elem.clone());
                }
            }
        }
        elements.sort();

        // Try coefficients from 1 to 10 for each molecule
        let n_reactants = reactants.len();
        let n_products = products.len();
        let n_molecules = n_reactants + n_products;

        // Brute force small coefficients (good enough for simple equations)
        for max_coef in 1..=10 {
            if let Some(coefficients) =
                self.find_coefficients(&reactant_molecules, &product_molecules, &elements, max_coef)
            {
                let reactant_coefs: Vec<u32> = coefficients[..n_reactants].to_vec();
                let product_coefs: Vec<u32> = coefficients[n_reactants..].to_vec();

                // Build balanced equation string
                let balanced =
                    self.format_balanced(&reactants, &products, &reactant_coefs, &product_coefs);

                return Ok(BalancedEquation {
                    original: equation.to_string(),
                    balanced,
                    reactant_coefficients: reactant_coefs,
                    product_coefficients: product_coefs,
                });
            }
        }

        Err("Could not balance equation with small coefficients".to_string())
    }

    /// Parse a molecule like "H2O" or "Ca(OH)2" into element counts
    fn parse_molecule(&self, molecule: &str) -> Result<HashMap<String, i32>, String> {
        let mut counts: HashMap<String, i32> = HashMap::new();
        let chars: Vec<char> = molecule.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i].is_ascii_uppercase() {
                let mut element = String::new();
                element.push(chars[i]);
                i += 1;

                // Check for lowercase continuation (e.g., "Ca")
                while i < chars.len() && chars[i].is_ascii_lowercase() {
                    element.push(chars[i]);
                    i += 1;
                }

                // Check for number
                let mut num_str = String::new();
                while i < chars.len() && chars[i].is_ascii_digit() {
                    num_str.push(chars[i]);
                    i += 1;
                }

                let count: i32 = if num_str.is_empty() {
                    1
                } else {
                    num_str.parse().unwrap_or(1)
                };

                *counts.entry(element).or_insert(0) += count;
            } else if chars[i] == '(' {
                // Handle groups like (OH)2
                i += 1;
                let mut group = String::new();
                let mut depth = 1;
                while i < chars.len() && depth > 0 {
                    if chars[i] == '(' {
                        depth += 1;
                    } else if chars[i] == ')' {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    group.push(chars[i]);
                    i += 1;
                }
                i += 1; // Skip closing )

                // Get multiplier
                let mut num_str = String::new();
                while i < chars.len() && chars[i].is_ascii_digit() {
                    num_str.push(chars[i]);
                    i += 1;
                }
                let multiplier: i32 = if num_str.is_empty() {
                    1
                } else {
                    num_str.parse().unwrap_or(1)
                };

                // Recursively parse group
                let group_counts = self.parse_molecule(&group)?;
                for (elem, count) in group_counts {
                    *counts.entry(elem).or_insert(0) += count * multiplier;
                }
            } else {
                i += 1;
            }
        }

        Ok(counts)
    }

    /// Find coefficients that balance the equation
    fn find_coefficients(
        &self,
        reactants: &[HashMap<String, i32>],
        products: &[HashMap<String, i32>],
        elements: &[String],
        max_coef: u32,
    ) -> Option<Vec<u32>> {
        let n_reactants = reactants.len();
        let n_products = products.len();
        let n_total = n_reactants + n_products;

        // Generate all combinations of coefficients
        self.try_coefficients(
            reactants,
            products,
            elements,
            &mut vec![1; n_total],
            0,
            max_coef,
        )
    }

    fn try_coefficients(
        &self,
        reactants: &[HashMap<String, i32>],
        products: &[HashMap<String, i32>],
        elements: &[String],
        coefficients: &mut Vec<u32>,
        pos: usize,
        max_coef: u32,
    ) -> Option<Vec<u32>> {
        if pos == coefficients.len() {
            // Check if balanced
            if self.is_balanced(reactants, products, elements, coefficients) {
                return Some(coefficients.clone());
            }
            return None;
        }

        for c in 1..=max_coef {
            coefficients[pos] = c;
            if let Some(result) = self.try_coefficients(
                reactants,
                products,
                elements,
                coefficients,
                pos + 1,
                max_coef,
            ) {
                return Some(result);
            }
        }

        None
    }

    fn is_balanced(
        &self,
        reactants: &[HashMap<String, i32>],
        products: &[HashMap<String, i32>],
        elements: &[String],
        coefficients: &[u32],
    ) -> bool {
        let n_reactants = reactants.len();

        for element in elements {
            let mut left_count: i32 = 0;
            let mut right_count: i32 = 0;

            // Sum reactants
            for (i, mol) in reactants.iter().enumerate() {
                left_count += mol.get(element).unwrap_or(&0) * coefficients[i] as i32;
            }

            // Sum products
            for (i, mol) in products.iter().enumerate() {
                right_count +=
                    mol.get(element).unwrap_or(&0) * coefficients[n_reactants + i] as i32;
            }

            if left_count != right_count {
                return false;
            }
        }

        true
    }

    fn format_balanced(
        &self,
        reactants: &[&str],
        products: &[&str],
        reactant_coefs: &[u32],
        product_coefs: &[u32],
    ) -> String {
        let left: Vec<String> = reactants
            .iter()
            .zip(reactant_coefs.iter())
            .map(|(mol, &coef)| {
                if coef == 1 {
                    mol.to_string()
                } else {
                    format!("{}{}", coef, mol)
                }
            })
            .collect();

        let right: Vec<String> = products
            .iter()
            .zip(product_coefs.iter())
            .map(|(mol, &coef)| {
                if coef == 1 {
                    mol.to_string()
                } else {
                    format!("{}{}", coef, mol)
                }
            })
            .collect();

        format!("{} -> {}", left.join(" + "), right.join(" + "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_molecule() {
        let solver = ChemistrySolver::new();

        let h2o = solver.parse_molecule("H2O").unwrap();
        assert_eq!(h2o.get("H"), Some(&2));
        assert_eq!(h2o.get("O"), Some(&1));

        let ca_oh_2 = solver.parse_molecule("Ca(OH)2").unwrap();
        assert_eq!(ca_oh_2.get("Ca"), Some(&1));
        assert_eq!(ca_oh_2.get("O"), Some(&2));
        assert_eq!(ca_oh_2.get("H"), Some(&2));
    }

    #[test]
    fn test_balance_water() {
        let solver = ChemistrySolver::new();

        let result = solver.balance("H2 + O2 -> H2O").unwrap();
        assert_eq!(result.balanced, "2H2 + O2 -> 2H2O");
    }

    #[test]
    fn test_balance_combustion() {
        let solver = ChemistrySolver::new();

        let result = solver.balance("CH4 + O2 -> CO2 + H2O").unwrap();
        // CH4 + 2O2 -> CO2 + 2H2O
        assert!(result.balanced.contains("CH4"));
        assert!(result.balanced.contains("CO2"));
    }
}
