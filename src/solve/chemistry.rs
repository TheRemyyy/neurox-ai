//! Chemistry Solver & Analysis Engine
//!
//! Provides stoichiometric balancing, molar mass calculations, and elemental analysis.
//! Uses a built-in periodic table for physical properties.

use std::collections::HashMap;

/// Chemical Element Data
#[derive(Debug, Clone)]
pub struct Element {
    pub number: u32,
    pub symbol: &'static str,
    pub name: &'static str,
    pub mass: f64, // g/mol
}

/// Static Periodic Table Data (Common Elements)
const PERIODIC_TABLE: &[Element] = &[
    Element { number: 1, symbol: "H", name: "Hydrogen", mass: 1.008 },
    Element { number: 2, symbol: "He", name: "Helium", mass: 4.0026 },
    Element { number: 3, symbol: "Li", name: "Lithium", mass: 6.94 },
    Element { number: 4, symbol: "Be", name: "Beryllium", mass: 9.0122 },
    Element { number: 5, symbol: "B", name: "Boron", mass: 10.81 },
    Element { number: 6, symbol: "C", name: "Carbon", mass: 12.011 },
    Element { number: 7, symbol: "N", name: "Nitrogen", mass: 14.007 },
    Element { number: 8, symbol: "O", name: "Oxygen", mass: 15.999 },
    Element { number: 9, symbol: "F", name: "Fluorine", mass: 18.998 },
    Element { number: 10, symbol: "Ne", name: "Neon", mass: 20.180 },
    Element { number: 11, symbol: "Na", name: "Sodium", mass: 22.990 },
    Element { number: 12, symbol: "Mg", name: "Magnesium", mass: 24.305 },
    Element { number: 13, symbol: "Al", name: "Aluminium", mass: 26.982 },
    Element { number: 14, symbol: "Si", name: "Silicon", mass: 28.085 },
    Element { number: 15, symbol: "P", name: "Phosphorus", mass: 30.974 },
    Element { number: 16, symbol: "S", name: "Sulfur", mass: 32.06 },
    Element { number: 17, symbol: "Cl", name: "Chlorine", mass: 35.45 },
    Element { number: 18, symbol: "Ar", name: "Argon", mass: 39.948 },
    Element { number: 19, symbol: "K", name: "Potassium", mass: 39.098 },
    Element { number: 20, symbol: "Ca", name: "Calcium", mass: 40.078 },
    Element { number: 26, symbol: "Fe", name: "Iron", mass: 55.845 },
    Element { number: 29, symbol: "Cu", name: "Copper", mass: 63.546 },
    Element { number: 30, symbol: "Zn", name: "Zinc", mass: 65.38 },
    Element { number: 35, symbol: "Br", name: "Bromine", mass: 79.904 },
    Element { number: 47, symbol: "Ag", name: "Silver", mass: 107.87 },
    Element { number: 53, symbol: "I", name: "Iodine", mass: 126.90 },
    Element { number: 79, symbol: "Au", name: "Gold", mass: 196.97 },
    Element { number: 82, symbol: "Pb", name: "Lead", mass: 207.2 },
    Element { number: 92, symbol: "U", name: "Uranium", mass: 238.03 },
];

/// Result of a chemical analysis
#[derive(Debug, Clone)]
pub struct ChemicalAnalysis {
    pub input: String,
    pub balanced_equation: Option<String>,
    pub molar_mass: Option<f64>,
    pub composition: Option<Vec<(String, f64)>>, // Element -> Mass %
    pub steps: Vec<String>, // Reasoning steps
}

impl std::fmt::Display for ChemicalAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Analysis for: {}", self.input)?;
        if let Some(eq) = &self.balanced_equation {
            writeln!(f, "Balanced: {}", eq)?;
        }
        if let Some(mass) = self.molar_mass {
            writeln!(f, "Molar Mass: {:.4} g/mol", mass)?;
        }
        if let Some(comp) = &self.composition {
            writeln!(f, "Composition:")?;
            for (elem, pct) in comp {
                writeln!(f, "  - {}: {:.2}%", elem, pct)?;
            }
        }
        Ok(())
    }
}

pub struct ChemistrySolver {
    elements: HashMap<&'static str, Element>,
}

impl Default for ChemistrySolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ChemistrySolver {
    pub fn new() -> Self {
        let mut elements = HashMap::new();
        for e in PERIODIC_TABLE {
            elements.insert(e.symbol, e.clone());
        }
        Self { elements }
    }

    /// Entry point for solving any chemistry problem
    pub fn solve(&self, input: &str) -> ChemicalAnalysis {
        let mut steps = Vec::new();
        steps.push(format!("Analyzing input: '{}'", input));

        // Detect type: Equation vs Formula
        if input.contains("->") || input.contains('=') {
            self.solve_equation(input, steps)
        } else {
            self.solve_formula(input, steps)
        }
    }

    fn solve_equation(&self, input: &str, mut steps: Vec<String>) -> ChemicalAnalysis {
        steps.push("Detected chemical equation. Attempting to balance...".to_string());
        
        let balanced = match self.balance(input) {
            Ok(b) => {
                steps.push("Successfully balanced equation.".to_string());
                Some(b.balanced)
            },
            Err(e) => {
                steps.push(format!("Failed to balance: {}", e));
                None
            }
        };

        ChemicalAnalysis {
            input: input.to_string(),
            balanced_equation: balanced,
            molar_mass: None,
            composition: None,
            steps,
        }
    }

    fn solve_formula(&self, formula: &str, mut steps: Vec<String>) -> ChemicalAnalysis {
        steps.push("Detected chemical formula. Calculating properties...".to_string());

        let parsed = match self.parse_molecule(formula) {
            Ok(p) => p,
            Err(e) => {
                steps.push(format!("Error parsing formula: {}", e));
                return ChemicalAnalysis {
                    input: formula.to_string(),
                    balanced_equation: None,
                    molar_mass: None,
                    composition: None,
                    steps,
                };
            }
        };

        // Calculate Molar Mass
        let mut total_mass = 0.0;
        let mut elem_masses = Vec::new();

        for (symbol, count) in &parsed {
            if let Some(element) = self.elements.get(symbol.as_str()) {
                let mass = element.mass * (*count as f64);
                total_mass += mass;
                elem_masses.push((symbol.clone(), mass));
                steps.push(format!("  - {}: {} atoms * {:.3} g/mol = {:.3} g/mol", 
                    element.name, count, element.mass, mass));
            } else {
                steps.push(format!("  - WARNING: Unknown element symbol '{}'", symbol));
            }
        }
        
        steps.push(format!("Total Molar Mass: {:.3} g/mol", total_mass));

        // Calculate Composition %
        let mut composition = Vec::new();
        if total_mass > 0.0 {
            for (sym, mass) in elem_masses {
                let pct = (mass / total_mass) * 100.0;
                composition.push((sym, pct));
            }
        }
        // Sort by percentage descending
        composition.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        ChemicalAnalysis {
            input: formula.to_string(),
            balanced_equation: None,
            molar_mass: Some(total_mass),
            composition: Some(composition),
            steps,
        }
    }

    /// Balance a chemical equation
    pub fn balance(&self, equation: &str) -> Result<BalancedEquation, String> {
         // Parse equation: "H2 + O2 -> H2O" or "H2 + O2 = H2O"
         let separator = if equation.contains("->") {
            "->"
        } else if equation.contains('=') {
            "="
        } else {
            return Err("Invalid equation format".to_string());
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

        // Matrix solver / Brute force approach
        // We stick to brute force for small integers as it's robust for simple cases
        for max_coef in 1..=12 {
            if let Some(coefficients) =
                self.find_coefficients(&reactant_molecules, &product_molecules, &elements, max_coef)
            {
                let n_reactants = reactants.len();
                let reactant_coefs: Vec<u32> = coefficients[..n_reactants].to_vec();
                let product_coefs: Vec<u32> = coefficients[n_reactants..].to_vec();

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

        Err("Could not balance equation (too complex or invalid)".to_string())
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

                // Check for lowercase continuation
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

                let group_counts = self.parse_molecule(&group)?;
                for (elem, count) in group_counts {
                    *counts.entry(elem).or_insert(0) += count * multiplier;
                }
            } else {
                i += 1; // Skip other chars
            }
        }

        Ok(counts)
    }

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

            for (i, mol) in reactants.iter().enumerate() {
                left_count += mol.get(element).unwrap_or(&0) * coefficients[i] as i32;
            }

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

/// A balanced chemical equation
#[derive(Debug, Clone)]
pub struct BalancedEquation {
    pub original: String,
    pub balanced: String,
    pub reactant_coefficients: Vec<u32>,
    pub product_coefficients: Vec<u32>,
}

impl std::fmt::Display for BalancedEquation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.balanced)
    }
}