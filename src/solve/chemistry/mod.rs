//! Chemistry Solver & Analysis Engine
//!
//! Features:
//! - Stoichiometric balancing
//! - Molar Mass / Composition analysis
//! - Mass Conservation Verification (Reality Check)
//! - Thermodynamics (Enthalpy, Entropy, Gibbs Free Energy)

mod data;

use self::data::{Element, ThermoData};
use std::collections::HashMap;

/// Result of a chemical analysis
#[derive(Debug, Clone)]
pub struct ChemicalAnalysis {
    pub input: String,
    pub balanced_equation: Option<String>,
    pub molar_mass: Option<f64>,
    pub composition: Option<Vec<(String, f64)>>, // Element -> Mass %
    pub steps: Vec<String>, // Reasoning steps
    pub mass_verification: Option<MassConservationCheck>,
    pub thermodynamics: Option<ThermodynamicResult>,
}

#[derive(Debug, Clone)]
pub struct MassConservationCheck {
    pub reactants_mass: f64,
    pub products_mass: f64,
    pub delta: f64,
    pub is_conserved: bool,
}

#[derive(Debug, Clone)]
pub struct ThermodynamicResult {
    pub delta_h: f64, // Enthalpy change (kJ/mol)
    pub delta_s: f64, // Entropy change (J/mol·K)
    pub delta_g: f64, // Gibbs Free Energy (kJ/mol)
    pub is_spontaneous: bool,
    pub is_exothermic: bool,
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
        
        if let Some(thermo) = &self.thermodynamics {
            writeln!(f, "Thermodynamics (at 298K):")?;
            writeln!(f, "  ΔH (Enthalpy): {:.2} kJ/mol ({})", thermo.delta_h, if thermo.is_exothermic { "Exothermic/Releases Heat" } else { "Endothermic/Absorbs Heat" })?;
            writeln!(f, "  ΔG (Gibbs):    {:.2} kJ/mol ({})", thermo.delta_g, if thermo.is_spontaneous { "Spontaneous" } else { "Non-spontaneous" })?;
        }

        if let Some(ver) = &self.mass_verification {
            writeln!(f, "Verification:")?;
            if ver.is_conserved {
                writeln!(f, "  Status: ✅ CONSERVED (Law of Conservation of Mass held)")?;
            } else {
                writeln!(f, "  Status: ❌ VIOLATION (Calculation Error)")?;
            }
        }
        
        Ok(())
    }
}

pub struct ChemistrySolver {
    elements: HashMap<&'static str, Element>,
    thermo_db: HashMap<&'static str, ThermoData>,
}

impl Default for ChemistrySolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ChemistrySolver {
    pub fn new() -> Self {
        let mut elements = HashMap::new();
        for e in data::get_periodic_table() {
            elements.insert(e.symbol, e);
        }

        let thermo_db = data::get_thermo_db();
        
        Self { elements, thermo_db }
    }

    pub fn solve(&self, input: &str) -> ChemicalAnalysis {
        let mut steps = Vec::new();
        steps.push(format!("Analyzing input: '{}'", input));

        if input.contains("->") || input.contains('=') {
            self.solve_equation(input, steps)
        } else {
            self.solve_formula(input, steps)
        }
    }

    fn solve_equation(&self, input: &str, mut steps: Vec<String>) -> ChemicalAnalysis {
        steps.push("Detected chemical equation.".to_string());
        
        match self.balance(input) {
            Ok(b) => {
                steps.push("Balanced equation successfully.".to_string());
                
                // 1. Mass Verification
                let (r_mass, p_mass) = self.calculate_equation_masses(&b);
                let delta = (r_mass - p_mass).abs();
                let is_conserved = delta < 1e-4;
                
                // 2. Thermodynamic Analysis
                let thermo = self.calculate_thermodynamics(&b, &mut steps);

                ChemicalAnalysis {
                    input: input.to_string(),
                    balanced_equation: Some(b.balanced),
                    molar_mass: None,
                    composition: None,
                    steps,
                    mass_verification: Some(MassConservationCheck {
                        reactants_mass: r_mass,
                        products_mass: p_mass,
                        delta,
                        is_conserved
                    }),
                    thermodynamics: thermo,
                }
            },
            Err(e) => {
                steps.push(format!("Failed to balance: {}", e));
                ChemicalAnalysis {
                    input: input.to_string(),
                    balanced_equation: None,
                    molar_mass: None,
                    composition: None,
                    steps,
                    mass_verification: None,
                    thermodynamics: None,
                }
            }
        }
    }

    fn calculate_thermodynamics(&self, eq: &BalancedEquation, steps: &mut Vec<String>) -> Option<ThermodynamicResult> {
        steps.push("Calculating Thermodynamics (Hess's Law)...".to_string());
        
        let mut h_reactants = 0.0;
        let mut s_reactants = 0.0;
        let mut h_products = 0.0;
        let mut s_products = 0.0;
        let mut unknown_compound = false;

        let parts: Vec<&str> = eq.original.split("->").collect();
        let r_strs: Vec<&str> = parts[0].split('+').map(|s| s.trim()).collect();
        let p_strs: Vec<&str> = parts[1].split('+').map(|s| s.trim()).collect();

        // Reactants
        for (i, formula) in r_strs.iter().enumerate() {
            let coef = *eq.reactant_coefficients.get(i).unwrap_or(&1) as f64;
            if let Some(data) = self.thermo_db.get(formula) {
                h_reactants += coef * data.enthalpy;
                s_reactants += coef * data.entropy;
            } else {
                steps.push(format!("  Warning: No thermo data for '{}'", formula));
                unknown_compound = true;
            }
        }

        // Products
        for (i, formula) in p_strs.iter().enumerate() {
            let coef = *eq.product_coefficients.get(i).unwrap_or(&1) as f64;
            if let Some(data) = self.thermo_db.get(formula) {
                h_products += coef * data.enthalpy;
                s_products += coef * data.entropy;
            } else {
                steps.push(format!("  Warning: No thermo data for '{}'", formula));
                unknown_compound = true;
            }
        }

        if unknown_compound {
            steps.push("  Cannot calculate precise thermodynamics due to missing data.".to_string());
            return None;
        }

        let delta_h = h_products - h_reactants;
        let delta_s = s_products - s_reactants; // J/mol K
        
        // Gibbs Free Energy: G = H - TS
        // T = 298.15 K
        // delta_s is in J, delta_h is in kJ. Need to convert S to kJ.
        let t = 298.15;
        let delta_g = delta_h - (t * delta_s / 1000.0);

        steps.push(format!("  ΔH (reaction) = {:.2} kJ/mol", delta_h));
        steps.push(format!("  ΔS (reaction) = {:.2} J/mol·K", delta_s));
        steps.push(format!("  ΔG = ΔH - TΔS = {:.2} kJ/mol", delta_g));

        Some(ThermodynamicResult {
            delta_h,
            delta_s,
            delta_g,
            is_spontaneous: delta_g < 0.0,
            is_exothermic: delta_h < 0.0,
        })
    }

    fn calculate_equation_masses(&self, eq: &BalancedEquation) -> (f64, f64) {
        let mut total_r = 0.0;
        let parts: Vec<&str> = eq.original.split("->").collect();
        if parts.len() < 2 { return (0.0, 0.0); }
        
        let r_strs: Vec<&str> = parts[0].split('+').map(|s| s.trim()).collect();
        let p_strs: Vec<&str> = parts[1].split('+').map(|s| s.trim()).collect();
        
        for (i, formula) in r_strs.iter().enumerate() {
            let coef = eq.reactant_coefficients.get(i).unwrap_or(&1);
            let mass = self.calculate_formula_mass(formula);
            total_r += mass * (*coef as f64);
        }

        let mut total_p = 0.0;
        for (i, formula) in p_strs.iter().enumerate() {
            let coef = eq.product_coefficients.get(i).unwrap_or(&1);
            let mass = self.calculate_formula_mass(formula);
            total_p += mass * (*coef as f64);
        }

        (total_r, total_p)
    }

    fn calculate_formula_mass(&self, formula: &str) -> f64 {
        if let Ok(parsed) = self.parse_molecule(formula) {
            let mut total = 0.0;
            for (sym, count) in parsed {
                if let Some(el) = self.elements.get(sym.as_str()) {
                    total += el.mass * (count as f64);
                }
            }
            total
        } else {
            0.0
        }
    }

    fn solve_formula(&self, formula: &str, mut steps: Vec<String>) -> ChemicalAnalysis {
        steps.push("Detected chemical formula.".to_string());

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
                    mass_verification: None,
                    thermodynamics: None,
                };
            }
        };

        let mut total_mass = 0.0;
        let mut elem_masses = Vec::new();

        for (symbol, count) in &parsed {
            if let Some(element) = self.elements.get(symbol.as_str()) {
                let mass = element.mass * (*count as f64);
                total_mass += mass;
                elem_masses.push((symbol.clone(), mass));
            } 
        }
        
        steps.push(format!("Calculated Molar Mass: {:.3} g/mol", total_mass));

        let mut composition = Vec::new();
        if total_mass > 0.0 {
            for (sym, mass) in elem_masses {
                let pct = (mass / total_mass) * 100.0;
                composition.push((sym, pct));
            }
        }
        composition.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        ChemicalAnalysis {
            input: formula.to_string(),
            balanced_equation: None,
            molar_mass: Some(total_mass),
            composition: Some(composition),
            steps,
            mass_verification: None,
            thermodynamics: None,
        }
    }

    pub fn balance(&self, equation: &str) -> Result<BalancedEquation, String> {
         let separator = if equation.contains("->") { "->" } else { "=" };
        let parts: Vec<&str> = equation.split(separator).collect();
        if parts.len() != 2 { return Err("Invalid format".to_string()); }

        let reactants: Vec<&str> = parts[0].split('+').map(|s| s.trim()).collect();
        let products: Vec<&str> = parts[1].split('+').map(|s| s.trim()).collect();

        let reactant_molecules: Vec<HashMap<String, i32>> = reactants.iter().map(|m| self.parse_molecule(m)).collect::<Result<Vec<_>, _>>()?;
        let product_molecules: Vec<HashMap<String, i32>> = products.iter().map(|m| self.parse_molecule(m)).collect::<Result<Vec<_>, _>>()?;

        let mut elements: Vec<String> = Vec::new();
        for mol in reactant_molecules.iter().chain(product_molecules.iter()) {
            for elem in mol.keys() {
                if !elements.contains(elem) { elements.push(elem.clone()); }
            }
        }
        elements.sort();

        for max_coef in 1..=12 {
            if let Some(coefficients) = self.find_coefficients(&reactant_molecules, &product_molecules, &elements, max_coef) {
                let n_reactants = reactants.len();
                let reactant_coefs: Vec<u32> = coefficients[..n_reactants].to_vec();
                let product_coefs: Vec<u32> = coefficients[n_reactants..].to_vec();
                let balanced = self.format_balanced(&reactants, &products, &reactant_coefs, &product_coefs);
                return Ok(BalancedEquation { original: equation.to_string(), balanced, reactant_coefficients: reactant_coefs, product_coefficients: product_coefs });
            }
        }
        Err("Could not balance equation".to_string())
    }

    fn parse_molecule(&self, molecule: &str) -> Result<HashMap<String, i32>, String> {
        let mut counts: HashMap<String, i32> = HashMap::new();
        let chars: Vec<char> = molecule.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i].is_ascii_uppercase() {
                let mut element = String::new();
                element.push(chars[i]);
                i += 1;
                while i < chars.len() && chars[i].is_ascii_lowercase() { element.push(chars[i]); i += 1; }
                let mut num_str = String::new();
                while i < chars.len() && chars[i].is_ascii_digit() { num_str.push(chars[i]); i += 1; }
                let count: i32 = if num_str.is_empty() { 1 } else { num_str.parse().unwrap_or(1) };
                *counts.entry(element).or_insert(0) += count;
            } else if chars[i] == '(' {
                i += 1;
                let mut group = String::new();
                let mut depth = 1;
                while i < chars.len() && depth > 0 {
                    if chars[i] == '(' { depth += 1; } else if chars[i] == ')' { depth -= 1; if depth == 0 { break; } }
                    group.push(chars[i]); i += 1;
                }
                i += 1; 
                let mut num_str = String::new();
                while i < chars.len() && chars[i].is_ascii_digit() { num_str.push(chars[i]); i += 1; }
                let multiplier: i32 = if num_str.is_empty() { 1 } else { num_str.parse().unwrap_or(1) };
                let group_counts = self.parse_molecule(&group)?;
                for (elem, count) in group_counts { *counts.entry(elem).or_insert(0) += count * multiplier; }
            } else { i += 1; }
        }
        Ok(counts)
    }

    fn find_coefficients(&self, reactants: &[HashMap<String, i32>], products: &[HashMap<String, i32>], elements: &[String], max_coef: u32) -> Option<Vec<u32>> {
        let n_total = reactants.len() + products.len();
        self.try_coefficients(reactants, products, elements, &mut vec![1; n_total], 0, max_coef)
    }

    fn try_coefficients(&self, reactants: &[HashMap<String, i32>], products: &[HashMap<String, i32>], elements: &[String], coefficients: &mut Vec<u32>, pos: usize, max_coef: u32) -> Option<Vec<u32>> {
        if pos == coefficients.len() {
            if self.is_balanced(reactants, products, elements, coefficients) { return Some(coefficients.clone()); }
            return None;
        }
        for c in 1..=max_coef {
            coefficients[pos] = c;
            if let Some(result) = self.try_coefficients(reactants, products, elements, coefficients, pos + 1, max_coef) { return Some(result); }
        }
        None
    }

    fn is_balanced(&self, reactants: &[HashMap<String, i32>], products: &[HashMap<String, i32>], elements: &[String], coefficients: &[u32]) -> bool {
        let n_reactants = reactants.len();
        for element in elements {
            let mut left = 0; let mut right = 0;
            for (i, mol) in reactants.iter().enumerate() { left += mol.get(element).unwrap_or(&0) * coefficients[i] as i32; }
            for (i, mol) in products.iter().enumerate() { right += mol.get(element).unwrap_or(&0) * coefficients[n_reactants + i] as i32; }
            if left != right { return false; }
        }
        true
    }

    fn format_balanced(&self, reactants: &[&str], products: &[&str], r_coefs: &[u32], p_coefs: &[u32]) -> String {
        let left: Vec<String> = reactants.iter().zip(r_coefs).map(|(m, &c)| if c == 1 { m.to_string() } else { format!("{}{}", c, m) }).collect();
        let right: Vec<String> = products.iter().zip(p_coefs).map(|(m, &c)| if c == 1 { m.to_string() } else { format!("{}{}", c, m) }).collect();
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
