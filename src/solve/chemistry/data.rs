use std::collections::HashMap;

/// Chemical Element Data
#[derive(Debug, Clone)]
pub struct Element {
    pub number: u32,
    pub symbol: &'static str,
    pub name: &'static str,
    pub mass: f64, // g/mol
}

/// Thermodynamic Data for Compounds (at 298K, 1 atm)
#[derive(Debug, Clone)]
pub struct ThermoData {
    pub enthalpy: f64, // ΔHf° (kJ/mol) - Heat of formation
    pub entropy: f64,  // S° (J/mol·K) - Standard molar entropy
}

/// Get the full periodic table
pub fn get_periodic_table() -> Vec<Element> {
    vec![
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
        // ... (can be extended further)
    ]
}

/// Get thermodynamic database
pub fn get_thermo_db() -> HashMap<&'static str, ThermoData> {
    let mut db = HashMap::new();
    // Elements (Standard State = 0)
    db.insert("H2", ThermoData { enthalpy: 0.0, entropy: 130.68 });
    db.insert("O2", ThermoData { enthalpy: 0.0, entropy: 205.15 });
    db.insert("N2", ThermoData { enthalpy: 0.0, entropy: 191.61 });
    db.insert("C", ThermoData { enthalpy: 0.0, entropy: 5.74 }); // Graphite
    db.insert("Fe", ThermoData { enthalpy: 0.0, entropy: 27.28 });

    // Compounds
    db.insert("H2O", ThermoData { enthalpy: -241.8, entropy: 188.8 }); // Gas
    db.insert("CO2", ThermoData { enthalpy: -393.5, entropy: 213.8 });
    db.insert("CH4", ThermoData { enthalpy: -74.8, entropy: 186.3 }); // Methane
    db.insert("C2H6", ThermoData { enthalpy: -84.0, entropy: 229.6 }); // Ethane
    db.insert("C3H8", ThermoData { enthalpy: -103.8, entropy: 269.9 }); // Propane
    db.insert("C4H10", ThermoData { enthalpy: -125.7, entropy: 310.2 }); // Butane
    db.insert("NH3", ThermoData { enthalpy: -45.9, entropy: 192.5 });
    db.insert("NO2", ThermoData { enthalpy: 33.2, entropy: 240.1 });
    db.insert("SO2", ThermoData { enthalpy: -296.8, entropy: 248.2 });
    db.insert("NaCl", ThermoData { enthalpy: -411.2, entropy: 72.1 });
    
    db
}
