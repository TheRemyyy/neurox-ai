//! Solve Command Handler

use crate::plugin::{ChemistrySolver, MathSolver};

// CLI Colors
const COLOR_RESET: &str = "\x1b[0m";
const COLOR_WHITE: &str = "\x1b[37m";
const COLOR_GRAY: &str = "\x1b[90m";
const COLOR_LIGHT_BLUE: &str = "\x1b[94m";
const BOLD: &str = "\x1b[1m";

/// Run the solve command
pub fn run(problem_type: &str, problem: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}╔════════════════════════════════════════════════════════════╗",
        COLOR_LIGHT_BLUE
    );
    println!("║  NeuroxAI Cognitive Symbolic Engine                        ║");
    println!(
        "╚════════════════════════════════════════════════════════════╝{}",
        COLOR_RESET
    );
    println!();

    match problem_type.to_lowercase().as_str() {
        "math" => {
            let mut solver = MathSolver::new();
            let analysis = solver.solve(problem);
            println!("{}Analysis Report:{}", BOLD, COLOR_RESET);
            println!("{}", analysis);

            println!("{}Reasoning Trace:{}", COLOR_GRAY, COLOR_RESET);
            for step in analysis.steps {
                println!("  › {}", step);
            }
        }
        "chemistry" | "chem" => {
            let solver = ChemistrySolver::new();
            let analysis = solver.solve(problem);

            println!("{}Analysis Report:{}", BOLD, COLOR_RESET);
            println!("{}", analysis);

            println!("{}Reasoning Trace:{}", COLOR_GRAY, COLOR_RESET);
            for step in analysis.steps {
                println!("  › {}", step);
            }
        }
        _ => {
            println!(
                "{}Unknown problem type: {}. Use 'math' or 'chemistry'.{}",
                COLOR_WHITE, problem_type, COLOR_RESET
            );
        }
    }

    Ok(())
}
