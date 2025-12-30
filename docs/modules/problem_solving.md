# Cognitive Symbolic Engine

The **Cognitive Symbolic Engine** acts as the "System 2" reasoning cortex for NeuroxAI. While the neural spiking networks handle probabilistic pattern matching and intuition, this engine handles precision, logic, and symbolic manipulation.

It bridges the gap between biological plausibility and the need for exact analytical answers in AGI systems.

## 📐 Symbolic Math Engine
The `MathSolver` is not a calculator; it is a recursive **Abstract Syntax Tree (AST)** processor capable of algebraic manipulation.

### Capabilities
*   **Symbolic Differentiation**: Calculates derivatives of complex functions.
    *   *Example*: `diff(x^2 + sin(x), x)` → `2*x + cos(x)`
*   **AST Simplification**: intelligently reduces expressions.
    *   *Example*: `x * 1 + 0` → `x`
*   **Variable Context**: Maintains a persistent state of variables (`x = 5`, `y = 10`).
*   **Supported Operations**:
    *   Arithmetic: `+`, `-`, `*`, `/`, `^`
    *   Trigonometry: `sin`, `cos`, `tan`
    *   Calculus: `diff`, `d/dx`
    *   Logarithms: `ln`, `log`

### Architecture
Input strings are tokenized and parsed into a recursive enum structure (`Expr`). Operations like differentiation are implemented as recursive tree transformations, not numerical approximations.

## 🧪 Computational Chemistry
The `ChemistrySolver` integrates a static database of physical properties with linear algebra solvers to simulate chemical reasoning.

### Capabilities
*   **Stoichiometric Balancing**: Uses matrix operations to balance reaction equations.
    *   *Input*: `C3H8 + O2 -> CO2 + H2O`
    *   *Output*: `C3H8 + 5O2 -> 3CO2 + 4H2O`
*   **Deep Analysis**: Parses molecular formulas to compute physical properties.
    *   **Molar Mass**: Calculates exact weight based on atomic composition (e.g., H2SO4 = 98.07 g/mol).
    *   **Elemental Composition**: Returns mass percentage breakdowns (e.g., "Oxygen: 65.2%").
*   **Reasoning Trace**: Generates a step-by-step log of its deduction process, allowing the cognitive system to "explain" its work.

## Integration Philosophy
In the human brain, explicit calculation recruits specific cortical regions (like the intraparietal sulcus) distinct from intuitive processing. NeuroxAI simulates this by having the `CognitiveSystem` delegate specific queries (detected via grammar or intent) to this Symbolic Engine, injecting the precise results back into the neural stream as "context".