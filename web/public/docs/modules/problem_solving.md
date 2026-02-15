# Cognitive Symbolic Engine

The **Cognitive Symbolic Engine** acts as the "System 2" reasoning cortex for NeuroxAI. While the neural spiking networks handle probabilistic pattern matching and intuition, this engine handles precision, logic, and symbolic manipulation.

It bridges the gap between biological plausibility and the need for exact analytical answers in AGI systems.

## Symbolic Math Engine
The `MathSolver` is not a calculator; it is a recursive **Abstract Syntax Tree (AST)** processor capable of algebraic manipulation.

### Capabilities
*   **Symbolic Differentiation**: Calculates derivatives of complex functions.
    *   *Example*: `diff(x^2 + sin(x), x)` → `2*x + cos(x)`
*   **AST Simplification**: intelligently reduces expressions.
*   **Self-Verification (Reality Check)**: After solving an equation, the engine performs a **back-substitution** (LHS vs RHS check) to prove the result's validity.

## Computational Chemistry
The `ChemistrySolver` integrates a static database of physical properties with linear algebra solvers to simulate chemical reasoning.

### Capabilities
*   **Stoichiometric Balancing**: Automatically calculates coefficients.
*   **Mass Conservation Verification**: After balancing, the engine calculates the exact mass of reactants and products (in g/mol) to verify the **Law of Conservation of Mass**.
*   **Deep Analysis**: Parses molecular formulas to compute molar mass and elemental composition.

## Integration Philosophy
In the human brain, explicit calculation recruits specific cortical regions (like the intraparietal sulcus) distinct from intuitive processing. NeuroxAI simulates this by having the `CognitiveSystem` delegate specific queries (detected via grammar or intent) to this Symbolic Engine, injecting the precise results back into the neural stream as "context".