# Problem Solving Modules

NeuroxAI includes dedicated modules for symbolic and algorithmic problem solving, bridging the gap between neural heuristics and precise logical operations.

## Math Solver
The `MathSolver` module (`src/solve/math.rs`) provides symbolic and numeric computation capabilities. It allows the system to handle explicit mathematical queries that require precision beyond probabilistic estimation.

### Features
*   **Numeric Computation**: Precise floating-point arithmetic.
*   **Symbolic Algebra**: Handling of variables and simple equations (e.g., "x + 5 = 10").
*   **Multiple Solutions**: Support for quadratic equations or systems yielding multiple roots.

### Usage
```rust
let solver = MathSolver::new();
// Solves arithmetic or algebraic expressions
let result = solver.solve("2 * x + 4 = 12"); 
```

## Chemistry Solver
The `ChemistrySolver` module (`src/solve/chemistry.rs`) implements a linear algebra-based approach to balancing chemical equations. This serves as a foundational component for scientific reasoning capabilities.

### Features
*   **Equation Balancing**: Automatically calculates stoichiometric coefficients.
*   **Format Support**: Accepts standard notation (e.g., "H2 + O2 -> H2O").
*   **Validation**: Ensures mass conservation.

### Usage
```rust
let solver = ChemistrySolver::new();
let balanced = solver.balance("C6H12O6 + O2 -> CO2 + H2O");
// Output: C6H12O6 + 6O2 -> 6CO2 + 6H2O
```

## Integration with Cognition
These solvers are designed to be triggered by the `CognitiveSystem` when specific "needs" or "queries" are detected in the inner dialogue stream that require high-precision answers, functioning similarly to tool-use in LLMs but integrated into the neuromorphic loop.
