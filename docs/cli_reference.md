# CLI Reference

NeuroxAI provides a powerful Command Line Interface for interaction, training, and benchmarking.

## Chat Interface (`chat`)

The primary mode for interacting with the cognitive architecture.

```bash
neurox-ai chat [OPTIONS]
```

### Configuration Arguments
*   `--vocab <N>`: Sets the size of the active vocabulary (default: `10000`).
*   `--pattern-dim <N>`: Dimensionality of the hyperdimensional vectors used for semantic representation (default: `512`).
*   `--neurons <N>`: Total number of neurons instantiated in the brain simulation (default: `10000`).
*   `--context <N>`: Size of the sliding context window (working memory capacity) (default: `128`).
*   `--sensitivity <FLOAT>`: Scaling factor for dopamine release, affecting how strongly the system learns from rewards (default: `1.0`).

## Problem Solving (`solve`)

Direct access to symbolic solving modules for mathematics and chemistry.

```bash
neurox-ai solve --problem-type <TYPE> "<PROBLEM>"
```

### Examples

#### Mathematics
Solve algebraic equations or arithmetic expressions:
```bash
neurox-ai solve --problem-type math "2 * x + 5 = 15"
# Output: x = 5
```

#### Chemistry
Balance chemical equations using the stoichiometric solver:
```bash
neurox-ai solve --problem-type chemistry "H2 + O2 -> H2O"
# Output: 2H2 + O2 -> 2H2O

neurox-ai solve --problem-type chemistry "C6H12O6 + O2 -> CO2 + H2O"
# Output: C6H12O6 + 6O2 -> 6CO2 + 6H2O
```

*   **Note**: Use quotes around the problem string to ensure special characters (like `*` or `->`) are handled correctly by your shell.



## Benchmarking (`benchmark`)

Tools for verifying system performance and accuracy.

```bash
neurox-ai benchmark [OPTIONS]
```

*   `--data-dir`: Path to MNIST data or "synthetic" / "auto".
*   `--epochs`: Number of training iterations.
*   `--bits`: Quantization level (4 or 8).
*   `--neurons`: Hidden layer size.
