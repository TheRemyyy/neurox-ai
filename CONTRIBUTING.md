# Contributing to NeuroxAI

Thank you for your interest in contributing to **NeuroxAI**! We are building the world's most biologically plausible neuromorphic brain simulation, and your help is vital to pushing the boundaries of AGI.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We expect all contributors to be respectful, inclusive, and constructive.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/neurox-ai.git
    cd neurox-ai
    ```
3.  **Install dependencies**:
    Ensure you have Rust (stable) and the CUDA Toolkit (12.6+) installed.
    ```bash
    cargo build
    ```
4.  **Create a branch** for your feature or bugfix:
    ```bash
    git checkout -b feature/amazing-plasticity
    ```

## Development Workflow

### Coding Standards

- **Rustfmt**: All code must be formatted using `cargo fmt`.
- **Clippy**: Run `cargo clippy` to catch common mistakes and ensure idiomatic Rust code.
- **Documentation**: Public APIs must be documented using `///` comments. We use `cargo doc` to generate HTML documentation.
- **Biological Accuracy**: When implementing new features, cite relevant neuroscience papers in the code comments where applicable.

### Testing

NeuroxAI relies on a rigorous testing suite to ensure stability across its complex dynamical systems.

- Run all unit tests:
    ```bash
    cargo test
    ```
- Run specific module tests:
    ```bash
    cargo test --package neurox-ai --lib -- learning::stdp
    ```
- Run benchmarks (requires significant time):
    ```bash
    cargo bench
    ```

### GPU Acceleration

If you are working on CUDA kernels (`src/cuda/`), ensure you have a compatible NVIDIA GPU. Tests for CUDA modules will automatically skip if no GPU is detected, but for development, hardware verification is strongly recommended.

## Pull Request Process

1.  Ensure all tests pass locally.
2.  Update `CHANGELOG.md` with your changes under the `[Unreleased]` section.
3.  Open a Pull Request against the `main` branch.
4.  Provide a clear description of the problem you are solving and your biological justification (if applicable).
5.  Wait for code review from the core team.

## Reporting Issues

Please use the GitHub Issue Tracker to report bugs. Include:
- Version of NeuroxAI (or commit hash).
- Operating System and CUDA version.
- A minimal reproduction code snippet or configuration.

## License

By contributing, you agree that your contributions will be licensed under the MIT License, same as the project.
