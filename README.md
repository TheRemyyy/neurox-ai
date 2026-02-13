<div align="center">

# NeuroxAI

**GPU-Accelerated Neuromorphic Computing Platform**

[![Website](https://img.shields.io/badge/Website-neuroxai.vercel.app-a5b4fc?style=flat-square&logo=vercel)](https://neuroxai.vercel.app/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*Biologically-inspired spiking neural networks targeting 1-10M neurons with real-time performance*

[Overview](#overview) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Performance](#performance) • [Documentation](#documentation)

</div>

---

## <a id="overview"></a>Overview

NeuroxAI is a high-performance neuromorphic computing platform that simulates biologically-realistic spiking neural networks. Built in Rust with CUDA acceleration, it implements state-of-the-art learning algorithms and cognitive architectures inspired by the human brain.

### Key Features

- **🚀 GPU Acceleration** — CUDA-optimized kernels for RTX GPUs, targeting 1-10M neurons
- **🧠 Biological Accuracy** — LIF neurons, STDP learning, realistic synaptic dynamics
- **🔬 Advanced Plasticity** — Triplet STDP, BCM metaplasticity, homeostatic regulation
- **🌊 Neural Oscillations** — Theta-gamma coupling, sleep consolidation cycles
- **💬 Language Processing** — Dual-stream architecture (ventral/dorsal pathways)
- **🗣️ Conversational AI** — Sequence generation, context windows, paraphrase detection
- **🎯 Cognitive Architecture** — Working memory, attention, hippocampal memory systems
- **⚡ Neuromodulation** — Dopamine, serotonin, norepinephrine dynamics
- **📊 Training System** — JSON-based supervised learning with emotional patterns

## Requirements

- **Rust** 1.75 or later
- **CUDA (optional)**: For GPU acceleration, install **CUDA Toolkit** 12.6+ and use an **NVIDIA GPU** (Compute Capability 7.0+, RTX recommended). Without CUDA, build with `--no-default-features` for a CPU-only binary (brain runs, no MNIST benchmark/training).
- **Windows/Linux** (macOS not supported when using CUDA)

## <a id="installation"></a>Installation

```bash
# Clone the repository
git clone https://github.com/TheRemyyy/neurox-ai.git
cd neurox-ai

# Build in release mode (with CUDA, default)
cargo build --release

# Build without CUDA (no NVIDIA stack required)
cargo build --release --no-default-features
```

## <a id="usage"></a>Usage

### System Information

```bash
cargo run -- info
```

Displays GPU capabilities and system configuration.

### Interactive Chat Mode

```bash
cargo run -- chat
```

Starts an interactive session with the neuromorphic brain. Available commands:

| Command | Description |
|---------|-------------|
| `/train <file>` | Train from file (.txt for skip-gram, .json for supervised) |
| `/vocab` | Display learned vocabulary |
| `/sleep` | Run memory consolidation cycle |
| `/debug` | Show internal brain state |
| `/help` | List all commands |
| `exit` | Quit the application |

## Project Structure

```
neurox-ai/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── lib.rs               # Library exports
│   ├── brain/               # Whole-brain architecture
│   ├── cortex/              # Cortical processing (V1, working memory, etc.)
│   ├── cuda/                # GPU kernels and context
│   ├── language/            # Dual-stream language + sequence generation
│   ├── learning/            # STDP, metaplasticity, homeostasis
│   ├── memory/              # Hippocampal memory systems
│   ├── semantics/           # Embeddings, paraphrase detection, intent clustering
│   ├── affect/              # Emotional states and transitions
│   ├── neuron/              # Neuron models (LIF, dendritic, interneurons)
│   ├── neuromodulation/     # Dopamine, serotonin, norepinephrine
│   ├── oscillations/        # Neural rhythms and coupling
│   ├── synapse/             # Synaptic dynamics, vesicle cycles
│   └── ...
├── data/                    # Training data
└── tests/                   # Test suite
```

## <a id="architecture"></a>Architecture Highlights

### Neuron Models

- Leaky Integrate-and-Fire (LIF) with adaptive threshold
- Dendritic compartment models
- Interneuron subtypes (PV, SST, VIP)

### Learning Mechanisms

- Triplet STDP with eligibility traces
- BCM metaplasticity for dynamic thresholds
- Reward-modulated learning (R-STDP)
- Homeostatic synaptic scaling

### Memory Systems

- Hippocampal pattern separation (DG) and completion (CA3/CA1)
- Working memory with capacity limits
- Sleep-dependent consolidation

### Conversational AI

- **Sequence Generator** — Beam search with temperature control
- **Context Window** — Maintains conversation history for coherent dialogue
- **Paraphrase Detection** — Recognizes semantically similar inputs
- **Intent Clustering** — Groups related intents with semantic similarity
- **Emotional Responses** — Sentiment patterns and emotion transitions

## <a id="performance"></a>Performance

Optimized for NVIDIA RTX GPUs with:

- Sparse matrix operations (cuSPARSE)
- Parallel neuron updates
- Efficient spike propagation
- Memory-optimized data layouts

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [**General Overview**](docs/overview.md) — Philosophy and high-level goals
- [**System Architecture**](docs/architecture/system_overview.md) — Detailed data flow and brain loops
- [**Cognitive Modules**](docs/modules/cognitive_upgrades.md) — Theory of Mind, Inner Dialogue, and 2025 upgrades
- [**Changelog**](CHANGELOG.md) — Version history and updates

## Contributing

Contributions are welcome! Whether you're fixing a bug, improving biological accuracy, or optimizing CUDA kernels, please read our [Contributing Guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Inspired by computational neuroscience research and modern deep learning frameworks. Special thanks to the Rust and CUDA communities.

---

<div align="center">
<sub>Built with ❤️ and Rust</sub>
</div>
