# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Docs**: Comprehensive documentation structure in `docs/`.
- **Docs**: Contribution guidelines.

## [0.1.0] - 2025-12-29

### Genesis
- Initial release of **NeuroxAI**, a biologically-inspired whole-brain simulation.

### Core Architecture
- **Hierarchical Cortex**: 5-level predictive coding hierarchy (V1 -> PFC).
- **Subcortical Systems**:
    - **Basal Ganglia**: Reinforcement learning with Go/NoGo pathways.
    - **Thalamus**: Sensory gating and attentional routing.
    - **Hippocampus**: Episodic memory with theta-gamma coupling.
    - **Amygdala**: Emotional processing and fear conditioning.
    - **Cerebellum**: Motor error correction.
- **Cognitive Modules (2025 Upgrades)**:
    - `TheoryOfMind`: Agent modeling and perspective taking.
    - `InnerDialogue`: Multi-perspective reasoning engine.
    - `Metacognition`: System 2 monitoring of thought processes.
    - `AbstractReasoning`: Analogical inference engine.

### Learning Mechanisms
- **STDP**: Spike-Timing-Dependent Plasticity (Hebbian learning).
- **R-STDP**: Reward-modulated plasticity via Dopamine.
- **Homeostasis**: BCM theory and synaptic scaling.
- **Structural Plasticity**: Dynamic synapse formation and pruning.
- **Sleep Consolidation**: Offline memory replay and optimization.

### Performance
- **CUDA Acceleration**:
    - `GpuV1OrientationSystem`: 100x speedup for visual processing.
    - `GpuMotionSystem`: 80x speedup for optic flow.
    - `GpuCognitiveSystem`: Accelerated attention and working memory.
- **Sparse Computing**: Optimized sparse matrix operations for synaptic connectivity.

### Interface
- **CLI**: Interactive terminal interface with `rustyline`.
- **Serialization**: Full save/load state support via Serde.
