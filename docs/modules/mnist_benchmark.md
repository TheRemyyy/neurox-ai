# MNIST Benchmark Module

The `MNISTBenchmark` module allows for rigorous testing of the neuromorphic architecture's classification capabilities using the standard MNIST handwritten digit dataset. It supports both real data loading and synthetic data generation for quick prototyping.

## CLI Usage

The benchmark is accessed via the `benchmark` subcommand:

```bash
neurox-ai benchmark [OPTIONS]
```

### Arguments

*   `--data-dir <PATH>`: Directory containing MNIST files. Use `"auto"` to download automatically or `"synthetic"` (default) to generate procedural test data.
*   `--epochs <N>`: Number of training passes (default: `10`).
*   `--bits <4|8>`: Quantization precision for weight compression (default: `4`).
*   `--neurons <N>`: Number of hidden layer neurons (default: `400`).
*   `--duration <MS>`: Stimulus presentation time in milliseconds (default: `50.0`).
*   `--isi <MS>`: Inter-stimulus interval (rest period) in milliseconds (default: `20.0`).

## Synthetic Data Generation

When `data_dir` is set to `"synthetic"`, the system procedurally generates digit-like patterns (circles for '0', lines for '1', etc.) with noise. This allows for:
1.  Testing pipeline integrity without external downloads.
2.  Verifying learning dynamics on controlled patterns.

## Performance Metrics

The benchmark reports:
1.  **FP32 Accuracy**: Baseline accuracy with full floating-point precision.
2.  **Quantized Accuracy**: Accuracy after compressing weights to 4 or 8 bits.
3.  **Compression Ratio**: Memory savings factor (e.g., 8.0x for 4-bit).
4.  **Training Time**: Elapsed time per epoch.

## Architecture

*   **Input**: 784 Poisson spike trains (28x28 pixels).
*   **Hidden**: Configurable (default 400) excitatory neurons with STDP.
*   **Output**: 10 neurons (one per digit) with lateral inhibition (WTA).
*   **Learning**: Reward-modulated STDP with homeostatic regulation.
