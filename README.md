# Attention-Free Diffusion for Sequence Classification

A novel approach to sequence classification that replaces self-attention with iterative diffusion-based information propagation, achieving competitive accuracy with dramatically reduced memory usage.

## Potential applications
Edge devices with low memory footprint - robotics, IoT and mobile applications.

## Overview

This repository implements an attention-free diffusion model for text classification that:
- Eliminates quadratic memory complexity of self-attention
- Uses iterative neighbor-based information propagation
- Achieves 30x lower memory usage compared to transformer baselines
- Maintains competitive performance on text classification tasks

## Key Results

| Dataset | Our Accuracy | Baseline | Memory Reduction | Speed Improvement |
|---------|--------------|----------|------------------|-------------------|
| AG News | 90.07% | 98.63% (DistilBERT) | 30x | 6x |
| IMDB | 86.30% | ~94-96% | 30x+ | 6x |
| LRA ListOps | 17.8% | ~35-60% | 10-20x | Variable |

## Installation

```bash
git clone https://github.com/pbanavara/attention-free-diffusion
cd attention-free-diffusion
pip install torch torchvision torchaudio datasets transformers scikit-learn
```

## Quick Start

### Basic Training
```python
from attention_free_diffusion_improved import run_experiment

# Run on IMDB dataset
results, model, log_dir = run_experiment(
    dataset_name='imdb',
    batch_size=32,
    epochs=10,
    experiment_name="imdb_baseline"
)
```

### LRA Benchmark
```python
# Run on LRA ListOps
results, model, log_dir = run_experiment(
    dataset_name='lra_listops',
    batch_size=16,
    epochs=20,
    data_dir="/path/to/lra_data",
    experiment_name="listops_baseline"
)
```

## Model Architecture

The diffusion model consists of:
1. **Token Embedding** with Gaussian noise injection
2. **Multi-head Neighbor Projections** for local interactions
3. **Iterative Diffusion Refinement** over T steps
4. **Layer Normalization** for training stability
5. **Mean Pooling** for sequence aggregation
6. **Classification Head** for final predictions

Core diffusion equation:
```
h_i^(t+1) = α * h_i^(t) + (1-α) * Σ W_ij * f(h_j^(t))
```

## Configuration

Key hyperparameters:
- `embed_dim`: Hidden dimension (default: 256)
- `num_iters`: Number of diffusion steps (default: 4)
- `alpha`: Decay factor controlling information retention (default: 0.5-0.7)
- `lr`: Learning rate (2e-5 for FP16, 5e-5 for FP32)
- `max_length`: Maximum sequence length (default: 4096)

## Memory Optimization

The implementation includes several optimizations:
- **FP16 mixed precision training** with gradient scaling
- **Efficient neighbor computation** using circular shifts
- **Memory-aware batch sizing** with automatic scaling
- **Gradient clipping** for numerical stability

## Dataset Support

Currently supports:
- **AG News**: Topic classification (4 classes)
- **IMDB**: Sentiment analysis (binary)
- **LRA ListOps**: Mathematical reasoning benchmark
- Custom datasets via the extensible data loading framework

## Experimental Results

### Memory Efficiency
- AG News: 673MB vs 21GB (DistilBERT) - 30x reduction
- IMDB: 0.14GB for batch_size=16 on 4K sequences
- LRA ListOps: 0.14GB for 2K token mathematical expressions

### Performance Analysis
- **Strong on text classification**: 90% AG News, 86% IMDB
- **Limited on reasoning tasks**: 17.8% ListOps (vs 35-60% transformers)
- **Consistent across configurations**: Minimal overfitting observed

## Files Structure

```
├── attention_free_diffusion_improved.py  # Main training pipeline
├── lra_dataloader.py                     # LRA benchmark data loading
├── logs/                                 # Experiment logs and checkpoints
├── paper/                                # Research paper and results
└── README.md                            # This file
```

## Limitations

The current architecture shows:
- **Performance ceiling** on complex compositional reasoning tasks
- **Neighborhood-based propagation** may be insufficient for hierarchical structures
- **Limited long-range dependencies** compared to global attention mechanisms

## Citation

```bibtex
@article{banavara2025diffusion,
  title={Diffusion is all you need: Attention-Free Sequence Classification},
  author={Banavara, Pradeep},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open issues for bugs or feature requests.

## Contact

Pradeep Banavara - pbanavara@gmail.com

## Acknowledgments

Built on PyTorch with inspiration from diffusion-based generative models and efficient transformer architectures.
