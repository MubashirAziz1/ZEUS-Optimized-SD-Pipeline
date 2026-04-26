# ZEUS-Optimized Stable Diffusion Pipeline

A training-free acceleration method for Stable Diffusion that achieves **2-3x speedup** while maintaining image generation quality through intelligent step skipping and second-order prediction.

## Overview

This repo creates the ZEUS (Zero-cost Extrapolation-based Unified Sparsity) optimized Stable Diffusion Pipeline. ZEUS is a novel approach that speeds up diffusion model inference by strategically skipping denoiser evaluations and approximating them using a second-order predictor.

### What is ZEUS?

ZEUS is a training-free acceleration method introduced in the paper ["ZEUS: Accelerating Diffusion Models with Only Second-Order Predictor"](https://arxiv.org/abs/2604.01552) (April 2026). It addresses the inference latency bottleneck in diffusion models through:

- **Second-Order Prediction**: Uses a simple yet optimal predictor that extrapolates skipped steps from the observed information set
- **Interleaved Skipping Scheme**: Maintains stability during aggressive acceleration by reusing observed information
- **Zero Overhead**: Requires no architectural modifications, feature caching, or additional memory

Key benefits:
- **2-3x faster inference** with minimal quality degradation
- Works across different backbones (U-Net, DiT, MMDiT)
- Compatible with multiple prediction objectives (ε, v, flow-matching)
- Fewer than 20 lines of code to integrate
- No training required

## Features

- **Fast Inference**: 2-3x speedup over standard diffusion sampling
- **Training-Free**: Works with off-the-shelf pretrained models without fine-tuning
- **Quality Preservation**: Maintains perceptual quality (LPIPS, FID) at aggressive speedups

## Requirements

- Python >= 3.12
- CUDA-compatible GPU
- Dependencies:
  - `torch >= 2.11.0`
  - `diffusers >= 0.37.1`
  - `transformers >= 5.6.2`
  - `accelerate >= 1.13.0`
  - `torchvision >= 0.26.0`

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/MubashirAziz1/ZEUS-Optimized-SD-Pipeline.git
cd ZEUS-Optimized-SD-Pipeline

# Install dependencies with uv
uv sync
```


## Quick Start

```python
import torch
from torchvision.utils import save_image
from zeus_sd_pipeline.zeus_pipeline import ZeusOptimizedStableDiffusionPipeline

# Load the optimized pipeline
pipe = ZeusOptimizedStableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Generate an image
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

# Save the result
image.save("output.png")
```



### Advanced Configuration

```python
# Fine-tune generation parameters
image = pipe(
    prompt="a portrait of a cyberpunk character",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=75,
    guidance_scale=8.0,
    width=768,
    height=768,
    seed=42
).images[0]
```

## Project Structure

```
ZEUS-Optimized-SD-Pipeline/
├── zeus_sd_pipeline/          # Main package directory
│   ├── __init__.py
│   ├── zeus_pipeline.py       # ZEUS-optimized SD pipeline implementation
│   └── ...                    # Additional modules
├── main.py                    # Example usage script
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Dependency lock file
├── .python-version           # Python version specification
├── .gitignore               # Git ignore rules
└── README.md                # This file
```


## Benchmarking

To measure speedup and quality metrics:

```python
import time
from zeus_sd_pipeline.zeus_pipeline import ZeusOptimizedStableDiffusionPipeline
from diffusers import StableDiffusionPipeline
import torch

prompt = "a serene mountain landscape at sunset"

# Baseline (full evaluation)
baseline_pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

start = time.time()
baseline_image = baseline_pipe(prompt, num_inference_steps=50).images[0]
baseline_time = time.time() - start

# ZEUS-accelerated
zeus_pipe = ZeusOptimizedStableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

start = time.time()
zeus_image = zeus_pipe(prompt, num_inference_steps=50).images[0]
zeus_time = time.time() - start

print(f"Baseline: {baseline_time:.2f}s")
print(f"ZEUS: {zeus_time:.2f}s")
print(f"Speedup: {baseline_time/zeus_time:.2f}x")

# Compare images using LPIPS, FID, etc.
```

## Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Open an issue describing the bug and steps to reproduce
2. **Suggest Features**: Share ideas for new features or improvements
3. **Submit PRs**: Fork the repo, make changes, and submit a pull request
4. **Improve Documentation**: Help make the docs clearer and more comprehensive

### Development Setup

```bash
# Clone the repository
git clone https://github.com/MubashirAziz1/ZEUS-Optimized-SD-Pipeline.git
cd ZEUS-Optimized-SD-Pipeline

# Install in development mode
pip install -e .

# Run tests (if available)
pytest tests/
```

## Acknowledgments

- **ZEUS Paper**: "ZEUS: Accelerating Diffusion Models with Only Second-Order Predictor" by Yixiao Wang, Ting Jiang, Zishan Shao, et al. (April 2026)
- **Original Implementation**: [ZEUS GitHub Repository](https://github.com/Ting-Justin-Jiang/ZEUS)
- **Hugging Face Diffusers**: For the excellent Stable Diffusion implementation



**Made with ⚡ by [MubashirAziz1](https://github.com/MubashirAziz1)**

*Making diffusion models faster, one skip at a time.*
