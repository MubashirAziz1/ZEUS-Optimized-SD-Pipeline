# ZEUS-Optimized Stable Diffusion Pipeline

A training-free acceleration method for Stable Diffusion that achieves **2-3x speedup** while maintaining image generation quality through intelligent step skipping and second-order prediction.

## Overview

This project implements the ZEUS (Zero-cost Extrapolation-based Unified Sparsity) acceleration framework for Stable Diffusion pipelines. ZEUS is a novel approach that speeds up diffusion model inference by strategically skipping denoiser evaluations and approximating them using a second-order predictor.

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
- **Model Agnostic**: Compatible with Stable Diffusion 1.5, 2.1, SDXL, and FLUX
- **Solver Flexible**: Works with Euler, DPM-Solver++, and other ODE solvers
- **Minimal Integration**: Drop-in replacement requiring minimal code changes
- **Zero Memory Overhead**: O(1) additional memory - no feature caching required

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

### Using pip

```bash
# Clone the repository
git clone https://github.com/MubashirAziz1/ZEUS-Optimized-SD-Pipeline.git
cd ZEUS-Optimized-SD-Pipeline

# Install dependencies
pip install -r requirements.txt
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

## Usage Examples

### Basic Image Generation

```python
from zeus_sd_pipeline.zeus_pipeline import ZeusOptimizedStableDiffusionPipeline

# Initialize pipeline
pipe = ZeusOptimizedStableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Generate image
image = pipe(
    prompt="a serene landscape with mountains and a lake",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

image.save("landscape.png")
```

### Batch Generation

```python
prompts = [
    "a futuristic cityscape at sunset",
    "a magical forest with glowing mushrooms",
    "a vintage car on a desert highway"
]

images = pipe(prompts, num_inference_steps=50).images

for idx, image in enumerate(images):
    image.save(f"output_{idx}.png")
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

## How It Works

ZEUS accelerates diffusion model inference through intelligent step skipping:

### The Core Concept

During diffusion sampling, the model iteratively denoises an image through many steps. Each step requires a computationally expensive forward pass through the denoiser network. ZEUS speeds this up by:

1. **Strategic Step Skipping**: Performing full denoiser evaluations only at selected steps (e.g., every 2nd or 3rd step)
2. **Second-Order Prediction**: Approximating skipped steps using a simple predictor based on the "observed information set"
3. **Interleaved Reuse**: Maintaining stability by alternating between extrapolation and reuse

### The Observed Information Set

At each full evaluation step `t`, ZEUS has access to:
- `ψ_t`: The current denoiser output
- `Δψ_t = ψ_t - ψ_{t+1}`: The backward difference (local trend information)

### Second-Order Predictor

For approximating a skipped step, ZEUS uses the optimal two-point predictor:

```
ψ̂_{t-1} = 2ψ_t - ψ_{t+1}
```

This predictor is:
- **Unbiased** for affine trends
- **Variance-optimal** (BLUE - Best Linear Unbiased Estimator)
- **Second-order accurate** with O(Δ²) bias

### Interleaved Approximation Scheme

For consecutive reduced steps (aggressive acceleration), ZEUS uses an interleaved pattern:

```
ψ̂_{t-j} = {
    2ψ_t - ψ_{t+1}  if j is odd
    ψ_t              if j is even
}
```

This prevents error accumulation while maintaining second-order precision where it matters most.

## Performance

### Speedup vs Quality Tradeoff

ZEUS offers multiple acceleration presets:

- **ZEUS-Medium** (~1.8-2.0x speedup): Best quality preservation, minimal artifacts
- **ZEUS-Fast** (~2.4x speedup): Balanced speed/quality tradeoff
- **ZEUS-Turbo** (~3.2x speedup): Maximum speed with acceptable quality

### Benchmark Results (from paper)

**FLUX.1-dev (Euler solver, 50 steps):**
- ZEUS-Medium: 2.09x speedup, LPIPS 0.047, FID 1.53
- ZEUS-Fast: 2.47x speedup, LPIPS 0.079, FID 2.49
- ZEUS-Turbo: 3.22x speedup, LPIPS 0.171, FID 4.52

**SDXL (DPM-Solver++, 50 steps):**
- ZEUS-Medium: 1.87x speedup, LPIPS 0.084, FID 3.59
- ZEUS-Fast: 1.93x speedup, LPIPS 0.129, FID 5.39

*Lower LPIPS/FID is better. ZEUS maintains competitive quality while significantly reducing sampling time.*

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

## Troubleshooting

### Common Issues

**Issue**: `CUDA out of memory`
- **Solution**: Reduce image resolution or batch size
- Try using `torch.float16` instead of `torch.float32`

**Issue**: ZEUS monitoring not working
- **Solution**: Ensure you have proper NVML/CUDA drivers installed
- Check GPU permissions and access

**Issue**: Slower than expected generation
- **Solution**: First run includes profiling overhead
- Subsequent runs should be faster

## Acknowledgments

- **ZEUS Paper**: "ZEUS: Accelerating Diffusion Models with Only Second-Order Predictor" by Yixiao Wang, Ting Jiang, Zishan Shao, et al. (April 2026)
- **Original Implementation**: [ZEUS GitHub Repository](https://github.com/Ting-Justin-Jiang/ZEUS)
- **Hugging Face Diffusers**: For the excellent Stable Diffusion implementation
- **Duke University**: Research team behind the ZEUS acceleration framework

### Citations

If you use this project in your research, please cite the ZEUS paper:

```bibtex
@article{wang2026zeus,
  title={ZEUS: Accelerating Diffusion Models with Only Second-Order Predictor},
  author={Wang, Yixiao and Jiang, Ting and Shao, Zishan and Ye, Hancheng and 
          Sun, Jingwei and Ma, Mingyuan and Zhang, Jianyi and Chen, Yiran and Li, Hai},
  journal={arXiv preprint arXiv:2604.01552},
  year={2026}
}
```

## License

This project is open-source. Please check the LICENSE file for details.

## Related Projects

- **ZEUS (Original)**: [https://github.com/Ting-Justin-Jiang/ZEUS](https://github.com/Ting-Justin-Jiang/ZEUS) - Original ZEUS acceleration framework
- **Hugging Face Diffusers**: [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers) - State-of-the-art diffusion models
- **DeepCache**: Feature reuse acceleration for diffusion models
- **ToCa**: Token-wise caching for transformer-based diffusion
- **Stable Diffusion**: [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) - Original Stable Diffusion implementation

## Support

- **Issues**: [GitHub Issues](https://github.com/MubashirAziz1/ZEUS-Optimized-SD-Pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MubashirAziz1/ZEUS-Optimized-SD-Pipeline/discussions)
- **ZEUS Paper**: [arXiv:2604.01552](https://arxiv.org/abs/2604.01552)
- **Original ZEUS**: [https://github.com/Ting-Justin-Jiang/ZEUS](https://github.com/Ting-Justin-Jiang/ZEUS)

## Roadmap

- [ ] Support for additional Stable Diffusion models (SDXL Turbo, Lightning)
- [ ] Integration with LoRA and ControlNet
- [ ] Adaptive step selection based on image complexity
- [ ] Video generation support (AnimateDiff, I2VGen-XL)
- [ ] Multi-GPU inference optimization
- [ ] Command-line interface for batch processing
- [ ] Quality-speed tradeoff presets for different use cases
- [ ] Comprehensive benchmarking suite with multiple metrics
- [ ] Integration with popular SD web interfaces (Automatic1111, ComfyUI)

---

**Made with ⚡ by [MubashirAziz1](https://github.com/MubashirAziz1)**

*Making diffusion models faster, one skip at a time.*
