# ZEUS-Optimized Stable Diffusion Pipeline

An energy-efficient implementation of Stable Diffusion that leverages the ZEUS optimization framework to reduce GPU energy consumption while maintaining image generation quality.

## Overview

This project integrates the [ZEUS energy optimization framework](https://ml.energy/zeus/) with Hugging Face's Diffusers library to create an energy-aware Stable Diffusion pipeline. ZEUS automatically optimizes GPU power consumption during inference, making AI image generation more sustainable and cost-effective.

### What is ZEUS?

ZEUS is an open-source framework developed at the University of Michigan for measuring and optimizing the energy consumption of deep learning workloads. It intelligently navigates the tradeoff between energy efficiency and performance by:

- **GPU Power Limit Optimization**: Automatically finding optimal power limits that minimize energy while maintaining performance
- **Real-time Energy Monitoring**: Tracking energy consumption with minimal overhead
- **Adaptive Optimization**: Adjusting configurations based on workload characteristics

Key benefits include:
- 15-75% reduction in energy consumption for various DNN workloads
- No offline profiling required
- Minimal performance overhead
- Compatibility with various GPU architectures

## Features

- **Energy-Efficient Image Generation**: Reduced power consumption without sacrificing image quality
- **Drop-in Replacement**: Compatible with standard Stable Diffusion pipelines
- **Automatic Optimization**: ZEUS handles energy optimization automatically
- **Production Ready**: Built on top of proven libraries (Diffusers, PyTorch, Transformers)
- **GPU Monitoring**: Real-time energy and power consumption tracking

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

The ZEUS-Optimized Stable Diffusion Pipeline wraps the standard Diffusers pipeline with ZEUS's energy optimization capabilities:

1. **Power Monitoring**: ZEUS monitors GPU power consumption in real-time
2. **Dynamic Optimization**: Automatically adjusts GPU power limits to find the optimal energy-performance tradeoff
3. **Transparent Integration**: Works seamlessly with existing Stable Diffusion models without code changes
4. **Minimal Overhead**: Energy optimization adds negligible computational overhead

## Energy Savings

Expected energy savings vary based on:
- GPU model and architecture
- Image resolution and generation parameters
- Number of inference steps
- Batch size

Typical savings range from **15% to 75%** compared to standard implementations, with minimal impact on generation quality or speed.

## Performance Considerations

- **First Run**: Initial optimization may take slightly longer as ZEUS profiles the workload
- **Subsequent Runs**: Optimization becomes more efficient with repeated use
- **Batch Processing**: Higher energy efficiency with larger batch sizes
- **GPU Compatibility**: Best results on modern NVIDIA GPUs (Ampere, Ada Lovelace architectures)

## Benchmarking

To measure energy consumption and compare with standard pipelines:

```python
from zeus_sd_pipeline.zeus_pipeline import ZeusOptimizedStableDiffusionPipeline
from zeus.monitor import ZeusMonitor

# Initialize monitor
monitor = ZeusMonitor(gpu_indices=[0])

# Start monitoring
monitor.begin_window("sd_generation")

# Generate image
pipe = ZeusOptimizedStableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
).to("cuda")
image = pipe("a beautiful sunset").images[0]

# Get measurements
measurement = monitor.end_window("sd_generation")
print(f"Energy consumed: {measurement.total_energy} J")
print(f"Time elapsed: {measurement.time} s")
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

- **ZEUS Framework**: [ML.ENERGY Initiative](https://ml.energy/zeus/) for the energy optimization framework
- **Hugging Face Diffusers**: For the excellent Stable Diffusion implementation
- **Research Paper**: *Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training* (NSDI '23)

### Citations

If you use this project in your research, please consider citing both the ZEUS framework and Stable Diffusion:

```bibtex
@inproceedings{zeus-nsdi23,
  title = {Zeus: Understanding and Optimizing {GPU} Energy Consumption of {DNN} Training},
  author = {Jie You and Jae-Won Chung and Mosharaf Chowdhury},
  booktitle = {USENIX NSDI},
  year = {2023}
}
```

## License

This project is open-source. Please check the LICENSE file for details.

## Related Projects

- [ZEUS](https://github.com/ml-energy/zeus) - The energy optimization framework
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - State-of-the-art diffusion models
- [ML.ENERGY Leaderboard](https://ml.energy/leaderboard) - Benchmark for LLM energy consumption
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Original Stable Diffusion implementation

## Support

- **Issues**: [GitHub Issues](https://github.com/MubashirAziz1/ZEUS-Optimized-SD-Pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MubashirAziz1/ZEUS-Optimized-SD-Pipeline/discussions)
- **ZEUS Documentation**: [ml.energy/zeus](https://ml.energy/zeus/)

## Roadmap

- [ ] Support for additional Stable Diffusion models (SD 2.0, SDXL)
- [ ] Integration with LoRA and ControlNet
- [ ] Energy consumption dashboard and visualization
- [ ] Multi-GPU optimization support
- [ ] Command-line interface for quick generation
- [ ] Pre-configured optimization profiles for different use cases
- [ ] Comprehensive benchmarking suite

---

**Made with ⚡ by [MubashirAziz1](https://github.com/MubashirAziz1)**

*Reducing AI's carbon footprint, one image at a time.*