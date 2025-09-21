# Cheaper-NeRF: Cost-Efficient Novel-View Synthesis 🚀


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/cheaper-nerf/blob/main/notebooks/demo.ipynb)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)



[![Watch the video](demo.gif)](https://drive.google.com/file/d/1f57Vnr41bB_-e3sRkI51_NlNVQQb_o7s/view?resourcekey)



## 🎯 Revolutionary Cost Reduction in 3D Scene Synthesis

Cheaper-NeRF is a modified Neural Radiance Fields architecture that achieves **~40% faster training** and **~60% memory reduction** while maintaining comparable visual quality. By implementing strategic data reduction techniques and optimizing the sampling process, we make NeRF technology accessible for real-time applications and resource-constrained environments.

> **TL;DR**: Train NeRF models 40% faster with minimal quality loss through intelligent sampling and zero-density filtering.

## 📊 Performance Comparison

<p align="center">
  <img src="assets/performance_chart.png" alt="Performance Metrics" width="600"/>
</p>

| Metric | Original NeRF | Cheaper-NeRF | Improvement |
|--------|--------------|--------------|-------------|
| **Training Time** | 10 hours | 6 hours | **40% faster** ⚡ |
| **Memory Usage** | 16GB | 6.4GB | **60% less** 💾 |
| **Samples/Ray** | 192 | 48 (effective: 24) | **87.5% reduction** 📉 |
| **PSNR** | 34 dB | 32 dB | -2 dB (acceptable) ✅ |
| **SSIM** | 0.95 | 0.94 | Maintained 📊 |

## 🌟 Key Innovations

### 1. **Mean Sampling Reduction** 
Combines every 4 sampled points through averaging, reducing network evaluations by 75%:
```python
p_combined = (1/4) × Σ(p_i) for i=1 to 4
```

### 2. **Zero-Density Filtering**
Intelligently removes points with σ ≈ 0 that don't contribute to the final image

### 3. **Optimized Network Architecture**
- Original: 8 layers × 256 channels
- Cheaper-NeRF: 6 layers × 128 channels
- **Result**: 75% parameter reduction

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/cheaper-nerf.git
cd cheaper-nerf

# Create conda environment
conda env create -f environment.yml
conda activate cheaper_nerf

# Download example data
bash scripts/download_example_data.sh

# Run training with Cheaper-NeRF optimizations
python scripts/train.py --config configs/config_cheaper.yaml
```

### 🎮 Try it in 30 seconds!
```bash
# Quick demo with pre-trained model
python scripts/demo.py --scene lego --model_path pretrained/lego_cheaper.ckpt
```

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture_diagram.png" alt="Cheaper-NeRF Architecture" width="700"/>
</p>

Cheaper-NeRF employs a sophisticated optimization pipeline:

### Phase 1: Intelligent Sampling
1. **Coarse Sampling** - 32 initial samples (reduced from 64)
2. **Mean Reduction** - Combine 4 points → 1 effective point
3. **Zero Filtering** - Remove non-contributive samples

### Phase 2: Efficient Rendering
1. **Hierarchical Sampling** - Focus computation on surfaces
2. **Optimized Networks** - Smaller, faster MLPs
3. **Volume Rendering** - With sparse optimization

## 💻 System Requirements

### Minimum Requirements
- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **RAM**: 16GB
- **Storage**: 10GB free space
- **OS**: Ubuntu 20.04 / Windows 10

### Recommended Setup
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **RAM**: 32GB
- **Storage**: 50GB SSD
- **OS**: Ubuntu 22.04

## 📁 Project Structure

```
cheaper-nerf/
│
├── 📂 cheaper_nerf/          # Core implementation
│   ├── encoding.py           # Positional encoding with caching
│   ├── network.py            # Optimized neural networks
│   ├── sampling.py           # KEY: Mean sampling & filtering
│   ├── rendering.py          # Volume rendering optimizations
│   └── trainer.py            # Training pipeline
│
├── 📂 scripts/               # Executable scripts
│   ├── train.py             # Main training script
│   ├── evaluate.py          # Evaluation metrics
│   └── benchmark.py         # Performance comparison
│
├── 📂 configs/              # Configuration files
│   ├── config_cheaper.yaml  # Cheaper-NeRF settings
│   └── config_baseline.yaml # Original NeRF settings
│
└── 📂 notebooks/            # Interactive demos
    └── demo.ipynb           # Colab-ready demonstration
```

## 🎯 Usage Examples

### Basic Training
```bash
# Train on synthetic Lego scene
python scripts/train.py \
    --datadir data/nerf_synthetic/lego \
    --expname cheaper_lego \
    --N_iters 200000
```

### Benchmark Mode
```bash
# Compare performance against vanilla NeRF
python scripts/benchmark.py --scene lego --compare_baseline
```

### Custom Configuration
```bash
# Fine-tune optimization parameters
python scripts/train.py \
    --sampling_reduction 4 \        # Combine 4 points
    --density_threshold 1e-10 \     # Filter threshold
    --netdepth 6 --netwidth 128 \   # Network size
    --N_samples 32 --N_importance 64 # Sampling counts
```

## 📊 Results Gallery

<table>
  <tr>
    <td align="center"><b>Scene</b></td>
    <td align="center"><b>Original NeRF</b></td>
    <td align="center"><b>Cheaper-NeRF</b></td>
    <td align="center"><b>Time Saved</b></td>
  </tr>
  <tr>
    <td align="center">Lego</td>
    <td align="center"><img src="assets/lego_original.gif" width="200"/></td>
    <td align="center"><img src="assets/lego_cheaper.gif" width="200"/></td>
    <td align="center">4 hours</td>
  </tr>
  <tr>
    <td align="center">Fern</td>
    <td align="center"><img src="assets/fern_original.gif" width="200"/></td>
    <td align="center"><img src="assets/fern_cheaper.gif" width="200"/></td>
    <td align="center">6 hours</td>
  </tr>
</table>

## 🔬 Technical Details

### Mean Sampling Algorithm
```python
def mean_sample_reduction(pts, reduction_factor=4):
    """
    Reduces sampling density by combining neighboring points.
    Mathematical: p_combined = (1/N) * Σ p_i
    """
    N = pts.shape[-2]
    pts_grouped = pts.reshape([..., N//reduction_factor, reduction_factor, 3])
    return pts_grouped.mean(axis=-2)
```

### Zero-Density Filtering
```python
def filter_zero_density(pts, sigma, threshold=1e-10):
    """
    Removes points that don't contribute to final image.
    Points with σ ≈ 0 have no visual impact.
    """
    mask = sigma > threshold
    return pts[mask], sigma[mask]
```

## 🚀 Deployment

### For Real-time Applications (Mobile/Edge)
```bash
# Use lightweight model variant
python scripts/train.py --model_type lightweight --netwidth 64
```

### For Production (Cloud)
```bash
# Docker deployment
docker build -t cheaper-nerf .
docker run -p 8080:8080 cheaper-nerf
```

## 📈 Performance Metrics

<p align="center">
  <img src="assets/metrics_comparison.png" alt="Detailed Metrics" width="700"/>
</p>

## 🛠️ Advanced Features

- **Adaptive Sampling**: Dynamically adjusts reduction based on scene complexity
- **Gradient-Aware Filtering**: Preserves high-frequency details
- **Multi-GPU Support**: Scale to multiple GPUs for faster training
- **Model Quantization**: Further reduce memory for edge deployment

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/yourusername/cheaper-nerf.git
cd cheaper-nerf
pip install -e ".[dev]"
pre-commit install
```

## 📚 Citation

If you find Cheaper-NeRF useful in your research, please cite:

```bibtex
@inproceedings{zhang2024cheapernerf,
  title={Cheaper-NeRF: A Cost-Efficient Approach for Novel-View Synthesis},
  author={Zhang, Zhen and Zhao, Ziyu},
  institution={University of Rochester},
  year={2024}
}
```

## 🏆 Acknowledgments

This work builds upon the original [NeRF](https://github.com/bmild/nerf) by Mildenhall et al. We thank:
- The original NeRF authors for their groundbreaking work
- The PyTorch NeRF community for inspiration
- NVIDIA for GPU support through academic programs

## 📊 Comparison with Other Methods

| Method | Training Time | Memory | PSNR | Year |
|--------|--------------|--------|------|------|
| NeRF | 10h | 16GB | 34.0 | 2020 |
| FastNeRF | 8h | 14GB | 33.5 | 2021 |
| InstantNGP | 0.1h | 12GB | 33.0 | 2022 |
| **Cheaper-NeRF** | **6h** | **6.4GB** | **32.0** | **2024** |

## 🎓 Authors

<table>
  <tr>
    <td align="center">
      <img src="assets/author1.jpg" width="100px;" alt=""/><br />
      <sub><b>Zhen Zhang</b></sub><br />
      <sub>University of Rochester</sub>
    </td>
    <td align="center">
      <img src="assets/author2.jpg" width="100px;" alt=""/><br />
      <sub><b>Ziyu Zhao</b></sub><br />
      <sub>University of Rochester</sub>
    </td>
  </tr>
</table>

## 📬 Contact

- **Email**: zzh131@u.rochester.edu, zzhao57@u.rochester.edu
- **Issues**: [GitHub Issues](https://github.com/yourusername/cheaper-nerf/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/cheaper-nerf/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Making NeRF accessible to everyone, one optimization at a time 🚀</b><br>
  <sub>Built with ❤️ at University of Rochester</sub>
</p>

<p align="center">
  <a href="#cheaper-nerf-cost-efficient-novel-view-synthesis-">Back to top ⬆️</a>
</p>
