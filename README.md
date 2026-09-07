# Gaussian VCR: 3D Gaussian Splatting Model Compression

A lightweight compression framework for 3D Gaussian Splatting (3DGS) based on View Contribution Rate (VCR) and Cross-view Contribution Volatility (CVF) joint decision pruning.

## Features

- **Joint Pruning Strategy**: Combines VCR and CVF metrics for intelligent redundancy identification
- **Multiple Pruning Modes**: Support VCR-only, CVF-only, and joint decision modes
- **K-means Compression**: Codebook-based quantization for further model compression
- **Comprehensive Evaluation**: Built-in metrics (PSNR, SSIM, LPIPS) and benchmark tools

## Installation

```bash
git clone https://github.com/EkiniasY/Gaussian_VCR.git
cd Gaussian_VCR
pip install -r requirements.txt

# Build CUDA extensions (required)
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

### Dependencies

- Python 3.7+
- PyTorch 1.12+
- CUDA 11.6+

### Note on Submodules

The `submodules/` directory contains **modified versions** of `diff-gaussian-rasterization` and `simple-knn` from the original 3DGS implementation. These have been extended with custom CUDA kernels (reduced 3dgs, variable SH bands, etc.) to support the VCR pruning method. They are included directly in this repository rather than as git submodules because the upstream versions are **not compatible** with this project.

## Usage

### Training

```bash
# Baseline (no pruning)
python train.py -s <dataset_path> --eval -m <output_path>

# VCR-only pruning
python train.py -s <dataset_path> --eval -m <output_path> \
    --mercy_points --prune_dead_points --store_grads \
    --lambda_alpha_regul=0.001 --mercy_type=ecr_only

# CVF-only pruning
python train.py -s <dataset_path> --eval -m <output_path> \
    --mercy_points --prune_dead_points --store_grads \
    --lambda_alpha_regul=0.001 --mercy_type=view_stability_only

# Joint decision (VCR + CVF)
python train.py -s <dataset_path> --eval -m <output_path> \
    --mercy_points --prune_dead_points --store_grads \
    --lambda_alpha_regul=0.001 --mercy_type=ecr_view_stability
```

### Rendering

```bash
python render.py -s <dataset_path> -m <model_path> --models baseline
```

### Evaluation

```bash
# Quality metrics (PSNR, SSIM, LPIPS)
python metrics.py -m <model_path>

# Performance benchmark (FPS)
python benchmark.py -s <dataset_path> -m <model_path> --models baseline
```

### Model Compression

```bash
# K-means clustering compression
python compress.py -p <ply_file_path> --half_float --store_codebook_path <output_dir>
```

### Full Evaluation Pipeline

```bash
python full_eval.py -s <dataset_path> -m <model_path>
```

## Project Structure

```
├── train.py              # Training script
├── render.py             # Rendering script
├── metrics.py            # Quality evaluation
├── benchmark.py          # Performance benchmark
├── compress.py           # Model compression
├── convert.py            # Format conversion
├── full_eval.py          # Full evaluation pipeline
├── arguments/            # Parameter definitions
├── gaussian_renderer/    # Rendering implementation
├── scene/                # Scene and camera handling
├── utils/                # Utility functions
├── lpipsPyTorch/         # LPIPS metric implementation
└── submodules/           # Modified CUDA extensions (see Note on Submodules)
    ├── diff-gaussian-rasterization/  # Custom rasterizer with reduced 3dgs support
    └── simple-knn/                   # Modified KNN implementation
```

## Datasets

Supported datasets:
- Mip-NeRF 360 (indoor/outdoor)
- Tanks and Temples
- Deep Blending

Dataset structure:
```
<dataset_name>/
├── <scene>_input/
│   ├── images/
│   └── sparse/ or cameras.bin
```

## Results

| Scene | Method | Gaussians | PSNR | SSIM | LPIPS | FPS |
|-------|--------|-----------|------|------|-------|-----|
| bicycle | Baseline | 6.07M | 25.18 | 0.764 | 0.212 | 85.1 |
| bicycle | Joint | 5.12M | 25.16 | 0.764 | 0.212 | 95.6 |
| room | Baseline | 1.46M | 32.02 | 0.939 | 0.116 | 261.9 |
| room | Joint | 0.92M | 32.04 | 0.940 | 0.117 | 302.6 |

## Citation

If you find this work useful, please cite:

```bibtex
@article{gaussian-vcr,
  title={Gaussian VCR: Efficient 3D Gaussian Splatting Compression via View Contribution Rate},
  author={EkiniasY},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Original implementation
- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) - Base CUDA rasterizer
- [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn) - Base KNN implementation
- [INRIA GRAPHDECO Team](https://team.inria.fr/graphdeco)
