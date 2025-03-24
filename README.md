# ViT Topology Analysis

This repository implements topology analysis for Vision Transformers (ViTs) based on the paper "Topology of Deep Neural Networks" (Naitzat et al., 2020). The project examines how ViTs transform the topology of input data through the network layers, comparing different activation functions (GELU vs ReLU).

## Overview

The implementation analyzes the topological properties (Betti numbers β₀, β₁, β₂, β₃) of features at different layers of a Vision Transformer model trained on MNIST for binary classification (digit "0" vs. non-"0"). 

## Features

- Simple Vision Transformer implementation for MNIST
- Binary classification (digit "0" vs. non-"0") with balanced classes
- Feature extraction from both attention and MLP blocks
- Persistent homology computation on three types of pointclouds:
  - CLS token features
  - Central patch (position 4,4) features
  - Aggregated patch features
- Visualization of Betti numbers across network layers

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.5
matplotlib>=3.3.0
scikit-learn>=0.24.0
ripser>=0.6.0
persim>=0.3.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/7pocheR/ViT_topo.git
cd ViT_topo
```

2. Create and activate a conda environment:
```bash
conda create -n topo_env python=3.9
conda activate topo_env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python vit_topology_main.py
```

This will:
1. Prepare a balanced MNIST dataset
2. Train ViT models with GELU and ReLU activations
3. Extract features from different layers
4. Compute persistent homology and Betti numbers
5. Generate visualization plots in the `results/` directory

## Model Architecture

- Image size: 28x28 (MNIST)
- Patch size: 4x4 (resulting in 7x7=49 patches)
- Embedding dimension: 16
- Number of heads: 4
- Transformer blocks: 12
- MLP ratio: 1.5
- Activation functions: GELU and ReLU (separate models)

## Topology Analysis

The analysis calculates persistent homology for different pointclouds extracted from each layer:
- CLS token representations
- Central patch (position 4,4) representations
- Total patches (aggregated patch representations)

For each pointcloud, we compute Betti numbers (β₀, β₁, β₂, β₃) and track their evolution across network layers.

## Results

Results are saved in the `results/` directory:
- JSON files containing Betti numbers for each layer
- Plots showing the evolution of Betti numbers across layers 