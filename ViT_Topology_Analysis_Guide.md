# Vision Transformer Topology Analysis Guide

This guide provides instructions for using the `vit_topology_combined_main.py` script to analyze the topological features of Vision Transformer (ViT) models trained on MNIST data.

## Basic Usage

```bash
python vit_topology_combined_main.py [arguments]
```

## Available Arguments

### Core Analysis Options
- `--method {euclidean,geodesic}`: Distance method for topology analysis (default: `euclidean`)
- `--model {gelu,relu,both}`: Which activation function model to analyze (default: `both`)
- `--auto-scales`: Use automatic scale selection algorithms instead of fixed scales

### Model Architecture
- `--depth INT`: Number of transformer blocks/layers (default: `8`)
- `--width INT`: Embedding dimension for the transformer (default: `24`)
- `--heads INT`: Number of attention heads (default: `4`)
- `--mlp-ratio FLOAT`: MLP ratio for transformer blocks (default: `1.5`)

### Geodesic Analysis Options
- `--k INT`: Number of nearest neighbors for geodesic distance method (default: `14`)

### PCA Options (Euclidean Distance Only)
- `--use-pca`: Apply PCA dimensionality reduction (only valid with Euclidean distance)
- `--pca-components INT`: Number of PCA components to use if PCA is enabled (default: `20`)

### Visualization Options
- `--visualize-3d`: Create 3D visualizations of topology results
- `--no-visualization`: Skip all visualizations (useful for batch processing)

### Training Options
- `--train`: Train models if not already trained

## Common Usage Patterns

### 1. Basic Analysis with Default Settings

```bash
python vit_topology_combined_main.py
```
This runs an analysis with Euclidean distance on both GELU and ReLU models with default architecture (depth=8, width=24).

### 2. Euclidean Distance Analysis with PCA and 3D Visualization

```bash
python vit_topology_combined_main.py --method euclidean --use-pca --visualize-3d
```
Performs Euclidean distance analysis with PCA dimensionality reduction and generates 3D visualizations.

### 3. Geodesic Distance Analysis with Automatic Scale Selection

```bash
python vit_topology_combined_main.py --method geodesic --auto-scales --k 20
```
Runs geodesic distance analysis with a custom k-nearest neighbors value (20) and automatic scale selection.

### 4. Single Activation Function Analysis with Custom Architecture

```bash
python vit_topology_combined_main.py --model gelu --depth 12 --width 48 --heads 8
```
Analyzes only the GELU model with a custom architecture (12 layers, 48 embedding dimensions, 8 attention heads).

### 5. Training and Analysis in One Command

```bash
python vit_topology_combined_main.py --train --model relu --depth 6 --width 32
```
Trains a ReLU model with the specified architecture (if not already trained) and then performs the analysis.

### 6. Batch Processing Without Visualizations

```bash
python vit_topology_combined_main.py --no-visualization --method euclidean --model both
```
Analyzes both model types without generating any visualizations (faster for batch processing).

### 7. Complete Analysis with All Visualization Types

```bash
python vit_topology_combined_main.py --method euclidean --auto-scales --visualize-3d --use-pca
```
Performs analysis with automatic scale selection, PCA, and generates both standard and 3D visualizations.

## Working with Results

### Visualizing Existing Results

You can generate 3D visualizations from previously saved results:

```bash
python visualize_topology_3d_for_integration.py --results_dir results_latest
```

### Result Directory Structure

Results are saved in timestamped directories with the following naming pattern:
```
results_{method}_{architecture}_{pca}_{timestamp}/
```

Example:
```
results_euclidean_d8w24_pca_20230615_102030/
```

## Best Practices

1. **Use the right distance method for your analysis**:
   - `euclidean`: Standard distance metric, can be combined with PCA
   - `geodesic`: Better preserves manifold structure, NOT compatible with PCA

2. **Choose appropriate model architectures**:
   - For MNIST, the default architecture (depth=8, width=24) is usually sufficient
   - Larger architectures (e.g., depth=12, width=48) may reveal more complex topological features

3. **PCA Considerations**:
   - Only valid with Euclidean distance
   - Can significantly reduce computation time for high-dimensional embeddings
   - Default component count (20) works well; for very wide models, half the width is used

4. **Visualization Strategy**:
   - Standard 2D visualizations are best for quick insights
   - 3D visualizations help understand relationships between layers, scales, and Betti numbers
   - Heatmaps provide clearer views of topological patterns

5. **Scale Selection**:
   - `--auto-scales` is recommended for the most meaningful results
   - For geodesic distance, appropriate values of `k` typically range from 10-35

## Examples of Analysis Questions

- How does topology differ between GELU and ReLU activations?
- What effect does model depth have on topological features?
- Does increasing model width create more complex topological structures?
- How does the CLS token's topology compare to patch embeddings?
- What effect does PCA have on the observed topological features? 