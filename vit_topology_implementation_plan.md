# ViT Topology Analysis - Implementation Plan

## Overview
This plan outlines the steps to replicate the topology analysis from "Topology of Deep Neural Networks" (Naitzat, Zhitnikov, Lim, 2020) using Vision Transformers (ViTs) on MNIST data.

## 1. Environment Setup
- [ ] Import necessary libraries
  - PyTorch for ViT implementation
  - scikit-learn for PCA
  - Ripser/Gudhi for persistent homology
  - NumPy, Matplotlib for computation and visualization

## 2. Data Preparation
- [ ] Load MNIST dataset
- [ ] Create binary classification task (digit "0" vs. non-"0")
- [ ] Split data into training and test sets
- [ ] Apply PCA to reduce dimensions to 50 (same as original paper)
- [ ] Normalize data

## 3. ViT Model Implementation
- [ ] Define simplified ViT architecture for MNIST
  - Input size: 28×28 images
  - Patch size: 4×4 (resulting in 7×7=49 patches)
  - Embedding dimension: 48 (reduced from 64)
  - Number of heads: 4
  - Number of transformer blocks: 4 (reduced from 6)
  - MLP ratio: 1.5 (hidden dimension 48×1.5=72)
  - **Activation functions**: Implement both options
    - [ ] GELU (smooth activation)
    - [ ] ReLU (non-smooth activation)
- [ ] Add hooks to extract representations at both points in each block:
  - After self-attention (post-residual)
  - After MLP (post-residual)
- [ ] Implement representation extraction function

## 4. Model Training
- [ ] Train two separate models with identical architecture except for activation:
  - [ ] Train ViT with GELU activation
  - [ ] Train ViT with ReLU activation
- [ ] Define training parameters
  - Optimizer: Adam
  - Learning rate: 1e-4
  - Batch size: 128
  - Epochs: 10 (or until high accuracy)
- [ ] Train both models to high accuracy (>98%)
- [ ] Save trained models

## 5. Feature Extraction
- [ ] Create function to extract embeddings at each extraction point
- [ ] Run test set through both models and collect embeddings
- [ ] Group embeddings by class (digit "0" vs. non-"0")
- [ ] Create multiple pointclouds:
  - [ ] Total pointcloud: All patch tokens combined (49*N points)
  - [ ] CLS token pointcloud (N points)
  - [ ] Central patch pointcloud: patch at position (4,4) (N points)
- [ ] Apply PCA to reduce each pointcloud to 50 dimensions (for consistency with original paper)
- [ ] Ensure all pointclouds have consistent size (e.g., 10000 points each)

## 6. Persistent Homology Computation
- [ ] Determine appropriate persistent homology parameters:
  - [ ] Use k = 14 nearest neighbors as initial value (from original paper)
  - [ ] Calibration step: For each model (GELU/ReLU) after training
    - [ ] Select representative layer outputs
    - [ ] Compute persistent homology across a wide range of scales (ε = 0.5 to 5.0)
    - [ ] Identify stable regions where Betti numbers reveal meaningful structure
    - [ ] Select 3-4 scales that best capture topological features
  - [ ] Document the selected scales for each model variant
- [ ] Implement batch processing for persistent homology computation to manage memory usage
- [ ] For each activation function (GELU and ReLU):
  - [ ] For each pointcloud (CLS token, central patch, total patches = 3 pointclouds):
    - [ ] For each extraction point (2 components × 4 blocks = 8 extraction points):
      - [ ] Compute persistent homology for each class using calibrated scales
      - [ ] Calculate Betti numbers (β₀, β₁, β₂) at each scale

## 7. Analysis and Visualization
- [ ] Create tables comparing Betti numbers across layers for both activations
- [ ] Generate persistence barcodes for selected layers
- [ ] Create visualizations showing topology transformation:
  - [ ] Two separate plots per activation function:
    - Plot 1: Topology after each attention layer (4 points)
    - Plot 2: Topology after each MLP layer (4 points)
  - [ ] Each plot contains 9 curves (one for each pointcloud)
  - [ ] X-axis: Layer index
  - [ ] Y-axis: Betti numbers at the selected scale
- [ ] Create comparative plots showing GELU vs. ReLU differences
- [ ] Visualize data projections using PCA/t-SNE at different layers

## 8. Comparison with Original Results
- [ ] Compare GELU vs. ReLU results with the paper's findings on smooth vs. non-smooth activations
- [ ] Analyze self-attention vs. MLP contributions to topology change
- [ ] Evaluate whether ViTs show similar topology simplification patterns to FFNs
- [ ] Document key differences in how ViTs transform topology

## 9. CNN Comparison (Future Extension)
- [ ] Implement a simple CNN architecture for MNIST
  - [ ] Similar parameter count to our ViT model
  - [ ] Use both ReLU and GELU variants
- [ ] Extract features from different CNN layers
- [ ] Apply the same topology analysis pipeline
- [ ] Compare topology transformation patterns:
  - [ ] CNN vs. ViT
  - [ ] How convolutional layers transform topology vs. self-attention

## Implementation Log
*This section will be updated as we progress through implementation*

### 2023-XX-XX: Initial Implementation

1. **Environment Setup**
   - ✅ Created basic project structure
   - ✅ Imported necessary libraries in the main script
   - ✅ Created three primary modules:
     - `vit_topology_main.py`: Main script to run the entire pipeline
     - `vit_model.py`: Implementation of Vision Transformer with hook support
     - `topology_analysis.py`: Functions for persistent homology computation

2. **Data Preparation**
   - ✅ Implemented MNIST dataset loading
   - ✅ Created binary classification (digit "0" vs. non-"0")
   - ✅ Set up data loaders for training and testing

3. **ViT Model Implementation**
   - ✅ Implemented simplified ViT architecture for MNIST
     - Patch embedding with 4×4 patches
     - Class token and position embeddings
     - 4 transformer blocks with 4 attention heads
     - Embedding dimension of 48 with MLP ratio of 1.5
   - ✅ Added support for both GELU and ReLU activations
   - ✅ Implemented hooks to extract features from both attention and MLP outputs

4. **Model Training**
   - ✅ Implemented training functions with early stopping
   - ✅ Added model saving/loading for efficient iteration
   - ✅ Implemented evaluation on test set

5. **Feature Extraction**
   - ✅ Implemented feature extraction from transformer blocks
   - ✅ Created functions to process features into pointclouds:
     - Total pointcloud (all patches)
     - CLS token pointcloud
     - Position-specific pointclouds (7, 14, 21, 28, 35, 42, 49)
   - ✅ Added PCA dimension reduction to 50 dimensions

6. **Persistent Homology Computation**
   - ✅ Implemented batch processing for persistent homology
   - ✅ Created scale calibration function
   - ✅ Implemented Betti number calculation at multiple scales

7. **Analysis and Visualization**
   - ✅ Added function to visualize Betti numbers across layers
   - ✅ Implemented saving of numerical results as JSON
   - ✅ Added plotting for topology changes across layers

**Next Steps:**
- Run the implementation on actual data
- Analyze the results to compare topology transformation between GELU and ReLU
- Fine-tune the analysis based on initial results 