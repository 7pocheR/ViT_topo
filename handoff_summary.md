# ViT Topology Analysis Project Handoff

## Project Overview
This project analyzes the topological evolution of Vision Transformers (ViT) through persistent homology. We've implemented a simple ViT for MNIST binary classification (digit 0 vs. non-0) and extracted features to compute persistent homology. We've added GPU acceleration with ripserplusplus for faster processing of higher-dimensional Betti numbers (β₀, β₁, β₂, β₃).

## Architecture and Implementation
- **Model**: SimpleViT with 12 transformer blocks
- **Input**: MNIST (28×28) split into 49 patches (4×4)
- **Embedding dimension**: 16
- **Attention heads**: 4
- **Activation functions**: GELU and ReLU variants (trained separately)
- **Classification**: Binary using CLS token (0 vs. non-0)
- **Point cloud size**: 1000 points for topology analysis

## Key Achievements
1. Successfully trained ViT models for MNIST (98.2-98.9% accuracy)
2. Implemented persistent homology computation up to dimension 3
3. Added GPU acceleration with ripserplusplus
4. Created comprehensive 3D visualizations of topology evolution
5. **Discovered a surprising pattern**: Unlike CNNs, ViTs show *increasing* Betti numbers (β₀, β₁) through layers, suggesting transformers build more complex topological structures rather than simplifying them

## Repository Structure
```
ViT_topo/
├── topology_analysis.py    # Core topology analysis functions
├── vit_model.py            # Vision Transformer implementation
├── vit_topology_main.py    # Main execution script
├── visualize_topology_3d.py # 3D visualization script
├── vit_architecture_explained.md # Architecture explanation
├── results/                # Results directory
│   ├── GELU_results.json   # GELU model topology results
│   ├── ReLU_results.json   # ReLU model topology results
│   ├── GELU_class0/        # GELU visualizations for class 0
│   ├── GELU_class1/        # GELU visualizations for class 1
│   ├── ReLU_class0/        # ReLU visualizations for class 0
│   └── ReLU_class1/        # ReLU visualizations for class 1
├── results_3d/             # 3D visualization results
├── model_gelu.pth          # Trained GELU model weights
└── model_relu.pth          # Trained ReLU model weights
```

## Installation Instructions

### Prerequisites
- CUDA-compatible GPU
- Python 3.9+
- PyTorch with CUDA support
- CMake for building ripserplusplus

### Setup Environment
```bash
# Create and activate virtual environment
python -m venv vit_topo_env
source vit_topo_env/bin/activate  # On Windows: vit_topo_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install GPU-accelerated ripserplusplus
# First ensure CUDA is in your PATH
export PATH=$CUDA_HOME/bin:$PATH  # On Windows: set PATH=%CUDA_HOME%\bin;%PATH%
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME  # On Windows: set CUDA_TOOLKIT_ROOT_DIR=%CUDA_HOME%

# Install ripser++
pip install ripserplusplus

# If pip installation fails:
git clone https://github.com/simonzhang00/ripser-plusplus.git
cd ripser-plusplus
mkdir -p build && cd build
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR
make -j4
cd ../python
pip install .
cd ../..
```

## Running the GPU-Accelerated Version

### Setting Up Data
```bash
# Create data directory structure
mkdir -p data/MNIST/raw

# Download MNIST files (if not automatically downloaded)
cd data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip -k *.gz
cd ../../..
```

### Running the Analysis
```bash
# Full training and analysis pipeline
python vit_topology_main.py

# If models are already trained and you just want to run topology analysis:
# 1. Edit vit_topology_main.py to skip training if models exist (already implemented)
# 2. Run the script
python vit_topology_main.py

# To visualize in 3D (after running the main script)
python visualize_topology_3d.py
```

## Troubleshooting Common Issues

### GPU Memory Issues
- **Symptom**: `CUDA out of memory` error during persistent homology computation
- **Solution**: Reduce `batch_size` and `target_size` in `topology_analysis.py`
  - Try 800 or 500 points instead of 1000
  - This can be adjusted in `create_pointclouds()` function

### CUDA Not Found
- **Symptom**: `CMake Error: Specify CUDA_TOOLKIT_ROOT_DIR` during ripserplusplus installation
- **Solution**: Ensure CUDA is properly installed and paths are set
  ```bash
  export PATH=$CUDA_HOME/bin:$PATH
  export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
  which nvcc  # Should show CUDA compiler path
  ```

### Ripser++ API Issues
- **Symptom**: `module 'ripserplusplus' has no attribute 'xyz'`
- **Solution**: The code now auto-detects the correct API. Check which function is available:
  ```python
  import ripserplusplus as rpp_py
  [func for func in dir(rpp_py) if not func.startswith('__')]
  ```

## Testing the GPU Acceleration
To verify that GPU acceleration is working:

1. **Check import success**:
```python
import ripserplusplus as rpp_py
print("Available functions:", [f for f in dir(rpp_py) if not f.startswith('__')])
```

2. **Monitor GPU usage** while running:
```bash
# Run in a separate terminal
nvidia-smi -l 1
```

3. **Compare performance**:
- Edit `USE_GPU = False` in `topology_analysis.py` to disable GPU
- Time the execution with CPU vs. GPU:
```bash
time python vit_topology_main.py  # With GPU
time python vit_topology_main.py  # With GPU disabled
```

## Next Steps for Future Work
1. Compare results with CNNs on the same dataset
2. Extend to multi-class classification
3. Analyze attention patterns in relation to topological features
4. Optimize for larger datasets and deeper networks
5. Investigate correlations between attention maps and persistent homology

## Citations
- Naitzat et al. (2020) "Topology of Deep Neural Networks" (original paper)
- Dosovitskiy et al. (2020) "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT paper)
- Zhang et al. (2020) "Ripser++: A High-Performance Computation Library for Persistent Cohomology" 