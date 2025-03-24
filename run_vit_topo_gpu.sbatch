#!/bin/bash
#SBATCH --job-name=vit_topo_gpu
#SBATCH --output=vit_topo_gpu.out
#SBATCH --error=vit_topo_gpu.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=pi-lekheng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G

# Print node and GPU info for debugging
echo "====== ENVIRONMENT INFO ======"
hostname
nvidia-smi
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "=============================="

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate topo_env

# Load CUDA module and set paths
module load cuda
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME

# Ensure MNIST data directory structure exists
echo "Setting up MNIST data directories..."
mkdir -p data/MNIST/raw

# Check if MNIST data files exist
if [ ! -f "data/MNIST/raw/train-images-idx3-ubyte" ]; then
    echo "MNIST data files not found. Checking if downloaded files exist..."
    
    # If the gzipped files exist, extract them
    if [ -f "data/MNIST/raw/train-images-idx3-ubyte.gz" ]; then
        echo "Extracting downloaded MNIST files..."
        gunzip -k data/MNIST/raw/*.gz
    else
        echo "WARNING: MNIST data files not found. You may need to manually download them."
        echo "The job will try to continue, but may fail if data cannot be located."
    fi
fi

# Print environment information
echo "====== PYTHON ENVIRONMENT ======"
which python
which nvcc
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}');"
python -c "import ripserplusplus as rpp_py; print('Ripserplusplus imported successfully'); print('Available functions:', [f for f in dir(rpp_py) if not f.startswith('__')])" || echo "Failed to import ripserplusplus"
echo "================================="

# Run with unbuffered output
python -u vit_topology_main.py 