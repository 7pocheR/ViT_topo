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

# Install the correct ripserplusplus package
echo "Installing ripserplusplus package..."
pip install ripserplusplus || pip install git+https://github.com/simonzhang00/ripser-plusplus.git

# Print environment information
echo "====== PYTHON ENVIRONMENT ======"
which python
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}');"
echo "================================="

# Run with unbuffered output
python -u vit_topology_main.py 