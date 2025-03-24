#!/bin/bash
#SBATCH --job-name=test_ripser
#SBATCH --output=test_ripser.out
#SBATCH --error=test_ripser.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --account=pi-lekheng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=16G

# Print node and GPU info for debugging
echo "====== ENVIRONMENT INFO ======"
hostname
nvidia-smi
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "=============================="

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate topo_env

# Run CUDA environment check
echo "Running CUDA environment check..."
./check_cuda.sh

# Install the ripserplusplus package
echo "Installing ripserplusplus package..."
./install_ripserplusplus.sh

# Print environment information
echo "====== PYTHON ENVIRONMENT ======"
which python
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}');"
echo "================================="

# Run test script
echo "Running ripserplusplus test..."
python -u test_ripserplusplus.py 