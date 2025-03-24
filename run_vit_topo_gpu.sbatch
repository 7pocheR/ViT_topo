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
hostname
nvidia-smi

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate topo_env

# Install ripser++ if not already installed
pip install ripser-plusplus

# Run with unbuffered output
python -u vit_topology_main.py 