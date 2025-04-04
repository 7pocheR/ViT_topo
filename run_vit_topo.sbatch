#!/bin/bash
#SBATCH --job-name=vit_topo
#SBATCH --output=vit_topo_%J.out
#SBATCH --error=vit_topo_%J.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Change to the project directory
cd $SLURM_SUBMIT_DIR

# Load required modules
module load python/anaconda-2023.03
module load cuda/11.7

# Create a conda environment if it doesn't exist
conda create -n vit_topo python=3.10 -y || echo "Environment already exists"

# Activate the environment
source activate vit_topo

# Install required packages
pip install -r requirements.txt
pip install torch torchvision torchaudio

# Run the main script
python vit_topology_combined_main.py