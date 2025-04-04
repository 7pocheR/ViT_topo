#!/bin/bash
#SBATCH --job-name=vit_topo
#SBATCH --output=vit_topo_%J.out
#SBATCH --error=vit_topo_%J.err
#SBATCH --time=02:00:00
#SBATCH --partition=caslake  # CPU partition on Midway3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G

# Change to the project directory
cd $SLURM_SUBMIT_DIR

# Load required modules
module load python

# Activate the environment
source activate topo_env

# Install required packages matching your local environment
pip install numpy scipy matplotlib scikit-learn pillow networkx joblib
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install gudhi ripser persim

# Install other packages from requirements.txt
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Run the main script
python vit_topology_combined_main.py