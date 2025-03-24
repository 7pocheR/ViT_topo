# Installation Guide for ViT Topology Analysis on Midway3

This guide provides detailed instructions for installing and running the topology analysis code on the Midway3 cluster.

## Prerequisites

- Access to Midway3 cluster with your account
- A GPU allocation on the cluster
- Git installed on Midway3

## Step 1: Clone the Repository

```bash
# Login to Midway3
ssh <your-username>@midway3.rcc.uchicago.edu

# Clone the repository
git clone https://github.com/7pocheR/ViT_topo.git
cd ViT_topo
```

## Step 2: Set Up the Conda Environment

```bash
# Load the miniconda module (or your preferred conda distribution)
module load miniconda

# Create a new conda environment
conda create -n topo_env python=3.9
source activate topo_env

# Install required packages
pip install -r requirements.txt
```

## Step 3: Install ripserplusplus with CUDA Support

The main challenge is installing `ripserplusplus` with proper CUDA support. We've provided several scripts to help with this:

### Option 1: Use the Automated Installation Script

```bash
# Make sure the scripts are executable
chmod +x check_cuda.sh install_ripserplusplus.sh

# Run the installation script
./install_ripserplusplus.sh
```

### Option 2: Manual Installation

If the automated script fails, follow these steps:

```bash
# Load required modules
module load cmake
module load cuda

# Check CUDA version and location
echo $CUDA_HOME
nvcc --version

# Set environment variable for CUDA
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME

# Install from GitHub
git clone https://github.com/simonzhang00/ripser-plusplus.git
cd ripser-plusplus
mkdir -p build && cd build
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR
make -j4
cd ../python
pip install .
```

## Step 4: Test the Installation

You can verify that everything is installed correctly by running our test script:

```bash
# Submit the test job
sbatch test_ripserplusplus.sbatch

# Check the status
squeue -u <your-username>

# Once completed, view the output
cat test_ripser.out
```

## Step 5: Run the Main Analysis

Once everything is installed correctly, you can run the main analysis:

```bash
# Submit the main job
sbatch run_vit_topo_gpu.sbatch

# Check the status
squeue -u <your-username>

# Monitor output
tail -f vit_topo_gpu.out
```

## Troubleshooting

### No CUDA visible devices

If you get an error about no CUDA devices being available, check:
- Are you on a GPU node?
- Is the CUDA module loaded?
- Does your account have GPU allocation?

### ripserplusplus installation fails

Try these approaches:
1. Make sure cmake and CUDA modules are loaded
2. Try a different CUDA version (module swap cuda cuda/11.4)
3. Clone from GitHub and install manually
4. Check if you have sufficient permissions for installation

### Memory issues

If you encounter memory issues:
- Reduce batch size in `vit_topology_main.py`
- Reduce point cloud size in `topology_analysis.py`
- Request more memory in the sbatch script 