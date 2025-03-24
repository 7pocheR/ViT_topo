#!/bin/bash

# First, load required modules
module load cmake
module load cuda

# Check which CUDA version is loaded
echo "Using CUDA version:"
nvcc --version

# Set CUDA toolkit root dir
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
echo "CUDA_TOOLKIT_ROOT_DIR set to: $CUDA_TOOLKIT_ROOT_DIR"

# Try pip install
echo "Attempting pip install..."
pip install ripserplusplus

# If pip install fails, try from GitHub
if [ $? -ne 0 ]; then
    echo "Pip install failed, trying from GitHub..."
    git clone https://github.com/simonzhang00/ripser-plusplus.git
    cd ripser-plusplus
    mkdir -p build && cd build
    
    # Use cmake with CUDA path
    cmake .. -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR
    make -j$(nproc)
    
    # Install the Python bindings
    cd ../python
    pip install .
    
    echo "Installation from GitHub complete"
fi

echo "Verifying installation:"
python -c "import ripserplusplus as rpp_py; print('Ripserplusplus imported successfully')" 