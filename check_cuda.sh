#!/bin/bash

echo "Available CUDA modules:"
module avail cuda 2>&1

echo -e "\nCurrently loaded modules:"
module list 2>&1

echo -e "\nCUDA Toolkit paths:"
for cuda_version in $(module avail -t cuda 2>&1 | grep -o "cuda/[0-9.]*" | cut -d/ -f2); do
    module load cuda/$cuda_version 2>/dev/null
    echo "CUDA $cuda_version: $CUDA_HOME"
    module unload cuda 2>/dev/null
done

echo -e "\nChecking nvcc:"
which nvcc 2>/dev/null || echo "nvcc not found"
if [ -x "$(which nvcc)" ]; then
    nvcc --version
fi 