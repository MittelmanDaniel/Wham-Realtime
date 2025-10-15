#!/bin/bash
# Fix corrupted environment and install DPVO without cudatoolkit-dev
set -e

echo "========================================"
echo "Fixing Corrupted Environment & Installing DPVO"
echo "========================================"

eval "$(conda shell.bash hook)"

# Remove corrupted file
echo "Removing corrupted conda metadata..."
rm -f ~/.conda/envs/wham/conda-meta/libgomp-15.2.0-h767d61c_7.json

# Clean up conda caches
echo "Cleaning conda caches..."
conda clean --all --yes

# Clean pip cache
echo "Cleaning pip cache..."
pip cache purge

# Remove DPVO build artifacts
echo "Cleaning DPVO artifacts..."
rm -rf third-party/DPVO/build third-party/DPVO/dist third-party/DPVO/*.egg-info

echo "Disk space freed!"
echo ""

# Activate environment
conda activate wham

echo "Installing DPVO (using system CUDA, skipping cudatoolkit-dev)..."
cd third-party/DPVO

# Check if eigen exists
if [ ! -d "thirdparty/eigen-3.4.0" ]; then
    echo "Downloading Eigen..."
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip -q eigen-3.4.0.zip -d thirdparty
    rm eigen-3.4.0.zip
fi

# Install pytorch-scatter if not installed
echo "Checking pytorch-scatter..."
python -c "import torch_scatter" 2>/dev/null || conda install -y pytorch-scatter=2.0.9 -c rusty1s

# Load system CUDA
module load cuda/11.8.0 2>/dev/null || echo "CUDA module not available, trying environment CUDA..."

# Set CUDA_HOME to system CUDA (not conda)
export CUDA_HOME=/usr/local/cuda-11.8

# Install GCC 9.5 if not present
gcc_version=$(gcc -dumpversion | cut -d. -f1)
if [ "$gcc_version" -gt 10 ]; then
    echo "Installing gxx=9.5..."
    conda install -y -c conda-forge gxx=9.5
fi

# Install DPVO
echo "Compiling DPVO..."
pip install --no-cache-dir -v .

cd ../..

# Test
echo ""
echo "Testing DPVO..."
python -c "from lib.models.preproc.slam import SLAMModel; print('✅ DPVO WORKING!')"

echo ""
echo "========================================"
echo "✅ DPVO Installation Complete!"
echo "========================================"

