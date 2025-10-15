#!/bin/bash
set -e

echo "========================================"
echo "Installing DPVO (Deep Patch Visual Odometry)"
echo "========================================"
echo "This enables world-grounded motion in WHAM"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wham

# Set temporary build directory to /tmp (more space)
export TMPDIR=/tmp/dpvo_build_$$
mkdir -p $TMPDIR
echo "Using temp directory: $TMPDIR"

cd /home/hice1/dmittelman6/WHAM

# Clone DPVO if not exists
if [ ! -d "third-party/DPVO" ]; then
    echo "Cloning DPVO repository..."
    mkdir -p third-party
    cd third-party
    git clone --recursive https://github.com/princeton-vl/DPVO.git
    cd ..
fi

echo "Installing DPVO dependencies..."

# Load CUDA module
module load cuda/11.8.0 2>/dev/null || echo "CUDA module already loaded or not available"

# Set CUDA_HOME
export CUDA_HOME=${CONDA_PREFIX}
if [ ! -f "${CUDA_HOME}/bin/nvcc" ]; then
    export CUDA_HOME=/usr/local/cuda-11.8
fi

echo "CUDA_HOME: ${CUDA_HOME}"
echo "nvcc version:"
nvcc --version || echo "nvcc not found, trying system CUDA..."

# Install dependencies
echo "Installing lietorch..."
cd third-party/DPVO/DPViewer
pip install --no-cache-dir .

cd ../lietorch
pip install --no-cache-dir .

cd ..
pip install --no-cache-dir .

cd /home/hice1/dmittelman6/WHAM

echo ""
echo "========================================"
echo "DPVO Installation Complete!"
echo "========================================"
echo "Testing import..."
python -c "from lib.models.preproc.slam import SLAMModel; print('✅ DPVO working!')" || echo "⚠️ DPVO import failed"

# Clean up temp directory
rm -rf $TMPDIR
echo "Cleaned up temp directory"

