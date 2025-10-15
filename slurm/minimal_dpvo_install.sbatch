#!/bin/bash
#SBATCH --job-name=minimal_dpvo
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --time=01:00:00
#SBATCH --output=logs/minimal_dpvo_%j.out
#SBATCH --error=logs/minimal_dpvo_%j.err

set -euo pipefail

cd /home/hice1/dmittelman6/WHAM
mkdir -p logs

echo "========================================"
echo "Minimal DPVO Install (System CUDA/GCC)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "========================================"

# Load modules
module load anaconda3/2023.03
module load cuda/11.8.0
# Skip GCC module (not available), will use conda gxx or system default

eval "$(conda shell.bash hook)"
conda activate wham

echo ""
echo "Environment:"
echo "Python: $(which python)"
echo "GCC: $(gcc --version | head -1)"
echo "nvcc: $(nvcc --version | grep release)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Use /tmp for build
export TMPDIR=/tmp/dpvo_${SLURM_JOB_ID}
mkdir -p $TMPDIR

# Set CUDA_HOME to system CUDA
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

cd third-party/DPVO

# Ensure Eigen exists
if [ ! -d "thirdparty/eigen-3.4.0" ]; then
    echo "Downloading Eigen..."
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip -q eigen-3.4.0.zip -d thirdparty
    rm eigen-3.4.0.zip
    echo "✅ Eigen downloaded"
fi

# Install pytorch-scatter with pip (not conda)
echo "Installing pytorch-scatter..."
pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html 2>&1 | tail -10

# Clean previous builds
rm -rf build dist *.egg-info

# Install DPVO
echo ""
echo "Compiling DPVO with system CUDA and GCC..."
pip install --no-cache-dir -v . 2>&1 | tee /tmp/dpvo_compile_${SLURM_JOB_ID}.log | tail -50

cd /home/hice1/dmittelman6/WHAM

echo ""
echo "========================================"
echo "Testing DPVO..."
echo "========================================"

python -c "
try:
    from lib.models.preproc.slam import SLAMModel
    print('✅ ✅ ✅ DPVO WORKS!')
    print('✅ 3D world-grounded coordinates now available!')
    import sys
    sys.exit(0)
except Exception as e:
    print(f'❌ Failed: {e}')
    import sys
    sys.exit(1)
"

exit_code=$?

# Clean up
rm -rf $TMPDIR

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "🎉🎉🎉 DPVO INSTALLATION SUCCESSFUL! 🎉🎉🎉"
else
    echo ""
    echo "❌ Installation failed. Check: /tmp/dpvo_compile_${SLURM_JOB_ID}.log"
fi

exit $exit_code

