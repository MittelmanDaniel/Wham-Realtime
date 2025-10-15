#!/bin/bash
#SBATCH --job-name=dpvo_syscuda
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --time=02:00:00
#SBATCH --output=logs/dpvo_syscuda_%j.out
#SBATCH --error=logs/dpvo_syscuda_%j.err

set -euo pipefail

cd /home/hice1/dmittelman6/WHAM
mkdir -p logs

echo "========================================"
echo "Install DPVO with System CUDA"
echo "(Skipping cudatoolkit-dev to save space)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "========================================"

# Load modules - system CUDA instead of conda cudatoolkit-dev
module load anaconda3/2023.03
module load cuda/11.8.0

eval "$(conda shell.bash hook)"
conda activate wham

echo ""
echo "Environment Check:"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"
echo "System CUDA: $(nvcc --version | grep release)"
echo ""

# Use /tmp for builds
export BUILD_DIR=/tmp/dpvo_${SLURM_JOB_ID}
mkdir -p $BUILD_DIR
export TMPDIR=$BUILD_DIR/tmp
mkdir -p $TMPDIR

# Set CUDA paths to SYSTEM CUDA (not conda)
# Find nvcc from loaded module
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"
echo "Build directory: $BUILD_DIR"
echo ""

cd third-party/DPVO

# Ensure Eigen exists
if [ ! -f "thirdparty/eigen-3.4.0/Eigen/Core" ]; then
    echo "Downloading Eigen 3.4.0..."
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip -q eigen-3.4.0.zip -d thirdparty
    rm eigen-3.4.0.zip
    echo "✅ Eigen downloaded"
else
    echo "✅ Eigen already exists"
fi

# Install pytorch-scatter (if not already installed)
echo ""
echo "Checking pytorch-scatter..."
python -c "import torch_scatter" 2>/dev/null || {
    echo "Installing pytorch-scatter..."
    conda install -y pytorch-scatter=2.0.9 -c rusty1s
}
echo "✅ pytorch-scatter ready"

# Install gxx=9.5 (if needed and if space allows)
echo ""
echo "Checking GCC version..."
gcc_version=$(gcc -dumpversion | cut -d. -f1)
echo "System GCC: $gcc_version"

if [ "$gcc_version" -gt 10 ]; then
    echo "GCC > 10, need gxx=9.5..."
    python -c "import os; exit(1 if os.path.exists('${CONDA_PREFIX}/bin/g++') and '9.5' in os.popen('${CONDA_PREFIX}/bin/g++ --version').read() else 0)" && {
        echo "Installing gxx=9.5..."
        conda install -y -c conda-forge gxx=9.5 || echo "⚠️ gxx install failed, will try with system GCC"
    } || echo "✅ gxx=9.5 already installed"
    
    # Use conda GCC if available
    if [ -f "${CONDA_PREFIX}/bin/gcc" ]; then
        export CC=${CONDA_PREFIX}/bin/gcc
        export CXX=${CONDA_PREFIX}/bin/g++
        echo "Using Conda GCC: $(gcc --version | head -1)"
    else
        echo "⚠️ Using system GCC (may have compatibility issues)"
    fi
else
    echo "✅ System GCC is compatible"
fi

echo ""
echo "Compiler Configuration:"
echo "GCC: $(gcc --version | head -1)"
echo "G++: $(g++ --version | head -1)"
echo "nvcc: $(nvcc --version | grep release)"
echo "CUDA_HOME: $CUDA_HOME"
echo ""

# Clean previous builds
rm -rf build dist *.egg-info

echo "Compiling DPVO (this may take 5-10 minutes)..."
# Add nvcc flag to allow GCC version mismatch  
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6"
export TORCH_NVCC_FLAGS="-allow-unsupported-compiler"
pip install --no-cache-dir -v . 2>&1 | tee $BUILD_DIR/dpvo_install.log | grep -E "(Building|Compiling|error|warning|Successfully)" | tail -50

cd /home/hice1/dmittelman6/WHAM

echo ""
echo "========================================"
echo "Testing DPVO..."
echo "========================================"

python -c "
try:
    from lib.models.preproc.slam import SLAMModel
    print('✅ ✅ ✅ DPVO SUCCESSFULLY INSTALLED!')
    print('✅ 3D world-grounded coordinates available!')
    print('')
    print('DPVO working with system CUDA!')
    import sys
    sys.exit(0)
except Exception as e:
    print(f'❌ DPVO import failed: {e}')
    import traceback
    traceback.print_exc()
    print('')
    print(f'Check: {BUILD_DIR}/dpvo_install.log')
    import sys
    sys.exit(1)
" 2>&1

exit_code=$?

# Cleanup
echo ""
echo "Cleaning up temp directory..."
rm -rf $BUILD_DIR
echo "✅ Cleanup complete"

echo ""
if [ $exit_code -eq 0 ]; then
    echo "========================================"
    echo "🎉🎉🎉 DPVO INSTALLED! 🎉🎉🎉"
    echo "========================================"
    echo ""
    echo "You can now use WHAM with:"
    echo "  --run_global   for world-grounded motion"
    echo ""
else
    echo "========================================"
    echo "❌ DPVO Installation Failed"
    echo "========================================"
    echo ""
    echo "Check log: $BUILD_DIR/dpvo_install.log"
    echo ""
    echo "You can still use WHAM without DPVO"
    echo "for real-time 2D pose tracking"
fi

echo ""
echo "Disk usage:"
quota -s | head -3

exit $exit_code

