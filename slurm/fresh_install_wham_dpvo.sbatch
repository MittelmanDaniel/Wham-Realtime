#!/bin/bash
#SBATCH --job-name=fresh_wham
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --time=03:00:00
#SBATCH --output=logs/fresh_install_%j.out
#SBATCH --error=logs/fresh_install_%j.err

set -euo pipefail

cd /home/hice1/dmittelman6/WHAM
mkdir -p logs

echo "========================================"
echo "Fresh WHAM + DPVO Installation"
echo "Following Official Installation Guide"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "========================================"

# Load modules
module load anaconda3/2023.03
module load cuda/11.8.0

eval "$(conda shell.bash hook)"

# Use /tmp for builds (compute node has space)
export TMPDIR=/tmp/wham_build_${SLURM_JOB_ID}
mkdir -p $TMPDIR
export CONDA_PKGS_DIRS=$TMPDIR/conda_pkgs
mkdir -p $CONDA_PKGS_DIRS

echo ""
echo "Step 1/11: Removing old corrupted environment..."
conda env remove -n wham --yes 2>&1 | tail -5 || echo "No existing environment to remove"
echo "✅ Old environment removed"

echo ""
echo "Step 2/11: Creating fresh conda environment..."
conda create -n wham python=3.9 --yes
conda activate wham
echo "✅ Environment created: $(which python)"

echo ""
echo "Step 3/11: Installing PyTorch 1.11.0 + CUDA 11.3..."
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch --yes
echo "✅ PyTorch installed: $(python -c 'import torch; print(torch.__version__)')"

echo ""
echo "Step 4/11: Installing PyTorch3D (for visualization)..."
conda install -c fvcore -c iopath -c conda-forge fvcore iopath --yes
pip install --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
echo "✅ PyTorch3D installed"

echo ""
echo "Step 5/11: Installing WHAM dependencies..."
pip install --no-cache-dir -r requirements.txt
echo "✅ WHAM dependencies installed"

echo ""
echo "Step 6/11: Installing ViTPose..."
pip install --no-cache-dir -v -e third-party/ViTPose
echo "✅ ViTPose installed"

echo ""
echo "Step 7/11: Preparing DPVO installation..."
cd third-party/DPVO

# Download Eigen
if [ ! -f "thirdparty/eigen-3.4.0/Eigen/Core" ]; then
    echo "Downloading Eigen 3.4.0..."
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip -q eigen-3.4.0.zip -d thirdparty
    rm eigen-3.4.0.zip
    echo "✅ Eigen 3.4.0 downloaded"
else
    echo "✅ Eigen already exists"
fi

echo ""
echo "Step 8/11: Installing pytorch-scatter..."
conda install pytorch-scatter=2.0.9 -c rusty1s --yes
echo "✅ pytorch-scatter installed"

echo ""
echo "Step 9/11: Installing cudatoolkit-dev..."
conda install cudatoolkit-dev=11.3.1 -c conda-forge --yes
echo "✅ cudatoolkit-dev installed"

echo ""
echo "Step 10/11: Checking GCC and installing gxx if needed..."
gcc_version=$(gcc -dumpversion | cut -d. -f1)
echo "System GCC version: $gcc_version"
if [ "$gcc_version" -gt 10 ]; then
    echo "GCC > 10, installing gxx=9.5 from conda..."
    conda install -c conda-forge gxx=9.5 --yes
    echo "✅ gxx=9.5 installed"
    export CC=$(which gcc)
    export CXX=$(which g++)
    echo "Using Conda GCC: $(gcc --version | head -1)"
else
    echo "GCC <= 10, using system GCC"
fi

echo ""
echo "Step 11/11: Compiling and installing DPVO..."
export CUDA_HOME=${CONDA_PREFIX}
echo "CUDA_HOME: $CUDA_HOME"
echo "GCC: $(gcc --version | head -1)"
echo "nvcc: $(nvcc --version | grep release)"
echo ""
echo "Starting DPVO compilation (this may take 5-10 minutes)..."

pip install --no-cache-dir -v . 2>&1 | tee $TMPDIR/dpvo_install.log | tail -100

cd /home/hice1/dmittelman6/WHAM

echo ""
echo "========================================"
echo "Testing Installation..."
echo "========================================"

# Test PyTorch
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

echo ""

# Test WHAM components
python -c "
print('Testing WHAM components...')
from lib.models import build_network, build_body_model
print('✅ WHAM network loadable')

from lib.models.preproc.detector import DetectionModel
print('✅ Detector loadable')

from lib.models.preproc.extractor import FeatureExtractor
print('✅ Feature extractor loadable')
"

echo ""

# Test DPVO
python -c "
try:
    from lib.models.preproc.slam import SLAMModel
    print('✅ ✅ ✅ DPVO SUCCESSFULLY INSTALLED!')
    print('✅ 3D world-grounded motion available!')
    import sys
    sys.exit(0)
except Exception as e:
    print(f'❌ DPVO import failed: {e}')
    import traceback
    traceback.print_exc()
    print('')
    print(f'DPVO install log: {TMPDIR}/dpvo_install.log')
    import sys
    sys.exit(1)
" 2>&1

exit_code=$?

# Cleanup
echo ""
echo "Cleaning up temp directory..."
rm -rf $TMPDIR
echo "✅ Cleanup complete"

echo ""
if [ $exit_code -eq 0 ]; then
    echo "========================================"
    echo "🎉🎉🎉 INSTALLATION SUCCESSFUL! 🎉🎉🎉"
    echo "========================================"
    echo ""
    echo "✅ Fresh WHAM environment created"
    echo "✅ All dependencies installed"
    echo "✅ DPVO working (3D + world-grounded)"
    echo ""
    echo "You can now:"
    echo "  - Run real-time WHAM with 2D pose"
    echo "  - Run full WHAM with 3D + world coordinates"
    echo "  - Use --run_global flag for DPVO"
    echo ""
    echo "To activate: conda activate wham"
else
    echo "========================================"
    echo "⚠️  Installation Complete with Issues"
    echo "========================================"
    echo ""
    echo "WHAM components installed, but DPVO failed"
    echo "You can still use real-time 2D pose tracking"
    echo ""
    echo "To debug DPVO, check the logs"
fi

echo ""
echo "Disk usage:"
du -sh ~/.conda/envs/wham ~/.conda/pkgs 2>/dev/null || true
quota -s | head -3

exit $exit_code

