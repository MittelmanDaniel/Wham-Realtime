#!/bin/bash
#SBATCH --job-name=dpvo_install
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --time=02:00:00
#SBATCH --output=logs/dpvo_install_%j.out
#SBATCH --error=logs/dpvo_install_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "========================================"
echo "DPVO Installation (Following Official Guide)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "========================================"

# Load modules
module load anaconda3/2023.03

eval "$(conda shell.bash hook)"
conda activate wham

echo ""
echo "Environment Info:"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Use /tmp for build (more space than home directory)
export TMPDIR=/tmp/dpvo_build_${SLURM_JOB_ID}
mkdir -p $TMPDIR
echo "Temp directory: $TMPDIR"
echo ""

# Clean previous attempts
rm -rf third-party/DPVO

# Clone DPVO
echo "Step 1/6: Cloning DPVO repository..."
mkdir -p third-party
cd third-party
git clone --recursive https://github.com/princeton-vl/DPVO.git
cd DPVO

echo ""
echo "Step 2/6: Downloading Eigen 3.4.0..."
wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip -q eigen-3.4.0.zip -d thirdparty
rm -rf eigen-3.4.0.zip
echo "✅ Eigen downloaded and extracted"

echo ""
echo "Step 3/6: Installing pytorch-scatter..."
conda install -y pytorch-scatter=2.0.9 -c rusty1s 2>&1 | tail -10
echo "✅ pytorch-scatter installed"

echo ""
echo "Step 4/6: Installing cudatoolkit-dev..."
# This might fail due to disk quota, but try anyway
conda install -y cudatoolkit-dev=11.3.1 -c conda-forge 2>&1 | tail -10 || echo "⚠️ cudatoolkit-dev install failed (may be okay)"

echo ""
echo "Step 5/6: Checking GCC version and installing gxx if needed..."
gcc_version=$(gcc -dumpversion | cut -d. -f1)
echo "GCC version: $gcc_version"
if [ "$gcc_version" -gt 10 ]; then
    echo "GCC > 10, installing gxx=9.5..."
    conda install -y -c conda-forge gxx=9.5 2>&1 | tail -10
    echo "✅ gxx=9.5 installed"
else
    echo "GCC <= 10, no need for gxx=9.5"
fi

echo ""
echo "Step 6/6: Installing DPVO..."
echo "This will compile CUDA extensions..."

# Set CUDA_HOME
export CUDA_HOME=${CONDA_PREFIX}
echo "CUDA_HOME: ${CUDA_HOME}"

# Install DPVO
pip install --no-cache-dir -v . 2>&1 | tee /tmp/dpvo_install_${SLURM_JOB_ID}.log | tail -50

cd /home/hice1/dmittelman6/WHAM

echo ""
echo "========================================"
echo "Testing DPVO Installation..."
echo "========================================"

python -c "
try:
    from lib.models.preproc.slam import SLAMModel
    print('✅ ✅ ✅ DPVO SUCCESSFULLY INSTALLED!')
    print('✅ World-grounded 3D motion now available!')
    print('')
    print('You can now use --run_global flag in demo.py')
except ImportError as e:
    print(f'❌ DPVO import failed: {e}')
    print('')
    print('Check logs at: /tmp/dpvo_install_${SLURM_JOB_ID}.log')
    import sys
    sys.exit(1)
"

exit_code=$?

# Clean up
rm -rf $TMPDIR
echo ""
echo "Cleaned up temp directory"

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "🎉 DPVO INSTALLATION SUCCESSFUL! 🎉"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "❌ DPVO Installation Failed"
    echo "Check log: /tmp/dpvo_install_${SLURM_JOB_ID}.log"
    echo "========================================"
fi

exit $exit_code

