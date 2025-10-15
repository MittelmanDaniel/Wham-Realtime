#!/bin/bash
#SBATCH --job-name=install_dpvo
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/install_dpvo_%j.out
#SBATCH --error=logs/install_dpvo_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "========================================"
echo "DPVO Installation Job"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "========================================"

# Load modules
module load anaconda3/2023.03
module load cuda/11.8.0

eval "$(conda shell.bash hook)"
conda activate wham

# Show environment
echo ""
echo "Python: $(which python)"
echo "CUDA_HOME: ${CUDA_HOME:-Not set}"
echo "nvcc: $(which nvcc 2>/dev/null || echo 'Not found')"
echo ""

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME}" ]; then
    export CUDA_HOME=/usr/local/cuda-11.8
fi

# Use /tmp for build (more space)
export TMPDIR=/tmp/dpvo_build_${SLURM_JOB_ID}
mkdir -p $TMPDIR
echo "Temp dir: $TMPDIR"
echo ""

# Clean any previous failed attempts
rm -rf third-party/DPVO

# Clone DPVO
echo "Cloning DPVO..."
mkdir -p third-party
cd third-party
git clone --recursive https://github.com/princeton-vl/DPVO.git
cd ..

echo ""
echo "Installing DPVO components..."
echo "----------------------------------------"

# Install DPViewer
echo "1/2: Installing DPViewer..."
cd third-party/DPVO/DPViewer
pip install --no-cache-dir --no-build-isolation . 2>&1 | tail -20

# Install DPVO (includes lietorch)
echo ""
echo "2/2: Installing DPVO..."
cd ..
pip install --no-cache-dir --no-build-isolation . 2>&1 | tail -20

cd /home/hice1/dmittelman6/WHAM

echo ""
echo "========================================"
echo "Testing DPVO Installation..."
echo "========================================"
python -c "
try:
    from lib.models.preproc.slam import SLAMModel
    print('✅ DPVO successfully installed and importable!')
    print('✅ World-grounded motion now available!')
except Exception as e:
    print(f'❌ DPVO import failed: {e}')
    import sys
    sys.exit(1)
"

# Clean up
rm -rf $TMPDIR
echo ""
echo "Cleaned up temp directory"
echo "========================================"
echo "DPVO Installation Complete!"
echo "========================================"

