#!/bin/bash
#SBATCH --job-name=dpvo_scratch
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --time=02:00:00
#SBATCH --output=logs/dpvo_scratch_%j.out
#SBATCH --error=logs/dpvo_scratch_%j.err

set -euo pipefail

cd /home/hice1/dmittelman6/WHAM
mkdir -p logs

echo "========================================"
echo "DPVO Install Using /scratch Space"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "========================================"

# Load modules
module load anaconda3/2023.03
module load cuda/11.8.0

eval "$(conda shell.bash hook)"
conda activate wham

echo ""
echo "Disk Space:"
df -h /scratch /home/hice1/dmittelman6 | grep -v Filesystem
echo ""

# Use /tmp on compute node (lots of space during job)
export BUILD_DIR=/tmp/dpvo_build_${SLURM_JOB_ID}
mkdir -p $BUILD_DIR
export TMPDIR=$BUILD_DIR/tmp
mkdir -p $TMPDIR

echo "Build directory: $BUILD_DIR"
echo "Temp directory: $TMPDIR"
df -h /tmp | grep -v Filesystem
echo ""

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Copy DPVO to scratch for compilation
echo "Copying DPVO to scratch space..."
cp -r third-party/DPVO $SCRATCH_DIR/
cd $SCRATCH_DIR/DPVO

echo ""
echo "Step 1/5: Downloading Eigen..."
if [ ! -d "thirdparty/eigen-3.4.0" ]; then
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip -q eigen-3.4.0.zip -d thirdparty
    rm eigen-3.4.0.zip
    echo "✅ Eigen downloaded"
else
    echo "✅ Eigen already exists"
fi

echo ""
echo "Step 2/5: Installing pytorch-scatter..."
pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

echo ""
echo "Step 3/5: Installing cudatoolkit-dev to /scratch..."
# Install cudatoolkit-dev with packages going to scratch
mkdir -p $SCRATCH_DIR/conda_pkgs
export CONDA_PKGS_DIRS=$SCRATCH_DIR/conda_pkgs
conda install -y cudatoolkit-dev=11.3.1 -c conda-forge 2>&1 | tail -20
echo "✅ cudatoolkit-dev installed"

echo ""
echo "Step 4/5: Installing gxx=9.5 to /scratch..."
conda install -y -c conda-forge gxx=9.5 2>&1 | tail -20
echo "✅ gxx=9.5 installed"

# Update paths to use conda's GCC
export CC=$(which gcc)
export CXX=$(which g++)
export CUDA_HOME=${CONDA_PREFIX}

echo ""
echo "Compiler Info:"
echo "GCC: $(gcc --version | head -1)"
echo "G++: $(g++ --version | head -1)"  
echo "nvcc: $(nvcc --version | grep release)"
echo "CUDA_HOME: $CUDA_HOME"
echo ""

echo "Step 5/5: Compiling DPVO..."
pip install --no-cache-dir -v . 2>&1 | tee $SCRATCH_DIR/dpvo_install.log | tail -50

# Test installation
cd /home/hice1/dmittelman6/WHAM
echo ""
echo "========================================"
echo "Testing DPVO Installation..."
echo "========================================"

python -c "
try:
    from lib.models.preproc.slam import SLAMModel
    print('✅ ✅ ✅ DPVO SUCCESSFULLY INSTALLED!')
    print('✅ 3D world-grounded coordinates available!')
    print('')
    print('DPVO is now functional for WHAM!')
    import sys
    sys.exit(0)
except Exception as e:
    print(f'❌ DPVO import failed: {e}')
    import traceback
    traceback.print_exc()
    print('')
    print(f'Check log: {SCRATCH_DIR}/dpvo_install.log')
    import sys
    sys.exit(1)
" 2>&1

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "========================================"
    echo "🎉🎉🎉 SUCCESS! 🎉🎉🎉"
    echo "========================================"
    echo ""
    echo "DPVO is now installed and working!"
    echo "You can now use --run_global flag for world-grounded motion"
    echo ""
    echo "Cleaning up scratch space..."
    # Keep the install log but remove build artifacts
    rm -rf $SCRATCH_DIR/DPVO $SCRATCH_DIR/tmp
    echo "✅ Cleanup complete"
else
    echo "========================================"
    echo "❌ Installation Failed"
    echo "========================================"
    echo ""
    echo "Build log saved at: $SCRATCH_DIR/dpvo_install.log"
    echo "Build directory: $SCRATCH_DIR"
    echo ""
    echo "You can check the log for detailed error messages"
fi

echo ""
echo "Disk usage after installation:"
df -h /scratch | grep -v Filesystem

exit $exit_code

