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
echo "DPVO Install Using 300GB Scratch Space!"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "========================================"

# Load modules
module load anaconda3/2023.03
module load cuda/11.8.0

eval "$(conda shell.bash hook)"
conda activate wham

# Disable unbound variable checking - conda activation scripts have issues with it
set +u

echo ""
echo "Storage Check:"
df -h /storage/ice1/3/2/dmittelman6 | tail -1
echo ""

# Use SCRATCH for ALL build operations (300GB!)
export SCRATCH_BUILD=/storage/ice1/3/2/dmittelman6/dpvo_build_${SLURM_JOB_ID}
mkdir -p $SCRATCH_BUILD
export TMPDIR=$SCRATCH_BUILD/tmp
mkdir -p $TMPDIR

# Set conda package cache to scratch
export CONDA_PKGS_DIRS=$SCRATCH_BUILD/conda_pkgs
mkdir -p $CONDA_PKGS_DIRS

echo "Build directory: $SCRATCH_BUILD"
echo "Temp directory: $TMPDIR"
echo "Conda packages: $CONDA_PKGS_DIRS"
echo ""

# Copy DPVO source to scratch
echo "Copying DPVO to scratch space..."
cp -r /home/hice1/dmittelman6/WHAM/third-party/DPVO $SCRATCH_BUILD/
cd $SCRATCH_BUILD/DPVO

# Set CUDA paths
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"
echo ""

# Download Eigen
if [ ! -f "thirdparty/eigen-3.4.0/Eigen/Core" ]; then
    echo "Downloading Eigen 3.4.0..."
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip -q eigen-3.4.0.zip -d thirdparty
    rm eigen-3.4.0.zip
    echo "✅ Eigen downloaded"
else
    echo "✅ Eigen already exists"
fi

# Install pytorch-scatter
echo ""
echo "Installing pytorch-scatter..."
python -c "import torch_scatter" 2>/dev/null || {
    conda install -y pytorch-scatter=2.0.9 -c rusty1s
}
echo "✅ pytorch-scatter ready"

# Install GCC 9.5 (simpler approach)
echo ""
echo "Installing GCC 9.5..."
gcc_version=$(gcc -dumpversion | cut -d. -f1)
echo "System GCC: $gcc_version"

if [ "$gcc_version" -gt 10 ]; then
    echo "Installing gxx=9.5..."
    conda install -y -c conda-forge gxx=9.5
    
    # Use conda GCC
    export CC=${CONDA_PREFIX}/bin/gcc
    export CXX=${CONDA_PREFIX}/bin/g++
    export CFLAGS="-I${CONDA_PREFIX}/include"
    export CXXFLAGS="-I${CONDA_PREFIX}/include"
    export LDFLAGS="-L${CONDA_PREFIX}/lib"
    
    # Ensure conda bin is first in PATH
    export PATH=${CONDA_PREFIX}/bin:$PATH
    
    echo "✅ Using Conda GCC:"
    echo "GCC: $(gcc --version | head -1)"
    echo "G++: $(g++ --version | head -1)"
else
    echo "✅ System GCC is compatible (version $gcc_version)"
fi

echo "nvcc: $(nvcc --version | grep release)"
echo ""

# Patch DPVO setup.py to add -allow-unsupported-compiler flag
echo "Patching DPVO setup.py to allow GCC version mismatch..."
sed -i "s/'cxx':  \['-O3'\]/'cxx':  ['-O3']/g" setup.py
sed -i "s/'nvcc': \['-O3'\]/'nvcc': ['-O3', '-allow-unsupported-compiler']/g" setup.py
echo "✅ setup.py patched"

echo ""
echo "Compiling DPVO in scratch space (5-10 minutes)..."
pip install --no-cache-dir -v . 2>&1 | tee $SCRATCH_BUILD/dpvo_install.log | tail -100

# Test from home directory
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
    print('DPVO compiled in scratch and installed to environment!')
    import sys
    sys.exit(0)
except Exception as e:
    print(f'❌ DPVO import failed: {e}')
    import traceback
    traceback.print_exc()
    print('')
    print(f'Build log: {SCRATCH_BUILD}/dpvo_install.log')
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
    echo "Built using 300GB scratch space"
    echo ""
    echo "You can now use:"
    echo "  python demo.py --run_global"
    echo "for 3D world-grounded motion!"
    echo ""
    echo "Cleaning up scratch space..."
    rm -rf $SCRATCH_BUILD
    echo "✅ Cleanup complete"
else
    echo "========================================"
    echo "❌ Installation Failed"
    echo "========================================"
    echo ""
    echo "Build log saved: $SCRATCH_BUILD/dpvo_install.log"
    echo "Scratch dir preserved: $SCRATCH_BUILD"
    echo ""
    echo "You can investigate the error or clean up manually"
fi

echo ""
echo "Final scratch disk usage:"
df -h /storage/ice1/3/2/dmittelman6 | tail -1
echo ""

exit $exit_code

