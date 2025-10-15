#!/bin/bash
#SBATCH --job-name=wham_overnight
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --time=08:00:00
#SBATCH --output=logs/overnight_%j.out
#SBATCH --error=logs/overnight_%j.err

echo "========================================"
echo "Fresh WHAM + DPVO Install (Overnight)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "========================================"

cd /home/hice1/dmittelman6/WHAM
mkdir -p logs

# Load modules
module load anaconda3/2023.03
module load cuda/11.8.0

# Use scratch for conda packages and builds
export CONDA_PKGS_DIRS=/storage/ice1/3/2/dmittelman6/conda_pkgs_${SLURM_JOB_ID}
export TMPDIR=/storage/ice1/3/2/dmittelman6/tmp_${SLURM_JOB_ID}
mkdir -p $CONDA_PKGS_DIRS $TMPDIR

echo ""
echo "Step 1: Creating conda environment..."
conda create -n wham python=3.9 --yes

eval "$(conda shell.bash hook)"
conda activate wham

echo ""
echo "Step 2: Installing PyTorch 1.11.0..."
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch --yes

echo ""
echo "Step 3: Installing PyTorch3D..."
conda install -c fvcore -c iopath -c conda-forge fvcore iopath --yes
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html --no-cache-dir

echo ""
echo "Step 4: Installing WHAM requirements..."
pip install -r requirements.txt --no-cache-dir

echo ""
echo "Step 5: Installing ViTPose..."
pip install -v -e third-party/ViTPose --no-cache-dir

echo ""
echo "Step 6: Setting up DPVO..."
cd third-party/DPVO

# Download Eigen if needed
if [ ! -f "thirdparty/eigen-3.4.0/Eigen/Core" ]; then
    echo "Downloading Eigen..."
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip -q eigen-3.4.0.zip -d thirdparty
    rm eigen-3.4.0.zip
fi

echo "Installing pytorch-scatter..."
conda install pytorch-scatter=2.0.9 -c rusty1s --yes

echo "Installing cudatoolkit-dev..."
conda install cudatoolkit-dev=11.3.1 -c conda-forge --yes

# Check GCC version and install gxx=9.5 if needed
gcc_version=$(gcc -dumpversion | cut -d. -f1)
echo "System GCC version: $gcc_version"
if [ "$gcc_version" -gt 10 ]; then
    echo "Installing gxx=9.5..."
    conda install -c conda-forge gxx=9.5 --yes
fi

echo ""
echo "Step 7: Installing DPVO..."
pip install . --no-cache-dir

cd /home/hice1/dmittelman6/WHAM

echo ""
echo "========================================"
echo "Testing Installation..."
echo "========================================"

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print()

try:
    from lib.models.preproc.slam import SLAMModel
    print('✅ DPVO installed successfully!')
    print('✅ 3D world-grounded coordinates available!')
except Exception as e:
    print(f'❌ DPVO import failed: {e}')
    import traceback
    traceback.print_exc()

print()
print('Testing imports...')
from lib.models import WHAM
print('✅ WHAM imported')
from lib.datasets.vitdet_dataset import ViTDetDataset
print('✅ ViTDet imported')
print()
print('🎉 Installation complete!')
"

echo ""
echo "========================================"
echo "Installation finished: $(date)"
echo "========================================"
echo ""
echo "Cleaning up scratch space..."
rm -rf $CONDA_PKGS_DIRS $TMPDIR
echo "Done!"

