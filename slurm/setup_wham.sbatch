#!/bin/bash
#SBATCH --job-name=wham_setup
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
## If your project requires an account, uncomment and set:
## #SBATCH --account=<YOUR_ACCOUNT>

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load anaconda3/2023.03 || true

# Ensure conda is available in non-interactive shell
eval "$(conda shell.bash hook)"

ENV_NAME=wham

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.9
fi

conda activate "$ENV_NAME"

# Pin CUDA/PyTorch to versions recommended by WHAM docs
conda install -y -c pytorch pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3

# Optional utils and a compatible compiler for C/C++ extensions
conda install -y -c conda-forge fvcore iopath gxx=9.5

# Install WHAM Python dependencies
pip install -r requirements.txt

# Install ViTPose (editable)
pip install -v -e third-party/ViTPose

# Install DPVO from source with CUDA 11.3 toolchain
pushd third-party/DPVO
wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip -q eigen-3.4.0.zip -d thirdparty && rm -f eigen-3.4.0.zip

conda install -y -c rusty1s pytorch-scatter=2.0.9
conda install -y -c conda-forge cudatoolkit-dev=11.3.1

export CUDA_HOME="$CONDA_PREFIX"
export FORCE_CUDA=1
# Target V100 (sm_70) for compilation on ice-gpu v100 nodes
export TORCH_CUDA_ARCH_LIST="7.0"

pip install .
popd

echo "Setup complete. To use the environment:"
echo "  module load anaconda3/2023.03"
echo "  eval \"\$(conda shell.bash hook)\" && conda activate $ENV_NAME"


