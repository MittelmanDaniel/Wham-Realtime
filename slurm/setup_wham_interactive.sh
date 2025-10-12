#!/bin/bash
set -euo pipefail

# Run this inside an salloc/srun GPU session
cd "$(dirname "$0")/.."
module load anaconda3/2023.03 || true
eval "$(conda shell.bash hook)"

ENV_NAME=wham
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.9
fi
conda activate "$ENV_NAME"

conda install -y -c pytorch pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3
conda install -y -c conda-forge fvcore iopath gxx=9.5

pip install -r requirements.txt
pip install -v -e third-party/ViTPose

pushd third-party/DPVO
wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip -q eigen-3.4.0.zip -d thirdparty && rm -f eigen-3.4.0.zip
conda install -y -c rusty1s pytorch-scatter=2.0.9
conda install -y -c conda-forge cudatoolkit-dev=11.3.1
export CUDA_HOME="$CONDA_PREFIX"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="7.0"
pip install .
popd

echo "Interactive setup complete."


