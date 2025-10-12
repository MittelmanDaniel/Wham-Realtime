#!/bin/bash
#SBATCH --job-name=wham_sanity
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/sanity_%j.out
#SBATCH --error=logs/sanity_%j.err
## #SBATCH --account=<YOUR_ACCOUNT>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load anaconda3/2023.03 || true
eval "$(conda shell.bash hook)"
conda activate wham

python - <<'PY'
import torch
print('Torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.mm(x, x)
    print('Matmul OK, mean:', y.mean().item())
PY


