#!/bin/bash
#SBATCH --job-name=wham_portrait
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/wham_portrait_%j.out
#SBATCH --error=logs/wham_portrait_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs || true

module load anaconda3/2023.03 || true
eval "$(conda shell.bash hook)"
conda activate wham

echo "====== WHAM PORTRAIT VIDEO DEMO ======"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "======================================"

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Running WHAM on rotated portrait video..."
echo "Input: examples/IMG_9732_portrait.mov (1080x1920)"

python demo.py \
    --video examples/IMG_9732_portrait.mov \
    --output_pth output/demo_portrait \
    --visualize \
    --estimate_local_only

echo ""
echo "====== DEMO COMPLETE ======"
echo "Output: output/demo_portrait/IMG_9732_portrait/output.mp4"
echo "Compare this with the original output to see if rotation is fixed!"
