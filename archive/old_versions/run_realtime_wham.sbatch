#!/bin/bash
#SBATCH --job-name=realtime_wham
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/realtime_wham_%j.out
#SBATCH --error=logs/realtime_wham_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs || true

module load anaconda3/2023.03 || true
eval "$(conda shell.bash hook)"
conda activate wham

echo "========================================"
echo "REAL-TIME WHAM TEST"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "========================================"

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Running REAL-TIME WHAM with actual pose estimation..."
echo "Input: examples/IMG_9732_portrait.mov"
echo "Processing with detection + tracking..."
echo ""

# Test 1: Process every other frame (15 FPS)
echo "Test 1: frame_skip=2 (15 FPS processing)"
echo "----------------------------------------"
python realtime_wham.py \
    examples/IMG_9732_portrait.mov \
    --output output/realtime_wham_test1 \
    --buffer-size 16 \
    --frame-skip 2 \
    --max-fps 30 \
    --duration 20

echo ""
echo ""

# Test 2: Process every 5th frame (6 FPS)
echo "Test 2: frame_skip=5 (6 FPS processing)"
echo "----------------------------------------"
python realtime_wham.py \
    examples/IMG_9732_portrait.mov \
    --output output/realtime_wham_test2 \
    --buffer-size 16 \
    --frame-skip 5 \
    --max-fps 30 \
    --duration 20

echo ""
echo "========================================"
echo "REAL-TIME WHAM TESTS COMPLETE"
echo "========================================"
echo "Check outputs:"
echo "  - output/realtime_wham_test1/"
echo "  - output/realtime_wham_test2/"
echo "========================================"
