#!/bin/bash
#SBATCH --job-name=online_wham
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/online_wham_%j.out
#SBATCH --error=logs/online_wham_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs || true

module load anaconda3/2023.03 || true
eval "$(conda shell.bash hook)"
conda activate wham

echo "========================================"
echo "ONLINE WHAM TEST (BATCH=1)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "========================================"

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Running ONLINE WHAM (batch=1 for lowest latency)..."
echo "Input: examples/IMG_9732_portrait.mov"
echo "This processes each frame immediately!"
echo ""

# Test 1: Process every frame (batch=1, no skip)
echo "Test 1: Every frame, batch=1"
echo "-------------------------------"
python realtime_wham_online.py \
    examples/IMG_9732_portrait.mov \
    --output output/online_test1 \
    --frame-skip 1 \
    --max-fps 30 \
    --duration 20

echo ""
echo ""

# Test 2: Process every other frame (batch=1, skip=2)
echo "Test 2: Every other frame, batch=1"
echo "-----------------------------------"
python realtime_wham_online.py \
    examples/IMG_9732_portrait.mov \
    --output output/online_test2 \
    --frame-skip 2 \
    --max-fps 30 \
    --duration 20

echo ""
echo ""

# Test 3: Process every 5th frame (batch=1, skip=5)
echo "Test 3: Every 5th frame, batch=1"
echo "---------------------------------"
python realtime_wham_online.py \
    examples/IMG_9732_portrait.mov \
    --output output/online_test3 \
    --frame-skip 5 \
    --max-fps 30 \
    --duration 20

echo ""
echo "========================================"
echo "ONLINE WHAM TESTS COMPLETE"
echo "========================================"
echo "Compare latency vs batch=16 version!"
echo "========================================"
