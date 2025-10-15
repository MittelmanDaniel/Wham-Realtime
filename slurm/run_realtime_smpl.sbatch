#!/bin/bash
#SBATCH --job-name=realtime_smpl
#SBATCH --output=logs/realtime_smpl_%j.out
#SBATCH --error=logs/realtime_smpl_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=01:00:00
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0

echo "========================================"
echo "Real-time WHAM with SMPL Parameters"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "========================================"

# Change to WHAM directory
cd /home/hice1/dmittelman6/WHAM

# Load conda
module load anaconda3/2023.03

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment
conda activate wham

# Run real-time WHAM with SMPL output
python realtime_wham_with_smpl.py \
    examples/IMG_9732_portrait.mov \
    --output-dir output/realtime_smpl_test \
    --frame-skip 2 \
    --max-fps 30 \
    --duration 30 \
    --window-size 16 \
    --save-interval 50

echo ""
echo "Done! Check output/realtime_smpl_test/ for results"

