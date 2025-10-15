#!/bin/bash

#SBATCH --job-name=framebyframe_wham
#SBATCH --output=logs/framebyframe_%j.out
#SBATCH --error=logs/framebyframe_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=01:00:00
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice

# Log start
echo "===================================================="
echo "Starting TRUE Frame-by-Frame WHAM with LSTM"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOBID"
echo "===================================================="

# Create logs dir
mkdir -p logs
mkdir -p output/framebyframe_wham

# Load modules
module load anaconda3/2023.03

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wham

# Verify GPU
echo "GPU Info:"
nvidia-smi
echo ""

# Set device
export CUDA_VISIBLE_DEVICES=0

# Run frame-by-frame WHAM
echo "Running frame-by-frame WHAM on test video..."
echo ""

python realtime_wham_framebyframe.py \
    examples/IMG_9732.mov \
    --output-dir output/framebyframe_wham \
    --frame-skip 1 \
    --max-fps 30

echo ""
echo "===================================================="
echo "Visualizing SMPL results..."
echo "===================================================="
echo ""

python visualize_framebyframe_smpl.py \
    examples/IMG_9732.mov \
    --smpl-results output/framebyframe_wham/smpl_results_final.pkl \
    --timing-stats output/framebyframe_wham/timing_stats.pkl \
    --output output/framebyframe_wham/visualization.mp4

echo ""
echo "===================================================="
echo "Done!"
echo "Time: $(date)"
echo "===================================================="

