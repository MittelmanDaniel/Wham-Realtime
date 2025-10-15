#!/bin/bash

#SBATCH --job-name=viz_fbf
#SBATCH --output=logs/viz_fbf_%j.out
#SBATCH --error=logs/viz_fbf_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:30:00
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice

# Log start
echo "===================================================="
echo "Visualizing Frame-by-Frame SMPL Results"
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

# Run visualization
echo "Visualizing SMPL results..."
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

