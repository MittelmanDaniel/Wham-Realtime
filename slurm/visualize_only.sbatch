#!/bin/bash
#SBATCH --job-name=viz_only
#SBATCH --output=logs/viz_only_%j.out
#SBATCH --error=logs/viz_only_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:20:00
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice

cd /home/hice1/dmittelman6/WHAM
mkdir -p logs

module load anaconda3/2023.03
eval "$(conda shell.bash hook)"
conda activate wham

export CUDA_VISIBLE_DEVICES=0

echo "Visualizing SMPL results..."

python visualize_framebyframe_smpl.py \
    examples/IMG_9732.mov \
    --smpl-results output/framebyframe_wham/smpl_results_final.pkl \
    --timing-stats output/framebyframe_wham/timing_stats.pkl \
    --output output/framebyframe_wham/visualization_fixed.mp4

echo "Done!"




