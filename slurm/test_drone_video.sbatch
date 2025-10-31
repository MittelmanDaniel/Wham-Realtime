#!/bin/bash
#SBATCH --job-name=test_drone
#SBATCH --output=logs/test_drone_%j.out
#SBATCH --error=logs/test_drone_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:15:00
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice

cd /home/hice1/dmittelman6/WHAM
mkdir -p logs output/drone_test

module load anaconda3/2023.03
eval "$(conda shell.bash hook)"
conda activate wham

export CUDA_VISIBLE_DEVICES=0

echo "Testing with drone video (more motion)..."
python realtime_wham_framebyframe.py \
    examples/drone_video.mp4 \
    --output-dir output/drone_test \
    --frame-skip 1 \
    --max-fps 30 \
    --duration 15

echo "Done! Check if drone video shows better pose tracking."


