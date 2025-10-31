#!/bin/bash
#SBATCH --job-name=test_lstm
#SBATCH --output=logs/test_lstm_%j.out
#SBATCH --error=logs/test_lstm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:10:00
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice

cd /home/hice1/dmittelman6/WHAM
mkdir -p logs output/framebyframe_wham

module load anaconda3/2023.03
eval "$(conda shell.bash hook)"
conda activate wham

export CUDA_VISIBLE_DEVICES=0

python realtime_wham_framebyframe.py \
    examples/IMG_9732.mov \
    --output-dir output/framebyframe_wham \
    --frame-skip 1 \
    --max-fps 30 \
    --duration 10

echo "Done! Check if LSTM reset messages appear in logs."


