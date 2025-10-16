#!/bin/bash
#SBATCH --job-name=test_render
#SBATCH --output=logs/test_render_%j.out
#SBATCH --error=logs/test_render_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:05:00
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice

module load anaconda3/2023.03
eval "$(conda shell.bash hook)"
conda activate wham

python test_render_one_frame.py


