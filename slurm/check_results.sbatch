#!/bin/bash
#SBATCH --job-name=check_res
#SBATCH --output=logs/check_res_%j.out
#SBATCH --error=logs/check_res_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice

module load anaconda3/2023.03
eval "$(conda shell.bash hook)"
conda activate wham

python check_results.py


