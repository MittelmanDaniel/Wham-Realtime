#!/bin/bash
#SBATCH --job-name=test_fbf
#SBATCH --output=logs/test_fbf_%j.out
#SBATCH --error=logs/test_fbf_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:10:00
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice

echo "========================================"
echo "Testing Frame-by-Frame WHAM"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "========================================"

cd /home/hice1/dmittelman6/WHAM

module load anaconda3/2023.03
eval "$(conda shell.bash hook)"
conda activate wham

python test_framebyframe.py

echo ""
echo "Test complete!"


