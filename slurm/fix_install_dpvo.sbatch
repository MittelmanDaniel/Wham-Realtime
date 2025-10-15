#!/bin/bash
#SBATCH --job-name=fix_dpvo
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --time=01:00:00
#SBATCH --output=logs/fix_dpvo_%j.out
#SBATCH --error=logs/fix_dpvo_%j.err

set -euo pipefail

cd /home/hice1/dmittelman6/WHAM
mkdir -p logs

echo "========================================"
echo "Fix Environment & Install DPVO"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "========================================"

# Load modules
module load anaconda3/2023.03
module load cuda/11.8.0

# Run fix script
bash fix_and_install_dpvo.sh

echo ""
echo "========================================"
echo "Done! Check if DPVO works:"
echo "python -c 'from lib.models.preproc.slam import SLAMModel'"
echo "========================================"

