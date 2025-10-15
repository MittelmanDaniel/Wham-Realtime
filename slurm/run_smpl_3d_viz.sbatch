#!/bin/bash
#SBATCH --job-name=smpl_3d_viz
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --time=01:00:00
#SBATCH --output=logs/smpl_3d_viz_%j.out
#SBATCH --error=logs/smpl_3d_viz_%j.err

cd /home/hice1/dmittelman6/WHAM
mkdir -p logs output/smpl_3d

echo "========================================"
echo "3D SMPL Mesh Visualization"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================"

# Load modules
module load anaconda3/2023.03

eval "$(conda shell.bash hook)"
conda activate wham

# Run visualization
python visualize_smpl_3d.py \
    examples/IMG_9732_portrait.mov \
    --output output/smpl_3d/output.mp4 \
    --frame-skip 1 \
    --max-fps 30 \
    --max-frames 300

echo ""
echo "Done! Check output/smpl_3d/output.mp4"


