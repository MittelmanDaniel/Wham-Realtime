#!/bin/bash
#SBATCH --job-name=smpl_3d_demo
#SBATCH --output=logs/smpl_3d_demo_%j.out
#SBATCH --error=logs/smpl_3d_demo_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=01:00:00
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0

# Print job info
echo "---------------------------------------"
echo "Begin Slurm Prolog: $(date)"
echo "Job ID:    $SLURM_JOB_ID"
echo "User ID:   $SLURM_JOB_USER"
echo "Account:   $SLURM_JOB_ACCOUNT"
echo "Job name:  $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "QOS:       $SLURM_JOB_QOS"
echo "---------------------------------------"

echo "========================================"
echo "3D SMPL Mesh Visualization from Demo"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "========================================"

# Change to WHAM directory
cd /home/hice1/dmittelman6/WHAM

# Load conda
module load anaconda3/2023.03

# Initialize conda for this shell
eval "$(conda shell.bash hook)"

# Activate environment
conda activate wham

# Run visualization
python visualize_smpl_3d_from_demo.py output/demo_portrait/IMG_9732_portrait \
    --video examples/IMG_9732_portrait.mov \
    --output output/smpl_3d/output.mp4 \
    --max-frames 300

echo ""
echo "Done! Check output/smpl_3d/output.mp4"

