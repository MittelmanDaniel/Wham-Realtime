#!/bin/bash
#SBATCH --job-name=wham_demo
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/wham_demo_%j.out
#SBATCH --error=logs/wham_demo_%j.err

set -euo pipefail

# Change to working directory
cd "${SLURM_SUBMIT_DIR}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module load anaconda3/2023.03

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wham

echo "====== JOB INFO ======"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Working directory: $(pwd)"
echo "======================"

# Verify PyTorch and CUDA
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Running WHAM demo..."
echo "Input video: examples/IMG_9732.mov"
echo "Note: Using --estimate_local_only flag (DPVO not installed)"
echo ""

# Run WHAM demo (without SLAM since DPVO is not installed)
python demo.py \
    --video examples/IMG_9732.mov \
    --visualize \
    --estimate_local_only

echo ""
echo "====== DEMO COMPLETE ======"
echo "Check output in the current directory"
echo "Logs saved to: logs/wham_demo_${SLURM_JOB_ID}.{out,err}"

