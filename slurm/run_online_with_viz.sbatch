#!/bin/bash
#SBATCH --job-name=online_viz
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/online_viz_%j.out
#SBATCH --error=logs/online_viz_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output/online_viz || true

module load anaconda3/2023.03 || true
eval "$(conda shell.bash hook)"
conda activate wham

echo "========================================"
echo "ONLINE WHAM WITH LIVE VISUALIZATION"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "========================================"

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Running ONLINE WHAM with visualization overlay..."
echo "Note: Visualization window requires X11 forwarding or VNC"
echo ""

# Process every 5th frame with live visualization
python realtime_wham_online.py \
    examples/IMG_9732_portrait.mov \
    --output output/online_viz \
    --frame-skip 5 \
    --max-fps 30 \
    --duration 10 \
    --visualize

echo ""
echo "========================================"
echo "DONE! Check output/online_viz/"
echo "========================================"

