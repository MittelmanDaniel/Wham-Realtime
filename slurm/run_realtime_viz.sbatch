#!/bin/bash
#SBATCH --job-name=realtime_viz
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --time=00:30:00
#SBATCH --output=logs/realtime_viz_%j.out
#SBATCH --error=logs/realtime_viz_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output/realtime_viz || true

module load anaconda3/2023.03 || true
eval "$(conda shell.bash hook)"
conda activate wham

echo "========================================"
echo "REAL-TIME WHAM VISUALIZATION"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "========================================"

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Creating visualization video with detection overlays..."
echo "Input: examples/IMG_9732_portrait.mov"
echo "Output: output/realtime_viz/output.mp4"
echo ""

# Create visualization (process first 300 frames = 10 seconds of video @ 30 FPS)
# No FPS throttling (max-fps 0) so it goes as fast as GPU allows
python visualize_realtime.py \
    examples/IMG_9732_portrait.mov \
    --output output/realtime_viz/output.mp4 \
    --frame-skip 1 \
    --max-fps 0 \
    --max-frames 300

echo ""
echo "========================================"
echo "DONE!"
echo "========================================"
echo "Output video: output/realtime_viz/output.mp4"
echo ""
echo "Download with:"
echo "scp dmittelman6@login-ice.pace.gatech.edu:WHAM/output/realtime_viz/output.mp4 ."
echo "========================================"

