#!/bin/bash
#SBATCH --job-name=wham_stream_test
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/stream_test_%j.out
#SBATCH --error=logs/stream_test_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs || true

module load anaconda3/2023.03 || true
eval "$(conda shell.bash hook)"
conda activate wham

echo "====== REAL-TIME STREAMING TEST ======"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "======================================"

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Test 1: Simulating real-time from video file"
echo "=============================================="
python realtime_simulation.py \
    examples/IMG_9732_portrait.mov \
    --output output/test_simulation \
    --frame_skip 2

echo ""
echo ""
echo "Test 2: HTTP Stream (background)"
echo "================================="
echo "Starting ffmpeg streamer in background..."
python ffmpeg_streamer.py examples/IMG_9732_portrait.mov \
    --port 5555 \
    --no-loop &

STREAMER_PID=$!
echo "Streamer PID: $STREAMER_PID"

# Wait for streamer to start
echo "Waiting for stream to start..."
sleep 10

echo ""
echo "Processing stream for 20 seconds..."
timeout 20 python realtime_processor.py \
    http://localhost:5555/video.mjpeg \
    --output output/test_stream \
    --buffer-size 16 \
    --frame-skip 2 || true

# Stop streamer
echo ""
echo "Stopping streamer..."
kill $STREAMER_PID 2>/dev/null || true

echo ""
echo "====== TESTS COMPLETE ======"
echo "Check output directories:"
echo "  - output/test_simulation/"
echo "  - output/test_stream/"
