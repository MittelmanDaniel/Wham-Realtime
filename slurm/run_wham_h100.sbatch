#!/bin/bash
#SBATCH --job-name=wham_h100
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/wham_h100_%j.out
#SBATCH --error=logs/wham_h100_%j.err

set -euo pipefail

# Change to working directory
cd "${SLURM_SUBMIT_DIR}"

# Load modules
module load anaconda3/2023.03

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wham

echo "====== H100 GPU BENCHMARK ======"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "================================"

# Verify PyTorch and CUDA
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')"

echo ""
echo "Running WHAM on H100 (NO VISUALIZATION - Data Only)..."
echo "Input video: examples/IMG_9732.mov"
echo ""

# Start timer
START_TIME=$(date +%s)

# Run WHAM WITHOUT visualization
python demo.py \
    --video examples/IMG_9732.mov \
    --output_pth output/h100_benchmark \
    --save_pkl \
    --estimate_local_only

# End timer
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "====== H100 BENCHMARK RESULTS ======"
echo "Processing time: ${ELAPSED} seconds"
echo "Video length: ~22 seconds"
echo ""
echo "Speed ratio: $(echo "scale=2; $ELAPSED / 22" | bc)x realtime"
echo "FPS estimate: $(echo "scale=2; 22 * 30 / $ELAPSED" | bc) FPS (assuming 30fps input)"
echo ""
echo "========== COMPARISON =========="
echo "RTX 6000: 215 seconds (3.1 FPS)"
echo "H100:     ${ELAPSED} seconds ($(echo "scale=2; 22 * 30 / $ELAPSED" | bc) FPS)"
echo "Speedup:  $(echo "scale=2; 215 / $ELAPSED" | bc)x faster! 🚀"
echo "==============================="

# Show output files
echo ""
echo "Output files:"
ls -lh output/h100_benchmark/ 2>/dev/null || echo "Output directory not found"
