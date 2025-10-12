#!/bin/bash
#SBATCH --job-name=wham_fast
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/wham_fast_%j.out
#SBATCH --error=logs/wham_fast_%j.err

set -euo pipefail

# Change to working directory
cd "${SLURM_SUBMIT_DIR}"

# Load modules
module load anaconda3/2023.03

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wham

echo "====== FAST MODE BENCHMARK ======"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "================================="

# Verify PyTorch and CUDA
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Running WHAM in FAST MODE (no visualization, data only)..."
echo "Input video: examples/IMG_9732.mov"
echo ""

# Start timer
START_TIME=$(date +%s)

# Run WHAM WITHOUT visualization (data output only)
python demo.py \
    --video examples/IMG_9732.mov \
    --output_pth output/fast_benchmark \
    --save_pkl \
    --estimate_local_only

# End timer
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "====== BENCHMARK RESULTS ======"
echo "Processing time: ${ELAPSED} seconds"
echo "Video length: ~22 seconds"
echo "Speed ratio: $(echo "scale=2; $ELAPSED / 22" | bc)x realtime"
echo "FPS estimate: $(echo "scale=2; 22 * 30 / $ELAPSED" | bc) FPS (assuming 30fps input)"
echo ""
echo "Compare to FULL MODE (with visualization): ~250 seconds"
echo "Speedup: $(echo "scale=2; 250 / $ELAPSED" | bc)x faster!"
echo "==============================="

# Show output files
echo ""
echo "Output files:"
ls -lh output/fast_benchmark/ 2>/dev/null || echo "Output directory not found"
