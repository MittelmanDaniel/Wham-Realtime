# 📁 WHAM Project File Organization

## 🗂️ Directory Structure

```
WHAM/
├── 🌟 realtime_wham_online.py       # Main real-time processor (BATCH=1)
├── wham_api.py                       # WHAM API interface
├── demo.py                           # Original WHAM demo script
├── train.py                          # Training script (not used)
│
├── 📂 slurm/                         # SLURM job submission scripts
│   ├── run_online_wham.sbatch       # ⭐ Run real-time tests
│   ├── run_wham_demo.sbatch         # Original demo
│   ├── run_wham_fast.sbatch         # Fast mode (no viz)
│   ├── run_wham_h100.sbatch         # H100-specific (has issues)
│   └── run_portrait_demo.sbatch     # Portrait video demo
│
├── 📂 docs/                          # Documentation
│   ├── QUICK_START_GUIDE.md         # ⭐ Start here!
│   └── FILE_ORGANIZATION.md         # This file
│
├── 📂 archive/                       # Archived/old files
│   ├── tests/                       # Test scripts
│   ├── camera_streaming/            # Mac→Cluster streaming
│   └── old_versions/                # Previous implementations
│
├── 📂 lib/                           # WHAM core library
├── 📂 configs/                       # Configuration files
├── 📂 checkpoints/                   # Model weights
├── 📂 examples/                      # Demo videos
├── 📂 output/                        # Processing results
└── 📂 logs/                          # SLURM job logs
```

---

## 🌟 Main Files (Root Directory)

### `realtime_wham_online.py` ⭐
**Purpose**: Real-time WHAM processing with online mode (batch=1)
**When to use**: Robot teleoperation, live motion capture, real-time applications
**Key features**:
- 150ms per-frame latency
- Processes frames immediately
- Simulates real-time streaming
- Supports video files, HTTP, RTSP, cameras

### `wham_api.py`
**Purpose**: WHAM API for programmatic access
**When to use**: Integration with other code, HumanPlus-style usage
**Key features**:
- Clean API interface
- Network, detector, extractor initialization
- Used by research papers

### `demo.py`
**Purpose**: Original WHAM demo script
**When to use**: Offline video processing, high-quality results
**Key features**:
- Full visualization
- SMPLify refinement
- DPVO integration
- Batch processing

---

## 📂 slurm/ - GPU Job Scripts

All SLURM batch scripts for running on the cluster.

### `run_online_wham.sbatch` ⭐
**GPU**: V100
**Time**: 30 min
**What it does**: 
- Runs 3 tests with different frame-skip rates
- Processes `examples/IMG_9732_portrait.mov`
- Measures latency statistics
- Outputs to `logs/online_wham_*.out`

### `run_wham_demo.sbatch`
**GPU**: V100
**Time**: 1 hour
**What it does**:
- Runs original WHAM demo
- Full visualization and output
- Good for reference

### `run_wham_fast.sbatch`
**GPU**: RTX 6000
**Time**: 1 hour
**What it does**:
- Fast mode without visualization
- Good for quick testing

---

## 📂 docs/ - Documentation

### `QUICK_START_GUIDE.md` ⭐
**The main guide!** Everything you need to get started:
- Quick start instructions
- Command-line options
- Performance benchmarks
- Use cases and tips

### `FILE_ORGANIZATION.md`
This file - explains the project structure.

---

## 📂 archive/ - Archived Files

Files kept for reference but not needed for normal usage.

### `archive/tests/` - Test Scripts
Scripts we created to test various components:

- **`test_camera.py`** - Basic camera test
- **`test_camera_mac.py`** - Mac-specific camera test
- **`fix_video_rotation.py`** - Fix iPhone video rotation
- **`realtime_simulation.py`** - Early real-time simulation
- **`ffmpeg_streamer.py`** - Stream video file via HTTP
- **`realtime_processor.py`** - Generic processor template
- **`test_realtime_stream.sbatch`** - Test streaming framework

### `archive/camera_streaming/` - Mac→Cluster Streaming
Everything needed to stream from Mac camera to cluster:

- **`REMOTE_CAMERA_SETUP.md`** - Complete setup guide
- **`remote_camera_server.py`** - Mac: Flask HTTP camera server
- **`stream_recorder.py`** - Cluster: Record from HTTP stream
- **`record_and_upload.py`** - Mac: Record and upload via SCP
- **`cluster_watcher.py`** - Cluster: Watch for uploaded videos

**Note**: Streaming setup is optional since simulated real-time gives accurate latency.

### `archive/old_versions/` - Previous Implementations
Earlier versions of the real-time processor:

- **`realtime_wham.py`** - Batch=16 version (13s latency)
- **`realtime_demo.py`** - Early demo version
- **`run_realtime_wham.sbatch`** - SLURM script for batch version
- **`REALTIME_STREAMING.md`** - Old documentation

**Why archived**: The new `realtime_wham_online.py` with batch=1 is 87x faster!

---

## 📂 lib/ - WHAM Core Library

WHAM's internal implementation (came with the repo):
- `lib/models/` - Neural network models
- `lib/data/` - Data loading and processing
- `lib/utils/` - Utility functions
- `lib/models/preproc/` - Preprocessing (detector, extractor, SLAM)

**Don't modify unless you know what you're doing!**

---

## 📂 configs/ - Configuration Files

YAML configuration files:
- `configs/yamls/demo.yaml` - Demo configuration
- `configs/yamls/train.yaml` - Training configuration

**Key setting**: `TRAIN.CHECKPOINT` points to model weights.

---

## 📂 checkpoints/ - Model Weights

Pre-trained model checkpoints:
- `wham_vit_w_3dpw.pth.tar` - Main WHAM model ⭐
- `hmr2a.ckpt` - HMR2 backbone
- `yolov8x.pt` - Person detector
- `vitpose-h-multi-coco.pth` - ViTPose detector

**Size**: ~5GB total

---

## 📂 examples/ - Demo Videos

Sample videos for testing:
- `IMG_9732_portrait.mov` - Portrait video (corrected rotation) ⭐
- `IMG_9732.mov` - Original landscape video
- `IMG_9730.mov` - Another test video

---

## 📂 output/ - Processing Results

All processing outputs go here:
- `output/online_test1/` - Every frame test
- `output/online_test2/` - Every 2nd frame test
- `output/online_test3/` - Every 5th frame test
- `output/demo/` - Original demo outputs
- `output/demo_portrait/` - Portrait demo outputs

**Contents**: Videos, SMPL parameters, visualizations

---

## 📂 logs/ - SLURM Job Logs

SLURM job outputs:
- `logs/online_wham_*.out` - Standard output
- `logs/online_wham_*.err` - Error output (includes timing stats)

**Tip**: Use `tail -f logs/online_wham_*.err` to monitor running jobs.

---

## 🧹 Cleanup Commands

If you want to clean up:

```bash
# Remove old test outputs
rm -rf output/online_test* output/demo output/demo_portrait

# Remove old log files
rm logs/*.out logs/*.err

# Archive is already organized, leave as-is
```

---

## 📋 File Count Summary

```
Main files:          4 files
SLURM scripts:       5 files
Documentation:       2 files
Archive (tests):     7 files
Archive (streaming): 5 files
Archive (old):       4 files
Core library:        ~50 files
Configs:             ~10 files
Checkpoints:         4 files
Examples:            3 files
```

**Total**: ~95 files (including library code)
**User-created**: 27 files (organized into archive/)

---

## 🎯 What You Actually Need

For normal usage, you only need:
1. ✅ `realtime_wham_online.py` - Main processor
2. ✅ `slurm/run_online_wham.sbatch` - Run tests
3. ✅ `docs/QUICK_START_GUIDE.md` - Documentation

Everything else is:
- Original WHAM code (lib/, configs/, etc.)
- Model weights (checkpoints/)
- Test data (examples/)
- Archives (archive/)

---

*Last updated: October 8, 2025*

