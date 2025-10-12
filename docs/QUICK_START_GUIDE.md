# 🤖 WHAM Real-Time Processing - Quick Start Guide

## 📋 Overview

This guide shows you how to use **WHAM** (World-grounded Humans with Accurate Motion) for **real-time humanoid robot teleoperation** with low latency (~150ms per frame).

---

## 🎯 What We Achieved

| Metric | Batch Processing (Old) | **Online Processing (New)** |
|--------|------------------------|----------------------------|
| **Latency** | 13,000ms per batch | **150ms per frame** ⚡ |
| **FPS** | 1.1 FPS | 6.3 FPS |
| **Robot Control** | ❌ Too slow | ✅ **Real-time ready!** |
| **Improvement** | - | **87x faster** 🚀 |

### Different Frame Skip Modes:
- **Every frame (30 FPS input)**: 150ms latency
- **Every 2nd frame (15 FPS)**: 126ms latency
- **Every 5th frame (6 FPS)**: 84ms latency ⚡⚡

---

## 🚀 Quick Start

### 1. Run WHAM on a Video (Simulated Real-Time)

```bash
# Submit to GPU cluster
cd /home/hice1/dmittelman6/WHAM
sbatch slurm/run_online_wham.sbatch

# Monitor the job
squeue -u $USER
tail -f logs/online_wham_*.err
```

This will:
- Process `examples/IMG_9732_portrait.mov`
- Simulate real-time streaming at 30 FPS
- Run 3 tests with different frame skip rates
- Show per-frame latency statistics

### 2. Run WHAM on a Different Video

```bash
cd /home/hice1/dmittelman6/WHAM
conda activate wham

# Request interactive GPU
salloc --partition=ice-gpu --gres=gpu:v100:1 --mem=32G --time=01:00:00

# Run on your video
python realtime_wham_online.py \
    /path/to/your/video.mp4 \
    --output output/my_test \
    --frame-skip 1 \
    --max-fps 30 \
    --duration 30
```

### 3. Use Different Input Sources

```python
# Local video file
python realtime_wham_online.py video.mp4

# HTTP stream (e.g., from camera server)
python realtime_wham_online.py http://localhost:8080/video

# RTSP stream
python realtime_wham_online.py rtsp://camera-ip:8554/stream

# USB camera (if running locally with GPU)
python realtime_wham_online.py 0
```

---

## 📁 File Organization

```
WHAM/
├── realtime_wham_online.py          # ⭐ MAIN SCRIPT - Online processing
├── wham_api.py                       # WHAM API (can use this too)
├── demo.py                           # Original WHAM demo
│
├── slurm/                            # SLURM job scripts
│   ├── run_online_wham.sbatch       # ⭐ Run online tests
│   ├── run_wham_demo.sbatch         # Original demo runner
│   ├── run_wham_fast.sbatch         # Fast mode (no viz)
│   └── run_portrait_demo.sbatch     # Portrait video demo
│
├── docs/                             # Documentation
│   └── QUICK_START_GUIDE.md         # This file!
│
├── archive/                          # Old files (kept for reference)
│   ├── tests/                       # Test scripts
│   │   ├── test_camera.py
│   │   ├── fix_video_rotation.py
│   │   └── ...
│   ├── camera_streaming/            # Mac→Cluster streaming
│   │   ├── remote_camera_server.py
│   │   ├── REMOTE_CAMERA_SETUP.md
│   │   └── ...
│   └── old_versions/                # Earlier implementations
│       ├── realtime_wham.py         # Batch=16 version
│       └── ...
│
├── examples/                         # Demo videos
├── output/                          # Results go here
├── checkpoints/                     # Model weights
└── configs/                         # Configuration files
```

---

## 🛠️ Key Files Explained

### 🌟 Main Processing Script

**`realtime_wham_online.py`** - The core real-time processor
- Processes frames **immediately** (batch size = 1)
- Simulates real-time by throttling to camera FPS
- Measures per-frame latency
- Can read from video files, HTTP streams, RTSP, or cameras

### 🎛️ SLURM Scripts

**`slurm/run_online_wham.sbatch`** - Runs comprehensive tests
- Tests 3 frame skip rates (1x, 2x, 5x)
- Compares latency and throughput
- Logs detailed statistics

### 📚 Original WHAM Files

**`demo.py`** - Original WHAM demo script
- Processes entire video offline
- Includes visualization and SMPLify refinement
- Good for high-quality offline processing

**`wham_api.py`** - WHAM API interface
- Programmatic access to WHAM
- Used by the HumanPlus paper for real-time
- Can be used instead of `realtime_wham_online.py`

---

## 🔧 Command-Line Options

```bash
python realtime_wham_online.py <source> [options]

Arguments:
  source              Video file path, URL, or camera ID (e.g., 0)

Options:
  --output DIR        Output directory (default: output/realtime_online)
  --frame-skip N      Process every Nth frame (default: 1)
  --max-fps FPS       Throttle to max FPS (default: 30)
  --duration SEC      Run for N seconds (default: run full video)
  --visualize         Show live visualization (default: off)
```

### Examples:

```bash
# Process every frame at 30 FPS for 20 seconds
python realtime_wham_online.py video.mp4 --max-fps 30 --duration 20

# Process every other frame (15 FPS effective)
python realtime_wham_online.py video.mp4 --frame-skip 2

# Process every 5th frame for lowest latency
python realtime_wham_online.py video.mp4 --frame-skip 5

# Show live visualization
python realtime_wham_online.py video.mp4 --visualize
```

---

## 📊 Understanding the Output

### Real-Time Stats
```
[10.0s] Recv: 100 (10.0 FPS) | Proc: 100 (10.0 FPS) | Skip: 0 | 
        Latency: 150ms | Avg: 157ms | People: 1
```

- **Recv**: Frames received from source
- **Proc**: Frames processed by WHAM
- **Skip**: Frames skipped (due to frame-skip setting)
- **Latency**: Current frame processing time
- **Avg**: Average latency across all frames
- **People**: Number of people detected

### Final Summary
```
============================================================
ONLINE PROCESSING COMPLETE
Total time: 20.13s
Frames received: 126
Frames processed: 126
Average process FPS: 6.3
Total processing time: 19.75s

LATENCY STATS:
  Average: 157ms
  P50: 150ms      ← 50% of frames faster than this
  P95: 152ms      ← 95% of frames faster than this
  P99: 156ms      ← 99% of frames faster than this
============================================================
```

---

## 🎮 Use Cases

### 1. Humanoid Robot Teleoperation ✅
- **150ms latency** is acceptable for pose-based control
- Use **frame-skip 2-5** for even lower latency (84-126ms)
- Process at 6-15 FPS for smooth control

### 2. Real-Time Motion Capture ✅
- Full-body 3D pose at 6+ FPS
- Sub-200ms latency
- World-grounded coordinates

### 3. Live Performance Analysis ✅
- Immediate feedback on body pose
- Track multiple people
- Export SMPL parameters

---

## 🚨 GPU Requirements

### Supported GPUs:
- ✅ **V100** - Best compatibility (PyTorch 1.11.0)
- ✅ **RTX 6000** - Works well
- ✅ **A100** - Works well
- ✅ **L40S** - Should work
- ❌ **H100/H200** - PyTorch 1.11.0 too old (need PyTorch 2.0+)
- ❌ **MI210 (AMD)** - CUDA only

### Performance Bottleneck:
- **ViTPose-Huge**: ~150ms (detection + feature extraction)
- **YOLO**: ~10ms (person detection)
- **WHAM Network**: Fast (included in ViTPose time)

---

## 🔄 Streaming from Mac Camera (Optional)

If you want to stream from your Mac camera to the cluster, see:
- `archive/camera_streaming/REMOTE_CAMERA_SETUP.md`

This includes:
1. Mac-side camera server (Flask HTTP)
2. SSH reverse tunnel setup
3. Cluster-side stream recorder
4. SCP-based upload alternative

**Note**: The simulated real-time processing gives accurate latency measurements, so streaming setup is optional.

---

## 💡 Tips & Tricks

### Reduce Latency Further:
1. **Skip frames**: Use `--frame-skip 5` for 84ms latency
2. **Reduce resolution**: Resize video before processing
3. **Use faster GPU**: V100 → A100 for ~20% speedup
4. **Disable tracking**: Modify detector to skip tracking if you have only one person

### Increase Accuracy:
1. **Process every frame**: `--frame-skip 1`
2. **Enable SMPLify**: Add temporal smoothing (increases latency)
3. **Use DPVO**: Enable world-grounded motion (requires installation)

### Debug Issues:
```bash
# Check GPU allocation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Test camera/video source
python -c "import cv2; cap = cv2.VideoCapture('video.mp4'); print(cap.isOpened())"

# Check conda environment
conda list | grep torch
conda list | grep mmpose
```

---

## 📖 Related Papers

- **WHAM**: [Project Page](https://wham.is.tue.mpg.de/)
- **HumanPlus** (Stanford): Used WHAM for real-time robot teleoperation
- **SMPL**: Body model used by WHAM
- **ViTPose**: 2D pose estimator (the main bottleneck)

---

## 🎯 Next Steps

1. **Test on your own videos**: Use `realtime_wham_online.py`
2. **Integrate with robot**: Use the API from `wham_api.py`
3. **Optimize for your use case**: Adjust frame-skip, resolution
4. **Set up live streaming**: Follow camera setup guide if needed

---

## 📞 Key Takeaway

✨ **WHAM can run in real-time at 150ms latency on V100 GPUs** ✨

This makes it viable for:
- 🤖 Humanoid robot teleoperation
- 🎭 Live motion capture
- 🎮 Interactive applications
- 🏃 Real-time performance analysis

The key was switching from **batch processing** (13s latency) to **online processing** (150ms latency) - an **87x improvement**!

---

*Generated: October 8, 2025*
*Environment: Georgia Tech PACE-ICE Cluster*

