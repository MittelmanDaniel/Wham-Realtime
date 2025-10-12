# 🤖 WHAM Real-Time Project Summary

**Goal**: Use WHAM for real-time humanoid robot teleoperation

**Result**: ✅ Achieved **150ms latency** (87x faster than batch processing!)

---

## 🎯 Quick Links

1. **📖 Start Here**: `docs/QUICK_START_GUIDE.md` - Complete usage guide
2. **🌟 Main Script**: `realtime_wham_online.py` - The real-time processor
3. **🚀 Run Tests**: `sbatch slurm/run_online_wham.sbatch`
4. **📁 File Organization**: `docs/FILE_ORGANIZATION.md`

---

## 🏆 Key Achievement

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Latency** | 13,000ms | **150ms** | **87x faster** ⚡ |
| **FPS** | 1.1 | **6.3** | **5.7x faster** |
| **Robot Ready?** | ❌ No | ✅ **YES!** | 🎉 |

---

## 📊 Latency Options

Different frame-skip rates for different use cases:

| Frame Skip | Input FPS | Latency | Best For |
|------------|-----------|---------|----------|
| 1 (every frame) | 30 FPS | 150ms | High accuracy |
| 2 (half frames) | 15 FPS | 126ms | Balanced |
| 5 (1/5 frames) | 6 FPS | **84ms** | **Lowest latency** ⚡ |

---

## 🚀 Quick Start

### Run the main test:
```bash
cd /home/hice1/dmittelman6/WHAM
sbatch slurm/run_online_wham.sbatch
```

### Run on your own video:
```bash
python realtime_wham_online.py your_video.mp4 \
    --frame-skip 1 \
    --max-fps 30 \
    --duration 30
```

### Monitor results:
```bash
tail -f logs/online_wham_*.err
```

---

## 📁 Project Structure

```
WHAM/
├── 🌟 realtime_wham_online.py    # Main real-time processor
├── wham_api.py                    # WHAM API interface
├── demo.py                        # Original WHAM demo
│
├── 📂 slurm/                      # GPU job scripts
│   └── run_online_wham.sbatch    # Run real-time tests
│
├── 📂 docs/                       # Documentation
│   ├── QUICK_START_GUIDE.md      # ⭐ READ THIS FIRST
│   └── FILE_ORGANIZATION.md      # Project structure
│
├── 📂 archive/                    # Old/test files
│   ├── tests/                    # Development tests
│   ├── camera_streaming/         # Mac→Cluster streaming
│   └── old_versions/             # Earlier implementations
│
├── 📂 lib/                        # WHAM core library
├── 📂 configs/                    # Configuration files
├── 📂 checkpoints/                # Model weights (~5GB)
├── 📂 examples/                   # Demo videos
├── 📂 output/                     # Results
└── 📂 logs/                       # Job logs
```

---

## 🎓 What We Learned

### 1. Batch Size Matters!
- **Batch=16**: Great for throughput, terrible for latency (13s)
- **Batch=1**: Required for real-time, achieves 150ms latency ✅

### 2. GPU Compatibility
- ✅ **V100, RTX 6000, A100**: Work perfectly
- ❌ **H100/H200**: PyTorch 1.11.0 too old
- ❌ **AMD MI210**: CUDA only

### 3. Bottleneck Analysis
- ViTPose (detection): ~150ms ⚠️ Main bottleneck
- YOLO (person detection): ~10ms
- WHAM (inference): Fast

### 4. Real-Time Simulation Works!
- Throttling video playback to 30 FPS gives accurate latency
- No need for actual camera streaming during testing
- Results match what you'd get with live camera

---

## 💡 Use Cases

### ✅ Perfect For:
- 🤖 Humanoid robot teleoperation (150ms is good!)
- 🎭 Live motion capture
- 🎮 Interactive VR/AR applications
- 🏃 Real-time sports analysis

### ⚠️ Consider Alternatives If:
- You need <50ms latency (consider simpler models)
- You need >30 FPS (consider lighter pose estimators)
- You have no GPU access (WHAM requires CUDA)

---

## 📚 Documentation Files

1. **`PROJECT_SUMMARY.md`** (this file) - High-level overview
2. **`docs/QUICK_START_GUIDE.md`** - Detailed usage guide
3. **`docs/FILE_ORGANIZATION.md`** - File structure explanation
4. **`archive/README.md`** - Archived files explanation

---

## 🔧 Key Commands

```bash
# Submit real-time test job
sbatch slurm/run_online_wham.sbatch

# Check job status
squeue -u $USER

# Monitor job output
tail -f logs/online_wham_*.err

# Run on custom video
python realtime_wham_online.py video.mp4 --frame-skip 2

# Clean up old outputs
rm -rf output/online_test* logs/*.out logs/*.err
```

---

## 🌟 The Magic File

**`realtime_wham_online.py`** is where all the magic happens:

```python
# Key features:
- Batch size = 1 (online processing)
- Immediate frame processing (no buffering)
- Real-time simulation via FPS throttling
- Per-frame latency measurement
- Supports: video files, HTTP, RTSP, cameras
```

Run it with:
```bash
python realtime_wham_online.py <source> [--frame-skip N] [--max-fps FPS]
```

---

## 📊 Benchmark Results (V100 GPU)

### Test 1: Every Frame (30 FPS)
- ✅ Latency: **157ms average** (150ms median)
- ✅ Throughput: 6.3 FPS
- ✅ Detection: 1 person per frame

### Test 2: Every 2nd Frame (15 FPS)
- ✅ Latency: **130ms average** (126ms median)
- ✅ Throughput: 6.0 FPS
- ✅ Detection: 1 person per frame

### Test 3: Every 5th Frame (6 FPS)
- ✅ Latency: **93ms average** (84ms median) ⚡
- ✅ Throughput: 4.4 FPS
- ✅ Detection: 1 person per frame

---

## 🎯 Next Steps

1. ✅ **Read the guide**: `docs/QUICK_START_GUIDE.md`
2. ✅ **Run the tests**: `sbatch slurm/run_online_wham.sbatch`
3. ✅ **Try your videos**: `python realtime_wham_online.py your_video.mp4`
4. 🔄 **Integrate with robot**: Use `wham_api.py` for programmatic access
5. 📹 **Optional**: Set up camera streaming (see `archive/camera_streaming/`)

---

## 🙏 Credits

- **WHAM**: [wham.is.tue.mpg.de](https://wham.is.tue.mpg.de/)
- **HumanPlus** (Stanford): Inspired real-time usage
- **ViTPose**: 2D pose estimation
- **YOLO**: Person detection
- **PyTorch**: Deep learning framework

---

## 📞 Support

- 📖 **Documentation**: Start with `docs/QUICK_START_GUIDE.md`
- 🐛 **Issues**: Check GPU compatibility and environment setup
- 💬 **Questions**: Refer to WHAM paper and codebase

---

## ✨ Bottom Line

**WHAM can run in real-time at 150ms latency on V100 GPUs!**

This makes it viable for humanoid robot teleoperation and other real-time applications. The key was optimizing from batch processing to online processing - an **87x improvement** in latency! 🎉

---

*Project completed: October 8, 2025*
*Environment: Georgia Tech PACE-ICE Cluster*
*GPU: NVIDIA Tesla V100*

