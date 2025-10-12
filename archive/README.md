# 📦 Archive Directory

This directory contains scripts and files created during development and testing. They're kept for reference but not needed for normal WHAM usage.

---

## 📁 Directory Structure

```
archive/
├── tests/              # Test and utility scripts
├── camera_streaming/   # Mac→Cluster camera streaming setup
└── old_versions/       # Earlier implementations (batch=16)
```

---

## 🧪 tests/ - Test Scripts

Development and testing scripts:

- **Video Tools**:
  - `fix_video_rotation.py` - Fix iPhone video rotation metadata
  - `ffmpeg_streamer.py` - Stream video file via HTTP/MJPEG
  
- **Camera Tests**:
  - `test_camera.py` - Basic OpenCV camera test
  - `test_camera_mac.py` - Mac-specific camera test with delays
  
- **Simulation**:
  - `realtime_simulation.py` - Early real-time streaming simulation
  - `realtime_processor.py` - Generic real-time processor template
  - `test_realtime_stream.sbatch` - SLURM script for simulation tests

**When to use**: Reference for building custom tools or debugging.

---

## 📹 camera_streaming/ - Remote Camera Setup

Complete setup for streaming Mac camera to cluster for processing:

### Files:
- **`REMOTE_CAMERA_SETUP.md`** - Complete setup guide
- **`remote_camera_server.py`** - Mac: Flask HTTP server for camera
- **`stream_recorder.py`** - Cluster: Records from HTTP stream
- **`record_and_upload.py`** - Mac: Records clips and uploads via SCP
- **`cluster_watcher.py`** - Cluster: Watches for uploaded videos

### Architecture:
```
Mac Camera → HTTP Stream → SSH Tunnel → Cluster → WHAM Processing
     or
Mac Camera → Record Clips → SCP Upload → Cluster → WHAM Processing
```

### Why Optional:
The main `realtime_wham_online.py` script can simulate real-time streaming from local video files with accurate latency measurements. Setting up actual streaming is only needed if you want to use a live camera feed.

---

## 🔄 old_versions/ - Earlier Implementations

Previous versions of the real-time WHAM processor:

- **`realtime_wham.py`** - Batch=16 version
  - Latency: 13 seconds per batch
  - FPS: 1.1 frames/sec
  - Result: Too slow for real-time
  
- **`run_realtime_wham.sbatch`** - SLURM script for batch version

- **`realtime_demo.py`** - Early demo implementation

- **`REALTIME_STREAMING.md`** - Old documentation

### Why Replaced:
The new **`realtime_wham_online.py`** (batch=1) is **87x faster**:
- Latency: 150ms per frame (vs 13,000ms)
- FPS: 6.3 frames/sec (vs 1.1)
- Real-time capable: YES ✅ (vs NO ❌)

---

## 🎯 Should You Use These Files?

### ✅ Yes, if you want to:
- Set up Mac camera streaming to cluster
- Build custom video processing tools
- Understand the development process
- Reference earlier implementations

### ❌ No, if you just want to:
- Run WHAM on videos → Use `realtime_wham_online.py`
- Process in real-time → Use `slurm/run_online_wham.sbatch`
- Get started quickly → Read `docs/QUICK_START_GUIDE.md`

---

## 📚 Key Lessons Learned

1. **Batch vs Online Processing**:
   - Batch=16: Good for throughput, bad for latency
   - Batch=1: Necessary for real-time control
   
2. **GPU Compatibility**:
   - V100, RTX 6000, A100: ✅ Work great
   - H100/H200: ❌ PyTorch 1.11.0 too old
   - AMD MI210: ❌ CUDA only
   
3. **Bottlenecks**:
   - ViTPose detection: ~150ms (main bottleneck)
   - YOLO detection: ~10ms
   - WHAM network: Fast
   
4. **Streaming vs Simulation**:
   - Simulated real-time gives accurate latency
   - Actual streaming adds minimal overhead
   - Use simulation for development/testing

---

*These files are kept for reference and educational purposes.*
*For production use, see `docs/QUICK_START_GUIDE.md`.*

