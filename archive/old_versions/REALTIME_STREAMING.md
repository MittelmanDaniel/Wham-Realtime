# Real-time WHAM Streaming Guide

This guide shows you how to test real-time video processing with WHAM using **file-based streaming** (no live camera needed!).

## 🎯 Overview

Three approaches for testing real-time WHAM:

1. **Python Simulation** - Simplest, reads video file at real FPS
2. **FFmpeg HTTP Streaming** - More realistic, actual network streaming
3. **Live Camera** - For actual robot teleoperation (see `REMOTE_CAMERA_SETUP.md`)

---

## 📁 Files Created

| File | Purpose |
|------|---------|
| `realtime_simulation.py` | Simulate real-time by reading video at native FPS |
| `ffmpeg_streamer.py` | Stream video file over HTTP using ffmpeg |
| `realtime_processor.py` | Process any video stream (HTTP/RTSP/camera) |
| `test_realtime_stream.sbatch` | SLURM job to test everything |

---

## 🚀 Quick Start

### Option 1: Run Full Test (Recommended)

Submit the test job that tries both approaches:

```bash
sbatch test_realtime_stream.sbatch
```

Monitor:
```bash
# Watch job status
squeue -u $USER

# Watch output
tail -f logs/stream_test_<JOBID>.out
```

---

### Option 2: Manual Testing

#### A. Python Simulation (Simplest)

```bash
# Interactive GPU session
srun --partition=ice-gpu --gres=gpu:rtx_6000:1 \
     --cpus-per-task=8 --mem=32G --time=00:30:00 --pty bash

# Activate environment
module load anaconda3/2023.03
eval "$(conda shell.bash hook)"
conda activate wham

# Run simulation
python realtime_simulation.py examples/IMG_9732_portrait.mov \
    --output output/realtime_test \
    --frame_skip 1
```

**What this does:**
- Reads video frame-by-frame at 30 FPS
- Mimics real-time camera capture
- Perfect for testing WHAM processing speed

---

#### B. FFmpeg HTTP Streaming (More Realistic)

**Terminal 1 - Start the streamer:**
```bash
python ffmpeg_streamer.py examples/IMG_9732_portrait.mov \
    --port 5000 \
    --loop
```

**Terminal 2 - Process the stream:**
```bash
# In GPU session
python realtime_processor.py http://localhost:5000/video.mjpeg \
    --output output/stream_test \
    --buffer-size 16 \
    --frame-skip 1
```

**View in browser:**
- Open: `http://localhost:5000/`
- See live MJPEG stream

**What this does:**
- Streams video over HTTP like a real camera
- Tests network latency effects
- OpenCV reads from stream URL

---

## 🎮 Usage Examples

### Test Different Speeds

```bash
# Process every frame (slowest, most accurate)
python realtime_simulation.py video.mov --frame_skip 1

# Process every other frame (2x faster)
python realtime_simulation.py video.mov --frame_skip 2

# Process every 5th frame (5x faster, less accurate)
python realtime_simulation.py video.mov --frame_skip 5
```

### Stream Custom Videos

```bash
# Stream your own video
python ffmpeg_streamer.py /path/to/your/video.mp4 \
    --port 8080 \
    --fps 30 \
    --loop

# Process it
python realtime_processor.py http://localhost:8080/video.mjpeg
```

### Process Live Camera (when available)

```bash
# From Mac camera (after setting up remote_camera_server.py)
python realtime_processor.py http://your-mac-ip:5001/video.mjpeg

# From local webcam
python realtime_processor.py 0  # Camera ID 0

# From RTSP camera
python realtime_processor.py rtsp://username:password@camera-ip:554/stream
```

---

## 📊 Performance Testing

### Benchmark Different Configurations

```bash
# Baseline: Full frames, small buffer
python realtime_simulation.py video.mov \
    --frame_skip 1 --buffer_size 16

# Faster: Skip frames, large buffer  
python realtime_simulation.py video.mov \
    --frame_skip 3 --buffer_size 32

# Maximum speed: Heavy skipping
python realtime_simulation.py video.mov \
    --frame_skip 5 --buffer_size 64
```

### Expected Performance

Based on previous benchmarks:
- **V100 GPU**: ~3-4 FPS (full processing)
- **RTX 6000**: ~3-4 FPS (full processing)
- **With frame_skip=2**: ~6-8 FPS
- **With frame_skip=5**: ~15-20 FPS

For real-time (30 FPS) robot teleoperation:
- Use `frame_skip=5` or higher
- Or optimize WHAM code
- Or use faster GPU + optimized PyTorch

---

## 🔧 Troubleshooting

### Stream won't connect

```bash
# Check if streamer is running
ps aux | grep ffmpeg_streamer

# Check port
netstat -tuln | grep 5000

# Try different port
python ffmpeg_streamer.py video.mov --port 8888
```

### Low FPS / Can't keep up

```bash
# Increase frame skip
python realtime_processor.py SOURCE --frame-skip 5

# Increase buffer size
python realtime_processor.py SOURCE --buffer-size 32

# Use faster GPU
sbatch --gres=gpu:rtx_6000:1 ...  # or h100:1
```

### FFmpeg errors

```bash
# Check ffmpeg is available
which ffmpeg
ffmpeg -version

# Should be in conda env:
# /home/hice1/dmittelman6/.conda/envs/wham/bin/ffmpeg
```

---

## 🎯 Next Steps

### For Robot Teleoperation

1. **Test streaming:** Use these scripts to measure FPS
2. **Optimize processing:** Find best frame_skip for your needs
3. **Setup camera:** Follow `REMOTE_CAMERA_SETUP.md` for Mac camera
4. **Integrate robot:** Use WHAM output to control robot

### For Better Performance

1. **Use H100 GPU** (if available) - 2-3x faster
2. **Optimize WHAM** - Remove visualization, reduce batch size
3. **Use WHAM API** - Lighter than demo.py
4. **Profile code** - Find bottlenecks with PyTorch profiler

---

## 📝 Notes

- **frame_skip=1**: Process every frame (30 FPS input)
- **frame_skip=2**: Process every other frame (15 FPS input)
- **frame_skip=5**: Process every 5th frame (6 FPS input)

Higher frame_skip = faster processing but less smooth motion.

For robot teleoperation, `frame_skip=2-3` is a good balance between smoothness and speed.

---

## 🆘 Help

If you encounter issues:

1. Check GPU is available: `nvidia-smi`
2. Check conda env: `conda activate wham`
3. Check logs: `logs/stream_test_*.out`
4. Test with simple video first: `examples/IMG_9732_portrait.mov`

---

**Happy streaming! 🎥🤖**
