# 🎨 WHAM Visualization Guide

## 📊 **Visualization Options**

### **Option 1: Real-Time Detection Overlay** ⭐ (Recommended)

Creates a video with bounding boxes, keypoints, and latency stats overlaid.

```bash
# Submit to cluster
sbatch slurm/run_realtime_viz.sbatch

# Or run directly
python visualize_realtime.py \
    examples/IMG_9732_portrait.mov \
    --output output/my_viz/output.mp4 \
    --frame-skip 1 \
    --duration 20
```

**Output:**
- Video with green bounding boxes around detected people
- 2D keypoint skeleton overlay
- Frame number and latency displayed
- Person ID tracking

**Preview:**
```
┌─────────────────────────────┐
│ Frame: 42 | Latency: 150ms  │
│ People: 1                   │
│                             │
│     [Person with green      │
│      bounding box and       │
│      skeleton overlay]      │
│                             │
└─────────────────────────────┘
```

---

### **Option 2: Full WHAM Demo (3D Visualization)**

Uses the original WHAM demo with full 3D body mesh.

```bash
# Submit to cluster
sbatch slurm/run_wham_demo.sbatch

# Or run directly
conda activate wham
python demo.py \
    --video examples/IMG_9732_portrait.mov \
    --output_pth output/full_demo \
    --visualize
```

**Output:**
- Full 3D SMPL body mesh overlaid on video
- World-grounded motion (if DPVO enabled)
- High quality but slower (not real-time)

**Location:** `output/demo_portrait/IMG_9732_portrait/output.mp4`

---

### **Option 3: Live Visualization (During Processing)**

Watch detections in real-time as they're processed (requires X11 or display).

```bash
# On interactive node with display
python realtime_wham_online.py \
    examples/IMG_9732_portrait.mov \
    --visualize \
    --frame-skip 2 \
    --duration 30
```

**Note:** Requires X11 forwarding (`ssh -X`) or running on a node with a display. Won't work on batch jobs.

---

## 🚀 **Quick Start**

### **Create a Visualization Right Now:**

```bash
cd /home/hice1/dmittelman6/WHAM
sbatch slurm/run_realtime_viz.sbatch

# Monitor progress
tail -f logs/realtime_viz_*.err

# When done, download the video
scp dmittelman6@login-ice.pace.gatech.edu:WHAM/output/realtime_viz/output.mp4 .
```

---

## 📊 **Comparison: Real-Time vs Full Demo**

| Feature | Real-Time (`visualize_realtime.py`) | Full Demo (`demo.py`) |
|---------|-------------------------------------|----------------------|
| **Processing** | Online (frame-by-frame) | Offline (entire video) |
| **Latency** | 150ms per frame | N/A (batch processing) |
| **Visualization** | 2D bounding boxes + keypoints | 3D SMPL body mesh |
| **Output Quality** | Fast, basic | Slow, high quality |
| **Use Case** | Test real-time performance | Final results |
| **Speed** | ~6 FPS | ~1 FPS |

---

## 🎬 **What Gets Visualized**

### **Real-Time Visualization Shows:**
✅ Green bounding boxes around detected people  
✅ 2D keypoint skeleton (17 COCO keypoints)  
✅ Person tracking IDs  
✅ Frame number  
✅ Per-frame latency in milliseconds  
✅ Number of people detected  

### **Full Demo Visualization Shows:**
✅ 3D SMPL body mesh overlay  
✅ World-grounded motion (with DPVO)  
✅ Smooth temporal motion  
✅ Body shape and pose parameters  

---

## 💡 **Tips**

### **For Faster Visualization:**
```bash
# Process every 5th frame (still smooth, 5x faster)
python visualize_realtime.py video.mp4 --frame-skip 5 --output viz.mp4
```

### **For High Quality:**
```bash
# Use full demo with all features
python demo.py --video video.mp4 --visualize --run_smplify
```

### **For Testing:**
```bash
# Process just 10 seconds
python visualize_realtime.py video.mp4 --duration 10 --output test.mp4
```

---

## 📥 **Download Visualizations**

### **From Cluster to Your Computer:**

```bash
# Real-time visualization
scp dmittelman6@login-ice.pace.gatech.edu:WHAM/output/realtime_viz/output.mp4 ./realtime_viz.mp4

# Full demo visualization
scp dmittelman6@login-ice.pace.gatech.edu:WHAM/output/demo_portrait/IMG_9732_portrait/output.mp4 ./full_demo.mp4
```

### **Or Use RSYNC:**
```bash
rsync -avz --progress \
    dmittelman6@login-ice.pace.gatech.edu:WHAM/output/realtime_viz/ \
    ./realtime_viz/
```

---

## 🎨 **Customization**

### **Change Visualization Colors:**

Edit `visualize_realtime.py`:

```python
# Line ~50: Bounding box color (B, G, R)
cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
# Change to: (255, 0, 0) for blue, (0, 0, 255) for red

# Line ~60: Keypoint color
cv2.circle(img, pt, 3, (0, 0, 255), -1)  # Red circles
# Change to: (255, 255, 0) for cyan
```

### **Add More Info:**

```python
# Add FPS to overlay
avg_fps = processed_count / elapsed
cv2.putText(viz_frame, f"FPS: {avg_fps:.1f}", 
           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
```

---

## 🐛 **Troubleshooting**

### **"No module named cv2"**
```bash
conda activate wham
pip install opencv-python
```

### **"Cannot open video"**
Check video path:
```bash
ls -lh examples/IMG_9732_portrait.mov
```

### **Video is blank/black**
The visualization might have failed. Check logs:
```bash
tail -50 logs/realtime_viz_*.err
```

### **No people detected**
Try lowering detection threshold in `lib/models/preproc/detector.py`:
```python
BBOX_CONF = 0.3  # Lower from 0.5
```

---

## 📊 **Example Output**

After running `sbatch slurm/run_realtime_viz.sbatch`, you'll see:

```
VISUALIZATION COMPLETE
Output: output/realtime_viz/output.mp4
Total frames: 600
Processed frames: 600
Total time: 95.2s
Average latency: 157ms (P50: 150ms, P95: 152ms)
```

**Video will show:**
- Original video with overlays
- Smooth detection tracking
- Real-time latency stats
- Professional-looking visualization

---

## 🎯 **Next Steps**

1. **Run the visualization**: `sbatch slurm/run_realtime_viz.sbatch`
2. **Monitor progress**: `tail -f logs/realtime_viz_*.err`
3. **Download result**: `scp ...output.mp4`
4. **Watch your video!** 🎬

---

*Generated: October 12, 2025*
*For questions, see `docs/QUICK_START_GUIDE.md`*

