# Remote Camera Setup for WHAM

Use your Mac's camera with WHAM running on the PACE-ICE cluster.

## 🎯 Problem
- Your Mac (Apple Silicon) can't run WHAM efficiently (no CUDA)
- The cluster has GPUs but no camera
- **Solution:** Stream/upload from Mac → Process on cluster → Get results back

---

## 📋 Two Approaches

### **Approach 1: HTTP Streaming** (Lower latency, requires network setup)
Stream live video over HTTP from Mac to cluster.

### **Approach 2: Record-and-Upload** (More reliable, easier setup)
Record short clips on Mac, upload to cluster, process, get results.

---

## 🔧 Approach 1: HTTP Streaming

### On Your Mac:

1. **Install dependencies:**
   ```bash
   pip install opencv-python flask
   ```

2. **Run the camera server:**
   ```bash
   python remote_camera_server.py --port 8080
   ```

3. **Find your Mac's IP address:**
   ```bash
   ipconfig getifaddr en0
   # Example output: 192.168.1.100
   ```

4. **Test in browser:** Open `http://localhost:8080` to see preview

### On the Cluster:

1. **Make sure you can reach your Mac:**
   - Both on same WiFi network, OR
   - Connected via GT VPN, OR
   - Use SSH tunnel (see below)

2. **Run WHAM with the stream URL:**
   ```bash
   python demo.py \
       --video http://YOUR_MAC_IP:8080/video.mjpeg \
       --output_pth output/remote \
       --save_pkl \
       --estimate_local_only
   ```

### SSH Tunnel (if direct connection doesn't work):

**On your Mac:**
```bash
# Forward cluster port to your Mac
ssh -R 8080:localhost:8080 dmittelman6@login-ice.pace.gatech.edu
```

**On cluster:**
```bash
# Access camera via localhost
python demo.py --video http://localhost:8080/video.mjpeg ...
```

---

## 🔧 Approach 2: Record-and-Upload (Recommended!)

This is more reliable and works even with unstable networks.

### Setup SSH Key (one-time):

```bash
# On your Mac, generate key if you don't have one
ssh-keygen -t ed25519

# Copy key to cluster
ssh-copy-id dmittelman6@login-ice.pace.gatech.edu

# Test passwordless login
ssh dmittelman6@login-ice.pace.gatech.edu "echo 'Success!'"
```

### On the Cluster:

1. **Create incoming directory:**
   ```bash
   cd /home/hice1/dmittelman6/WHAM
   mkdir incoming
   ```

2. **Install watchdog (if not installed):**
   ```bash
   module load anaconda3/2023.03
   eval "$(conda shell.bash hook)"
   conda activate wham
   pip install watchdog
   ```

3. **Start the video watcher:**
   ```bash
   # In an interactive session or screen/tmux
   python cluster_watcher.py --watch-dir incoming --use-gpu
   ```
   
   Or submit as a job:
   ```bash
   sbatch run_watcher.sbatch  # (create this if needed)
   ```

### On Your Mac:

1. **Install dependencies:**
   ```bash
   pip install opencv-python
   ```

2. **Run the recorder:**
   ```bash
   python record_and_upload.py \
       --camera 0 \
       --cluster dmittelman6@login-ice.pace.gatech.edu \
       --remote-dir /home/hice1/dmittelman6/WHAM/incoming \
       --clip-duration 2 \
       --interval 0.5
   ```

### How it Works:

1. 📹 Mac records 2-second video clips continuously
2. 📤 Uploads each clip to cluster via SCP
3. 👁️  Cluster watcher detects new video
4. ⚙️  Processes with WHAM
5. 📥 Saves results as `.pkl` file
6. 📊 Mac downloads results (optional)
7. 🗑️  Deletes processed files

**Latency:** ~2-5 seconds total (recording + upload + processing)

---

## 📊 Performance Comparison

| Approach | Latency | Reliability | Setup Difficulty | Best For |
|----------|---------|-------------|------------------|----------|
| **HTTP Streaming** | ~0.5-1s | Medium | Hard (networking) | Real-time demos |
| **Record-Upload** | ~2-5s | High | Easy | Development, teleoperation |

---

## 🤖 Integration with Robot Control

Once working, you can extract poses and send to your robot:

```python
import joblib

# Load results
results = joblib.load('results/clip_XXXXX_results.pkl')

# Extract pose data (for first detected person, latest frame)
person_id = list(results.keys())[0]
latest_pose = results[person_id]['pose_world'][-1]  # (72,) array
latest_trans = results[person_id]['trans_world'][-1]  # (3,) array

# Send to robot
# your_robot.set_joint_angles(latest_pose)
# your_robot.set_root_position(latest_trans)
```

---

## 🐛 Troubleshooting

### Mac can't open camera:
```bash
# Check camera permissions in System Settings → Privacy & Security → Camera
# Make sure Terminal/Python has camera access
```

### Cluster can't reach Mac:
```bash
# Use SSH tunnel approach instead of direct connection
# Or use Record-and-Upload method
```

### Processing too slow:
```bash
# Use faster GPU on cluster (RTX 6000 > V100)
# Reduce video resolution on Mac
# Increase --interval between clips
```

### "Permission denied" on SCP:
```bash
# Set up SSH key authentication (see setup above)
# Make sure incoming directory exists and is writable
```

---

## 💡 Tips

1. **Start with Record-Upload** - easier to debug
2. **Use GTwifi or VPN** - for network connectivity
3. **Test locally first** - run `test_camera.py` on Mac to verify camera works
4. **Monitor bandwidth** - video uploads use ~5-10 MB/s
5. **Use screen/tmux** - keep watcher running on cluster

---

## 📝 Next Steps for Robot Teleoperation

1. ✅ Get this pipeline working (Mac camera → cluster → poses)
2. ✅ Test latency and reliability
3. ✅ Create pose retargeting (WHAM poses → robot joints)
4. ✅ Add robot controller integration
5. ✅ Deploy to robot's onboard computer (later)

---

## 📧 Need Help?

Check the WHAM repository issues or the HumanPlus paper for more details on real-time usage.

