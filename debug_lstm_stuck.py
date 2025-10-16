"""Debug why LSTM gets stuck after first frame"""
import joblib
import numpy as np

# Try to load timing stats (should work)
try:
    timing = joblib.load('output/framebyframe_wham/timing_stats.pkl')
    print("Timing stats loaded successfully")
    print(f"  Total frames processed: {timing.get('total_frames', 'N/A')}")
    print(f"  SMPL outputs: {timing.get('smpl_outputs', 'N/A')}")
except Exception as e:
    print(f"Timing stats error: {e}")

print("\n" + "="*60)
print("LSTM STUCK ANALYSIS:")
print("="*60)

print("\n1. LSTM Hidden States Issue:")
print("   - First frame: LSTM initialized with zeros")
print("   - Subsequent frames: LSTM uses previous hidden states")
print("   - If input features are similar, LSTM converges to fixed point")

print("\n2. Possible Causes:")
print("   a) Input features (x) not changing between frames")
print("   b) LSTM learning rate too low (stuck in local minimum)")
print("   c) Poor initialization of hidden states")
print("   d) Detector giving same keypoints every frame")

print("\n3. Solutions to try:")
print("   a) Add noise to input features")
print("   b) Reset LSTM hidden states periodically")
print("   c) Use different initialization")
print("   d) Check if detector is actually tracking motion")

print("\n4. Quick test - check if we can recover from backup:")
print("   - Look for intermediate .pkl files")
print("   - Check if visualization shows motion (it should!)")
print("   - If visualization works, the issue is in pose parameters, not vertices")

# Check if visualization exists and has reasonable size
import os
viz_path = 'output/framebyframe_wham/visualization.mp4'
if os.path.exists(viz_path):
    size_mb = os.path.getsize(viz_path) / (1024*1024)
    print(f"\n✅ Visualization exists: {size_mb:.1f}MB")
    if size_mb > 50:
        print("   → Large file suggests rendering worked")
        print("   → If mesh moves in video, vertices are correct")
        print("   → Issue is pose parameters getting stuck")
    else:
        print("   → Small file suggests rendering failed")
else:
    print("\n❌ No visualization found")
