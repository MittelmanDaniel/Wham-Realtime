"""Debug if the detector is actually tracking motion between frames"""
import cv2
import numpy as np

# Load the video and check if keypoints are changing
cap = cv2.VideoCapture('examples/IMG_9732.mov')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("Checking detector motion tracking...")

# Read first few frames and check if they're different
frames = []
for i in range(5):
    ret, frame = cap.read()
    if not ret:
        break
    # Rotate to match processing
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frames.append(frame)
    print(f"Frame {i}: shape {frame.shape}")

# Check if frames are actually different
if len(frames) >= 2:
    diff = np.abs(frames[1].astype(float) - frames[0].astype(float))
    print(f"\nFrame difference (1-0):")
    print(f"  Mean diff: {diff.mean():.2f}")
    print(f"  Max diff: {diff.max():.2f}")
    print(f"  Non-zero pixels: {(diff > 10).sum()} / {diff.size}")
    
    if diff.mean() < 5:
        print("⚠️  WARNING: Frames are very similar! Video might be static!")
    else:
        print("✅ Frames are different - video has motion")

cap.release()

print("\n" + "="*60)
print("POSSIBLE ISSUES:")
print("="*60)
print("1. Detector giving same keypoints every frame")
print("2. LSTM input features not changing")
print("3. Person not moving much in video")
print("4. WHAM model trained on different data distribution")
print("5. Camera motion not being tracked properly")

print("\n" + "="*60)
print("SOLUTIONS TO TRY:")
print("="*60)
print("1. Check if person is actually moving in the video")
print("2. Add noise to input features to force variation")
print("3. Use different initialization for each frame")
print("4. Check if detector confidence is too low")
print("5. Try a different video with more motion")


