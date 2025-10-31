"""Rotate the visualization video to portrait orientation"""
import cv2
from pathlib import Path

input_video = 'output/framebyframe_wham/visualization.mp4'
output_video = 'output/framebyframe_wham/visualization_rotated.mp4'

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Input: {width}x{height} @ {fps} FPS")
print(f"Rotating 90° clockwise...")

# After rotation, dimensions swap
out_width = height
out_height = width

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (out_width, out_height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Rotate 90° clockwise
    frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    out.write(frame_rotated)
    frame_count += 1
    
    if frame_count % 100 == 0:
        print(f"  Processed {frame_count} frames...")

cap.release()
out.release()

print(f"✅ Saved rotated video: {output_video}")
print(f"   {out_width}x{out_height}, {frame_count} frames")




