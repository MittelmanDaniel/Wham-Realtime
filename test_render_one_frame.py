"""Test rendering one frame to debug visualization"""
import cv2
import torch
import joblib
import numpy as np
from pathlib import Path

from configs.config import get_cfg_defaults
from lib.models import build_body_model
from lib.vis.renderer import Renderer

# Load config
cfg = get_cfg_defaults()
cfg.merge_from_file('configs/yamls/demo.yaml')

# Load SMPL model
smpl = build_body_model(cfg.DEVICE, 1)
faces = smpl.faces

# Load results
results = joblib.load('output/framebyframe_wham/smpl_results_final.pkl')
print(f"Loaded {len(results)} results")

# Group by frame
results_by_frame = {}
for result in results:
    frame_id = result['frame_id']
    if frame_id not in results_by_frame:
        results_by_frame[frame_id] = []
    results_by_frame[frame_id].append(result)

# Open video and get one frame (frame 100)
test_frame_idx = 100
cap = cv2.VideoCapture('examples/IMG_9732.mov')
cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_idx)
ret, frame = cap.read()

if not ret:
    print(f"Could not read frame {test_frame_idx}")
    exit(1)

print(f"Frame shape: {frame.shape}")

# NO rotation - keep original dimensions to match WHAM processing
# Convert to RGB
img_rgb = frame[..., ::-1].copy()
print(f"RGB image shape: {img_rgb.shape}")

# Save original frame
cv2.imwrite('output/framebyframe_wham/test_frame_original.jpg', frame)
print("Saved original frame")

# Setup renderer
height, width = frame.shape[:2]
focal_length = (width ** 2 + height ** 2) ** 0.5
renderer = Renderer(width, height, focal_length, cfg.DEVICE, faces)

# Create camera
default_R = torch.eye(3)
default_T = torch.zeros(3)
renderer.create_camera(default_R, default_T)

# Get vertices for this frame
if test_frame_idx in results_by_frame:
    result = results_by_frame[test_frame_idx][0]
    
    if 'verts' in result:
        verts = torch.from_numpy(result['verts']).float().to(cfg.DEVICE)
        print(f"Verts shape: {verts.shape}, range: [{verts.min():.2f}, {verts.max():.2f}]")
        
        # Render mesh
        print("Rendering mesh...")
        img_rendered = renderer.render_mesh(verts, img_rgb)
        print(f"Rendered image shape: {img_rendered.shape}, dtype: {img_rendered.dtype}")
        
        # Convert back to BGR
        frame_out = img_rendered[..., ::-1].copy()
        
        # Save rendered frame
        cv2.imwrite('output/framebyframe_wham/test_frame_rendered.jpg', frame_out)
        print("Saved rendered frame to test_frame_rendered.jpg")
        
        # Also save a version with high contrast mesh
        print("\nNow trying with bright colors...")
        img_rgb2 = frame[..., ::-1].copy()
        renderer2 = Renderer(width, height, focal_length, cfg.DEVICE, faces)
        renderer2.create_camera(default_R, default_T)
        img_rendered2 = renderer2.render_mesh(verts, img_rgb2, colors=[255, 0, 0])  # Bright red
        frame_out2 = img_rendered2[..., ::-1].copy()
        cv2.imwrite('output/framebyframe_wham/test_frame_red.jpg', frame_out2)
        print("Saved RED mesh version to test_frame_red.jpg")
        
    else:
        print(f"NO VERTS in result for frame {test_frame_idx}")
else:
    print(f"No results for frame {test_frame_idx}")

cap.release()
print("\nDone! Check output/framebyframe_wham/ for test images")

