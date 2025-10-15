"""
3D SMPL Mesh Visualization from Demo Output
Renders the 3D SMPL body mesh from existing WHAM demo results.
"""

import os
import os.path as osp
import sys
import cv2
import torch
import joblib
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Add lib to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# WHAM imports
from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.models import build_network, build_body_model
from lib.vis.renderer import Renderer

def visualize_smpl_from_demo(
    demo_output_dir,
    output_video,
    original_video_path,
    max_frames=None
):
    """Visualize 3D SMPL mesh from existing demo output"""
    
    print("\n" + "="*80)
    print("3D SMPL Mesh Visualization from Demo Output")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Demo output: {demo_output_dir}")
    print(f"Original video: {original_video_path}")
    
    # Load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    # Build SMPL model
    print("\nLoading SMPL model...")
    smpl = build_body_model(device, 1)
    
    # Get video info first to create renderer
    cap = cv2.VideoCapture(original_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    
    # Create renderer with CLIFF focal length estimation
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, device, smpl.faces)
    
    # Build network
    print("Loading WHAM network...")
    network = build_network(cfg, smpl)
    network.eval()
    network = network.to(device)
    
    # Load preprocessed data
    print(f"\nLoading preprocessed data...")
    tracking_results = joblib.load(osp.join(demo_output_dir, 'tracking_results.pth'))
    slam_results = joblib.load(osp.join(demo_output_dir, 'slam_results.pth'))
    
    # Build dataset
    print("Building dataset...")
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    # Run WHAM
    print(f"\nRunning WHAM on {len(dataset)} subjects...")
    results = defaultdict(dict)
    
    with torch.no_grad():
        for subj in tqdm(range(len(dataset)), desc="Processing subjects"):
            batch = dataset.load_data(subj)
            _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
            pred = network(x, inits, features, mask=mask, init_root=init_root, 
                          cam_angvel=cam_angvel, return_y_up=True, **kwargs)
            
            # Get vertices - network returns them in camera coordinates
            # Use the same formula as demo.py
            verts = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).squeeze(0).cpu().numpy()
            
            # Store results
            results[subj]['verts'] = verts  # (T, 6890, 3)
            results[subj]['frame_ids'] = frame_id
    
    # Create output video
    print(f"\nRendering 3D mesh visualization...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Create default camera (identity rotation, zero translation)
    default_R = torch.eye(3)
    default_T = torch.zeros(3)
    renderer.create_camera(default_R, default_T)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    
    pbar = tqdm(total=total_frames, desc="Rendering frames")
    
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for rendering
        img = frame[..., ::-1].copy()
        
        # Find which subject(s) are in this frame and render
        for subj in results:
            frame_ids = results[subj]['frame_ids']
            if frame_idx in frame_ids:
                # Get the index within this subject's sequence
                seq_idx = np.where(frame_ids == frame_idx)[0][0]
                
                # Get vertices for this frame
                verts = torch.from_numpy(results[subj]['verts'][seq_idx]).to(device)  # Shape: (6890, 3)
                
                # Render mesh onto the image using WHAM's renderer
                img = renderer.render_mesh(verts, img)
        
        # Convert RGB back to BGR for video writer
        frame_out = img[..., ::-1].copy()
        out.write(frame_out)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✓ Done! Output saved to: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D SMPL mesh from demo output")
    parser.add_argument('demo_output', type=str, help='Path to demo output directory')
    parser.add_argument('--video', type=str, required=True, help='Path to original video')
    parser.add_argument('--output', type=str, default='output/smpl_3d/output.mp4', help='Output video path')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames to process')
    
    args = parser.parse_args()
    
    os.makedirs(osp.dirname(args.output), exist_ok=True)
    
    visualize_smpl_from_demo(
        args.demo_output,
        args.output,
        args.video,
        max_frames=args.max_frames
    )
