"""
Real-time WHAM with 3D SMPL Mesh Visualization
Renders full 3D body meshes on video
"""

import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle

# WHAM imports
from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model

# Visualization imports
import pyrender
import trimesh
from pyrender.constants import RenderFlags


class SMPLRenderer:
    """Render SMPL mesh using pyrender"""
    def __init__(self, width=1920, height=1080, faces=None):
        self.width = width
        self.height = height
        self.faces = faces
        
        # Create pyrender scene
        self.scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0, 0, 0, 0])
        
        # Add camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=width/height)
        self.camera_node = self.scene.add(camera, pose=np.eye(4))
        
        # Add light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self.scene.add(light, pose=np.eye(4))
        
        # Create renderer
        self.renderer = pyrender.OffscreenRenderer(width, height)
        
        self.mesh_node = None
        
    def render(self, vertices, camera_translation, camera_rotation=None):
        """
        Render SMPL mesh
        Args:
            vertices: (6890, 3) SMPL vertices
            camera_translation: (3,) camera translation
            camera_rotation: (3, 3) camera rotation (optional)
        Returns:
            color: (H, W, 3) rendered RGB image
            mask: (H, W) alpha mask
        """
        # Create mesh
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.8, 1.0]
        tri_mesh = trimesh.Trimesh(vertices, self.faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
        
        # Update mesh in scene
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(mesh)
        
        # Update camera pose
        camera_pose = np.eye(4)
        if camera_rotation is not None:
            camera_pose[:3, :3] = camera_rotation
        camera_pose[:3, 3] = camera_translation
        
        self.scene.set_pose(self.camera_node, pose=camera_pose)
        
        # Render
        color, depth = self.renderer.render(self.scene, flags=RenderFlags.RGBA)
        
        return color[:, :, :3], color[:, :, 3]
    
    def __del__(self):
        if hasattr(self, 'renderer'):
            self.renderer.delete()


def overlay_mesh_on_image(image, mesh_render, alpha_mask, alpha=0.7):
    """
    Overlay rendered mesh on original image
    Args:
        image: (H, W, 3) original image
        mesh_render: (H, W, 3) rendered mesh
        alpha_mask: (H, W) mask for blending
        alpha: transparency factor
    Returns:
        blended: (H, W, 3) blended image
    """
    # Normalize alpha mask
    alpha_mask = alpha_mask.astype(np.float32) / 255.0
    alpha_mask = alpha_mask[:, :, None] * alpha
    
    # Blend
    blended = (1 - alpha_mask) * image + alpha_mask * mesh_render
    
    return blended.astype(np.uint8)


def visualize_smpl(video_path, output_path, frame_skip=1, max_fps=30, max_frames=None):
    """
    Process video with real-time WHAM and render 3D SMPL meshes
    """
    print("="*80)
    print("Real-time WHAM with 3D SMPL Visualization")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    model_cfg = cfg.MODEL
    
    # Build SMPL body model
    print("\nLoading SMPL model...")
    smpl = build_body_model(device, 1)
    faces = smpl.faces
    
    # Build network
    print("Loading WHAM network...")
    network = build_network(cfg, smpl)
    network.eval()
    network = network.to(device)
    
    # Build dataset (for detector/extractor)
    print("Loading detector and pose extractor...")
    dataset = CustomDataset(cfg, video_path, skip=frame_skip)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {width}x{height} @ {fps:.2f} fps")
    print(f"Total frames: {total_frames}")
    print(f"Frame skip: {frame_skip}")
    
    if max_frames:
        print(f"Max frames: {max_frames}")
        total_frames = min(total_frames, max_frames * frame_skip)
    
    # Setup output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out_fps = min(fps / frame_skip, max_fps) if max_fps > 0 else fps / frame_skip
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
    
    print(f"Output: {output_path} @ {out_fps:.2f} fps")
    
    # Create SMPL renderer
    print("\nInitializing 3D renderer...")
    renderer = SMPLRenderer(width, height, faces)
    
    print("\n" + "="*80)
    print("Processing video with 3D mesh rendering...")
    print("="*80 + "\n")
    
    frame_idx = 0
    processed_count = 0
    
    with torch.no_grad():
        for data_idx in tqdm(range(len(dataset)), desc="Rendering"):
            # Get frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Check max frames
            if max_frames and processed_count >= max_frames:
                break
            
            try:
                # Get detection and features
                batch = dataset[data_idx]
                
                if batch is None or len(batch['img']) == 0:
                    # No detection - just write original frame
                    out.write(frame)
                    processed_count += 1
                    frame_idx += 1
                    continue
                
                # Move to device
                img = batch['img'].unsqueeze(0).to(device)
                bbox = batch['bbox'].unsqueeze(0).to(device)
                
                # Run WHAM
                pred = network(img, bbox)
                
                # Extract SMPL parameters
                smpl_pose = pred['pose'].cpu().numpy()[0]  # (24, 3, 3)
                smpl_betas = pred['betas'].cpu().numpy()[0]  # (10,)
                cam_trans = pred['cam_trans'].cpu().numpy()[0]  # (3,)
                
                # Convert rotation matrices to axis-angle
                smpl_pose_aa = matrix_to_axis_angle(torch.from_numpy(smpl_pose)).numpy()
                smpl_pose_aa = smpl_pose_aa.reshape(-1)  # (72,)
                
                # Forward through SMPL to get vertices
                smpl_output = smpl(
                    betas=torch.from_numpy(smpl_betas).unsqueeze(0).float().to(device),
                    body_pose=torch.from_numpy(smpl_pose_aa[3:]).unsqueeze(0).float().to(device),
                    global_orient=torch.from_numpy(smpl_pose_aa[:3]).unsqueeze(0).float().to(device)
                )
                
                vertices = smpl_output.vertices[0].cpu().numpy()  # (6890, 3)
                
                # Adjust camera translation for rendering
                # WHAM outputs are in a specific coordinate system
                cam_trans_render = cam_trans.copy()
                cam_trans_render[2] += 5.0  # Move camera back
                
                # Render SMPL mesh
                mesh_img, alpha_mask = renderer.render(vertices, cam_trans_render)
                
                # Overlay on original frame
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                blended = overlay_mesh_on_image(frame_bgr, mesh_img, alpha_mask, alpha=0.8)
                blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
                
                # Add info text
                cv2.putText(blended_bgr, f"Frame: {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(blended_bgr, "3D SMPL Mesh", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                out.write(blended_bgr)
                
            except Exception as e:
                print(f"\nError processing frame {frame_idx}: {e}")
                out.write(frame)
            
            processed_count += 1
            frame_idx += 1
    
    cap.release()
    out.release()
    
    print("\n" + "="*80)
    print(f"✅ Done! Processed {processed_count} frames")
    print(f"Output saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize real-time WHAM with 3D SMPL meshes')
    parser.add_argument('video', type=str, help='Input video path')
    parser.add_argument('--output', type=str, default='output/smpl_3d/output.mp4',
                        help='Output video path')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Process every Nth frame (1 = all frames)')
    parser.add_argument('--max-fps', type=float, default=30,
                        help='Max output FPS (0 = no limit)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Max number of frames to process')
    
    args = parser.parse_args()
    
    visualize_smpl(
        args.video,
        args.output,
        frame_skip=args.frame_skip,
        max_fps=args.max_fps,
        max_frames=args.max_frames
    )

