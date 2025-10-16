"""
Visualize frame-by-frame SMPL results on video
"""
import cv2
import torch
import joblib
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from configs.config import get_cfg_defaults
from lib.models import build_network, build_body_model
from lib.vis.renderer import Renderer


def visualize_smpl_results(video_path, smpl_results_path, output_path, fps_stats_path=None):
    """
    Render SMPL meshes on video frames
    """
    logger.info(f"Loading SMPL results from {smpl_results_path}")
    smpl_results = joblib.load(smpl_results_path)
    logger.info(f"Loaded {len(smpl_results)} SMPL results")
    
    # Load FPS stats if available
    if fps_stats_path and Path(fps_stats_path).exists():
        timing_stats = joblib.load(fps_stats_path)
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE STATISTICS")
        logger.info("=" * 60)
        if len(timing_stats['total']) > 0:
            avg_total = np.mean(timing_stats['total'])
            avg_det = np.mean(timing_stats['detection'])
            avg_wham = np.mean(timing_stats['wham_inference'])
            logger.info(f"Average total latency: {avg_total*1000:.1f}ms")
            logger.info(f"Average detection: {avg_det*1000:.1f}ms")
            logger.info(f"Average WHAM inference: {avg_wham*1000:.1f}ms")
            logger.info(f"Effective FPS: {1.0/avg_total:.2f}")
            logger.info("=" * 60 + "\n")
    
    # Group results by frame
    logger.info("Grouping results by frame...")
    results_by_frame = {}
    for result in smpl_results:
        frame_id = result['frame_id']
        if frame_id not in results_by_frame:
            results_by_frame[frame_id] = []
        results_by_frame[frame_id].append(result)
    
    logger.info(f"Results span {len(results_by_frame)} frames")
    
    # Setup config
    logger.info("Setting up config...")
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    # Load SMPL model to get faces for renderer
    from lib.models import build_body_model
    smpl = build_body_model(cfg.DEVICE, 1)
    faces = smpl.faces
    
    # Open video
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Swap dimensions for 90° rotation to match processing
    width = orig_height
    height = orig_width
    
    logger.info(f"Original: {orig_width}x{orig_height}, Rotated: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    
    # Setup renderer with rotated dimensions (matching processing)
    logger.info("Initializing renderer...")
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, faces)
    
    # Create default camera (identity rotation, zero translation)
    default_R = torch.eye(3)
    default_T = torch.zeros(3)
    renderer.create_camera(default_R, default_T)
    
    # Setup video writer with rotated dimensions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    logger.info(f"Output video: {output_path}")
    logger.info("Rendering frames...")
    logger.info(f"Results frames available: {sorted(list(results_by_frame.keys()))[:10]}...")  # Show first 10
    
    frame_idx = 0
    rendered_count = 0
    pbar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rotate frame to match processing orientation
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Convert BGR to RGB for rendering
        img = frame[..., ::-1].copy()
        
        # Get SMPL results for this frame
        if frame_idx in results_by_frame:
            for result in results_by_frame[frame_idx]:
                # Get pre-computed vertices (already in camera space with translation applied)
                if 'verts' not in result:
                    logger.warning(f"Frame {frame_idx}: NO VERTS IN RESULT!")
                    continue
                    
                verts = torch.from_numpy(result['verts']).float().to(cfg.DEVICE)  # (6890, 3)
                logger.info(f"Frame {frame_idx}: Rendering mesh with {verts.shape[0]} vertices, range [{verts.min():.2f}, {verts.max():.2f}]")
                
                # Render mesh on image (renderer expects RGB and tensor)
                img = renderer.render_mesh(verts, img)
                rendered_count += 1
        
        # Convert RGB back to BGR for video writer
        frame_out = img[..., ::-1].copy()
        out.write(frame_out)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    logger.info(f"✅ Video saved to {output_path}")
    logger.info(f"Rendered {frame_idx} frames")
    logger.info(f"🎨 Actually rendered SMPL meshes on {rendered_count} frames")


def main():
    parser = argparse.ArgumentParser(description="Visualize frame-by-frame SMPL results")
    parser.add_argument('video', type=str, help='Input video file')
    parser.add_argument('--smpl-results', type=str, 
                       default='output/framebyframe_wham/smpl_results_final.pkl',
                       help='Path to SMPL results pickle file')
    parser.add_argument('--timing-stats', type=str,
                       default='output/framebyframe_wham/timing_stats.pkl',
                       help='Path to timing statistics pickle file')
    parser.add_argument('--output', type=str, 
                       default='output/framebyframe_wham/visualization.mp4',
                       help='Output video path')
    
    args = parser.parse_args()
    
    visualize_smpl_results(
        args.video,
        args.smpl_results,
        args.output,
        args.timing_stats
    )


if __name__ == '__main__':
    main()

