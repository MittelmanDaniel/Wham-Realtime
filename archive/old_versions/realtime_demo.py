"""
Real-time WHAM for Robot Teleoperation
Processes live camera feed and outputs poses for robot control
"""
import os
import time
import argparse
from collections import deque, defaultdict

import cv2
import torch
import numpy as np
from loguru import logger

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor


class RealtimeWHAM:
    """
    Real-time WHAM for continuous pose estimation from camera feed.
    
    Usage:
        wham = RealtimeWHAM(camera_id=0)  # 0 for default webcam
        
        for pose_data in wham.stream():
            # Send pose_data to your robot
            print(pose_data['pose_world'])  # 72-dim pose vector
            print(pose_data['trans_world'])  # 3D position
    """
    
    def __init__(self, 
                 camera_id=0, 
                 window_size=16,
                 stride=8,
                 max_fps=30,
                 device='cuda'):
        """
        Args:
            camera_id: Camera device ID (0 for default webcam) or stream URL
            window_size: Number of frames to process together (WHAM uses 16-frame windows)
            stride: How many frames to skip between windows (lower = smoother but slower)
            max_fps: Maximum processing FPS (throttle to avoid overload)
            device: 'cuda' or 'cpu'
        """
        self.camera_id = camera_id
        self.window_size = window_size
        self.stride = stride
        self.max_fps = max_fps
        self.device = device
        
        logger.info(f"Initializing Real-time WHAM...")
        
        # Load config
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file('configs/yamls/demo.yaml')
        self.cfg.DEVICE = device
        
        # Build WHAM network
        logger.info("Loading WHAM model...")
        smpl_batch_size = self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN
        smpl = build_body_model(self.cfg.DEVICE, smpl_batch_size)
        self.network = build_network(self.cfg, smpl)
        self.network.eval()
        
        # Build detector and feature extractor
        logger.info("Loading detector and feature extractor...")
        self.detector = DetectionModel(self.cfg.DEVICE.lower())
        self.extractor = FeatureExtractor(self.cfg.DEVICE.lower())
        
        # Frame buffer for sliding window
        self.frame_buffer = deque(maxlen=window_size)
        self.detection_buffer = []
        
        logger.info("✅ Real-time WHAM initialized!")
        
    def open_camera(self):
        """Open camera stream"""
        if isinstance(self.camera_id, int):
            logger.info(f"Opening camera {self.camera_id}...")
        else:
            logger.info(f"Opening stream: {self.camera_id}...")
            
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera/stream: {self.camera_id}")
        
        # Get camera properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if unknown
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Camera opened: {width}x{height} @ {fps} FPS")
        
        return cap, fps, width, height
    
    def process_frame(self, frame, fps):
        """
        Process a single frame: detection and feature extraction
        Returns detection result or None if no person detected
        """
        # Run person detection
        # Note: detector.track() expects to process entire video, so we hack it a bit
        detection = self.detector.detect_single_frame(frame)
        
        if detection is None or len(detection) == 0:
            return None
        
        # Extract features from detected person
        # For simplicity, just take the first person if multiple detected
        bbox = detection[0]['bbox']  # [x, y, w, h]
        
        return {
            'frame': frame.copy(),
            'bbox': bbox,
            'frame_id': len(self.detection_buffer)
        }
    
    def stream(self):
        """
        Main streaming loop. Yields pose data for each processed window.
        
        Yields:
            dict: {
                'pose_world': np.array (72,),  # Full body pose in world coordinates
                'trans_world': np.array (3,),   # Root translation in world coordinates  
                'pose_body': np.array (69,),    # Body pose (without root)
                'betas': np.array (10,),        # Body shape parameters
                'frame_id': int,                # Frame number
                'fps': float,                   # Processing FPS
                'timestamp': float              # Time in seconds
            }
        """
        cap, fps, width, height = self.open_camera()
        
        frame_count = 0
        last_inference_time = 0
        min_frame_interval = 1.0 / self.max_fps
        
        try:
            logger.info("🎥 Starting real-time processing... Press Ctrl+C to stop")
            
            while True:
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame, reconnecting...")
                    cap.release()
                    cap, fps, width, height = self.open_camera()
                    continue
                
                frame_count += 1
                
                # Throttle processing to max_fps
                elapsed = start_time - last_inference_time
                if elapsed < min_frame_interval:
                    continue
                
                # Process frame (detection + features)
                result = self.process_frame(frame, fps)
                
                if result is None:
                    # No person detected - skip
                    continue
                
                self.detection_buffer.append(result)
                
                # Wait until we have enough frames for a window
                if len(self.detection_buffer) < self.window_size:
                    continue
                
                # Run WHAM inference on the window
                with torch.no_grad():
                    pose_data = self.run_inference_on_window(
                        self.detection_buffer[-self.window_size:],
                        width, height, fps
                    )
                
                if pose_data is not None:
                    # Add metadata
                    pose_data['frame_id'] = frame_count
                    pose_data['timestamp'] = time.time()
                    pose_data['fps'] = 1.0 / (time.time() - start_time)
                    
                    last_inference_time = time.time()
                    
                    yield pose_data
                
                # Slide the window forward
                self.detection_buffer = self.detection_buffer[-self.stride:]
                
        except KeyboardInterrupt:
            logger.info("\n⏹️  Stopped by user")
        finally:
            cap.release()
            logger.info("Camera released")
    
    def run_inference_on_window(self, window_data, width, height, fps):
        """
        Run WHAM inference on a window of frames
        
        This is a simplified version - you may need to adapt based on 
        WHAM's exact requirements for tracking_results format
        """
        # TODO: This needs to be properly implemented based on WHAM's data format
        # For now, this is a placeholder showing the structure
        
        logger.warning("⚠️  run_inference_on_window() needs full implementation!")
        logger.warning("⚠️  This requires adapting WHAM's preprocessing to work on frame windows")
        
        # Placeholder return
        return {
            'pose_world': np.zeros(72),
            'trans_world': np.zeros(3),
            'pose_body': np.zeros(69),
            'betas': np.zeros(10)
        }


def main():
    parser = argparse.ArgumentParser(description='Real-time WHAM for robot teleoperation')
    
    parser.add_argument('--camera', type=str, default='0',
                        help='Camera device ID (0, 1, ...) or stream URL')
    parser.add_argument('--window_size', type=int, default=16,
                        help='Number of frames per window')
    parser.add_argument('--stride', type=int, default=8,
                        help='Frame stride between windows')
    parser.add_argument('--max_fps', type=int, default=10,
                        help='Maximum processing FPS')
    parser.add_argument('--visualize', action='store_true',
                        help='Show live visualization')
    
    args = parser.parse_args()
    
    # Convert camera arg to int if it's a number
    try:
        camera_id = int(args.camera)
    except:
        camera_id = args.camera
    
    # Initialize real-time WHAM
    wham = RealtimeWHAM(
        camera_id=camera_id,
        window_size=args.window_size,
        stride=args.stride,
        max_fps=args.max_fps
    )
    
    # Stream poses
    logger.info("\n" + "="*50)
    logger.info("Real-time WHAM started!")
    logger.info("This will output pose data continuously.")
    logger.info("Send this data to your robot controller.")
    logger.info("="*50 + "\n")
    
    for pose_data in wham.stream():
        # This is where you'd send data to your robot!
        logger.info(f"Frame {pose_data['frame_id']} | "
                   f"FPS: {pose_data['fps']:.1f} | "
                   f"Root pos: [{pose_data['trans_world'][0]:.2f}, "
                   f"{pose_data['trans_world'][1]:.2f}, "
                   f"{pose_data['trans_world'][2]:.2f}]")
        
        # Example: Send to robot
        # your_robot.set_pose(pose_data['pose_world'])
        # your_robot.set_position(pose_data['trans_world'])


if __name__ == '__main__':
    main()

