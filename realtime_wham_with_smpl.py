"""
Real-time WHAM with SMPL Parameter Output
Process frames and output SMPL parameters with minimal latency
Uses micro-batching for efficiency
"""
import cv2
import time
import argparse
import numpy as np
import torch
import joblib
from pathlib import Path
from loguru import logger
from collections import defaultdict

from configs.config import get_cfg_defaults
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor


class RealtimeWHAMWithSMPL:
    """
    Near-realtime WHAM processor with SMPL parameter output
    Processes in micro-batches for efficiency while maintaining low latency
    """
    def __init__(self, output_dir='output/realtime_smpl', batch_size=16):
        logger.info("Initializing Realtime WHAM with SMPL output...")
        
        # Setup config
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file('configs/yamls/demo.yaml')
        
        # Build SMPL body model
        logger.info("Loading SMPL model...")
        smpl_batch_size = self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN
        self.smpl = build_body_model(self.cfg.DEVICE, smpl_batch_size)
        
        # Build network
        logger.info("Loading WHAM network...")
        self.network = build_network(self.cfg, self.smpl)
        self.network.eval()
        
        # Initialize detector and feature extractor
        logger.info("Loading detector and feature extractor...")
        self.detector = DetectionModel(self.cfg.DEVICE.lower())
        self.extractor = FeatureExtractor(self.cfg.DEVICE.lower())
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Batch settings
        self.batch_size = batch_size
        self.frame_buffer = []
        
        # Track state
        self.frame_count = 0
        self.processed_count = 0
        
        # Store SMPL results
        self.smpl_results = []
        
        logger.info(f"✅ Realtime WHAM with SMPL initialized! Batch size: {batch_size}")
    
    def process_frame(self, frame, fps=30):
        """
        Add frame to buffer and process when batch is ready
        Returns: (process_time, smpl_params_list or None)
        """
        start_time = time.time()
        
        # Add frame to buffer
        self.frame_buffer.append(frame)
        self.frame_count += 1
        
        smpl_params_list = None
        
        # Process batch when ready
        if len(self.frame_buffer) >= self.batch_size:
            smpl_params_list = self._process_batch(fps)
            self.frame_buffer = []  # Clear buffer
        
        process_time = time.time() - start_time
        
        return process_time, smpl_params_list
    
    @torch.no_grad()
    def _process_batch(self, fps):
        """
        Process accumulated frames and return SMPL parameters
        """
        try:
            batch_frames = self.frame_buffer.copy()
            num_frames = len(batch_frames)
            
            logger.info(f"Processing batch of {num_frames} frames...")
            
            # Run detection on all frames in batch
            for frame in batch_frames:
                self.detector.track(frame, fps, len(batch_frames))
            
            # Get tracking results
            tracking_results = self.detector.process(fps=fps)
            
            if not tracking_results or len(tracking_results) == 0:
                logger.warning("No tracking results")
                return None
            
            # Extract features
            height, width = batch_frames[0].shape[:2]
            tracking_results = self.extractor.run(
                video=None,
                tracking_results=tracking_results,
                width=width,
                height=height,
                fps=fps
            )
            
            # Create dummy SLAM results
            num_tracked_frames = len(tracking_results[list(tracking_results.keys())[0]]['frame_ids'])
            slam_results = {
                'frame_ids': np.arange(num_tracked_frames),
                'cam_angvel': np.zeros((num_tracked_frames, 3))
            }
            
            # Run WHAM inference on each tracked person
            smpl_params_list = []
            
            for person_id, person_data in tracking_results.items():
                # Prepare input for WHAM
                x = torch.from_numpy(person_data['img_feat']).float().to(self.cfg.DEVICE).unsqueeze(0)
                inits = torch.from_numpy(person_data['init_pose']).float().to(self.cfg.DEVICE).unsqueeze(0)
                features = torch.from_numpy(person_data['features']).float().to(self.cfg.DEVICE).unsqueeze(0)
                
                bbox = torch.from_numpy(person_data['bbox']).float().to(self.cfg.DEVICE).unsqueeze(0)
                mask = torch.ones(1, features.shape[1], dtype=torch.bool, device=self.cfg.DEVICE)
                
                init_root = None
                cam_angvel = torch.from_numpy(slam_results['cam_angvel'][:features.shape[1]]).float().to(self.cfg.DEVICE).unsqueeze(0)
                
                res = torch.tensor([[width, height]], dtype=torch.float32, device=self.cfg.DEVICE).unsqueeze(1).expand(-1, features.shape[1], -1)
                
                # Run WHAM network
                pred = self.network(
                    x, inits, features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    return_y_up=True,
                    bbox=bbox,
                    res=res,
                    cam_intrinsics=None
                )
                
                # Extract SMPL parameters for all frames in batch
                T = pred['pose'].shape[1]
                for t in range(T):
                    frame_id = person_data['frame_ids'][t]
                    
                    smpl_params = {
                        'frame_id': int(frame_id),
                        'person_id': person_id,
                        'pose': pred['pose'][0, t].cpu().numpy(),  # (24, 3, 3)
                        'betas': pred['betas'][0, t].cpu().numpy(),  # (10,)
                        'trans_cam': pred['trans_cam'][0, t].cpu().numpy(),  # (3,)
                        'verts': (pred['verts_cam'][0, t] + pred['trans_cam'][0, t].unsqueeze(0)).cpu().numpy(),  # (6890, 3)
                    }
                    
                    smpl_params_list.append(smpl_params)
                    self.smpl_results.append(smpl_params)
            
            self.processed_count += num_frames
            logger.info(f"✅ Extracted SMPL parameters for {len(smpl_params_list)} detections")
            
            return smpl_params_list
            
        except Exception as e:
            logger.error(f"WHAM inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def finalize(self):
        """Process any remaining frames in buffer"""
        if len(self.frame_buffer) > 0:
            logger.info(f"Processing final {len(self.frame_buffer)} frames...")
            return self._process_batch(fps=30)
        return None
    
    def save_results(self, filename='smpl_results.pkl'):
        """Save all SMPL results to file"""
        output_path = self.output_dir / filename
        joblib.dump(self.smpl_results, output_path)
        logger.info(f"Saved {len(self.smpl_results)} SMPL results to {output_path}")


class RealtimeStreamWithSMPL:
    """
    Stream processor with SMPL parameter output
    """
    def __init__(self, source, output_dir='output/realtime_smpl',
                 frame_skip=1, max_fps=30, batch_size=16):
        self.source = source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_skip = frame_skip
        self.max_fps = max_fps
        self.frame_time = 1.0 / max_fps if max_fps > 0 else 0
        
        # Open video source
        logger.info(f"Opening video source: {source}")
        
        try:
            source_id = int(source)
            self.cap = cv2.VideoCapture(source_id)
        except:
            self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Resolution: {self.width}x{self.height}")
        logger.info(f"FPS: {self.fps:.2f}")
        logger.info(f"Frame skip: {frame_skip}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"⚡ MICRO-BATCH MODE with SMPL parameter output")
        
        # Initialize WHAM with SMPL
        self.wham = RealtimeWHAMWithSMPL(output_dir=output_dir, batch_size=batch_size)
        
        # Stats
        self.stats = {
            'frames_received': 0,
            'frames_processed': 0,
            'frames_skipped': 0,
            'smpl_outputs': 0,
            'total_processing_time': 0,
            'start_time': None,
        }
    
    def run(self, duration=None):
        """
        Run realtime WHAM processing with SMPL output
        """
        logger.info("=" * 60)
        logger.info("REALTIME WHAM WITH SMPL PARAMETERS")
        logger.info("=" * 60)
        logger.info("Starting... Press Ctrl+C to stop")
        
        self.stats['start_time'] = time.time()
        last_print = self.stats['start_time']
        last_frame_time = self.stats['start_time']
        
        try:
            while True:
                current_time = time.time()
                
                # Check duration limit
                if duration and (current_time - self.stats['start_time']) > duration:
                    logger.info(f"Duration limit reached ({duration}s)")
                    break
                
                # Throttle frame reading
                if self.frame_time > 0:
                    time_since_last = current_time - last_frame_time
                    if time_since_last < self.frame_time:
                        time.sleep(self.frame_time - time_since_last)
                        current_time = time.time()
                    last_frame_time = current_time
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("End of stream")
                    break
                
                self.stats['frames_received'] += 1
                
                # Skip frames if requested
                if (self.stats['frames_received'] - 1) % self.frame_skip != 0:
                    self.stats['frames_skipped'] += 1
                    continue
                
                # Process frame (adds to batch, processes when ready)
                process_time, smpl_params_list = self.wham.process_frame(frame, fps=self.fps)
                
                self.stats['total_processing_time'] += process_time
                self.stats['frames_processed'] += 1
                
                if smpl_params_list is not None:
                    num_smpl = len(smpl_params_list)
                    self.stats['smpl_outputs'] += num_smpl
                    logger.info(f"✅ Got {num_smpl} SMPL parameter sets!")
                
                # Print stats every 2 seconds
                if current_time - last_print >= 2.0:
                    self._print_stats(current_time)
                    last_print = current_time
        
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        
        finally:
            self.cleanup()
    
    def _print_stats(self, current_time):
        """Print current statistics"""
        elapsed = current_time - self.stats['start_time']
        fps_received = self.stats['frames_received'] / elapsed if elapsed > 0 else 0
        fps_processed = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        
        logger.info(f"[{elapsed:.1f}s] "
                   f"Received: {self.stats['frames_received']} ({fps_received:.1f} FPS) | "
                   f"Processed: {self.stats['frames_processed']} ({fps_processed:.1f} FPS) | "
                   f"SMPL outputs: {self.stats['smpl_outputs']}")
    
    def cleanup(self):
        """Cleanup and save final results"""
        logger.info("\n" + "=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 60)
        
        # Process any remaining frames
        final_smpl = self.wham.finalize()
        if final_smpl:
            self.stats['smpl_outputs'] += len(final_smpl)
        
        elapsed = time.time() - self.stats['start_time']
        
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Frames received: {self.stats['frames_received']}")
        logger.info(f"Frames processed: {self.stats['frames_processed']}")
        logger.info(f"Frames skipped: {self.stats['frames_skipped']}")
        logger.info(f"SMPL parameters output: {self.stats['smpl_outputs']}")
        logger.info(f"Average FPS (processed): {self.stats['frames_processed']/elapsed:.2f}")
        
        # Save final results
        logger.info("\nSaving results...")
        self.wham.save_results('smpl_results_final.pkl')
        
        self.cap.release()
        logger.info("✅ Cleanup complete!")


def main():
    parser = argparse.ArgumentParser(description="Realtime WHAM with SMPL parameters")
    parser.add_argument('video', type=str, help='Video file or camera ID (0, 1, etc.)')
    parser.add_argument('--output-dir', type=str, default='output/realtime_smpl',
                       help='Output directory')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Process every Nth frame (1 = process all frames)')
    parser.add_argument('--max-fps', type=float, default=30,
                       help='Maximum FPS to process (0 = no limit)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Duration to run in seconds (None = until video ends)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Micro-batch size for processing')
    
    args = parser.parse_args()
    
    # Run streaming
    stream = RealtimeStreamWithSMPL(
        args.video,
        output_dir=args.output_dir,
        frame_skip=args.frame_skip,
        max_fps=args.max_fps,
        batch_size=args.batch_size
    )
    
    stream.run(duration=args.duration)


if __name__ == '__main__':
    main()
