"""
TRUE Real-time WHAM - Frame-by-Frame SMPL Output with LSTM
Process each frame individually and output SMPL parameters immediately!
"""
import cv2
import time
import argparse
import numpy as np
import torch
import joblib
from pathlib import Path
from loguru import logger

from configs.config import get_cfg_defaults
from configs import constants as _C
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.utils import transforms


class FrameByFrameWHAM:
    """
    TRUE frame-by-frame WHAM with LSTM hidden state
    Output SMPL parameters for EVERY frame!
    """
    def __init__(self, output_dir='output/framebyframe_wham'):
        logger.info("Initializing Frame-by-Frame WHAM...")
        
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
        
        # Initialize detector
        logger.info("Loading detector...")
        self.detector = DetectionModel(self.cfg.DEVICE.lower())
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track state per person
        self.person_states = {}  # Track LSTM hidden states per person
        
        # Frame counter
        self.frame_count = 0
        
        # Store SMPL results
        self.smpl_results = []
        
        # Timing breakdown
        self.timing = {
            'detection': [],
            'wham_inference': [],
            'total': [],
        }
        
        logger.info(f"✅ Frame-by-Frame WHAM initialized!")
    
    @torch.no_grad()
    def process_frame(self, frame, fps=30):
        """
        Process a single frame and return SMPL parameters immediately
        Returns: (total_time, det_time, wham_time, smpl_params_list)
        """
        frame_start = time.time()
        height, width = frame.shape[:2]
        
        # 1. Detection and tracking
        det_start = time.time()
        self.detector.track(frame, fps, 10000)
        det_time = time.time() - det_start
        
        # Get current frame detections
        if not hasattr(self.detector, 'pose_results_last') or len(self.detector.pose_results_last) == 0:
            self.frame_count += 1
            return time.time() - frame_start, det_time, 0, None
        
        # 2. WHAM inference for detected persons
        wham_start = time.time()
        pose_results = self.detector.pose_results_last
        
        smpl_params_list = []
        
        for person in pose_results:
            person_id = person.get('track_id', 0)
            
            # Extract bbox and keypoints
            bbox = person['bbox']  # [x1, y1, x2, y2, conf]
            keypoints = person['keypoints']  # (num_kpts, 3)
            
            # Normalize keypoints to [-1, 1]
            kpts_2d = keypoints[:, :2].copy()
            kpts_2d[:, 0] = 2 * (kpts_2d[:, 0] / width) - 1
            kpts_2d[:, 1] = 2 * (kpts_2d[:, 1] / height) - 1
            
            # Prepare input for WHAM (2D keypoints + camera motion)
            n_joints = _C.KEYPOINTS.NUM_JOINTS
            x_input = np.concatenate([
                kpts_2d.flatten(),  # Flatten keypoints
                np.zeros(3)  # Dummy camera motion (can be improved with IMU)
            ])
            
            # Convert to tensor (batch=1, seq=1, features)
            x_tensor = torch.from_numpy(x_input).float().to(self.cfg.DEVICE).unsqueeze(0).unsqueeze(0)
            
            # Prepare bbox for WHAM (center_x, center_y, scale)
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[:4]
            bbox_cx = (bbox_x1 + bbox_x2) / 2 / width
            bbox_cy = (bbox_y1 + bbox_y2) / 2 / height
            bbox_scale = max(bbox_x2 - bbox_x1, bbox_y2 - bbox_y1) / max(width, height)
            bbox_tensor = torch.tensor([[[bbox_cx, bbox_cy, bbox_scale]]]).float().to(self.cfg.DEVICE)
            
            # Resolution tensor
            res_tensor = torch.tensor([[width, height]]).float().to(self.cfg.DEVICE)
            
            # Camera intrinsics (simple CLIFF-style)
            focal_length = (width ** 2 + height ** 2) ** 0.5
            cam_intrinsics = torch.zeros(1, 1, 3, 3).to(self.cfg.DEVICE)
            cam_intrinsics[:, :, 0, 0] = focal_length
            cam_intrinsics[:, :, 1, 1] = focal_length
            cam_intrinsics[:, :, 0, 2] = width / 2
            cam_intrinsics[:, :, 1, 2] = height / 2
            cam_intrinsics[:, :, 2, 2] = 1.0
            
            # Initialize person state if new
            if person_id not in self.person_states:
                self.person_states[person_id] = {
                    'hidden_states': None,
                    'init_kp': torch.zeros(1, 1, n_joints * 3).to(self.cfg.DEVICE),
                    'init_smpl': torch.zeros(1, 1, 24 * 6).to(self.cfg.DEVICE),
                    'init_root': torch.zeros(1, 1, 6).to(self.cfg.DEVICE),
                }
            
            state = self.person_states[person_id]
            
            # Prepare inits
            inits = (state['init_kp'], state['init_smpl'])
            mask = torch.ones(1, 1, n_joints, dtype=torch.bool).to(self.cfg.DEVICE)
            cam_angvel = torch.zeros(1, 1, 6).to(self.cfg.DEVICE)  # No camera motion for now
            
            # Run WHAM forward_single_frame!
            output, new_hidden_states = self.network.forward_single_frame(
                x_tensor, inits, 
                mask=mask, 
                init_root=state['init_root'], 
                cam_angvel=cam_angvel,
                bbox=bbox_tensor,
                res=res_tensor,
                cam_intrinsics=cam_intrinsics,
                return_y_up=True,
                hidden_states=state['hidden_states']
            )
            
            # Update person state
            self.person_states[person_id]['hidden_states'] = new_hidden_states
            self.person_states[person_id]['init_kp'] = new_hidden_states['prev_kp3d']
            self.person_states[person_id]['init_smpl'] = new_hidden_states['prev_pose']
            self.person_states[person_id]['init_root'] = new_hidden_states['prev_root']
            
            # Extract SMPL parameters and vertices  
            # NOTE: forward_single_frame output has NO time dimension
            # verts_cam shape: (B, 6890, 3), trans_cam shape: (B, 3)
            verts_cam = output['verts_cam'][0]  # (6890, 3)
            trans_cam = output['trans_cam'][0]  # (3,)
            
            # DEBUG: Print for first frame
            if self.frame_count == 0:
                logger.info(f"DEBUG Frame 0:")
                logger.info(f"  verts_cam alone range: [{verts_cam.min():.2f}, {verts_cam.max():.2f}]")
                logger.info(f"  trans_cam: {trans_cam.cpu().numpy()}")
                final_verts = verts_cam + trans_cam.unsqueeze(0)
                logger.info(f"  verts + trans range: [{final_verts.min():.2f}, {final_verts.max():.2f}]")
            
            smpl_params = {
                'frame_id': self.frame_count,
                'person_id': person_id,
                'pose': output['pose'][0, 0].cpu().numpy(),  # (144,) = 24 * 6
                'betas': output['betas'][0, 0].cpu().numpy(),  # (10,)
                'trans_cam': trans_cam.cpu().numpy(),  # (3,)
                'trans_world': output['trans_world'][0].cpu().numpy() if 'trans_world' in output else None,
                # Save vertices WITHOUT trans_cam - verts_cam is already correct!
                'verts': verts_cam.cpu().numpy(),  # (6890, 3) - DO NOT ADD trans_cam!
            }
            
            smpl_params_list.append(smpl_params)
            self.smpl_results.append(smpl_params)
        
        wham_time = time.time() - wham_start
        
        # Update timing
        self.timing['detection'].append(det_time)
        self.timing['wham_inference'].append(wham_time)
        
        self.frame_count += 1
        
        total_time = time.time() - frame_start
        self.timing['total'].append(total_time)
        
        return total_time, det_time, wham_time, smpl_params_list
    
    def save_results(self, filename='smpl_results.pkl'):
        """Save all SMPL results to file"""
        output_path = self.output_dir / filename
        joblib.dump(self.smpl_results, output_path)
        logger.info(f"Saved {len(self.smpl_results)} SMPL results to {output_path}")
        
        # Save timing statistics
        timing_path = self.output_dir / 'timing_stats.pkl'
        joblib.dump(self.timing, timing_path)
        logger.info(f"Saved timing statistics to {timing_path}")


class FrameByFrameStream:
    """
    Stream processor with frame-by-frame SMPL output
    """
    def __init__(self, source, output_dir='output/framebyframe_wham',
                 frame_skip=1, max_fps=30):
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
        logger.info(f"⚡ TRUE FRAME-BY-FRAME MODE - LSTM REAL-TIME")
        
        # Initialize WHAM
        self.wham = FrameByFrameWHAM(output_dir=output_dir)
        
        # Stats
        self.stats = {
            'frames_received': 0,
            'frames_processed': 0,
            'frames_skipped': 0,
            'smpl_outputs': 0,
            'start_time': None,
        }
    
    def run(self, duration=None):
        """
        Run frame-by-frame WHAM processing
        """
        logger.info("=" * 60)
        logger.info("TRUE FRAME-BY-FRAME WHAM WITH LSTM")
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
                
                # Process frame immediately
                total_time, det_time, wham_time, smpl_params = self.wham.process_frame(
                    frame, fps=self.fps
                )
                
                self.stats['frames_processed'] += 1
                
                if smpl_params:
                    self.stats['smpl_outputs'] += len(smpl_params)
                    for sp in smpl_params:
                        logger.info(f"✅ Frame {sp['frame_id']}, Person {sp['person_id']}: "
                                   f"SMPL output! (det: {det_time*1000:.0f}ms, wham: {wham_time*1000:.0f}ms, total: {total_time*1000:.0f}ms)")
                
                # Print stats every 5 seconds
                if current_time - last_print >= 5.0:
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
        
        recent_times = self.wham.timing['total'][-30:]
        recent_det = self.wham.timing['detection'][-30:]
        recent_wham = self.wham.timing['wham_inference'][-30:]
        
        avg_total = np.mean(recent_times) if recent_times else 0
        avg_det = np.mean(recent_det) if recent_det else 0
        avg_wham = np.mean(recent_wham) if recent_wham else 0
        
        logger.info(f"\n[{elapsed:.1f}s] STATS:")
        logger.info(f"  Received: {self.stats['frames_received']} ({fps_received:.1f} FPS)")
        logger.info(f"  Processed: {self.stats['frames_processed']} ({fps_processed:.1f} FPS)")
        logger.info(f"  SMPL outputs: {self.stats['smpl_outputs']}")
        logger.info(f"  Avg latency: {avg_total*1000:.0f}ms (det: {avg_det*1000:.0f}ms, wham: {avg_wham*1000:.0f}ms)")
    
    def cleanup(self):
        """Cleanup and save final results"""
        logger.info("\n" + "=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 60)
        
        elapsed = time.time() - self.stats['start_time']
        
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Frames received: {self.stats['frames_received']}")
        logger.info(f"Frames processed: {self.stats['frames_processed']}")
        logger.info(f"Frames skipped: {self.stats['frames_skipped']}")
        logger.info(f"SMPL outputs: {self.stats['smpl_outputs']}")
        logger.info(f"Average FPS: {self.stats['frames_processed']/elapsed:.2f}")
        
        if len(self.wham.timing['total']) > 0:
            avg_total = np.mean(self.wham.timing['total'])
            avg_det = np.mean(self.wham.timing['detection'])
            avg_wham = np.mean(self.wham.timing['wham_inference'])
            logger.info(f"Average total latency: {avg_total*1000:.1f}ms")
            logger.info(f"Average detection: {avg_det*1000:.1f}ms")
            logger.info(f"Average WHAM inference: {avg_wham*1000:.1f}ms")
        
        # Save results
        logger.info("\nSaving results...")
        self.wham.save_results('smpl_results_final.pkl')
        
        self.cap.release()
        logger.info("✅ Cleanup complete!")


def main():
    parser = argparse.ArgumentParser(description="Frame-by-frame realtime WHAM with LSTM")
    parser.add_argument('video', type=str, help='Video file or camera ID (0, 1, etc.)')
    parser.add_argument('--output-dir', type=str, default='output/framebyframe_wham',
                       help='Output directory')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Process every Nth frame (1 = process all frames)')
    parser.add_argument('--max-fps', type=float, default=30,
                       help='Maximum FPS to process (0 = no limit)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Duration to run in seconds (None = until video ends)')
    
    args = parser.parse_args()
    
    # Run streaming
    stream = FrameByFrameStream(
        args.video,
        output_dir=args.output_dir,
        frame_skip=args.frame_skip,
        max_fps=args.max_fps
    )
    
    stream.run(duration=args.duration)


if __name__ == '__main__':
    main()
