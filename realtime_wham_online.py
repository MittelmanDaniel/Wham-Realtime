"""
Real-time WHAM with ONLINE processing (batch=1)
Process each frame immediately for lowest latency
"""
import cv2
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from loguru import logger

from configs.config import get_cfg_defaults
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor


class OnlineWHAM:
    """
    Online WHAM processor - processes frames one at a time for lowest latency
    """
    def __init__(self, output_dir='output/online_wham'):
        logger.info("Initializing Online WHAM...")
        
        # Setup config
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file('configs/yamls/demo.yaml')
        
        # Build network
        logger.info("Loading WHAM network...")
        self.network = build_network(
            self.cfg, 
            build_body_model(self.cfg.DEVICE, self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN)
        )
        self.network.eval()
        
        # Initialize detector and feature extractor ONCE (not per batch!)
        logger.info("Loading detector and feature extractor...")
        self.detector = DetectionModel(self.cfg.DEVICE.lower())
        self.extractor = FeatureExtractor(self.cfg.DEVICE.lower())
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track state
        self.frame_count = 0
        self.total_frames = 1000  # Placeholder
        
        logger.info("✅ Online WHAM initialized!")
    
    @torch.no_grad()
    def process_frame(self, frame, fps=30):
        """
        Process a single frame through detection (online, no batching)
        Returns processing time and number of detections
        """
        start_time = time.time()
        
        # Run detection + tracking on this single frame
        self.frame_count += 1
        self.detector.track(frame, fps, self.total_frames)
        
        process_time = time.time() - start_time
        
        # Count current detections (from last frame's results)
        num_detections = len(self.detector.pose_results_last) if hasattr(self.detector, 'pose_results_last') else 0
        
        return process_time, num_detections
    
    def get_current_results(self):
        """
        Get current tracking results
        """
        # Get accumulated results so far
        tracking_results = self.detector.process(fps=30)
        return tracking_results


class OnlineWHAMStream:
    """
    Stream processor with online (batch=1) WHAM processing
    """
    def __init__(self, source, output_dir='output/online_wham',
                 frame_skip=1, max_fps=30):
        self.source = source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_skip = frame_skip
        self.max_fps = max_fps
        self.frame_time = 1.0 / max_fps if max_fps > 0 else 0
        
        # Open video source
        logger.info(f"Opening video source: {source}")
        
        # Try to convert to int (camera ID)
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
        logger.info(f"⚡ ONLINE MODE: Batch size = 1 (lowest latency)")
        
        # Initialize WHAM
        self.wham = OnlineWHAM(output_dir=output_dir)
        
        # Stats
        self.stats = {
            'frames_received': 0,
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_processing_time': 0,
            'total_detections': 0,
            'start_time': None,
            'frame_times': [],  # Track individual frame latencies
        }
    
    def run(self, duration=None, visualize=False):
        """
        Run online WHAM processing (batch=1)
        """
        logger.info("=" * 60)
        logger.info("ONLINE WHAM PROCESSING (BATCH=1)")
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
                
                # Throttle frame reading to max FPS
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
                
                # Process IMMEDIATELY (no batching!)
                process_time, num_detections = self.wham.process_frame(frame, fps=self.fps)
                
                self.stats['total_processing_time'] += process_time
                self.stats['frames_processed'] += 1
                self.stats['total_detections'] += num_detections
                self.stats['frame_times'].append(process_time)
                
                # Print stats every second
                if time.time() - last_print >= 1.0:
                    self.print_stats(process_time, num_detections)
                    last_print = time.time()
                
                # Optional visualization
                if visualize:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, 
                               f"Frame: {self.stats['frames_received']} | Latency: {process_time*1000:.0f}ms", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Online WHAM', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        
        finally:
            self.cap.release()
            if visualize:
                cv2.destroyAllWindows()
            
            self.print_final_stats()
    
    def print_stats(self, last_process_time, last_detections):
        """Print current statistics"""
        elapsed = time.time() - self.stats['start_time']
        receive_fps = self.stats['frames_received'] / elapsed
        process_fps = self.stats['frames_processed'] / elapsed
        avg_latency = (self.stats['total_processing_time'] / self.stats['frames_processed']
                      if self.stats['frames_processed'] > 0 else 0)
        
        logger.info(
            f"[{elapsed:.1f}s] "
            f"Recv: {self.stats['frames_received']} ({receive_fps:.1f} FPS) | "
            f"Proc: {self.stats['frames_processed']} ({process_fps:.1f} FPS) | "
            f"Skip: {self.stats['frames_skipped']} | "
            f"Latency: {last_process_time*1000:.0f}ms | "
            f"Avg: {avg_latency*1000:.0f}ms | "
            f"People: {last_detections}"
        )
    
    def print_final_stats(self):
        """Print final summary"""
        elapsed = time.time() - self.stats['start_time']
        
        avg_latency = (self.stats['total_processing_time'] / self.stats['frames_processed']
                      if self.stats['frames_processed'] > 0 else 0)
        
        # Calculate latency percentiles
        if self.stats['frame_times']:
            sorted_times = sorted(self.stats['frame_times'])
            p50 = sorted_times[len(sorted_times)//2] * 1000
            p95 = sorted_times[int(len(sorted_times)*0.95)] * 1000
            p99 = sorted_times[int(len(sorted_times)*0.99)] * 1000
        else:
            p50 = p95 = p99 = 0
        
        logger.info("=" * 60)
        logger.info("ONLINE PROCESSING COMPLETE")
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Frames received: {self.stats['frames_received']}")
        logger.info(f"Frames processed: {self.stats['frames_processed']}")
        logger.info(f"Frames skipped: {self.stats['frames_skipped']}")
        logger.info(f"Average receive FPS: {self.stats['frames_received']/elapsed:.1f}")
        logger.info(f"Average process FPS: {self.stats['frames_processed']/elapsed:.1f}")
        logger.info(f"Total processing time: {self.stats['total_processing_time']:.2f}s")
        logger.info(f"")
        logger.info(f"LATENCY STATS:")
        logger.info(f"  Average: {avg_latency*1000:.0f}ms")
        logger.info(f"  P50: {p50:.0f}ms")
        logger.info(f"  P95: {p95:.0f}ms")
        logger.info(f"  P99: {p99:.0f}ms")
        logger.info(f"")
        logger.info(f"Total detections: {self.stats['total_detections']}")
        if self.stats['frames_processed'] > 0:
            logger.info(f"Avg detections per frame: {self.stats['total_detections']/self.stats['frames_processed']:.1f}")
        logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Online (batch=1) WHAM with lowest latency'
    )
    parser.add_argument('source', 
                        help='Video source: file path, URL, or camera ID')
    parser.add_argument('--output', default='output/online_wham',
                        help='Output directory')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Process every Nth frame')
    parser.add_argument('--max-fps', type=int, default=30,
                        help='Maximum input FPS')
    parser.add_argument('--duration', type=int, default=None,
                        help='Max duration in seconds')
    parser.add_argument('--visualize', action='store_true',
                        help='Show live visualization')
    
    args = parser.parse_args()
    
    processor = OnlineWHAMStream(
        args.source,
        output_dir=args.output,
        frame_skip=args.frame_skip,
        max_fps=args.max_fps
    )
    
    processor.run(
        duration=args.duration,
        visualize=args.visualize
    )
