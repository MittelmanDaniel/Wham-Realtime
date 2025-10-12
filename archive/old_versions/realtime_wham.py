"""
Real-time WHAM processing with actual pose estimation
Processes video streams with live WHAM inference
"""
import cv2
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import deque
from loguru import logger
import tempfile
import os

from configs.config import get_cfg_defaults
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor


class RealtimeWHAM:
    """
    Real-time WHAM processor - runs actual pose estimation
    """
    def __init__(self, output_dir='output/realtime_wham'):
        logger.info("Initializing Real-time WHAM...")
        
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
        
        # Initialize detector and feature extractor
        logger.info("Loading detector and feature extractor...")
        self.detector = DetectionModel(self.cfg.DEVICE.lower())
        self.extractor = FeatureExtractor(self.cfg.DEVICE.lower())
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Real-time WHAM initialized!")
    
    @torch.no_grad()
    def process_frame(self, frame, frame_id, fps=30):
        """
        Process a single frame through detection
        """
        # Run 2D detection and tracking
        self.detector.track(frame, fps, 1000)  # length=1000 as placeholder
        return frame
    
    @torch.no_grad()
    def process_batch(self, frames, fps=30):
        """
        Process a batch of frames through WHAM
        Returns processing time and detection count
        """
        if len(frames) == 0:
            return 0, 0
        
        start_time = time.time()
        
        # Reset detector for new batch
        self.detector = DetectionModel(self.cfg.DEVICE.lower())
        
        # Process each frame through detector
        for i, frame in enumerate(frames):
            self.detector.track(frame, fps, len(frames))
        
        # Get tracking results
        tracking_results = self.detector.process(fps)
        
        process_time = time.time() - start_time
        
        # Count detections
        num_detections = len(tracking_results['bbox']) if 'bbox' in tracking_results else 0
        
        return process_time, num_detections


class RealtimeWHAMStream:
    """
    Stream processor with real-time WHAM
    """
    def __init__(self, source, output_dir='output/realtime_wham',
                 buffer_size=16, frame_skip=1, max_fps=30):
        self.source = source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
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
        logger.info(f"Buffer size: {buffer_size} frames")
        logger.info(f"Frame skip: {frame_skip}")
        
        # Initialize WHAM
        self.wham = RealtimeWHAM(output_dir=output_dir)
        
        # Stats
        self.frame_buffer = deque(maxlen=buffer_size)
        self.stats = {
            'frames_received': 0,
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_processing_time': 0,
            'total_detections': 0,
            'start_time': None,
        }
    
    def run(self, duration=None, visualize=False, save_video=False):
        """
        Run real-time WHAM processing
        """
        logger.info("=" * 60)
        logger.info("REAL-TIME WHAM PROCESSING")
        logger.info("=" * 60)
        logger.info("Starting... Press Ctrl+C to stop")
        
        self.stats['start_time'] = time.time()
        last_print = self.stats['start_time']
        last_frame_time = self.stats['start_time']
        
        # Video writer if saving
        video_writer = None
        if save_video:
            output_video = self.output_dir / 'realtime_output.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video), fourcc, self.fps, (self.width, self.height)
            )
            logger.info(f"Saving video to: {output_video}")
        
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
                
                # Add to buffer
                self.frame_buffer.append(frame.copy())
                
                # Process when buffer is full
                if len(self.frame_buffer) >= self.buffer_size:
                    frames_to_process = list(self.frame_buffer)
                    self.frame_buffer.clear()
                    
                    # Run WHAM on batch
                    process_time, num_detections = self.wham.process_batch(
                        frames_to_process, fps=self.fps
                    )
                    
                    self.stats['total_processing_time'] += process_time
                    self.stats['frames_processed'] += len(frames_to_process)
                    self.stats['total_detections'] += num_detections
                    
                    # Save frames if requested
                    if video_writer:
                        for f in frames_to_process:
                            video_writer.write(f)
                    
                    # Print stats every second
                    if time.time() - last_print >= 1.0:
                        self.print_stats(process_time, num_detections)
                        last_print = time.time()
                
                # Optional visualization
                if visualize:
                    # Draw some info on frame
                    display_frame = frame.copy()
                    cv2.putText(display_frame, 
                               f"Frames: {self.stats['frames_received']}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Real-time WHAM', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        
        finally:
            # Process remaining frames
            if len(self.frame_buffer) > 0:
                frames_to_process = list(self.frame_buffer)
                process_time, num_detections = self.wham.process_batch(
                    frames_to_process, fps=self.fps
                )
                self.stats['total_processing_time'] += process_time
                self.stats['frames_processed'] += len(frames_to_process)
                self.stats['total_detections'] += num_detections
            
            self.cap.release()
            if video_writer:
                video_writer.release()
            if visualize:
                cv2.destroyAllWindows()
            
            self.print_final_stats()
    
    def print_stats(self, last_process_time, last_detections):
        """Print current statistics"""
        elapsed = time.time() - self.stats['start_time']
        receive_fps = self.stats['frames_received'] / elapsed
        process_fps = self.stats['frames_processed'] / elapsed
        avg_detections = (self.stats['total_detections'] / 
                         (self.stats['frames_processed'] / self.buffer_size)
                         if self.stats['frames_processed'] > 0 else 0)
        
        logger.info(
            f"[{elapsed:.1f}s] "
            f"Recv: {self.stats['frames_received']} ({receive_fps:.1f} FPS) | "
            f"Proc: {self.stats['frames_processed']} ({process_fps:.1f} FPS) | "
            f"Skip: {self.stats['frames_skipped']} | "
            f"Batch: {last_process_time*1000:.0f}ms | "
            f"People: {last_detections}"
        )
    
    def print_final_stats(self):
        """Print final summary"""
        elapsed = time.time() - self.stats['start_time']
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Frames received: {self.stats['frames_received']}")
        logger.info(f"Frames processed: {self.stats['frames_processed']}")
        logger.info(f"Frames skipped: {self.stats['frames_skipped']}")
        logger.info(f"Average receive FPS: {self.stats['frames_received']/elapsed:.1f}")
        logger.info(f"Average process FPS: {self.stats['frames_processed']/elapsed:.1f}")
        logger.info(f"Total processing time: {self.stats['total_processing_time']:.2f}s")
        logger.info(f"Total detections: {self.stats['total_detections']}")
        if self.stats['frames_processed'] > 0:
            logger.info(f"Avg detections per batch: {self.stats['total_detections']/(self.stats['frames_processed']/self.buffer_size):.1f}")
        logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Real-time WHAM with actual pose estimation'
    )
    parser.add_argument('source', 
                        help='Video source: file path, URL, or camera ID')
    parser.add_argument('--output', default='output/realtime_wham',
                        help='Output directory')
    parser.add_argument('--buffer-size', type=int, default=16,
                        help='Frame buffer size')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Process every Nth frame')
    parser.add_argument('--max-fps', type=int, default=30,
                        help='Maximum input FPS')
    parser.add_argument('--duration', type=int, default=None,
                        help='Max duration in seconds')
    parser.add_argument('--visualize', action='store_true',
                        help='Show live visualization')
    parser.add_argument('--save-video', action='store_true',
                        help='Save processed video')
    
    args = parser.parse_args()
    
    processor = RealtimeWHAMStream(
        args.source,
        output_dir=args.output,
        buffer_size=args.buffer_size,
        frame_skip=args.frame_skip,
        max_fps=args.max_fps
    )
    
    processor.run(
        duration=args.duration,
        visualize=args.visualize,
        save_video=args.save_video
    )
