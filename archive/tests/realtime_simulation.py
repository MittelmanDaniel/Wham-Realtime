"""
Simulate real-time video streaming by reading a video file at its native FPS.
This mimics what would happen with a live camera feed.
"""
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
import torch
from loguru import logger

# Import WHAM API
from wham_api import WHAM_API


class RealtimeSimulator:
    """
    Simulates real-time video streaming from a file
    """
    def __init__(self, video_path, playback_speed=1.0):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        self.playback_speed = playback_speed
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_time = 1.0 / self.fps * self.playback_speed
        
        logger.info(f"Video: {video_path}")
        logger.info(f"Resolution: {self.width}x{self.height}")
        logger.info(f"FPS: {self.fps:.2f}")
        logger.info(f"Total frames: {self.total_frames}")
        logger.info(f"Frame time: {self.frame_time*1000:.2f}ms")
    
    def __iter__(self):
        """
        Iterator that yields frames at the correct real-time rate
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Calculate when this frame should be displayed
            expected_time = start_time + (frame_count * self.frame_time)
            current_time = time.time()
            
            # Sleep if we're ahead of schedule
            if current_time < expected_time:
                time.sleep(expected_time - current_time)
            
            # Calculate actual timing stats
            actual_time = time.time()
            actual_fps = frame_count / (actual_time - start_time) if actual_time > start_time else 0
            
            yield {
                'frame': frame,
                'frame_number': frame_count,
                'timestamp': actual_time - start_time,
                'fps': actual_fps,
                'total_frames': self.total_frames
            }
    
    def close(self):
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def process_realtime_simulation(video_path, output_dir='output/realtime_sim', 
                                 visualize=False, frame_skip=1):
    """
    Process a video file as if it were a real-time stream
    
    Args:
        video_path: Path to video file to simulate streaming
        output_dir: Where to save results
        visualize: Whether to visualize results
        frame_skip: Process every Nth frame (1 = all frames, 2 = every other frame)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("REAL-TIME SIMULATION MODE")
    logger.info("=" * 50)
    
    # Initialize WHAM
    logger.info("Initializing WHAM...")
    wham = WHAM_API()
    logger.info("WHAM initialized!")
    
    # Start simulator
    with RealtimeSimulator(video_path) as sim:
        logger.info(f"Starting real-time simulation (frame_skip={frame_skip})...")
        logger.info("Processing frames as they arrive...")
        
        frame_buffer = []
        buffer_size = 16  # Process in small batches
        processed_count = 0
        skipped_count = 0
        
        start_time = time.time()
        last_print = start_time
        
        for frame_data in sim:
            frame_num = frame_data['frame_number']
            
            # Skip frames if requested
            if (frame_num - 1) % frame_skip != 0:
                skipped_count += 1
                continue
            
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            fps = frame_data['fps']
            
            # Add to buffer
            frame_buffer.append(frame)
            
            # Process when buffer is full
            if len(frame_buffer) >= buffer_size:
                process_start = time.time()
                
                # TODO: Here you would call WHAM on the frame buffer
                # For now, just simulate processing time
                # results = wham.process_frames(frame_buffer)
                time.sleep(0.01)  # Simulate processing
                
                process_time = time.time() - process_start
                processed_count += len(frame_buffer)
                frame_buffer = []
                
                # Print stats every second
                current_time = time.time()
                if current_time - last_print >= 1.0:
                    total_time = current_time - start_time
                    processing_fps = processed_count / total_time if total_time > 0 else 0
                    
                    logger.info(
                        f"[{timestamp:.1f}s] "
                        f"Frame {frame_num}/{frame_data['total_frames']} | "
                        f"Stream: {fps:.1f} FPS | "
                        f"Processing: {processing_fps:.1f} FPS | "
                        f"Buffer: {len(frame_buffer)} | "
                        f"Batch time: {process_time*1000:.1f}ms"
                    )
                    last_print = current_time
        
        # Process remaining frames
        if frame_buffer:
            processed_count += len(frame_buffer)
        
        total_time = time.time() - start_time
        avg_fps = processed_count / total_time if total_time > 0 else 0
        
        logger.info("=" * 50)
        logger.info("SIMULATION COMPLETE")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Frames processed: {processed_count}")
        logger.info(f"Frames skipped: {skipped_count}")
        logger.info(f"Average processing FPS: {avg_fps:.1f}")
        logger.info("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate real-time video processing with WHAM'
    )
    parser.add_argument('video', help='Video file to simulate streaming')
    parser.add_argument('--output', default='output/realtime_sim',
                        help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results in real-time')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Process every Nth frame (1=all, 2=every other, etc)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier (1.0=normal, 2.0=2x speed)')
    
    args = parser.parse_args()
    
    process_realtime_simulation(
        args.video,
        output_dir=args.output,
        visualize=args.visualize,
        frame_skip=args.frame_skip
    )
