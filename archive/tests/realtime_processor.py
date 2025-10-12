"""
Real-time WHAM processor for live video streams
Can process from:
- HTTP streams (http://...)
- RTSP streams (rtsp://...)
- Local camera (0, 1, 2, ...)
- Video files (simulated real-time)
"""
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from loguru import logger


class RealtimeWHAMProcessor:
    """
    Process video streams in real-time with WHAM
    """
    def __init__(self, source, output_dir='output/realtime', 
                 buffer_size=16, frame_skip=1):
        self.source = source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.frame_skip = frame_skip
        
        # Open video source
        logger.info(f"Opening video source: {source}")
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
        
        # Stats tracking
        self.frame_buffer = deque(maxlen=buffer_size)
        self.stats = {
            'frames_received': 0,
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_processing_time': 0,
            'start_time': None,
        }
    
    def process_frame_batch(self, frames):
        """
        Process a batch of frames with WHAM
        TODO: Integrate actual WHAM processing
        """
        process_start = time.time()
        
        # Placeholder: Here you would call WHAM
        # For now, just simulate processing
        # results = wham.process_frames(frames)
        
        # Simulate processing time
        time.sleep(0.05)  # ~50ms per batch
        
        process_time = time.time() - process_start
        self.stats['total_processing_time'] += process_time
        self.stats['frames_processed'] += len(frames)
        
        return process_time
    
    def run(self, duration=None, visualize=False):
        """
        Start processing the video stream
        
        Args:
            duration: Max duration in seconds (None = unlimited)
            visualize: Show live visualization
        """
        logger.info("=" * 50)
        logger.info("REAL-TIME WHAM PROCESSOR")
        logger.info("=" * 50)
        logger.info("Starting... Press Ctrl+C to stop")
        
        self.stats['start_time'] = time.time()
        last_print = self.stats['start_time']
        
        try:
            while True:
                # Check duration limit
                if duration and (time.time() - self.stats['start_time']) > duration:
                    logger.info(f"Duration limit reached ({duration}s)")
                    break
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                self.stats['frames_received'] += 1
                
                # Skip frames if requested
                if (self.stats['frames_received'] - 1) % self.frame_skip != 0:
                    self.stats['frames_skipped'] += 1
                    continue
                
                # Add to buffer
                self.frame_buffer.append(frame)
                
                # Process when buffer is full
                if len(self.frame_buffer) >= self.buffer_size:
                    frames_to_process = list(self.frame_buffer)
                    self.frame_buffer.clear()
                    
                    process_time = self.process_frame_batch(frames_to_process)
                    
                    # Print stats every second
                    current_time = time.time()
                    if current_time - last_print >= 1.0:
                        self.print_stats()
                        last_print = current_time
                
                # Optional visualization
                if visualize:
                    cv2.imshow('Real-time WHAM', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        
        finally:
            # Process remaining frames
            if len(self.frame_buffer) > 0:
                self.process_frame_batch(list(self.frame_buffer))
            
            self.cap.release()
            if visualize:
                cv2.destroyAllWindows()
            
            self.print_final_stats()
    
    def print_stats(self):
        """Print current processing statistics"""
        elapsed = time.time() - self.stats['start_time']
        receive_fps = self.stats['frames_received'] / elapsed
        process_fps = self.stats['frames_processed'] / elapsed
        avg_process_time = (self.stats['total_processing_time'] / 
                           (self.stats['frames_processed'] / self.buffer_size)
                           if self.stats['frames_processed'] > 0 else 0)
        
        logger.info(
            f"[{elapsed:.1f}s] "
            f"Received: {self.stats['frames_received']} ({receive_fps:.1f} FPS) | "
            f"Processed: {self.stats['frames_processed']} ({process_fps:.1f} FPS) | "
            f"Skipped: {self.stats['frames_skipped']} | "
            f"Batch time: {avg_process_time*1000:.1f}ms"
        )
    
    def print_final_stats(self):
        """Print final summary"""
        elapsed = time.time() - self.stats['start_time']
        
        logger.info("=" * 50)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Frames received: {self.stats['frames_received']}")
        logger.info(f"Frames processed: {self.stats['frames_processed']}")
        logger.info(f"Frames skipped: {self.stats['frames_skipped']}")
        logger.info(f"Average receive FPS: {self.stats['frames_received']/elapsed:.1f}")
        logger.info(f"Average process FPS: {self.stats['frames_processed']/elapsed:.1f}")
        logger.info(f"Total processing time: {self.stats['total_processing_time']:.2f}s")
        logger.info("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process video streams in real-time with WHAM'
    )
    parser.add_argument('source', 
                        help='Video source: URL (http://...), camera (0), or file')
    parser.add_argument('--output', default='output/realtime',
                        help='Output directory')
    parser.add_argument('--buffer-size', type=int, default=16,
                        help='Frame buffer size (default: 16)')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Process every Nth frame (default: 1)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Max duration in seconds (default: unlimited)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show live visualization')
    
    args = parser.parse_args()
    
    # Try to parse as integer (camera ID)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    processor = RealtimeWHAMProcessor(
        source,
        output_dir=args.output,
        buffer_size=args.buffer_size,
        frame_skip=args.frame_skip
    )
    
    processor.run(duration=args.duration, visualize=args.visualize)
