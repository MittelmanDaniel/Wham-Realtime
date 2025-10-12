"""
Visualize real-time WHAM results with detection overlays
Creates an output video with bounding boxes and keypoints
"""
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from loguru import logger
from realtime_wham_online import OnlineWHAM

def draw_keypoints(img, keypoints, threshold=0.3):
    """Draw 2D keypoints on image"""
    # COCO skeleton connections
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    # Draw skeleton
    for conn in skeleton:
        pt1_idx, pt2_idx = conn[0] - 1, conn[1] - 1
        if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
            kp1, kp2 = keypoints[pt1_idx], keypoints[pt2_idx]
            if kp1[2] > threshold and kp2[2] > threshold:
                pt1 = (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2[0]), int(kp2[1]))
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if kp[2] > threshold:
            pt = (int(kp[0]), int(kp[1]))
            cv2.circle(img, pt, 3, (0, 0, 255), -1)
    
    return img

def visualize_realtime(video_path, output_path, frame_skip=1, max_fps=30, duration=None, max_frames=None):
    """
    Process video with real-time WHAM and create visualization
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Input: {video_path}")
    logger.info(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    
    # Create output video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps / frame_skip, (width, height))
    
    # Initialize WHAM
    logger.info("Initializing WHAM...")
    wham = OnlineWHAM(output_dir=output_path.parent)
    
    # Process video
    logger.info("Processing video...")
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    latencies = []
    
    try:
        while cap.isOpened():
            # Check duration
            if duration and (time.time() - start_time) > duration:
                break
            
            # Check max frames
            if max_frames and processed_count >= max_frames:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames
            if (frame_count - 1) % frame_skip != 0:
                continue
            
            # Process frame
            process_time, num_detections = wham.process_frame(frame, fps=fps)
            latencies.append(process_time * 1000)
            processed_count += 1
            
            # Create visualization
            viz_frame = frame.copy()
            
            # Get detection results
            if hasattr(wham.detector, 'pose_results_last'):
                for person in wham.detector.pose_results_last:
                    # Draw bounding box
                    if 'bbox' in person:
                        bbox = person['bbox']
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw ID
                        if 'track_id' in person:
                            track_id = person['track_id']
                            cv2.putText(viz_frame, f"ID: {track_id}", 
                                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.6, (0, 255, 0), 2)
                    
                    # Draw keypoints
                    if 'keypoints' in person:
                        keypoints = person['keypoints']
                        viz_frame = draw_keypoints(viz_frame, keypoints)
            
            # Add stats overlay
            cv2.putText(viz_frame, f"Frame: {frame_count} | Latency: {process_time*1000:.0f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(viz_frame, f"People: {num_detections}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(viz_frame)
            
            # Progress
            if processed_count % 30 == 0:
                avg_latency = np.mean(latencies[-30:])
                logger.info(f"Processed {processed_count} frames | Avg latency: {avg_latency:.0f}ms")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        cap.release()
        out.release()
        
        # Final stats
        elapsed = time.time() - start_time
        avg_latency = np.mean(latencies) if latencies else 0
        p50 = np.percentile(latencies, 50) if latencies else 0
        p95 = np.percentile(latencies, 95) if latencies else 0
        
        logger.info("=" * 60)
        logger.info("VISUALIZATION COMPLETE")
        logger.info(f"Output: {output_path}")
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"Processed frames: {processed_count}")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Average latency: {avg_latency:.0f}ms (P50: {p50:.0f}ms, P95: {p95:.0f}ms)")
        logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize real-time WHAM with detection overlays'
    )
    parser.add_argument('video', help='Input video path')
    parser.add_argument('--output', default='output/realtime_viz/output.mp4',
                        help='Output video path')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Process every Nth frame')
    parser.add_argument('--max-fps', type=int, default=30,
                        help='Maximum input FPS')
    parser.add_argument('--duration', type=int, default=None,
                        help='Max duration in seconds')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Max number of frames to process')
    
    args = parser.parse_args()
    
    visualize_realtime(
        args.video,
        args.output,
        frame_skip=args.frame_skip,
        max_fps=args.max_fps,
        duration=args.duration,
        max_frames=args.max_frames
    )

