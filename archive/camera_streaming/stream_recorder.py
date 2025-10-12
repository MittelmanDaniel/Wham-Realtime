"""
Stream Recorder - Captures video from HTTP stream and saves to file
Then processes with WHAM

Usage on cluster (after SSH tunnel is established):
    python stream_recorder.py --stream http://localhost:8080/video.mjpeg
"""

import cv2
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def record_from_stream(stream_url, duration=3, output_file='temp_clip.mp4'):
    """
    Record video from HTTP stream to file
    
    Args:
        stream_url: URL of video stream
        duration: Recording duration in seconds
        output_file: Output filename
    """
    print(f"📡 Connecting to stream: {stream_url}")
    
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print(f"❌ Failed to connect to stream")
        return None
    
    # Get stream properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Connected: {width}x{height} @ {fps} FPS")
    print(f"🔴 Recording {duration}s clip...", end='', flush=True)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            frame_count += 1
        else:
            print(f"\n⚠️  Failed to read frame")
            break
    
    out.release()
    cap.release()
    
    print(f" ✅ Recorded {frame_count} frames")
    
    return output_file

def process_with_wham(video_file, output_dir='output/stream'):
    """Process video with WHAM"""
    print(f"⚙️  Processing with WHAM...")
    
    cmd = [
        'python', 'demo.py',
        '--video', video_file,
        '--output_pth', output_dir,
        '--save_pkl',
        '--estimate_local_only'
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ WHAM processing complete ({elapsed:.1f}s)")
            return True
        else:
            print(f"❌ WHAM failed:")
            print(result.stderr[-500:])  # Last 500 chars of error
            return False
    
    except subprocess.TimeoutExpired:
        print("❌ Processing timeout")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Record from HTTP stream and process with WHAM'
    )
    parser.add_argument('--stream', type=str, 
                       default='http://localhost:8080/video.mjpeg',
                       help='Stream URL (after SSH tunnel)')
    parser.add_argument('--duration', type=float, default=3.0,
                       help='Recording duration in seconds')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Time between recordings (seconds)')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuously record and process')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("📹 Stream Recorder for WHAM")
    print("="*60)
    print(f"\n📡 Stream: {args.stream}")
    print(f"⏱️  Duration: {args.duration}s per clip")
    print(f"🔄 Continuous: {args.continuous}")
    
    if args.continuous:
        print("\n🔁 Continuous mode - Press Ctrl+C to stop\n")
    
    clip_num = 0
    
    try:
        while True:
            clip_num += 1
            print(f"\n--- Clip #{clip_num} ({datetime.now().strftime('%H:%M:%S')}) ---")
            
            # Record clip
            video_file = f"temp_clip_{clip_num}.mp4"
            recorded = record_from_stream(args.stream, args.duration, video_file)
            
            if recorded:
                # Process with WHAM
                output_dir = f"output/stream/clip_{clip_num:03d}"
                success = process_with_wham(recorded, output_dir)
                
                if success:
                    print(f"📊 Results saved to: {output_dir}/")
                
                # Clean up temp file
                Path(video_file).unlink()
            
            # If not continuous, exit after one clip
            if not args.continuous:
                break
            
            # Wait before next clip
            if args.interval > 0:
                print(f"⏸️  Waiting {args.interval}s...")
                time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
    
    print("\n✅ Done")

if __name__ == '__main__':
    main()

