"""
Record-and-Upload Client - Run this on your Mac
Records short video clips and uploads to cluster for processing

This is more reliable than streaming if network is unstable.

Usage on Mac:
    python record_and_upload.py \\
        --cluster user@cluster.gatech.edu \\
        --remote-dir /path/on/cluster/WHAM/incoming \\
        --clip-duration 2

On cluster, run the companion script to watch for videos and process them.
"""

import cv2
import os
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

class VideoRecorder:
    def __init__(self, camera_id=0, clip_duration=2, output_dir='clips'):
        self.camera_id = camera_id
        self.clip_duration = clip_duration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎥 Opening camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        print(f"✅ Camera ready: {self.width}x{self.height} @ {self.fps} FPS")
        
    def record_clip(self):
        """Record a short video clip"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.output_dir / f"clip_{timestamp}.mp4"
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(filename), fourcc, self.fps, 
                            (self.width, self.height))
        
        print(f"🔴 Recording {self.clip_duration}s clip...", end='', flush=True)
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < self.clip_duration:
            ret, frame = self.cap.read()
            if ret:
                out.write(frame)
                frame_count += 1
            else:
                print("\n⚠️  Failed to read frame")
                break
        
        out.release()
        
        # Get file size
        size_mb = filename.stat().st_size / (1024 * 1024)
        
        print(f" ✅ Saved {frame_count} frames ({size_mb:.1f} MB)")
        
        return filename
    
    def close(self):
        """Release camera"""
        self.cap.release()

def upload_to_cluster(local_file, cluster, remote_dir):
    """Upload file to cluster via SCP"""
    remote_path = f"{cluster}:{remote_dir}/{local_file.name}"
    
    print(f"📤 Uploading to cluster...", end='', flush=True)
    
    cmd = ['scp', '-q', str(local_file), remote_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(" ✅ Done")
            return True
        else:
            print(f" ❌ Failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(" ❌ Timeout")
        return False
    except Exception as e:
        print(f" ❌ Error: {e}")
        return False

def wait_for_results(cluster, remote_dir, clip_name):
    """Wait for results file from cluster"""
    # Results file will be named: clip_XXXXX_results.pkl
    results_name = clip_name.replace('.mp4', '_results.pkl')
    local_results = Path('results') / results_name
    local_results.parent.mkdir(exist_ok=True)
    
    remote_results = f"{cluster}:{remote_dir}/{results_name}"
    
    print(f"⏳ Waiting for results...", end='', flush=True)
    
    # Poll for results file (cluster will create it after processing)
    for i in range(60):  # Wait up to 60 seconds
        time.sleep(1)
        
        # Try to download results
        cmd = ['scp', '-q', remote_results, str(local_results)]
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode == 0:
            print(f" ✅ Received! ({i+1}s)")
            return local_results
    
    print(" ⏰ Timeout")
    return None

def main():
    parser = argparse.ArgumentParser(
        description='Record video clips and upload to cluster for processing'
    )
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--clip-duration', type=float, default=2.0,
                        help='Clip duration in seconds')
    parser.add_argument('--cluster', type=str, required=True,
                        help='Cluster SSH address (user@host)')
    parser.add_argument('--remote-dir', type=str, required=True,
                        help='Directory on cluster for video exchange')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='Time between clips (seconds)')
    parser.add_argument('--no-wait', action='store_true',
                        help='Don\'t wait for results (just upload)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎥 Record-and-Upload for Remote WHAM Processing")
    print("="*60)
    print(f"\n📹 Camera: {args.camera}")
    print(f"⏱️  Clip duration: {args.clip_duration}s")
    print(f"🖥️  Cluster: {args.cluster}")
    print(f"📁 Remote dir: {args.remote_dir}")
    print(f"\n⚠️  Make sure:")
    print(f"   1. SSH key authentication is set up (no password prompts)")
    print(f"   2. Remote directory exists: {args.remote_dir}")
    print(f"   3. Cluster watcher script is running")
    print("\nPress Ctrl+C to stop\n")
    
    recorder = VideoRecorder(args.camera, args.clip_duration)
    
    try:
        clip_num = 0
        while True:
            clip_num += 1
            print(f"\n--- Clip #{clip_num} ---")
            
            # Record
            clip_file = recorder.record_clip()
            
            # Upload
            success = upload_to_cluster(clip_file, args.cluster, args.remote_dir)
            
            if success and not args.no_wait:
                # Wait for results
                results = wait_for_results(args.cluster, args.remote_dir, 
                                          clip_file.name)
                if results:
                    print(f"📊 Results saved: {results}")
                    # Here you could load and display the poses
            
            # Clean up local clip to save space
            clip_file.unlink()
            print("🗑️  Local clip deleted")
            
            # Wait before next clip
            if args.interval > 0:
                time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")
    finally:
        recorder.close()
        print("✅ Camera released")

if __name__ == '__main__':
    main()

