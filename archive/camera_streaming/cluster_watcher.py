"""
Cluster Video Watcher - Run this on the cluster
Watches for incoming videos and processes them with WHAM

Usage on cluster:
    python cluster_watcher.py --watch-dir incoming --use-gpu
"""

import os
import time
import argparse
import joblib
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import torch
from loguru import logger

from configs.config import get_cfg_defaults
from lib.models import build_network, build_body_model

class VideoProcessor:
    """Processes videos with WHAM"""
    
    def __init__(self, output_dir='output/remote', device='cuda'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        logger.info("Loading WHAM model...")
        cfg = get_cfg_defaults()
        cfg.merge_from_file('configs/yamls/demo.yaml')
        cfg.DEVICE = device
        
        smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
        smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
        self.network = build_network(cfg, smpl)
        self.network.eval()
        
        logger.info("✅ WHAM model loaded")
    
    def process_video(self, video_path):
        """Process a video file with WHAM"""
        video_path = Path(video_path)
        
        logger.info(f"🎬 Processing: {video_path.name}")
        start_time = time.time()
        
        # Run demo.py on the video
        output_subdir = self.output_dir / video_path.stem
        
        cmd = [
            'python', 'demo.py',
            '--video', str(video_path),
            '--output_pth', str(output_subdir),
            '--save_pkl',
            '--estimate_local_only'  # Faster, no SLAM
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                   timeout=120)
            
            if result.returncode != 0:
                logger.error(f"❌ Processing failed: {result.stderr}")
                return None
            
            # Load results
            results_file = output_subdir / "wham_output.pkl"
            
            if results_file.exists():
                elapsed = time.time() - start_time
                logger.info(f"✅ Processed in {elapsed:.1f}s")
                return results_file
            else:
                logger.error(f"❌ No results file generated")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Processing timeout")
            return None
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return None

class VideoWatcherHandler(FileSystemEventHandler):
    """Handles new video file events"""
    
    def __init__(self, processor, watch_dir):
        self.processor = processor
        self.watch_dir = Path(watch_dir)
        self.processing = set()  # Track files being processed
    
    def on_created(self, event):
        """Called when a new file is created"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process video files
        if file_path.suffix.lower() not in ['.mp4', '.mov', '.avi', '.mkv']:
            return
        
        # Avoid processing same file twice
        if file_path in self.processing:
            return
        
        logger.info(f"📥 New video detected: {file_path.name}")
        
        # Wait for file to be completely written
        time.sleep(0.5)
        
        self.processing.add(file_path)
        
        try:
            # Process video
            results_file = self.processor.process_video(file_path)
            
            if results_file:
                # Copy results back to watch directory for client to download
                results_copy = self.watch_dir / f"{file_path.stem}_results.pkl"
                import shutil
                shutil.copy(results_file, results_copy)
                logger.info(f"📤 Results ready: {results_copy.name}")
            
            # Clean up processed video
            file_path.unlink()
            logger.info(f"🗑️  Deleted: {file_path.name}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
        
        finally:
            self.processing.discard(file_path)

def main():
    parser = argparse.ArgumentParser(
        description='Watch for videos and process with WHAM'
    )
    parser.add_argument('--watch-dir', type=str, default='incoming',
                        help='Directory to watch for incoming videos')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for processing')
    parser.add_argument('--output-dir', type=str, default='output/remote',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    watch_dir = Path(args.watch_dir)
    watch_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*60)
    print("👁️  Cluster Video Watcher for WHAM")
    print("="*60)
    print(f"\n📁 Watching: {watch_dir.absolute()}")
    print(f"🖥️  Device: {device}")
    print(f"📊 Output: {args.output_dir}")
    print("\n⚙️  Initializing WHAM...")
    
    processor = VideoProcessor(args.output_dir, device)
    
    print("\n✅ Ready! Waiting for videos...")
    print("   Upload videos to this directory and they'll be processed automatically")
    print("\nPress Ctrl+C to stop\n")
    
    event_handler = VideoWatcherHandler(processor, watch_dir)
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping watcher...")
        observer.stop()
    
    observer.join()
    print("✅ Stopped")

if __name__ == '__main__':
    main()

