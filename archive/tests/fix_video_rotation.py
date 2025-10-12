"""
Fix video rotation issues from iPhone/mobile videos
Reads rotation metadata and applies it to create a properly oriented video
"""
import cv2
import numpy as np
import argparse
from pathlib import Path
import subprocess
import json


def get_rotation_metadata(video_path):
    """
    Try to get rotation metadata from video using ffprobe
    Falls back to 0 if not available
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    # Check for rotation tag
                    rotation = stream.get('tags', {}).get('rotate', 0)
                    return int(rotation)
    except Exception as e:
        print(f"⚠️  Could not read metadata: {e}")
    return 0


def rotate_frame(frame, rotation):
    """
    Rotate frame based on rotation angle
    """
    if rotation == 0:
        return frame
    elif rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        print(f"⚠️  Unsupported rotation: {rotation}°")
        return frame


def fix_video_rotation(input_path, output_path=None, rotation=None):
    """
    Read video, apply rotation, and save
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_rotated{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    print(f"📹 Input: {input_path}")
    print(f"💾 Output: {output_path}")
    
    # Get rotation from metadata if not specified
    if rotation is None:
        rotation = get_rotation_metadata(str(input_path))
    
    print(f"🔄 Rotation: {rotation}°")
    
    if rotation == 0:
        print("✅ No rotation needed!")
        return str(input_path)
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {input_path}")
        return None
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Original size: {original_width}x{original_height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Frames: {total_frames}")
    
    # For 90/270 rotation, width and height swap
    if rotation in [90, 270]:
        output_width, output_height = original_height, original_width
    else:
        output_width, output_height = original_width, original_height
    
    print(f"   Output size: {output_width}x{output_height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        print(f"❌ Failed to create output video writer")
        cap.release()
        return None
    
    # Process frames
    print(f"\n🎬 Processing...")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Rotate frame
            rotated_frame = rotate_frame(frame, rotation)
            out.write(rotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = frame_count / total_frames * 100 if total_frames > 0 else 0
                print(f"   {frame_count}/{total_frames} frames ({progress:.1f}%)", end='\r')
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Stopped by user")
    
    finally:
        cap.release()
        out.release()
        
        print(f"\n✅ Done! Processed {frame_count} frames")
        print(f"💾 Saved to: {output_path}")
    
    return str(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fix video rotation from iPhone/mobile videos'
    )
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-o', '--output', help='Output video file (default: input_rotated.ext)')
    parser.add_argument('-r', '--rotation', type=int, choices=[0, 90, 180, 270],
                        help='Rotation angle (default: auto-detect from metadata)')
    
    args = parser.parse_args()
    
    fix_video_rotation(args.input, args.output, args.rotation)
