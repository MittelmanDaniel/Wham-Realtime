"""
Mac-specific camera test with better initialization
"""
import cv2
import time
import argparse

def test_camera_mac(camera_id=0, duration=10):
    """
    Test camera capture on Mac with proper initialization
    """
    print(f"🎥 Opening camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
        print("\nTroubleshooting:")
        print("  1. System Settings → Privacy & Security → Camera")
        print("     Make sure Terminal has camera permission")
        print("  2. Try different camera_id: 0, 1, 2")
        print("  3. Close other apps using the camera (Zoom, FaceTime, etc.)")
        return
    
    # Set camera properties (some might not work on Mac)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # IMPORTANT: Give Mac camera time to warm up
    print("⏳ Warming up camera (this takes a few seconds on Mac)...")
    time.sleep(2)
    
    # Discard first few frames (often black/bad on Mac)
    for _ in range(5):
        cap.read()
    
    # Now get actual properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera ready!")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS setting: {fps}")
    print(f"\n📸 Capturing for {duration} seconds...\n")
    
    frame_count = 0
    start_time = time.time()
    last_print = start_time
    successful_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if ret:
                frame_count += 1
                successful_frames += 1
                
                # Save first successful frame as test
                if successful_frames == 1:
                    cv2.imwrite('test_frame.jpg', frame)
                    print(f"✅ First frame captured! Saved as test_frame.jpg")
                    print(f"   Frame shape: {frame.shape}")
                    print("")
            else:
                frame_count += 1
                # Don't print every failed frame, just count them
            
            # Print stats every second
            current_time = time.time()
            if current_time - last_print >= 1.0:
                actual_fps = successful_frames / (current_time - start_time)
                print(f"[{current_time - start_time:.1f}s] "
                      f"{successful_frames}/{frame_count} frames "
                      f"({actual_fps:.1f} FPS)")
                last_print = current_time
            
            # Stop after duration
            if current_time - start_time >= duration:
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    
    finally:
        cap.release()
        
        total_time = time.time() - start_time
        avg_fps = successful_frames / total_time if total_time > 0 else 0
        success_rate = successful_frames / frame_count * 100 if frame_count > 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"   Total frames read: {frame_count}")
        print(f"   Successful frames: {successful_frames} ({success_rate:.1f}%)")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        
        if successful_frames > 0:
            print(f"\n✅ Camera works! Check test_frame.jpg")
        else:
            print(f"\n❌ No frames captured successfully")
            print("   Try a different camera ID or check permissions")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, default='0',
                        help='Camera ID (0, 1, ...) or stream URL')
    parser.add_argument('--duration', type=int, default=10,
                        help='Test duration in seconds')
    args = parser.parse_args()
    
    try:
        camera_id = int(args.camera)
    except:
        camera_id = args.camera
    
    test_camera_mac(camera_id, args.duration)

