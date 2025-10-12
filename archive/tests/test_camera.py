"""
Simple test script to verify camera capture works
This demonstrates the basics of cv2.VideoCapture for live camera
"""
import cv2
import time
import argparse

def test_camera(camera_id=0, duration=10):
    """
    Test camera capture and display FPS
    
    Args:
        camera_id: 0 for default webcam, 1+ for other cameras, or URL for IP cam
        duration: How long to run (seconds)
    """
    print(f"🎥 Opening camera {camera_id}...")
    
    # Open camera - THIS IS THE KEY LINE!
    # Works with: 0 (webcam), "rtsp://...", "http://...", etc.
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
        print("Troubleshooting:")
        print("  - Try camera_id=0, 1, 2, ...")
        print("  - Check if camera is already in use")
        print("  - For IP camera, use full URL: 'rtsp://user:pass@ip:port/stream'")
        return
    
    # Get camera info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    print(f"✅ Camera opened successfully!")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"\nCapturing for {duration} seconds...")
    print("(This is what WHAM would receive as input)\n")
    
    frame_count = 0
    start_time = time.time()
    last_print = start_time
    
    try:
        while True:
            # Read a frame - THIS IS WHAT WHAM DOES IN THE LOOP
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Print stats every second
            current_time = time.time()
            if current_time - last_print >= 1.0:
                actual_fps = frame_count / (current_time - start_time)
                elapsed = current_time - start_time
                print(f"[{elapsed:.1f}s] Captured {frame_count} frames @ {actual_fps:.1f} FPS")
                last_print = current_time
            
            # Optional: Display the frame (requires X11/display)
            # cv2.imshow('Camera Feed', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            # Stop after duration
            if current_time - start_time >= duration:
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    
    finally:
        cap.release()
        # cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"   Total frames: {frame_count}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Frame shape: {frame.shape if frame is not None else 'N/A'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, default='0',
                        help='Camera ID (0, 1, ...) or stream URL')
    parser.add_argument('--duration', type=int, default=10,
                        help='Test duration in seconds')
    args = parser.parse_args()
    
    # Try to convert to int
    try:
        camera_id = int(args.camera)
    except:
        camera_id = args.camera
    
    test_camera(camera_id, args.duration)

