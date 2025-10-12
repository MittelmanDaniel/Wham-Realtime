"""
Remote Camera Server - Run this on your Mac
Streams camera feed over HTTP so the cluster can access it

Usage on Mac:
    python remote_camera_server.py --port 8080

Then on cluster:
    python demo.py --video http://YOUR_MAC_IP:8080/video
"""

import cv2
import argparse
from flask import Flask, Response
import threading
import time

app = Flask(__name__)

class CameraStream:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        
    def start(self):
        """Start camera capture in background thread"""
        print(f"🎥 Opening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set camera properties for better streaming
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera opened: {width}x{height} @ {fps} FPS")
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
    def _capture_loop(self):
        """Background thread that continuously captures frames"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print("⚠️  Failed to read frame")
                time.sleep(0.1)
    
    def get_frame(self):
        """Get the latest frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("🛑 Camera stopped")

# Global camera instance
camera = None

@app.route('/')
def index():
    return """
    <html>
    <head><title>Remote Camera Server</title></head>
    <body>
        <h1>🎥 Remote Camera Server</h1>
        <p>Camera is streaming!</p>
        <h2>Preview:</h2>
        <img src="/preview" width="640">
        <h2>Usage:</h2>
        <p>On your cluster, use this URL with WHAM:</p>
        <pre>python demo.py --video http://THIS_IP:PORT/video.mjpeg</pre>
        <p>Or save a video file first and then process it.</p>
    </body>
    </html>
    """

@app.route('/preview')
def preview():
    """Show a single frame as JPEG for preview"""
    frame = camera.get_frame()
    if frame is None:
        return "No frame available", 503
    
    ret, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/video.mjpeg')
def video_mjpeg():
    """Stream video as MJPEG (Motion JPEG)"""
    def generate():
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Yield in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   buffer.tobytes() + b'\r\n')
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    parser = argparse.ArgumentParser(
        description='Stream Mac camera to cluster via HTTP'
    )
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (usually 0)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host (0.0.0.0 to allow external connections)')
    
    args = parser.parse_args()
    
    global camera
    camera = CameraStream(args.camera)
    camera.start()
    
    print("\n" + "="*60)
    print("🎥 Remote Camera Server Started!")
    print("="*60)
    print(f"\n📡 Server running on: http://{args.host}:{args.port}")
    print(f"\n🔗 Access from cluster using:")
    print(f"   http://YOUR_MAC_IP:{args.port}/video.mjpeg")
    print(f"\n💡 To find your Mac's IP:")
    print(f"   - System Settings → Network → Your Connection")
    print(f"   - Or run: ipconfig getifaddr en0")
    print("\n⚠️  Important: Make sure cluster can reach your Mac:")
    print("   - Both on same network, OR")
    print("   - VPN connected, OR")  
    print("   - Use SSH tunnel (see documentation)")
    print("\n🌐 Open http://localhost:{} in browser to test".format(args.port))
    print("\nPress Ctrl+C to stop\n")
    
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping server...")
    finally:
        camera.stop()

if __name__ == '__main__':
    main()

