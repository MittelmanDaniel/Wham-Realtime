"""
Stream a video file using ffmpeg and process it in real-time
Uses HTTP streaming (simpler than RTSP, no server needed)
"""
import subprocess
import cv2
import time
import argparse
from pathlib import Path
from flask import Flask, Response
import threading
from loguru import logger


class FFmpegVideoStreamer:
    """
    Stream a video file over HTTP using ffmpeg
    """
    def __init__(self, video_path, fps=None, loop=False):
        self.video_path = Path(video_path)
        self.loop = loop
        
        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps or cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        logger.info(f"Video: {video_path}")
        logger.info(f"Resolution: {self.width}x{self.height}")
        logger.info(f"FPS: {self.fps}")
        logger.info(f"Loop: {self.loop}")
    
    def stream_generator(self):
        """
        Generate MJPEG stream using ffmpeg
        """
        while True:
            # ffmpeg command to stream video
            cmd = [
                'ffmpeg',
                '-re',  # Read at native frame rate (real-time)
                '-i', str(self.video_path),
                '-f', 'image2pipe',  # Output as image stream
                '-pix_fmt', 'bgr24',  # OpenCV format
                '-vcodec', 'rawvideo',
                '-'  # Output to stdout
            ]
            
            if not self.loop:
                logger.info("Starting stream (one-time)...")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            frame_size = self.width * self.height * 3  # BGR = 3 bytes per pixel
            
            try:
                while True:
                    # Read one frame
                    raw_frame = process.stdout.read(frame_size)
                    
                    if len(raw_frame) != frame_size:
                        break  # End of stream
                    
                    # Convert to numpy array (OpenCV format)
                    import numpy as np
                    frame = np.frombuffer(raw_frame, dtype=np.uint8)
                    frame = frame.reshape((self.height, self.width, 3))
                    
                    # Encode as JPEG for MJPEG stream
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    if not ret:
                        continue
                    
                    # Yield as multipart MJPEG
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            except Exception as e:
                logger.error(f"Stream error: {e}")
            
            finally:
                process.terminate()
                process.wait()
            
            if not self.loop:
                logger.info("Stream ended")
                break
            else:
                logger.info("Looping video...")


def start_http_stream_server(video_path, port=5000, fps=None, loop=True):
    """
    Start HTTP server that streams video
    """
    app = Flask(__name__)
    streamer = FFmpegVideoStreamer(video_path, fps=fps, loop=loop)
    
    @app.route('/video.mjpeg')
    def video_feed():
        return Response(
            streamer.stream_generator(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    
    @app.route('/')
    def index():
        return f"""
        <html>
        <head><title>WHAM Video Stream</title></head>
        <body>
            <h1>WHAM Real-time Video Stream</h1>
            <p>Video: {video_path}</p>
            <p>Resolution: {streamer.width}x{streamer.height} @ {streamer.fps} FPS</p>
            <p>Loop: {loop}</p>
            <hr>
            <img src="/video.mjpeg" width="640">
            <hr>
            <p>To capture this stream in OpenCV:</p>
            <pre>cap = cv2.VideoCapture('http://localhost:{port}/video.mjpeg')</pre>
        </body>
        </html>
        """
    
    logger.info("=" * 50)
    logger.info("FFMPEG HTTP VIDEO STREAMER")
    logger.info("=" * 50)
    logger.info(f"Stream URL: http://localhost:{port}/video.mjpeg")
    logger.info(f"Web interface: http://localhost:{port}/")
    logger.info(f"")
    logger.info(f"To process with WHAM, run:")
    logger.info(f"  python realtime_processor.py http://localhost:{port}/video.mjpeg")
    logger.info("=" * 50)
    
    app.run(host='0.0.0.0', port=port, threaded=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stream a video file over HTTP for real-time processing'
    )
    parser.add_argument('video', help='Video file to stream')
    parser.add_argument('--port', type=int, default=5000,
                        help='HTTP port (default: 5000)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Override FPS (default: use video FPS)')
    parser.add_argument('--loop', action='store_true',
                        help='Loop video continuously')
    parser.add_argument('--no-loop', dest='loop', action='store_false',
                        help='Play video once and stop')
    parser.set_defaults(loop=True)
    
    args = parser.parse_args()
    
    start_http_stream_server(
        args.video,
        port=args.port,
        fps=args.fps,
        loop=args.loop
    )
