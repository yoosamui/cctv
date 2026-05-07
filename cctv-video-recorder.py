"""
CCTV Recorder with Person Detection
====================================

Usage Examples:
---------------
Gate camera:
    python cctv_recorder.py --name Gate --url rtsp://<USER>:<PASSWORD>@192.168.1.99:554/Streaming/channels/101

Center camera:
    python cctv_recorder.py --name Center --url rtsp://<USER>:<PASSWORD>@192.168.1.100:554/Streaming/channels/102

Using short form parameters:
    python cctv_recorder.py -n Gate -u rtsp://<USER>:<PASSWORD>@192.168.1.99:554/Streaming/channels/101
"""

import subprocess
import time
import os
import shutil
import logging
import argparse
from datetime import datetime
from flask import Flask, request
from threading import Thread, Lock
import signal
import sys

# ================= SILENCE FLASK LOGS =================
log = logging.getLogger('werkzeug')
log.setLevel(logging.INFO)
# ======================================================

# ================= GLOBAL VARIABLES =================
CAM_NAME = None
URL = None
BASE_DIR = "/media/share/cameras/cctv-storage" 
TEMP_DIR = "/tmp/cctv_staging"
RETENTION_DAYS = 7
SEGMENT_DURATION = 300  # 5 minutes

# Session Control Variables
MAX_IMAGES_PER_SESSION = 3
session_image_count = 0
last_session_prefix = ""

# Global tracker with thread safety
current_video_prefix = ""
prefix_lock = Lock()
shutdown_flag = False
# =================================================

os.makedirs(TEMP_DIR, exist_ok=True)
app = Flask(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CCTV Recorder with Person Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--name', '-n', type=str, required=True, help='Camera name')
    parser.add_argument('--url', '-u', type=str, required=True, help='Full RTSP URL')
    
    return parser.parse_args()

def get_camera_dir():
    """Returns camera-specific folder path: BASE_DIR/YYYY-MM-DD/CAM_NAME/"""
    today = datetime.now().strftime("%Y-%m-%d")
    camera_path = os.path.join(BASE_DIR, today, CAM_NAME)
    try:
        os.makedirs(camera_path, mode=0o777, exist_ok=True)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] DIR ERROR: {e}", flush=True)
        return BASE_DIR
    return camera_path

def cleanup_old_folders(days):
    """Cleanup old history from the share."""
    try:
        now = time.time()
        cutoff_time = now - (days * 86400)
        
        for date_folder in os.listdir(BASE_DIR):
            date_path = os.path.join(BASE_DIR, date_folder)
            if os.path.isdir(date_path) and len(date_folder) == 10:
                for camera_folder in os.listdir(date_path):
                    camera_path = os.path.join(date_path, camera_folder)
                    if os.path.isdir(camera_path):
                        if os.path.getmtime(camera_path) < cutoff_time:
                            shutil.rmtree(camera_path)
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] CLEANUP: Removed {camera_path}", flush=True)
                
                if os.path.exists(date_path) and not os.listdir(date_path):
                    os.rmdir(date_path)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] CLEANUP ERROR: {e}", flush=True)

@app.route("/upload", methods=["POST"])
def upload_image():
    """Receives detection images and enforces per-session limits."""
    global session_image_count, last_session_prefix
    
    with prefix_lock:
        current_prefix = current_video_prefix
        
        # If the video file name has changed, it's a new "Session"
        if current_prefix != last_session_prefix:
            last_session_prefix = current_prefix
            session_image_count = 0
            
        # Check if we have already saved the max images for this segment
        if session_image_count >= MAX_IMAGES_PER_SESSION:
            return {"status": "ignored", "message": "Limit reached for this session"}, 200

    if 'image' in request.files:
        file = request.files['image']
        target_dir = get_camera_dir()
        actual_time = datetime.now().strftime("%H-%M-%S")
        
        new_filename = f"{current_prefix}_DETECTION_{actual_time}.jpg"
        save_path = os.path.join(target_dir, new_filename)
        
        try:
            file.save(save_path)
            
            with prefix_lock:
                session_image_count += 1
                
            print(f"[{datetime.now().strftime('%H:%M:%S')}] IMAGE SAVED ({session_image_count}/{MAX_IMAGES_PER_SESSION}): {save_path}", flush=True)
            return {"status": "success", "path": new_filename}, 200
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] IMG SAVE ERROR: {e}", flush=True)
            return {"status": "error", "message": str(e)}, 500
    return {"status": "error", "message": "No image part"}, 400

def run_flask():
    """Run Flask server."""
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] FLASK ERROR: {e}", flush=True)

def move_to_share_background(local_path, start_epoch, filename):
    """Background thread: Moves video to share."""
    try:
        final_dir = get_camera_dir()
        final_path = os.path.join(final_dir, filename)
        
        if os.path.exists(local_path):
            shutil.move(local_path, final_path)
            try:
                os.utime(final_path, (start_epoch, start_epoch))
            except:
                pass
            print(f"[{datetime.now().strftime('%H:%M:%S')}] File moved to share: {filename}", flush=True)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] BACKGROUND MOVE FAILED: {e}", flush=True)

def kill_ffmpeg():
    """Kill ffmpeg processes."""
    try:
        subprocess.run(["pkill", "-9", "-f", "ffmpeg.*rtsp"], stderr=subprocess.DEVNULL)
        time.sleep(1)
    except:
        pass

def recording_loop():
    """Main recording loop."""
    global current_video_prefix, shutdown_flag
    consecutive_errors = 0
    
    while not shutdown_flag:
        kill_ffmpeg()
        start_epoch = time.time()
        timestamp = datetime.fromtimestamp(start_epoch).strftime("%Y-%m-%d_%H-%M-%S")
        
        with prefix_lock:
            current_video_prefix = f"{timestamp}_{CAM_NAME}"
            current_prefix = current_video_prefix
        
        filename = f"{current_prefix}.mp4"
        local_path = os.path.join(TEMP_DIR, filename)
        
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-stimeout", "5000000",
            "-i", URL, "-c:v", "copy", "-map", "0:v:0",
            "-t", str(SEGMENT_DURATION), "-reset_timestamps", "1",
            local_path
        ]
        
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] RECORDING: {filename}", flush=True)
            subprocess.run(cmd, check=True, timeout=SEGMENT_DURATION + 30)
            
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                Thread(target=move_to_share_background, 
                       args=(local_path, start_epoch, filename),
                       daemon=True).start()
                consecutive_errors = 0
            else:
                consecutive_errors += 1
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] RECORDING ERROR: {e}", flush=True)
            consecutive_errors += 1
            time.sleep(min(10 * consecutive_errors, 60))
        
        if int(start_epoch) % (SEGMENT_DURATION * 10) < SEGMENT_DURATION:
            cleanup_old_folders(RETENTION_DAYS)

def signal_handler(signum, frame):
    global shutdown_flag
    shutdown_flag = True
    kill_ffmpeg()
    sys.exit(0)

if __name__ == "__main__":
    args = parse_arguments()
    CAM_NAME = args.name
    URL = args.url
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        subprocess.run(["fuser", "-k", "5000/tcp"], stderr=subprocess.DEVNULL)
        time.sleep(1)
    except:
        pass
    
    print(f"Starting CCTV Recorder for: {CAM_NAME}")
    server_thread = Thread(target=run_flask, daemon=True)
    server_thread.start()
    
    try:
        recording_loop()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
    finally:
        kill_ffmpeg()
