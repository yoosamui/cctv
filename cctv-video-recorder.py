"""
CCTV Recorder with Person Detection
====================================
Version: 1.6 - Strict Limit Enforcement
"""

import subprocess
import time
import os
import shutil
import logging
import argparse
import configparser
from datetime import datetime
from flask import Flask, request
from threading import Thread, Lock
import signal
import sys
import re
from concurrent.futures import ThreadPoolExecutor
VERSION = "1.5"

# ================= CONFIGURATION LOADING =================
config = configparser.ConfigParser()
config_file = os.path.join(os.path.dirname(__file__), 'config.ini')

def load_config():
    """Reads settings from config.ini with fallback defaults."""
    if not os.path.exists(config_file):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {config_file} not found!")
        sys.exit(1)
    config.read(config_file)

load_config()
print(" * Configuration loaded successfully.")

BASE_DIR = config.get('STORAGE', 'base_dir', fallback='/media/share/cameras/cctv-storage')
TEMP_DIR = config.get('STORAGE', 'temp_dir', fallback='/tmp/cctv_staging')
RETENTION_DAYS = config.getint('STORAGE', 'retention_days', fallback=7)
SEGMENT_DURATION = config.getint('RECORDING', 'segment_duration', fallback=300)
MAX_IMAGES_PER_SESSION = config.getint('RECORDING', 'max_images_per_session', fallback=3)
FLASK_PORT = config.getint('NETWORK', 'flask_port', fallback=5000)

# ================= GLOBAL STATE =================
CAM_NAME = None
URL = None
session_image_count = 0
last_session_prefix = ""
current_video_prefix = ""
prefix_lock = Lock()
shutdown_flag = False
shutdown_lock = Lock()
move_executor = ThreadPoolExecutor(max_workers=4)
video_start_time_secs = datetime.now().timestamp()

os.makedirs(TEMP_DIR, exist_ok=True)
app = Flask(__name__)

# Silence Flask logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def get_camera_dir():
    """Returns camera-specific folder path."""
    today = datetime.now().strftime("%Y-%m-%d")
    camera_path = os.path.join(BASE_DIR, today, CAM_NAME)
    os.makedirs(camera_path, mode=0o755, exist_ok=True)
    return camera_path

@app.route("/upload", methods=["POST"])
def upload_image():
    """Receives detection images and enforces per-session limits."""
    global session_image_count, last_session_prefix
    
    with prefix_lock:
        if current_video_prefix == "":
            return {"status": "error", "message": "Recording not started yet"}, 503
            
        current_prefix = current_video_prefix
        
        # New session detection
        if current_prefix != last_session_prefix:
            last_session_prefix = current_prefix
            session_image_count = 0

        # ENFORCE LIMIT: Send error if threshold reached
        if session_image_count >= MAX_IMAGES_PER_SESSION:
            print(f"Limit reached ({MAX_IMAGES_PER_SESSION}). No more images accepted.")
            #start_epoch = time.time()
           # video_start_time_minus_segment_duration = datetime.fromtimestamp(start_epoch - SEGMENT_DURATION).strftime("%H-%M-%S")
            now = datetime.now().timestamp()
            duration_secs = abs(int(0 - (now - video_start_time_secs)))
            if duration_secs > 300:
                duration_secs = 200

            print(f"duration_secs = {duration_secs}")
            return {
                "status": "error","duration_secs": duration_secs,
                "message": f"Limit reached ({MAX_IMAGES_PER_SESSION}). No more images accepted."
            }, 429 
        
        session_image_count += 1

    if 'image' in request.files:
        file = request.files['image']
        target_dir = get_camera_dir()
        actual_time = datetime.now().strftime("%H-%M-%S-%f")[:-3]
        
        new_filename = f"{current_prefix}_DETECTION_{actual_time}.jpg"
        save_path = os.path.join(target_dir, new_filename)
        
        try:
            file.save(save_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚡IMAGE SAVED ({session_image_count}/{MAX_IMAGES_PER_SESSION}): {new_filename}", flush=True)
            return {"status": "success", "path": new_filename}, 200
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500

    return {"status": "error", "message": "No image part"}, 400

def run_flask():
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False, threaded=True)

def move_to_share_background(local_path, start_epoch, filename):
    try:
        final_dir = get_camera_dir()
        final_path = os.path.join(final_dir, filename)
        if os.path.exists(local_path):
            shutil.move(local_path, final_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] File moved to share: {filename}", flush=True)
    except Exception as e:
        print(f"BACKGROUND MOVE FAILED: {e}", flush=True)

def kill_ffmpeg():
    if CAM_NAME:
        subprocess.run(["pkill", "-9", "-f", f"ffmpeg.*{CAM_NAME}"], stderr=subprocess.DEVNULL)

def recording_loop():
    global current_video_prefix, shutdown_flag
    last_cleanup_segment = -1
    
    while not shutdown_flag:
        kill_ffmpeg()
        start_epoch = time.time()
        timestamp = datetime.fromtimestamp(start_epoch).strftime("%Y-%m-%d_%H-%M-%S")
        video_start_time_secs = datetime.now().timestamp()


        with prefix_lock:
            current_video_prefix = f"{timestamp}_{CAM_NAME}"
        
        filename = f"{current_video_prefix}.mp4"
        local_path = os.path.join(TEMP_DIR, filename)
        quoted_url = URL.replace('"', '\\"')
        
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-stimeout", "5000000",
            "-i", quoted_url, "-c:v", "copy", "-map", "0:v:0",
            "-t", str(SEGMENT_DURATION), "-reset_timestamps", "1",
            local_path
        ]
        
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] RECORDING: {filename}", flush=True)
            subprocess.run(cmd, check=True, timeout=SEGMENT_DURATION + 30)
            
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                move_executor.submit(move_to_share_background, local_path, start_epoch, filename)
        except Exception as e:
            print(f"RECORDING ERROR: {e}", flush=True)
            time.sleep(5)
        
        # Periodic cleanup
        current_segment = int(start_epoch / SEGMENT_DURATION)
        if current_segment != last_cleanup_segment and current_segment % 10 == 0:
            last_cleanup_segment = current_segment

def signal_handler(signum, frame):
    global shutdown_flag
    shutdown_flag = True
    kill_ffmpeg()
    move_executor.shutdown(wait=False)
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--url', '-u', type=str, required=True)
    args = parser.parse_args()
    CAM_NAME, URL = args.name, args.url
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server_thread = Thread(target=run_flask, daemon=True)
    server_thread.start()
    
    recording_loop()
