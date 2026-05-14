"""
CCTV Recorder with Person Detection
====================================
Version: 1.8.2 - Simplified Reset with Detector Deduplication
"""

import subprocess
import time
import os
import shutil
import logging
import argparse
import configparser
import requests
from datetime import datetime
from flask import Flask, request
from threading import Thread, Lock
import signal
import sys
import re
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

VERSION = "1.8.2"

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
MAX_IMAGES_PER_SESSION = config.getint('RECORDING', 'max_images_per_session', fallback=6)
FLASK_PORT = config.getint('NETWORK', 'flask_port', fallback=5000)

# ==========================================
# AUTHENTICATION
# ==========================================
credentials_file = "/etc/cctv/credentials.env"
WEBHOOK_SECRET = None

if os.path.exists(credentials_file):
    try:
        load_dotenv(credentials_file)
        WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
        if not WEBHOOK_SECRET:
            print(f"WARNING: WEBHOOK_SECRET not found in {credentials_file}")
            print("Authentication to detector will fail!")
        else:
            print(f" * Webhook secret loaded successfully")
    except Exception as e:
        print(f"ERROR loading credentials: {e}")
else:
    print(f"WARNING: Credentials file not found: {credentials_file}")
    print("Please run: sudo mkdir -p /etc/cctv && sudo tee /etc/cctv/credentials.env <<< 'WEBHOOK_SECRET=\"your-secret\"'")
    print("Authentication to detector will fail!")

# Detector webhook URL
DETECTOR_WEBHOOK_URL = "http://192.168.1.103:5001/session-reset"

# ================= GLOBAL STATE =================
CAM_NAME = None
URL = None
session_image_count = 0
last_session_prefix = ""
last_detection_id = None  # Track unique detection session ID
current_video_prefix = ""
prefix_lock = Lock()
shutdown_flag = False
shutdown_lock = Lock()
move_executor = ThreadPoolExecutor(max_workers=4)
video_start_time_secs = datetime.now().timestamp()

# Create necessary directories
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

# ================= AI ANALYZER TRIGGER =================
def send_to_analyzer(file_path):
    """Send reset signal to detector when recording is complete"""
    if not WEBHOOK_SECRET:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Cannot send reset: No WEBHOOK_SECRET configured")
        return False
    
    camera_name = CAM_NAME
    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': WEBHOOK_SECRET
    }
    payload = {"camera": camera_name}
    
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                DETECTOR_WEBHOOK_URL, 
                json=payload, 
                headers=headers, 
                timeout=3
            )
            
            if response.status_code == 200:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {camera_name} session reset confirmed")
                return True
            elif response.status_code == 401:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ AUTH ERROR: API key mismatch for {camera_name}")
                return False
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Attempt {attempt+1}/{max_retries}: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Attempt {attempt+1}/{max_retries}: Cannot connect to detector at {DETECTOR_WEBHOOK_URL}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Attempt {attempt+1}/{max_retries}: Timeout connecting to detector")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Failed to send reset after {max_retries} attempts")
    return False

# ================= FLASK ENDPOINTS =================
@app.route("/upload", methods=["POST"])
def upload_image():
    """Receives detection images with detection_id for session tracking."""
    global session_image_count, last_session_prefix, last_detection_id
    
    # Get detection session info from form data
    detection_id = request.form.get('detection_id', None)
    frame_num = int(request.form.get('frame_num', 0))
    total_frames = int(request.form.get('total_frames', 6))
    camera_name = request.form.get('camera', CAM_NAME)
    
    with prefix_lock:
        if current_video_prefix == "":
            return {"status": "error", "message": "Recording not started yet"}, 503
        
        # New detection session? Reset counter
        if detection_id and detection_id != last_detection_id:
            last_detection_id = detection_id
            session_image_count = 0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🆔 New detection session: {detection_id} for {camera_name}")
        
        # Check if we've reached the limit for this session
        if session_image_count >= MAX_IMAGES_PER_SESSION:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ REJECTED: {camera_name} - limit reached ({MAX_IMAGES_PER_SESSION})")
            return {
                "status": "error",
                "message": f"Limit reached ({MAX_IMAGES_PER_SESSION}).",
                "session_count": session_image_count
            }, 429
        
        # Increment counter for this session
        session_image_count += 1
        current_count = session_image_count
    
    # Process the image
    if 'image' not in request.files:
        return {"status": "error", "message": "No image part"}, 400
    
    file = request.files['image']
    target_dir = get_camera_dir()
    actual_time = datetime.now().strftime("%H-%M-%S-%f")[:-3]
    
    # Include detection_id in filename for easier debugging
    detection_tag = f"_{detection_id}" if detection_id else ""
    new_filename = f"{current_video_prefix}_DETECTION{detection_tag}_{actual_time}.jpg"
    save_path = os.path.join(target_dir, new_filename)
    
    try:
        file.save(save_path)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📸 {camera_name}: IMAGE {current_count}/{total_frames} (ID: {detection_id})", flush=True)
        
        # If this is the last frame, log completion
        if current_count == total_frames:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {camera_name}: Detection session {detection_id} complete ({total_frames}/{total_frames})")
        
        return {
            "status": "success", 
            "path": new_filename,
            "frame_num": current_count,
            "total_frames": total_frames,
            "detection_id": detection_id
        }, 200
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Save failed: {e}")
        return {"status": "error", "message": str(e)}, 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    with prefix_lock:
        return {
            "status": "running",
            "camera": CAM_NAME,
            "version": VERSION,
            "session_count": session_image_count,
            "max_images": MAX_IMAGES_PER_SESSION,
            "current_video": current_video_prefix,
            "last_detection_id": last_detection_id,
            "webhook_configured": WEBHOOK_SECRET is not None
        }, 200

@app.route("/reset", methods=["POST"])
def reset_session():
    """Manually reset the session counter (for testing/debugging)"""
    global session_image_count, last_detection_id
    
    with prefix_lock:
        session_image_count = 0
        last_detection_id = None
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Manual reset of session counter for {CAM_NAME}")
        return {"status": "success", "message": "Session reset"}, 200

@app.route("/status", methods=["GET"])
def status():
    """Detailed status endpoint"""
    with prefix_lock:
        return {
            "camera": CAM_NAME,
            "version": VERSION,
            "recording": current_video_prefix != "",
            "current_segment": current_video_prefix,
            "session_images": session_image_count,
            "last_detection": last_detection_id,
            "max_per_session": MAX_IMAGES_PER_SESSION,
            "temp_dir": TEMP_DIR,
            "base_dir": BASE_DIR
        }, 200

def run_flask():
    """Start the Flask server in a background thread"""
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False, threaded=True)

# ================= FILE MANAGEMENT =================
def move_to_share_background(local_path, filename):
    """Move file to final location and trigger detector reset"""
    try:
        final_dir = get_camera_dir()
        final_path = os.path.join(final_dir, filename)
        if os.path.exists(local_path):
            shutil.move(local_path, final_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📁 File moved to share: {filename}", flush=True)
            
            # Send reset signal to detector
            send_to_analyzer(final_path)
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ BACKGROUND MOVE FAILED: {e}", flush=True)

def kill_ffmpeg():
    """Kill any running ffmpeg processes for this camera"""
    if CAM_NAME:
        subprocess.run(["pkill", "-9", "-f", f"ffmpeg.*{CAM_NAME}"], stderr=subprocess.DEVNULL)

# ================= RECORDING LOOP =================
def recording_loop():
    """Main recording loop - captures video segments"""
    global current_video_prefix, shutdown_flag, video_start_time_secs
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 RECORDING LOOP STARTED for {CAM_NAME}", flush=True)
    
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
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎥 RECORDING: {filename}", flush=True)
            subprocess.run(cmd, check=True, timeout=SEGMENT_DURATION + 30)
            
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                move_executor.submit(move_to_share_background, local_path, filename)
                
                # ✅ Send reset to detector after successful recording (NO COOLDOWN CHECK)
                # Detector will handle duplicate resets with its own deduplication
                try:
                    detector_reset_url = "http://192.168.1.103:5001/session-reset"
                    reset_payload = {
                        "camera": CAM_NAME,
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "action": "reset"
                    }
                    
                    headers = {
                        'Content-Type': 'application/json',
                        'X-API-KEY': WEBHOOK_SECRET
                    }
                    
                    response = requests.post(
                        detector_reset_url, 
                        json=reset_payload, 
                        headers=headers,
                        timeout=2
                    )
                    
                    if response.status_code == 200:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 Reset sent to detector for {CAM_NAME}")
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Reset failed: {response.status_code}")
                        
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Failed to send reset: {e}")
                    
        except subprocess.TimeoutExpired:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏰ Recording timeout for {CAM_NAME}")
        except subprocess.CalledProcessError as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ FFmpeg failed with code {e.returncode}: {e}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ RECORDING ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            time.sleep(5)

# ================= SIGNAL HANDLING =================
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🛑 Shutting down recorder for {CAM_NAME}...")
    shutdown_flag = True
    kill_ffmpeg()
    move_executor.shutdown(wait=False)
    sys.exit(0)

# ================= MAIN =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CCTV Recorder with Person Detection')
    parser.add_argument('--name', '-n', type=str, required=True, help='Camera name (e.g., Gate, Center)')
    parser.add_argument('--url', '-u', type=str, required=True, help='RTSP URL of the camera')
    args = parser.parse_args()
    CAM_NAME, URL = args.name, args.url
    
    print("=" * 60)
    print(f"CCTV RECORDER v{VERSION}")
    print(f"Camera: {CAM_NAME}")
    print(f"Detector URL: {DETECTOR_WEBHOOK_URL}")
    print(f"Flask port: {FLASK_PORT}")
    print(f"Max images per session: {MAX_IMAGES_PER_SESSION}")
    print(f"Segment duration: {SEGMENT_DURATION}s")
    print(f"Webhook configured: {WEBHOOK_SECRET is not None}")
    print("=" * 60)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start Flask server in background
    server_thread = Thread(target=run_flask, daemon=True)
    server_thread.start()
    
    # Give Flask time to start
    time.sleep(2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Recorder ready for {CAM_NAME}")
    
    # Start recording loop (blocks until shutdown)
    recording_loop()
