import subprocess
import time
import os
import shutil
import logging
from datetime import datetime
from flask import Flask, request
from threading import Thread

# ================= SILENCE FLASK LOGS =================
# We only use the logger level to avoid the KeyError 
# while keeping the terminal clean.
log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)
log.setLevel(logging.INFO)

# ======================================================

# ================= CONFIGURATION =================
CAM_NAME = "Gate"
URL = "rtsp://admin:master!31416Pi@192.168.1.99:554/Streaming/channels/101"
BASE_DIR = "/media/share/cameras/cctv-storage" 
TEMP_DIR = "/tmp/cctv_staging" 
# =================================================

current_video_prefix = f"{CAM_NAME}_Init"
os.makedirs(TEMP_DIR, exist_ok=True)
app = Flask(__name__)

def get_daily_dir():
    """Returns daily folder path and ensures it exists."""
    today = datetime.now().strftime("%Y-%m-%d")
    daily_path = os.path.join(BASE_DIR, today)
    if not os.path.exists(daily_path):
        try:
            os.makedirs(daily_path, mode=0o777, exist_ok=True)
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] DIR ERROR: {e}", flush=True)
            return BASE_DIR 
    return daily_path

@app.route("/upload", methods=["POST"])
def upload_image():
    """Receives detection images from the PC."""
    global current_video_prefix
    if 'image' in request.files:
        file = request.files['image']
        target_dir = get_daily_dir()
        actual_time = datetime.now().strftime("%H-%M-%S")
        new_filename = f"{current_video_prefix}_DETECTION_{actual_time}.jpg"
        save_path = os.path.join(target_dir, new_filename)
        
        try:
            file.save(save_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] IMAGE SAVED: {new_filename}", flush=True)
            return {"status": "success"}, 200
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] IMG SAVE ERROR: {e}", flush=True)
            return {"status": "error", "message": str(e)}, 500
    return {"status": "error"}, 400

def run_flask():
    """Runs the listener for AI detection."""
    # use_reloader=False is critical for service stability
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)

def move_to_share_background(local_path, start_epoch, filename):
    """Background thread: Moves video to share and fixes timestamp."""
    try:
        final_dir = get_daily_dir()
        final_path = os.path.join(final_dir, filename)
        
        if os.path.exists(local_path):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Background Transfer Starting: {filename}", flush=True)
            shutil.move(local_path, final_path)
            
            # Update timestamp to start of recording for perfect sorting
            try:
                os.utime(final_path, (start_epoch, start_epoch))
            except:
                pass
                
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Background Transfer Finished: {filename}", flush=True)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] BACKGROUND MOVE FAILED: {e}", flush=True)

def recording_loop():
    """Main loop: Focuses strictly on recording segments."""
    global current_video_prefix
    while True:
        # Kill any runaway FFmpeg instances
        subprocess.run(["pkill", "-9", "ffmpeg"], stderr=subprocess.DEVNULL)
        
        start_epoch = time.time()
        timestamp = datetime.fromtimestamp(start_epoch).strftime("%Y-%m-%d_%H-%M-%S")
        
        current_video_prefix = f"{CAM_NAME}_{timestamp}"
        filename = f"{current_video_prefix}.mp4"
        local_path = os.path.join(TEMP_DIR, filename)

        # FFmpeg Command: Direct Stream Copy (Low CPU)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", URL,
            "-c:v", "copy", "-map", "0:v:0",
            "-t", "300", local_path
        ]

        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] RECORDING SEGMENT: {filename}", flush=True)
            subprocess.run(cmd, check=True)
            
            # Kick off the background move
            if os.path.exists(local_path):
                Thread(target=move_to_share_background, 
                       args=(local_path, start_epoch, filename),
                       daemon=True).start()
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] RECORDING ERROR: {e}", flush=True)
            time.sleep(10)
        
        cleanup_old_folders(days=7)

def cleanup_old_folders(days):
    """Cleanup old history from the share."""
    try:
        now = time.time()
        for folder_name in os.listdir(BASE_DIR):
            folder_path = os.path.join(BASE_DIR, folder_name)
            if os.path.isdir(folder_path) and len(folder_name) == 10:
                if os.stat(folder_path).st_mtime < now - (days * 86400):
                    shutil.rmtree(folder_path)
    except:
        pass

if __name__ == "__main__":
    print(">>> SERVICE STARTING...", flush=True)

    # 1. Clean up port 5000 
    try:
        subprocess.run(["fuser", "-k", "5000/tcp"], stderr=subprocess.DEVNULL)
    except:
        pass

    # 2. Start Flask as a background daemon
    print(">>> Launching Flask API Thread...", flush=True)
    server_thread = Thread(target=run_flask, daemon=True)
    server_thread.start()
    
    time.sleep(1)

    # 3. Start Main Loop
    print(">>> Entering Recording Loop...", flush=True)
    try:
        recording_loop()
    except Exception as e:
        print(f">>> CRITICAL ERROR: {e}", flush=True)
