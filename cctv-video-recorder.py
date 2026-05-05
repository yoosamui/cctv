"""
CCTV Recorder with Person Detection
====================================

Usage Examples:
---------------
Gate camera:
    python cctv_recorder.py --name Gate --url rtsp://admin:master!31416Pi@192.168.1.99:554/Streaming/channels/101

Center camera:
    python cctv_recorder.py --name Center --url rtsp://admin:master!31416Pi@192.168.1.100:554/Streaming/channels/102

Backyard camera with different credentials:
    python cctv_recorder.py --name Backyard --url rtsp://admin:password123@192.168.1.50:554/Streaming/channels/101

Using short form parameters:
    python cctv_recorder.py -n Gate -u rtsp://admin:pass@192.168.1.99:554/Streaming/channels/101

Show help:
    python cctv_recorder.py --help
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
# =================================================

# Global tracker with thread safety
current_video_prefix = ""
prefix_lock = Lock()
shutdown_flag = False

os.makedirs(TEMP_DIR, exist_ok=True)
app = Flask(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CCTV Recorder with Person Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python cctv_recorder.py --name Gate --url "rtsp://admin:master!31416Pi@192.168.1.99:554/Streaming/channels/101"
  python cctv_recorder.py --name Center --url "rtsp://admin:master!31416Pi@192.168.1.100:554/Streaming/channels/102"
  python cctv_recorder.py -n Backyard -u "rtsp://admin:pass@192.168.1.50:554/Streaming/channels/101"
        '''
    )
    
    parser.add_argument(
        '--name', '-n',
        type=str,
        required=True,
        help='Camera name (e.g., Gate, Center, Backyard)'
    )
    
    parser.add_argument(
        '--url', '-u',
        type=str,
        required=True,
        help='Full RTSP URL (e.g., rtsp://username:password@ip:port/Streaming/channels/101)'
    )
    
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
    """Cleanup old history from the share with camera-aware structure."""
    try:
        now = time.time()
        cutoff_time = now - (days * 86400)
        
        for date_folder in os.listdir(BASE_DIR):
            date_path = os.path.join(BASE_DIR, date_folder)
            
            # Check if it's a date folder (YYYY-MM-DD format)
            if os.path.isdir(date_path) and len(date_folder) == 10 and date_folder[4] == '-' and date_folder[7] == '-':
                # Check camera subfolders
                for camera_folder in os.listdir(date_path):
                    camera_path = os.path.join(date_path, camera_folder)
                    if os.path.isdir(camera_path):
                        # Delete if folder is older than retention period
                        if os.path.getmtime(camera_path) < cutoff_time:
                            shutil.rmtree(camera_path)
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] CLEANUP: Removed {camera_path}", flush=True)
                
                # Remove empty date folder
                if os.path.exists(date_path) and not os.listdir(date_path):
                    os.rmdir(date_path)
                    
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] CLEANUP ERROR: {e}", flush=True)

@app.route("/upload", methods=["POST"])
def upload_image():
    """Receives detection images from the PC."""
    with prefix_lock:
        current_prefix = current_video_prefix
    
    if 'image' in request.files:
        file = request.files['image']
        target_dir = get_camera_dir()  # Now saves to Camera folder
        actual_time = datetime.now().strftime("%H-%M-%S")
        
        # Format: 2026-05-05_23-43-17_Gate_DETECTION_23-43-50.jpg
        new_filename = f"{current_prefix}_DETECTION_{actual_time}.jpg"
        save_path = os.path.join(target_dir, new_filename)
        
        try:
            file.save(save_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] IMAGE SAVED: {save_path}", flush=True)
            return {"status": "success", "path": new_filename}, 200
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] IMG SAVE ERROR: {e}", flush=True)
            return {"status": "error", "message": str(e)}, 500
    return {"status": "error"}, 400

def run_flask():
    """Run Flask server with better error handling."""
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] FLASK ERROR: {e}", flush=True)

def move_to_share_background(local_path, start_epoch, filename):
    """Background thread: Moves video to share and fixes timestamp."""
    try:
        final_dir = get_camera_dir()  # Saves to camera subfolder
        final_path = os.path.join(final_dir, filename)
        
        if os.path.exists(local_path):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Background Transfer Starting: {filename}", flush=True)
            shutil.move(local_path, final_path)
            
            # Set file timestamp to match recording start time
            try:
                os.utime(final_path, (start_epoch, start_epoch))
            except:
                pass
                
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Background Transfer Finished: {final_path}", flush=True)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] BACKGROUND MOVE FAILED: {e}", flush=True)

def kill_ffmpeg():
    """Safely kill all ffmpeg processes."""
    try:
        subprocess.run(["pkill", "-9", "-f", "ffmpeg.*rtsp"], stderr=subprocess.DEVNULL)
        time.sleep(1)  # Give time for processes to die
    except:
        pass

def recording_loop():
    """Main loop: Focuses strictly on recording segments."""
    global current_video_prefix, shutdown_flag
    consecutive_errors = 0
    max_consecutive_errors = 5
    base_sleep = 10
    
    while not shutdown_flag:
        # Kill any lingering ffmpeg processes
        kill_ffmpeg()
        
        start_epoch = time.time()
        timestamp = datetime.fromtimestamp(start_epoch).strftime("%Y-%m-%d_%H-%M-%S")
        
        # Update current video prefix with thread safety
        with prefix_lock:
            current_video_prefix = f"{timestamp}_{CAM_NAME}"
            current_prefix = current_video_prefix
        
        filename = f"{current_prefix}.mp4"
        local_path = os.path.join(TEMP_DIR, filename)
        
        # Build ffmpeg command with better parameters
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-stimeout", "5000000",  # 5 second timeout for RTSP
            "-i", URL,
            "-c:v", "copy", 
            "-map", "0:v:0",
            "-t", str(SEGMENT_DURATION),
            "-reset_timestamps", "1",  # Reset timestamps for cleaner files
            local_path
        ]
        
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] RECORDING SEGMENT: {filename}", flush=True)
            result = subprocess.run(cmd, check=True, timeout=SEGMENT_DURATION + 30)
            
            # Check if file was created and has content
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                Thread(target=move_to_share_background, 
                       args=(local_path, start_epoch, filename),
                       daemon=True).start()
                consecutive_errors = 0  # Reset error counter on success
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Empty or missing file: {filename}", flush=True)
                consecutive_errors += 1
                
        except subprocess.TimeoutExpired:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] RECORDING TIMEOUT for {filename}", flush=True)
            consecutive_errors += 1
        except subprocess.CalledProcessError as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] RECORDING ERROR: {e}", flush=True)
            consecutive_errors += 1
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] UNEXPECTED ERROR: {e}", flush=True)
            consecutive_errors += 1
        
        # Exponential backoff on repeated errors
        if consecutive_errors > 0:
            sleep_time = min(base_sleep * consecutive_errors, 60)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error {consecutive_errors}/{max_consecutive_errors}. Retrying in {sleep_time}s", flush=True)
            time.sleep(sleep_time)
        
        # Periodic cleanup (every 10 segments)
        if int(start_epoch) % (SEGMENT_DURATION * 10) < SEGMENT_DURATION:
            cleanup_old_folders(RETENTION_DAYS)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_flag
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Received signal {signum}. Shutting down...", flush=True)
    shutdown_flag = True
    kill_ffmpeg()
    sys.exit(0)

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set global configuration from arguments
    CAM_NAME = args.name
    URL = args.url
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60, flush=True)
    print(f"CCTV RECORDER SERVICE - {CAM_NAME}", flush=True)
    print("=" * 60, flush=True)
    print(f"RTSP URL: {URL}", flush=True)
    print(f"Storage Path: {BASE_DIR}", flush=True)
    print(f"Camera Subfolder: {CAM_NAME}", flush=True)
    print(f"Segment Duration: {SEGMENT_DURATION}s", flush=True)
    print(f"Retention: {RETENTION_DAYS} days", flush=True)
    print("=" * 60, flush=True)
    
    # Test directory creation
    test_dir = get_camera_dir()
    print(f"✓ Storage ready: {test_dir}", flush=True)
    
    # Kill any process using port 5000
    try:
        subprocess.run(["fuser", "-k", "5000/tcp"], stderr=subprocess.DEVNULL)
        time.sleep(1)
    except:
        pass
    
    print(">>> Launching Flask API Thread...", flush=True)
    server_thread = Thread(target=run_flask, daemon=True)
    server_thread.start()
    
    time.sleep(2)
    print(">>> Entering Recording Loop...", flush=True)
    print("Press Ctrl+C to stop", flush=True)
    print("-" * 60, flush=True)
    
    try:
        recording_loop()
    except KeyboardInterrupt:
        print("\n>>> Stopped by user", flush=True)
    except Exception as e:
        print(f">>> CRITICAL ERROR: {e}", flush=True)
    finally:
        print(">>> Cleaning up...", flush=True)
        kill_ffmpeg()
        print(">>> Service stopped", flush=True)
