# Full CCTV Recorder Code v1.8.7 - Async Session Reset

"""
CCTV Recorder with Person Detection
====================================
Version: 1.8.7 - Async Session Reset (eliminates 10-second delay)
"""

import subprocess
import time
import os
import shutil
import logging
import argparse
import configparser
import requests
from datetime import datetime, timedelta
from flask import Flask, request
from threading import Thread, Lock
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

VERSION = "1.8.7"

# ================= CONFIGURATION LOADING =================
config = configparser.ConfigParser()
config_file = os.path.join(os.path.dirname(__file__), 'config.ini')


def load_config():
    """Reads settings from config.ini with fallback defaults."""

    if not os.path.exists(config_file):

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"ERROR: {config_file} not found!"
        )

        sys.exit(1)

    config.read(config_file)


load_config()

print(" * Configuration loaded successfully.")

BASE_DIR = config.get(
    'STORAGE',
    'base_dir',
    fallback='/media/share/cameras/cctv-storage'
)

TEMP_DIR = config.get(
    'STORAGE',
    'temp_dir',
    fallback='/tmp/cctv_staging'
)

RETENTION_DAYS = config.getint(
    'STORAGE',
    'retention_days',
    fallback=7
)

CLEANUP_INTERVAL_HOURS = config.getint(
    'STORAGE',
    'cleanup_interval_hours',
    fallback=6
)

SEGMENT_DURATION = config.getint(
    'RECORDING',
    'segment_duration',
    fallback=300
)

MAX_IMAGES_PER_SESSION = config.getint(
    'RECORDING',
    'max_images_per_session',
    fallback=6
)

FLASK_PORT = config.getint(
    'NETWORK',
    'flask_port',
    fallback=5000
)

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

            print(
                f"WARNING: WEBHOOK_SECRET not found in "
                f"{credentials_file}"
            )

            print("Authentication to detector will fail!")

        else:
            print(" * Webhook secret loaded successfully")

    except Exception as e:

        print(f"ERROR loading credentials: {e}")

else:

    print(f"WARNING: Credentials file not found: {credentials_file}")

    print(
        "Please run: sudo mkdir -p /etc/cctv && "
        "sudo tee /etc/cctv/credentials.env <<< "
        "'WEBHOOK_SECRET=\"your-secret\"'"
    )

    print("Authentication to detector will fail!")

# Detector webhook URL
DETECTOR_WEBHOOK_URL = "http://192.168.1.103:5001/session-reset"

# ================= GLOBAL STATE =================
CAM_NAME = None
URL = None

session_image_count = 0
last_detection_id = None
current_video_prefix = ""
last_detection_time = 0

prefix_lock = Lock()

shutdown_flag = False

SESSION_STALE_TIMEOUT = 300

move_executor = ThreadPoolExecutor(max_workers=4)

ffmpeg_process = None

# Create necessary directories
os.makedirs(TEMP_DIR, exist_ok=True)

app = Flask(__name__)

# upload limit
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Upload authentication
UPLOAD_API_KEY = WEBHOOK_SECRET

# Silence Flask logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def get_camera_dir():
    """Returns camera-specific folder path."""

    today = datetime.now().strftime("%Y-%m-%d")

    camera_path = os.path.join(
        BASE_DIR,
        today,
        CAM_NAME
    )

    os.makedirs(camera_path, mode=0o755, exist_ok=True)

    return camera_path


# ================= DETECTOR RESET =================
def send_reset_to_detector():
    """Send reset signal to detector."""

    if not WEBHOOK_SECRET:

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"❌ Cannot send reset: No WEBHOOK_SECRET configured"
        )

        return False

    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': WEBHOOK_SECRET
    }

    payload = {
        "camera": CAM_NAME
    }

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

                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"✅ {CAM_NAME} session reset confirmed"
                )

                return True

            elif response.status_code == 401:

                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"❌ AUTH ERROR: API key mismatch for {CAM_NAME}"
                )

                return False

            else:

                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"⚠️ Attempt {attempt+1}/{max_retries}: "
                    f"HTTP {response.status_code}"
                )

        except requests.exceptions.ConnectionError:

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"⚠️ Attempt {attempt+1}/{max_retries}: "
                f"Cannot connect to detector"
            )

        except requests.exceptions.Timeout:

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"⚠️ Attempt {attempt+1}/{max_retries}: "
                f"Timeout connecting to detector"
            )

        except Exception as e:

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"❌ Reset error: {e}"
            )

        if attempt < max_retries - 1:
            time.sleep(2)

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"❌ Failed to send reset after {max_retries} attempts"
    )

    return False


# ================= ASYNC SESSION RESET (NEW) =================
def async_session_reset():
    """Background thread to reset session state asynchronously."""
    global session_image_count
    global last_detection_id
    global last_detection_time
    
    # Brief delay to allow any in-flight uploads to complete
    time.sleep(0.5)
    
    with prefix_lock:
        old_count = session_image_count
        old_id = last_detection_id
        
        session_image_count = 0
        last_detection_id = None
        last_detection_time = time.time()
        
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"🔄 Async session reset for {CAM_NAME}: "
            f"cleared {old_count} images (ID: {old_id})"
        )
    
    # Now send reset signal to detector (outside the lock)
    send_reset_to_detector()


def queue_async_reset():
    """Queue a session reset to happen in background."""
    reset_thread = Thread(
        target=async_session_reset,
        daemon=True
    )
    reset_thread.start()
    return reset_thread


# ================= FLASK ENDPOINTS =================
@app.route("/upload", methods=["POST"])
def upload_image():
    """Receives detection images."""

    # Authentication
    api_key = request.headers.get("X-API-KEY")

    if not UPLOAD_API_KEY:
        return {
            "status": "error",
            "message": "Upload authentication not configured"
        }, 503

    if api_key != UPLOAD_API_KEY:

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"❌ Unauthorized upload attempt"
        )

        return {
            "status": "error",
            "message": "Unauthorized"
        }, 401

    global session_image_count
    global last_detection_id
    global last_detection_time

    if 'image' not in request.files:
        return {
            "status": "error",
            "message": "No image part"
        }, 400

    detection_id = request.form.get('detection_id', None)

    total_frames = int(request.form.get('total_frames', 6))

    camera_name = request.form.get('camera', CAM_NAME)

    file = request.files['image']

    with prefix_lock:

        if current_video_prefix == "":
            return {
                "status": "error",
                "message": "Recording not started yet"
            }, 503

        # stale session recovery
        if (
            last_detection_time > 0
            and
            time.time() - last_detection_time > SESSION_STALE_TIMEOUT
        ):

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"⚠️ Stale detection session reset for {camera_name}"
            )

            session_image_count = 0
            last_detection_id = None

        # new session
        if detection_id and detection_id != last_detection_id:

            last_detection_id = detection_id
            session_image_count = 0

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"🆔 New detection session: "
                f"{detection_id} for {camera_name}"
            )

        if session_image_count >= MAX_IMAGES_PER_SESSION:

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"⚠️ REJECTED: {camera_name} - "
                f"limit reached ({MAX_IMAGES_PER_SESSION})"
            )

            return {
                "status": "error",
                "message": f"Limit reached ({MAX_IMAGES_PER_SESSION})",
                "session_count": session_image_count
            }, 429

        current_prefix = current_video_prefix

    target_dir = get_camera_dir()

    actual_time = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    )[:-3]

    detection_tag = f"_{detection_id}" if detection_id else ""

    new_filename = (
        f"{current_prefix}_DETECTION"
        f"{detection_tag}_{actual_time}.jpg"
    )

    save_path = os.path.join(target_dir, new_filename)

    try:

        file.save(save_path)

        with prefix_lock:
            session_image_count += 1
            current_count = session_image_count
            last_detection_time = time.time()

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"📸 {camera_name}: IMAGE "
            f"{current_count}/{total_frames} "
            f"(ID: {detection_id})",
            flush=True
        )

        if current_count == total_frames:

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"✅ {camera_name}: Detection session "
                f"{detection_id} complete "
                f"({total_frames}/{total_frames})"
            )
            
            # QUEUE ASYNC RESET - this will call send_reset_to_detector()
            # from inside the background thread (non-blocking)
            queue_async_reset()

        return {
            "status": "success",
            "path": new_filename,
            "frame_num": current_count,
            "total_frames": total_frames,
            "detection_id": detection_id
        }, 200

    except Exception as e:

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"❌ Save failed: {e}"
        )

        return {
            "status": "error",
            "message": str(e)
        }, 500


@app.route("/health", methods=["GET"])
def health_check():

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

    global session_image_count
    global last_detection_id

    with prefix_lock:

        session_image_count = 0
        last_detection_id = None

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"🔄 Manual reset of session counter for {CAM_NAME}"
        )

        return {
            "status": "success",
            "message": "Session reset"
        }, 200


@app.route("/status", methods=["GET"])
def status():

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


@app.route("/ready", methods=["GET"])
def ready_check():
    """Quick check if recorder is ready to accept uploads."""
    with prefix_lock:
        is_ready = (session_image_count < MAX_IMAGES_PER_SESSION)
        
        return {
            "ready": is_ready,
            "current_count": session_image_count,
            "max": MAX_IMAGES_PER_SESSION,
            "camera": CAM_NAME
        }, 200


def run_flask():
    """Start Flask server."""

    app.run(
        host="0.0.0.0",
        port=FLASK_PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )


# ================= FILE MANAGEMENT =================
def move_to_share_background(local_path, filename):
    """Move file to final location."""
    # NOTE: Reset is now triggered from upload_image() when session completes
    # No reset signal sent from here anymore

    try:

        final_dir = get_camera_dir()

        final_path = os.path.join(final_dir, filename)

        if os.path.exists(local_path):

            shutil.move(local_path, final_path)

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"📁 File moved to share: {filename}",
                flush=True
            )

            # REMOVED: send_reset_to_detector()
            # Reset is now triggered asynchronously from upload_image()

    except Exception as e:

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"❌ BACKGROUND MOVE FAILED: {e}",
            flush=True
        )


# ================= FFMPEG PROCESS MANAGEMENT =================
def kill_ffmpeg():
    """Safely terminate active ffmpeg process."""

    global ffmpeg_process

    if ffmpeg_process is None:
        return

    try:

        if ffmpeg_process.poll() is None:

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"🛑 Stopping ffmpeg process for {CAM_NAME}"
            )

            ffmpeg_process.terminate()

            try:
                ffmpeg_process.wait(timeout=10)

            except subprocess.TimeoutExpired:

                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"⚠️ ffmpeg did not terminate gracefully, killing"
                )

                ffmpeg_process.kill()

    except Exception as e:

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"❌ Error stopping ffmpeg: {e}"
        )

    finally:
        ffmpeg_process = None


# ==========================================
# FILE RETENTION CLEANUP (IMPROVED)
# ==========================================
def cleanup_old_recordings():
    """Delete recordings older than RETENTION_DAYS."""
    
    try:
        if not os.path.exists(BASE_DIR):
            return

        now = time.time()
        retention_seconds = RETENTION_DAYS * 86400
        
        deleted_folders = 0
        freed_space = 0
        
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"🧹 Starting cleanup (retention: {RETENTION_DAYS} days)"
        )
        
        # Walk through date folders in BASE_DIR
        for date_folder in os.listdir(BASE_DIR):
            date_path = os.path.join(BASE_DIR, date_folder)
            
            # Check if it's a directory
            if not os.path.isdir(date_path):
                continue
                
            # Parse date from folder name (format: YYYY-MM-DD)
            try:
                folder_date = datetime.strptime(date_folder, "%Y-%m-%d")
                folder_timestamp = folder_date.timestamp()
                
                # If folder is older than retention period
                if (now - folder_timestamp) > retention_seconds:
                    
                    # Calculate folder size before deletion
                    folder_size = 0
                    for dirpath, dirnames, filenames in os.walk(date_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            try:
                                folder_size += os.path.getsize(filepath)
                            except (OSError, IOError):
                                pass
                    
                    # Delete entire folder
                    shutil.rmtree(date_path)
                    
                    deleted_folders += 1
                    freed_space += folder_size
                    
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"🗑️ Deleted old date folder: {date_folder} "
                        f"({folder_size / (1024**3):.2f} GB)"
                    )
                    
            except ValueError:
                # Not a date folder, skip (could be legacy structure)
                # For legacy structure, clean files individually
                try:
                    for camera_folder in os.listdir(date_path):
                        camera_path = os.path.join(date_path, camera_folder)
                        if os.path.isdir(camera_path):
                            for file in os.listdir(camera_path):
                                file_path = os.path.join(camera_path, file)
                                try:
                                    file_age = now - os.path.getmtime(file_path)
                                    if file_age > retention_seconds:
                                        os.remove(file_path)
                                        print(
                                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                                            f"🗑️ Deleted old file: {file_path}"
                                        )
                                except Exception:
                                    pass
                except Exception:
                    pass
                
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"✅ Cleanup complete "
            f"(folders deleted: {deleted_folders}, "
            f"freed: {freed_space / (1024**3):.2f} GB)"
        )
        
    except Exception as e:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"❌ Cleanup failed: {e}"
        )


def cleanup_worker():
    """Background thread that runs cleanup on a schedule."""
    
    # Run cleanup immediately on startup
    cleanup_old_recordings()
    
    # Then run at configured interval
    while not shutdown_flag:
        time.sleep(CLEANUP_INTERVAL_HOURS * 3600)
        
        if not shutdown_flag:
            cleanup_old_recordings()


# ================= RECORDING LOOP =================
def recording_loop():
    """Main recording loop."""

    global current_video_prefix
    global shutdown_flag
    global ffmpeg_process

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"🔄 RECORDING LOOP STARTED for {CAM_NAME}",
        flush=True
    )
    
    # Start cleanup thread
    cleanup_thread = Thread(
        target=cleanup_worker,
        daemon=True
    )
    cleanup_thread.start()
    
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"🧹 Cleanup thread started (interval: {CLEANUP_INTERVAL_HOURS} hours)",
        flush=True
    )

    while not shutdown_flag:

        kill_ffmpeg()

        start_epoch = time.time()

        timestamp = datetime.fromtimestamp(
            start_epoch
        ).strftime("%Y-%m-%d_%H-%M-%S")

        with prefix_lock:
            current_video_prefix = f"{timestamp}_{CAM_NAME}"

        filename = f"{current_video_prefix}.mp4"

        local_path = os.path.join(TEMP_DIR, filename)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-stimeout",
            "5000000",
            "-i",
            URL,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-t",
            str(SEGMENT_DURATION),
            "-reset_timestamps",
            "1",
            local_path
        ]

        try:

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"🎥 RECORDING: {filename}",
                flush=True
            )

            ffmpeg_process = subprocess.Popen(cmd)

            try:

                ffmpeg_process.wait(
                    timeout=SEGMENT_DURATION + 30
                )

            except subprocess.TimeoutExpired:

                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"⏰ ffmpeg timeout for {CAM_NAME}"
                )

                kill_ffmpeg()
                continue

            if ffmpeg_process.returncode != 0:

                raise subprocess.CalledProcessError(
                    ffmpeg_process.returncode,
                    cmd
                )

            ffmpeg_process = None

            if (
                os.path.exists(local_path)
                and
                os.path.getsize(local_path) > 0
            ):

                move_executor.submit(
                    move_to_share_background,
                    local_path,
                    filename
                )

        except subprocess.CalledProcessError as e:

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"❌ FFmpeg failed with code "
                f"{e.returncode}: {e}"
            )

            ffmpeg_process = None

            time.sleep(5)

        except Exception as e:

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"❌ RECORDING ERROR: {e}",
                flush=True
            )

            import traceback
            traceback.print_exc()

            ffmpeg_process = None

            time.sleep(5)


# ================= SIGNAL HANDLING =================
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""

    global shutdown_flag

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"🛑 Shutting down recorder for {CAM_NAME}..."
    )

    shutdown_flag = True

    kill_ffmpeg()

    move_executor.shutdown(wait=True)

    sys.exit(0)


# ================= MAIN =================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='CCTV Recorder with Person Detection'
    )

    parser.add_argument(
        '--name',
        '-n',
        type=str,
        required=True,
        help='Camera name (e.g., Gate, Center)'
    )

    parser.add_argument(
        '--url',
        '-u',
        type=str,
        required=True,
        help='RTSP URL of the camera'
    )

    args = parser.parse_args()

    CAM_NAME = args.name
    URL = args.url

    print("=" * 60)
    print(f"CCTV RECORDER v{VERSION}")
    print(f"Camera: {CAM_NAME}")
    print(f"Detector URL: {DETECTOR_WEBHOOK_URL}")
    print(f"Flask port: {FLASK_PORT}")
    print(f"Max images per session: {MAX_IMAGES_PER_SESSION}")
    print(f"Segment duration: {SEGMENT_DURATION}s")
    print(f"Retention: {RETENTION_DAYS} days")
    print(f"Cleanup interval: {CLEANUP_INTERVAL_HOURS} hours")
    print(f"Webhook configured: {WEBHOOK_SECRET is not None}")
    print("=" * 60)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server_thread = Thread(
        target=run_flask,
        daemon=True
    )

    server_thread.start()

    time.sleep(2)

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"✅ Recorder ready for {CAM_NAME}"
    )

    recording_loop()
