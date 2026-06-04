#!/usr/bin/env python3
# ==============================================================================
# CCTV IMAGE DETECTOR - VERSION 3.25.0
# ==============================================================================
#
# IMPROVEMENTS in v3.25.0:
#   1. Day/Night hierarchical configuration support
#      - Base daytime defaults in CAMERA_* sections
#      - Night overrides in NIGHT_CAMERA_* sections (only changed values needed)
#      - Automatic switching based on system time
#   2. Added merge_camera_config() and load_camera_config() helper functions
#   3. Periodic config reload thread (checks every hour for day/night transition)
#
# IMPROVEMENTS in v3.24.9:
#   1. Changed email alerts: send ONE email per session (after reset) instead of per frame
#   2. Email now includes ALL frames from the session (up to MAX_IMAGES)
#   3. Added session_frames buffer in CameraState to store frames for email
#   4. Frames are sorted by confidence, best frames kept for email
#   5. Added rejection emails for false positives (rate-limited to 1 per 5 minutes per camera)
#
# AUTHOR: yoosamui
# DATE: 2026-06-04
# ==============================================================================
import glob
import cv2
import multiprocessing
import threading
import time
import requests
import os
import sys
import uuid
import onnxruntime as ort
import numpy as np
import configparser
import datetime
from queue import Empty, Full
from urllib.parse import quote
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import logging
from collections import defaultdict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

VERSION = "3.25.0"

# ==========================================
# SILENCE LOGS
# ==========================================
werkzeug_log = logging.getLogger('werkzeug')
werkzeug_log.disabled = True
logging.getLogger('werkzeug').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


# ==========================================
# LOAD CONFIGURATION
# ==========================================

config = configparser.ConfigParser()

if not config.read("/etc/cctv/config.ini"):
    print("ERROR: Failed to load /etc/cctv/config.ini")
    sys.exit(1)

# ==========================================
# CONFIGURATION HELPERS
# ==========================================
def cfg_int(section, key):
    return config.getint(section, key)

def cfg_float(section, key):
    return config.getfloat(section, key)

def cfg_bool(section, key):
    return config.getboolean(section, key)

def cfg_str(section, key):
    return config.get(section, key)


# ==========================================
# TIME-BASED CONFIGURATION HELPERS
# ==========================================
DAY_START_HOUR = 7   # 7 AM
DAY_END_HOUR = 19    # 7 PM

def is_daytime():
    """Return True if current time is daytime."""
    now = datetime.datetime.now()
    current_hour = now.hour
    return DAY_START_HOUR <= current_hour < DAY_END_HOUR

def merge_camera_config(base_dict, override_dict):
    """
    Merge two camera configuration dictionaries.
    override_dict takes precedence over base_dict.
    """
    result = base_dict.copy()
    result.update(override_dict)
    return result

def load_camera_config(section_prefix):
    """
    Load camera configuration with support for day/night overrides.

    Args:
        section_prefix: 'CAMERA_MIN_AREA', 'CAMERA_MAX_AREA', or 'CAMERA_ASPECT_RATIO'

    Returns:
        Dictionary with camera configurations merged from base and night overrides
    """
    # Load base configuration (daytime defaults)
    base_config = {}
    for camera, value in config[section_prefix].items():
        camera_name = camera.capitalize()
        if section_prefix == "CAMERA_ASPECT_RATIO":
            min_ratio, max_ratio = map(float, value.split(","))
            base_config[camera_name] = (min_ratio, max_ratio)
        else:
            base_config[camera_name] = int(value)

    # If it's nighttime, try to load night overrides
    if not is_daytime():
        night_section = f"NIGHT_{section_prefix}"
        if config.has_section(night_section):
            for camera, value in config[night_section].items():
                camera_name = camera.capitalize()
                if section_prefix == "CAMERA_ASPECT_RATIO":
                    min_ratio, max_ratio = map(float, value.split(","))
                    base_config[camera_name] = (min_ratio, max_ratio)
                    print(f"🌙 NIGHT override: {camera_name} aspect ratio = ({min_ratio}, {max_ratio})")
                else:
                    base_config[camera_name] = int(value)
                    print(f"🌙 NIGHT override: {camera_name} {section_prefix.split('_')[1]} = {value}")
        else:
            print(f"🌙 No night overrides for {section_prefix}, using daytime defaults")

    return base_config


# ==========================================
# CONFIGURATION
# ==========================================
ANALYSIS_INTERVAL = cfg_int("GENERAL", "ANALYSIS_INTERVAL")
MAX_IMAGES = cfg_int("GENERAL", "MAX_IMAGES")
COOLDOWN = cfg_float("GENERAL", "COOLDOWN")
CAM_THREAD_SLEEP = cfg_float("GENERAL", "CAM_THREAD_SLEEP")
WEBHOOK_PORT = cfg_int("GENERAL", "WEBHOOK_PORT")

YOLO_CONFIDENCE = cfg_float("YOLO", "CONFIDENCE")
YOLO_INPUT_SIZE = cfg_int("YOLO", "INPUT_SIZE")
YOLO_IOU = cfg_float("YOLO", "IOU")
YOLO_RESTART_DELAY = cfg_int("YOLO", "RESTART_DELAY")

JPEG_QUALITY = cfg_int("IMAGE", "JPEG_QUALITY")
DRAW_BOUNDING_BOXES = cfg_bool("IMAGE", "DRAW_BOUNDING_BOXES")

SESSION_TIMEOUT = cfg_int("SESSION", "SESSION_TIMEOUT")
WATCHDOG_TIMEOUT = cfg_int("SESSION", "WATCHDOG_TIMEOUT")
WATCHDOG_CHECK = cfg_int("SESSION", "WATCHDOG_CHECK")
RESET_DEDUP_WINDOW = cfg_int("SESSION", "RESET_DEDUP_WINDOW")

DEBUG_LOG_INTERVAL = cfg_float("DEBUG", "DEBUG_LOG_INTERVAL")
ENABLE_DEBUG_PRINTS = cfg_bool("DEBUG", "ENABLE_DEBUG_PRINTS")
SAVE_IMAGE_INTERVAL = cfg_float("DEBUG", "SAVE_IMAGE_INTERVAL")

SAVE_REJECTED_IMAGES = cfg_bool("DEBUG", "SAVE_REJECTED_IMAGES")
REJECTED_IMAGES_DIR = cfg_str("DEBUG", "REJECTED_IMAGES_DIR")
MAX_REJECTED_IMAGES = cfg_int("DEBUG", "MAX_REJECTED_IMAGES")

ENABLE_AREA_FILTER = cfg_bool("FILTERS", "ENABLE_AREA_FILTER")
ENABLE_ASPECT_FILTER = cfg_bool("FILTERS", "ENABLE_ASPECT_FILTER")
ENABLE_TOP_EDGE_FILTER = cfg_bool("FILTERS", "ENABLE_TOP_EDGE_FILTER")
ENABLE_DARK_FILTER = cfg_bool("FILTERS", "ENABLE_DARK_FILTER")

MIN_PERSON_AREA = cfg_int("FILTERS", "MIN_PERSON_AREA")
MIN_ASPECT_RATIO = cfg_float("FILTERS", "MIN_ASPECT_RATIO")
MAX_ASPECT_RATIO = cfg_float("FILTERS", "MAX_ASPECT_RATIO")

TOP_EDGE_MARGIN = cfg_int("FILTERS", "TOP_EDGE_MARGIN")
TOP_EDGE_HIGH_CONF = cfg_float("FILTERS", "TOP_EDGE_HIGH_CONF")

# Dark Pixel Filter Configuration
DARKNESS_THRESHOLD = cfg_int("FILTERS", "DARKNESS_THRESHOLD")
DARK_PIXEL_RATIO = cfg_float("FILTERS", "DARK_PIXEL_RATIO")

UPLOAD_WORKERS = cfg_int("UPLOAD", "UPLOAD_WORKERS")
UPLOAD_QUEUE_SIZE = cfg_int("UPLOAD", "UPLOAD_QUEUE_SIZE")
UPLOAD_MAX_RETRIES = cfg_int("UPLOAD", "UPLOAD_MAX_RETRIES")
UPLOAD_RETRY_DELAY_BASE = cfg_int("UPLOAD", "UPLOAD_RETRY_DELAY_BASE")

TASK_QUEUE_SIZE = cfg_int("UPLOAD", "TASK_QUEUE_SIZE")
RESULT_QUEUE_SIZE = cfg_int("UPLOAD", "RESULT_QUEUE_SIZE")

FILTER_CONFIDENCE = cfg_float("FILTERS", "FILTER_CONFIDENCE")

# ==========================================
# CAMERA-SPECIFIC CONFIGURATION LOADING (with day/night support)
# ==========================================
CAMERA_MIN_AREA = load_camera_config("CAMERA_MIN_AREA")
CAMERA_MAX_AREA = load_camera_config("CAMERA_MAX_AREA")
CAMERA_ASPECT_RATIOS = load_camera_config("CAMERA_ASPECT_RATIO")

# Print loaded configuration
print("\n" + "=" * 60)
print(f"TIME: {'☀️ DAY' if is_daytime() else '🌙 NIGHT'}")
print("CAMERA_MIN_AREA (final):")
for camera, value in CAMERA_MIN_AREA.items():
    print(f"  {camera} = {value}")
print("CAMERA_MAX_AREA (final):")
for camera, value in CAMERA_MAX_AREA.items():
    print(f"  {camera} = {value}")
print("CAMERA_ASPECT_RATIOS (final):")
for camera, value in CAMERA_ASPECT_RATIOS.items():
    print(f"  {camera} = {value[0]},{value[1]}")
print("=" * 60)


# ==========================================
# REJECTED IMAGES FUNCTIONS
# ==========================================
def ensure_rejected_dir():
    """Create rejected images directory if it doesn't exist"""
    if SAVE_REJECTED_IMAGES:
        try:
            os.makedirs(REJECTED_IMAGES_DIR, exist_ok=True)

            existing = glob.glob(os.path.join(REJECTED_IMAGES_DIR, "*.jpg"))
            if len(existing) > MAX_REJECTED_IMAGES:
                existing.sort()
                for f in existing[:-MAX_REJECTED_IMAGES]:
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"[WARN] Failed to remove old rejected image {f}: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to create rejected images directory {REJECTED_IMAGES_DIR}: {e}")


# ==========================================
# DRAW TEXT WITH BACKGROUND
# ==========================================
def draw_text_with_background(img, text, position, font_scale, bg_color, text_color, thickness=1):
    """Draw text with a colored background rectangle."""
    x, y = position
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    padding = 4
    # Draw background rectangle (filled)
    cv2.rectangle(img,
                  (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + padding),
                  bg_color,
                  -1)

    # Draw text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)


def save_rejected_image(camera_name, frame, box, confidence, reason, aspect_ratio=None, area=None, min_value=None, top_y=None, min_conf=None):
    """Save rejected detection image for debugging with clean filename format"""

    if not SAVE_REJECTED_IMAGES:
        return

    try:
        ensure_rejected_dir()

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Draw rejection info on image
        debug_frame = frame.copy()
        yellow = (0, 255, 255)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), yellow, 1)

        timestamp = time.strftime('%Y-%m-%d_%H%M%S')

        if reason == "area":
            filename = f"[{timestamp}] _[DEBUG] {camera_name}: Area too small ({area}px < {min_value}px) - box: {width}x{height}px conf={confidence:.2f} REJECTED.jpg"
        elif reason == "area_too_large":
            filename = f"[{timestamp}] _[DEBUG] {camera_name}: Area too large ({area}px > {min_value}px) - box: {width}x{height}px conf={confidence:.2f} REJECTED.jpg"
        elif reason == "aspect":
            ar = aspect_ratio if aspect_ratio is not None else 0
            filename = f"[{timestamp}] _[DEBUG] {camera_name}: Bad aspect ratio ({ar:.2f}) - box: {width}x{height}px conf={confidence:.2f} REJECTED.jpg"
        elif reason == "top_edge":
            filename = f"[{timestamp}] _[DEBUG] {camera_name}: Top-edge rejection - y1={top_y}, conf={confidence:.2f}<{min_conf} - box: {width}x{height}px REJECTED.jpg"
        elif reason == "dark_pixels":
            dark_percent = int(DARK_PIXEL_RATIO * 100)
            filename = f"[{timestamp}] _[DEBUG] {camera_name}: Too many dark pixels ({min_value}% > {dark_percent}%) - box: {width}x{height}px conf={confidence:.2f} REJECTED.jpg"
        else:
            filename = f"[{timestamp}] _[DEBUG] {camera_name}: {reason} - box: {width}x{height}px conf={confidence:.2f} REJECTED.jpg"

        filename = filename.replace(" ", "_").replace(":", "-")
        filepath = os.path.join(REJECTED_IMAGES_DIR, filename)

        if reason == "area":
            text = f"REJECTED: Area too small ({area}px < {min_value}px)"
        elif reason == "area_too_large":
            text = f"REJECTED: Area too large ({area}px > {min_value}px)"
        elif reason == "aspect":
            ar = aspect_ratio if aspect_ratio is not None else 0
            text = f"REJECTED: Bad aspect ratio ({ar:.2f})"
        elif reason == "top_edge":
            text = f"REJECTED: Top-edge (y1={top_y}, conf={confidence:.2f}<{min_conf})"
        elif reason == "dark_pixels":
            dark_percent = int(DARK_PIXEL_RATIO * 100)
            text = f"REJECTED: Too many dark pixels ({min_value}% > {dark_percent}%)"
        else:
            text = f"REJECTED: {reason}"

        draw_text_with_background(debug_frame, text, (x1, y1 - 10), 0.5, (0, 255, 255), (0, 0, 0))
        draw_text_with_background(debug_frame, f"Conf: {confidence:.2f}", (x1, y1 - 28), 0.4, (0, 255, 255), (0, 0, 0))

        cv2.imwrite(filepath, debug_frame)

    except Exception as e:
        print(f"[ERROR] save_rejected_image failed for {camera_name}: {e}")

# ==========================================
# DARK PIXEL FILTER FUNCTION
# ==========================================
def is_dark_detection(frame, box, darkness_threshold=50, dark_pixel_ratio=0.3):
    """
    Check if a detection contains too many dark pixels (shadows, headlights, false positives).
    Uses OpenCV for fast pixel analysis (no Python loops).

    Args:
        frame: Original frame (BGR)
        box: Bounding box [x1, y1, x2, y2]
        darkness_threshold: Pixel value < this is considered dark (0-255)
        dark_pixel_ratio: Reject if > this percentage of pixels are dark (0.0-1.0)

    Returns:
        True if detection is too dark (should be rejected)
    """
    x1, y1, x2, y2 = box

    # Ensure coordinates are within frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return False

    # Extract region of interest
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    # Convert to grayscale (fast)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Count dark pixels using OpenCV (very fast)
    _, dark_mask = cv2.threshold(gray, darkness_threshold, 255, cv2.THRESH_BINARY_INV)
    dark_pixels = cv2.countNonZero(dark_mask)
    total_pixels = roi.shape[0] * roi.shape[1]

    dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0

    return dark_ratio > dark_pixel_ratio


# ==========================================
# THREAD POOL SETTINGS
# ==========================================
UPLOAD_WORKERS = 2
UPLOAD_QUEUE_SIZE = 20
YOLO_RESTART_DELAY = 5
UPLOAD_MAX_RETRIES = 3
UPLOAD_RETRY_DELAY_BASE = 2
TASK_QUEUE_SIZE = 10
RESULT_QUEUE_SIZE = 20

# ==========================================
# SESSION STATE ENUM
# ==========================================
class SessionState(Enum):
    IDLE = 0
    ACTIVE = 1
    WAITING_RESET = 2
    COMPLETED = 3

# ==========================================
# THREAD-SAFE METRICS COLLECTOR
# ==========================================
class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.pending_uploads = 0
        self.dropped_frames = 0
        self.dropped_uploads = 0

    def inc_dropped_frames(self):
        with self._lock:
            self.dropped_frames += 1

    def inc_dropped_uploads(self):
        with self._lock:
            self.dropped_uploads += 1

    def inc_pending_uploads(self):
        with self._lock:
            self.pending_uploads += 1

    def dec_pending_uploads(self):
        with self._lock:
            self.pending_uploads = max(0, self.pending_uploads - 1)

    def can_add_upload(self):
        with self._lock:
            return self.pending_uploads < UPLOAD_QUEUE_SIZE

    def get_stats(self):
        with self._lock:
            return {
                'pending_uploads': self.pending_uploads,
                'dropped_frames': self.dropped_frames,
                'dropped_uploads': self.dropped_uploads
            }

metrics = Metrics()
yolo_process = None

# ==========================================
# AUTHENTICATION
# ==========================================
load_dotenv("/etc/cctv/credentials.env")
CAM_PASS = os.getenv("CAM_PASS")
if not CAM_PASS:
    print("ERROR: CAM_PASS not found!")
    sys.exit(1)
password = quote(CAM_PASS)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-this-default-secret")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

# ==========================================
# EMAIL ALERT CONFIGURATION
# ==========================================
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
ALERT_TO   = os.getenv("ALERT_TO")
ENABLE_EMAIL_ALERTS = bool(SMTP_USER and SMTP_PASS and ALERT_TO)

# ==========================================
# CAMERA NODES
# ==========================================
NODES = {
    "Gate":     {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.99:554/Streaming/channels/102", "rpi_url": "http://192.168.1.14:5000/upload"},
    "Center":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.82:554/Streaming/channels/102", "rpi_url": "http://192.168.1.13:5000/upload"},
    "Entrance": {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.89:554/Streaming/channels/102", "rpi_url": "http://192.168.1.15:5000/upload"},
    "Garage":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.81:554/Streaming/channels/102", "rpi_url": "http://192.168.1.16:5000/upload"},
    "Behind":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.92:554/Streaming/channels/102", "rpi_url": "http://192.168.1.17:5000/upload"},
    "Left":     {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.93:554/Streaming/channels/102", "rpi_url": "http://192.168.1.18:5000/upload"}
}

# ==========================================
# PER-CAMERA SESSION STATE
# ==========================================
class CameraState:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = SessionState.IDLE
        self.count = 0
        self.detection_id = None
        self.last_upload = 0
        self.last_queued_time = 0
        self.last_result_time = 0
        self.last_activity = 0
        self.last_waiting_start = 0
        self.last_reset_time = 0
        self.last_reset_processed = 0
        self.active_session_id = None
        self.last_processed_count = 0

        # Store frames for email (top MAX_IMAGES by confidence)
        self.session_frames = []
        self.session_frames_lock = threading.Lock()

camera_states = {n: CameraState() for n in NODES}
upload_executor = ThreadPoolExecutor(max_workers=UPLOAD_WORKERS, thread_name_prefix="upload")

# Create a separate executor for email alerts (non-critical)
email_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="email")

# ==========================================
# FLASK WEBHOOK SERVER
# ==========================================
app = Flask(__name__)

def verify_auth():
    api_key = request.headers.get('X-API-KEY')
    return api_key and api_key == WEBHOOK_SECRET

@app.route('/session-reset', methods=['POST'])
def session_reset():
    if not verify_auth():
        return jsonify({"status": "unauthorized"}), 401
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error"}), 400
        camera_name = data.get('camera')
        if not camera_name or camera_name not in camera_states:
            return jsonify({"status": "ok"}), 200
        state = camera_states[camera_name]
        now = time.time()
        with state.lock:
            if now - state.last_reset_processed < RESET_DEDUP_WINDOW:
                return '', 200
            state.last_reset_processed = now
            was_waiting = (state.state == SessionState.WAITING_RESET)
            old_count = state.count
            old_detection_id = state.detection_id
            state.state = SessionState.COMPLETED
            state.count = 0
            state.detection_id = None
            state.last_waiting_start = 0
            state.last_reset_time = now
            state.active_session_id = None
            state.last_processed_count = 0

            if was_waiting:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📡 Recorder signaled: {camera_name} reset - DETECTION RESUMED [SID:{old_detection_id}]")
            elif old_count > 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📡 Recorder signaled: {camera_name} reset - CLEANED UP PARTIAL SESSION ({old_count}/{MAX_IMAGES} frames) [SID:{old_detection_id}]")

            # Send email with all frames from the session
            if old_detection_id and old_count > 0:
                email_executor.submit(send_session_email, camera_name, old_detection_id)

        return '', 200
    except Exception as e:
        print(f"Error in session_reset: {e}")
        return jsonify({"status": "error"}), 500

@app.route('/reset', methods=['POST'])
def reset_legacy():
    response = make_response(session_reset())
    response.headers['Warning'] = '299 - "Deprecated endpoint: Use /session-reset instead"'
    return response

@app.route('/health', methods=['GET'])
def health_check():
    camera_stats = {}
    for name, state in camera_states.items():
        with state.lock:
            camera_stats[name] = {"state": state.state.name, "count": state.count}
    return jsonify({"status": "running", "version": VERSION, "cameras": camera_stats}), 200

def start_webhook_server():
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=WEBHOOK_PORT, debug=False, use_reloader=False, threaded=True), daemon=True).start()
    print(f"🌐 Webhook server on port {WEBHOOK_PORT}")

# ==========================================
# UPLOAD FUNCTIONS
# ==========================================
def upload_task(camera_name, url, image_buffer, ts, current_count, max_images, detection_id):
    max_retries = UPLOAD_MAX_RETRIES
    state = camera_states[camera_name]

    with state.lock:
        if state.active_session_id and detection_id != state.active_session_id:
            return
        original_session_id = state.active_session_id

    for attempt in range(max_retries):
        try:
            files = {'image': (f"{camera_name}_{ts}.jpg", image_buffer, 'image/jpeg')}
            data = {'frame_num': current_count, 'total_frames': max_images, 'camera': camera_name, 'detection_id': detection_id}
            headers = {'X-API-KEY': WEBHOOK_SECRET}
            response = requests.post(url, files=files, data=data, headers=headers, timeout=5)
            response.raise_for_status()

            if current_count == max_images:
                with state.lock:
                    if (state.state == SessionState.ACTIVE and
                        state.active_session_id == original_session_id and
                        state.detection_id == detection_id):
                        state.state = SessionState.WAITING_RESET
                        state.last_waiting_start = time.time()
                        print(f"[{ts}] 🛑 {camera_name}: Last frame sent ({current_count}/{max_images}) [SID:{detection_id}]")
                    else:
                        print(f"[{ts}] ⚠️ {camera_name}: Session changed during upload, skipping state change [SID:{detection_id}]")
            return
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                wait_time = UPLOAD_RETRY_DELAY_BASE * (attempt + 1)
                time.sleep(wait_time)
            elif e.response.status_code == 401:
                break
            else:
                break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(UPLOAD_RETRY_DELAY_BASE)
            else:
                break

def draw_and_upload(camera_name, url, frame, detections, ts, current_count, max_images, detection_id):
    working_frame = frame.copy()
    yellow = (0, 255, 255)

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"PERSON {d['conf']:.2f}"
        if DRAW_BOUNDING_BOXES:
            cv2.rectangle(working_frame, (x1, y1), (x2, y2), yellow, 1)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(working_frame, (x1, y1 - 20), (x1 + w, y1), yellow, -1)
        cv2.putText(working_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    success, buffer = cv2.imencode('.jpg', working_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if success:
        if not metrics.can_add_upload():
            metrics.inc_dropped_uploads()
            return
        metrics.inc_pending_uploads()
        def upload_wrapper(*args, **kwargs):
            try:
                return upload_task(*args, **kwargs)
            finally:
                metrics.dec_pending_uploads()
        upload_executor.submit(upload_wrapper, camera_name, url, buffer.tobytes(), ts, current_count, max_images, detection_id)

# ==========================================
# EMAIL ALERT - SEND SESSION EMAIL (ONE PER SESSION)
# ==========================================
def send_session_email(camera_name, detection_id):
    """Send email after session reset with ALL frames from the session."""
    if not ENABLE_EMAIL_ALERTS:
        return

    if not SMTP_USER or not SMTP_PASS or not ALERT_TO:
        return

    state = camera_states[camera_name]

    # Get all frames from the session
    with state.session_frames_lock:
        session_frames = state.session_frames.copy()
        # Clear after sending to avoid duplicate emails
        state.session_frames = []

    if not session_frames:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No frames to send for session {detection_id}")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = ALERT_TO
        msg['Subject'] = f"🚨 CCTV Alert: Person detected on {camera_name} (Session {detection_id})"

        # Calculate average confidence
        avg_conf = sum(f['confidence'] for f in session_frames) / len(session_frames)

        body = (
            f"<b>Camera:</b> {camera_name}<br>"
            f"<b>Session ID:</b> {detection_id}<br>"
            f"<b>Frames captured:</b> {len(session_frames)}/{MAX_IMAGES}<br>"
            f"<b>Average Confidence:</b> {avg_conf:.0%}<br>"
            f"<b>Time:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}<br><br>"
            f"<i>Attached: All frames from this detection session (sorted by confidence).</i>"
        )
        msg.attach(MIMEText(body, 'html'))

        # Attach all frames
        for idx, frame_data in enumerate(session_frames, 1):
            conf = frame_data['confidence']
            frame = frame_data['frame']
            frame_num = frame_data.get('frame_num', idx)

            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if success:
                img = MIMEImage(buffer.tobytes(), name=f"{camera_name}_{detection_id}_frame{frame_num}_conf{conf:.0%}.jpg")
                msg.attach(img)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Email sent for {camera_name} - Session {detection_id} ({len(session_frames)} frames)")

    except Exception as e:
        print(f"[ERROR] Email alert failed for {camera_name}: {e}")

# ==========================================
# EMAIL ALERT - SEND REJECTION EMAIL (RATE LIMITED)
# ==========================================
def send_rejection_email(camera_name, frame, box, confidence, reason, aspect_ratio=None, area=None, min_value=None):
    """Send email with rejected image for debugging."""
    if not ENABLE_EMAIL_ALERTS:
        return

    if not SMTP_USER or not SMTP_PASS or not ALERT_TO:
        return

    try:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Draw rejection info on image
        annotated_frame = frame.copy()
        yellow = (0, 255, 255)  # yellow for rejected
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), yellow, 1)

        if reason == "area":
            text = f"REJECTED: Area too small ({area}px < {min_value}px)"
            subject = f"⚠️ CCTV Rejection: Area too small on {camera_name}"
        elif reason == "area_too_large":
            text = f"REJECTED: Area too large ({area}px > {min_value}px)"
            subject = f"⚠️ CCTV Rejection: Area too large on {camera_name}"
        elif reason == "aspect":
            text = f"REJECTED: Bad aspect ratio ({aspect_ratio:.2f})"
            subject = f"⚠️ CCTV Rejection: Bad aspect ratio on {camera_name}"
        elif reason == "top_edge":
            text = f"REJECTED: Top-edge (y1={y1})"
            subject = f"⚠️ CCTV Rejection: Top-edge on {camera_name}"
        elif reason == "dark_pixels":
            text = f"REJECTED: Too many dark pixels"
            subject = f"⚠️ CCTV Rejection: Dark pixels on {camera_name}"
        else:
            text = f"REJECTED: {reason}"
            subject = f"⚠️ CCTV Rejection: {reason} on {camera_name}"

        draw_text_with_background(annotated_frame, text, (x1, y1 - 10), 0.5, (0, 255, 255), (0, 0, 0))
        draw_text_with_background(annotated_frame, f"Conf: {confidence:.2f}", (x1, y1 - 28), 0.4, (0, 255, 255), (0, 0, 0))


        #cv2.putText(annotated_frame, text, (x1, y1 - 10),
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1)
        #cv2.putText(annotated_frame, f"Conf: {confidence:.2f}", (x1, y2 + 15),
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.4, yellow, 1)

        success, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not success:
            return

        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = ALERT_TO
        msg['Subject'] = subject

        body = (
            f"<b>Camera:</b> {camera_name}<br>"
            f"<b>Reason:</b> {text}<br>"
            f"<b>Confidence:</b> {confidence:.0%}<br>"
            f"<b>Box:</b> {width}x{height}px<br>"
            f"<b>Time:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        msg.attach(MIMEText(body, 'html'))

        img = MIMEImage(buffer.tobytes(), name=f"rejected_{camera_name}_{reason}.jpg")
        msg.attach(img)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📧 Rejection email sent for {camera_name}: {reason}")

    except Exception as e:
        print(f"[ERROR] Rejection email failed for {camera_name}: {e}")

# ==========================================
# CONFIGURATION RELOAD THREAD (for day/night switching)
# ==========================================
def config_reload_thread():
    """Reload camera configurations every hour to handle day/night transitions."""
    global CAMERA_MIN_AREA, CAMERA_MAX_AREA, CAMERA_ASPECT_RATIOS

    while True:
        time.sleep(3600)  # Check every hour

        # Reload configurations
        new_min_area = load_camera_config("CAMERA_MIN_AREA")

        # Only update if changed (avoid unnecessary updates)
        if CAMERA_MIN_AREA != new_min_area:
            CAMERA_MIN_AREA = new_min_area
            CAMERA_MAX_AREA = load_camera_config("CAMERA_MAX_AREA")
            CAMERA_ASPECT_RATIOS = load_camera_config("CAMERA_ASPECT_RATIO")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Camera configs updated for {'DAY' if is_daytime() else 'NIGHT'}")

# Start config reload thread
threading.Thread(target=config_reload_thread, daemon=True).start()

# ==========================================
# YOLO WORKER PROCESS
# ==========================================
def yolo_worker_process(input_q, output_q, min_person_area, min_aspect_ratio, max_aspect_ratio,
                        enable_area_filter, enable_aspect_filter, enable_dark_filter,
                        camera_min_area_dict, camera_max_area_dict, camera_aspect_ratios_dict,
                        yolo_iou, debug_log_interval):

    import threading
    # Separate timers for debug logs (prevents cross-filter suppression)
    _last_area_log_time = {}          # Area too small
    _last_maxarea_log_time = {}       # Area too large (headlights, vehicles)
    _last_aspect_log_time = {}        # Bad aspect ratio
    _last_topedge_log_time = {}       # Top-edge rejection
    _last_topedge_accept_log_time = {} # Top-edge acceptance
    _last_dark_log_time = {}          # Dark pixel rejection

    # Separate timers for image saving (rate-limited)
    _last_area_save_time = {}          # Area too small save timer
    _last_maxarea_save_time = {}       # Area too large save timer
    _last_aspect_save_time = {}        # Bad aspect ratio save timer
    _last_topedge_save_time = {}       # Top-edge rejection save timer
    _last_dark_save_time = {}          # Dark pixel rejection save timer

    # Timers for rejection email (rate-limited to 5 minutes per camera)
    _last_rejection_email_time = {}

    _log_lock = threading.Lock()

    def is_valid_person_detection_worker(camera_name, box, confidence, image_height, frame_id=None, original_frame=None):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        min_area = camera_min_area_dict.get(camera_name, min_person_area)
        max_area = camera_max_area_dict.get(camera_name) if camera_max_area_dict else None
        min_ratio, max_ratio = camera_aspect_ratios_dict.get(camera_name, (min_aspect_ratio, max_aspect_ratio))

        # Initialize aspect_ratio to a default value
        aspect_ratio = 0.0

        # ==========================================
        # HIGH CONFIDENCE BYPASS
        # ==========================================
        if confidence >= FILTER_CONFIDENCE:
            return True

        # ==========================================
        # DARK PIXEL FILTER
        # ==========================================
        if enable_dark_filter and confidence < FILTER_CONFIDENCE:
            if is_dark_detection(original_frame, box, DARKNESS_THRESHOLD, DARK_PIXEL_RATIO):
                with _log_lock:
                    now = time.time()
                    last_time = _last_dark_log_time.get(camera_name, 0)
                    if now - last_time >= debug_log_interval and ENABLE_DEBUG_PRINTS:
                        _last_dark_log_time[camera_name] = now
                        dark_percent = int(DARK_PIXEL_RATIO * 100)
                        msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: Too many dark pixels (> {dark_percent}%) - box: {width}x{height}px conf={confidence:.2f} REJECTED!"
                        print(msg)

                # Rate-limited image saving
                if original_frame is not None and SAVE_REJECTED_IMAGES:
                    now = time.time()
                    last_save = _last_dark_save_time.get(camera_name, 0)
                    if now - last_save >= SAVE_IMAGE_INTERVAL:
                        _last_dark_save_time[camera_name] = now
                        dark_percent = int(DARK_PIXEL_RATIO * 100)
                        save_rejected_image(camera_name, original_frame, box, confidence, "dark_pixels",
                                          min_value=dark_percent)

                # Rate-limited rejection email (1 per 5 minutes per camera)
                if original_frame is not None:
                    now = time.time()
                    last_email = _last_rejection_email_time.get(camera_name, 0)
                    if now - last_email >= 300:
                        _last_rejection_email_time[camera_name] = now
                        email_executor.submit(send_rejection_email, camera_name, original_frame, box, confidence, "dark_pixels")
                return False

        # ==========================================
        # MIN AREA FILTER (too small)
        # ==========================================
        if enable_area_filter and area < min_area:
            with _log_lock:
                now = time.time()
                last_time = _last_area_log_time.get(camera_name, 0)
                if now - last_time >= debug_log_interval and ENABLE_DEBUG_PRINTS:
                    _last_area_log_time[camera_name] = now
                    msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: Area too small ({area}px < {min_area}px) - box: {width}x{height}px REJECTED!"
                    print(msg)

            # Rate-limited image saving
            if original_frame is not None and SAVE_REJECTED_IMAGES:
                now = time.time()
                last_save = _last_area_save_time.get(camera_name, 0)
                if now - last_save >= SAVE_IMAGE_INTERVAL:
                    _last_area_save_time[camera_name] = now
                    save_rejected_image(camera_name, original_frame, box, confidence, "area",
                                      area=area, min_value=min_area)

            # Rate-limited rejection email
            if original_frame is not None:
                now = time.time()
                last_email = _last_rejection_email_time.get(camera_name, 0)
                if now - last_email >= 300:
                    _last_rejection_email_time[camera_name] = now
                    email_executor.submit(send_rejection_email, camera_name, original_frame, box, confidence, "area",
                                         area=area, min_value=min_area)
            return False

        # ==========================================
        # MAX AREA FILTER (too large)
        # ==========================================
        if max_area is not None and area > max_area:
            with _log_lock:
                now = time.time()
                last_time = _last_maxarea_log_time.get(camera_name, 0)
                if now - last_time >= debug_log_interval:
                    _last_maxarea_log_time[camera_name] = now
                    msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: Area too large ({area}px > {max_area}px) - box: {width}x{height}px conf={confidence:.2f} REJECTED!"
                    print(msg)

            # Rate-limited image saving
            if original_frame is not None and SAVE_REJECTED_IMAGES:
                now = time.time()
                last_save = _last_maxarea_save_time.get(camera_name, 0)
                if now - last_save >= SAVE_IMAGE_INTERVAL:
                    _last_maxarea_save_time[camera_name] = now
                    save_rejected_image(camera_name, original_frame, box, confidence, "area_too_large",
                                      area=area, min_value=max_area)

            # Rate-limited rejection email
            if original_frame is not None:
                now = time.time()
                last_email = _last_rejection_email_time.get(camera_name, 0)
                if now - last_email >= 300:
                    _last_rejection_email_time[camera_name] = now
                    email_executor.submit(send_rejection_email, camera_name, original_frame, box, confidence, "area_too_large",
                                         area=area, min_value=max_area)
            return False

        # ==========================================
        # ASPECT RATIO FILTER
        # ==========================================
        if enable_aspect_filter and width > 0:
            aspect_ratio = height / width
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                with _log_lock:
                    now = time.time()
                    last_time = _last_aspect_log_time.get(camera_name, 0)
                    if now - last_time >= debug_log_interval:
                        _last_aspect_log_time[camera_name] = now
                        msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: Bad aspect ratio ({aspect_ratio:.2f}) - box: {width}x{height}px conf={confidence:.2f} REJECTED!"
                        print(msg)

                if original_frame is not None and SAVE_REJECTED_IMAGES:
                    now = time.time()
                    last_save = _last_aspect_save_time.get(camera_name, 0)
                    if now - last_save >= SAVE_IMAGE_INTERVAL:
                        _last_aspect_save_time[camera_name] = now
                        save_rejected_image(camera_name, original_frame, box, confidence, "aspect",
                                          aspect_ratio=aspect_ratio)

                # Rate-limited rejection email
                if original_frame is not None:
                    now = time.time()
                    last_email = _last_rejection_email_time.get(camera_name, 0)
                    if now - last_email >= 300:
                        _last_rejection_email_time[camera_name] = now
                        email_executor.submit(send_rejection_email, camera_name, original_frame, box, confidence, "aspect",
                                             aspect_ratio=aspect_ratio)
                return False

        # ==========================================
        # TOP-EDGE (GROUND) FILTER
        # ==========================================
        if ENABLE_TOP_EDGE_FILTER and image_height is not None:
            LABEL_MARGIN = 12  # Space reserved for label
            top_y = y1
            if top_y <= TOP_EDGE_MARGIN + LABEL_MARGIN:
                if confidence < TOP_EDGE_HIGH_CONF:
                    with _log_lock:
                        now = time.time()
                        last_time = _last_topedge_log_time.get(camera_name, 0)
                        if now - last_time >= debug_log_interval:
                            _last_topedge_log_time[camera_name] = now
                            msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: Top-edge rejection - y1={top_y}, conf={confidence:.2f}<{TOP_EDGE_HIGH_CONF} REJECTED!"
                            print(msg)

                    # Rate-limited image saving
                    if original_frame is not None and SAVE_REJECTED_IMAGES:
                        now = time.time()
                        last_save = _last_topedge_save_time.get(camera_name, 0)
                        if now - last_save >= SAVE_IMAGE_INTERVAL:
                            _last_topedge_save_time[camera_name] = now
                            save_rejected_image(camera_name, original_frame, box, confidence, "top_edge",
                                              top_y=top_y, min_conf=TOP_EDGE_HIGH_CONF)

                    # Rate-limited rejection email
                    if original_frame is not None:
                        now = time.time()
                        last_email = _last_rejection_email_time.get(camera_name, 0)
                        if now - last_email >= 300:
                            _last_rejection_email_time[camera_name] = now
                            email_executor.submit(send_rejection_email, camera_name, original_frame, box, confidence, "top_edge")
                    return False
                else:
                    with _log_lock:
                        now = time.time()
                        last_time = _last_topedge_accept_log_time.get(camera_name, 0)
                        if now - last_time >= debug_log_interval:
                            _last_topedge_accept_log_time[camera_name] = now
                            msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ [DEBUG] {camera_name}: Top-edge HIGH CONF ACCEPT - y1={top_y}, conf={confidence:.2f}>={TOP_EDGE_HIGH_CONF}"
                            print(msg)

        return True

    try:
        print("Loading yolov8n.onnx with pure ONNX Runtime...")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        session = ort.InferenceSession(
            "yolov8n.onnx",
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )

        input_name = session.get_inputs()[0].name
        print("Pure ONNX Runtime initialized")
        print("Ultra-light + smart NMS (minimal CPU increase)")

    except Exception as e:
        print(f"❌ ONNX load failed: {e}")
        return

    scale_x = scale_y = 1.0

    while True:
        try:
            name, frame, ts, capture_time = input_q.get(timeout=0.5)
            frame_id = f"{name}_{capture_time}"

            h, w = frame.shape[:2]
            scale_x = w / YOLO_INPUT_SIZE
            scale_y = h / YOLO_INPUT_SIZE

            img = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1).reshape(1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)

            outputs = session.run(None, {input_name: img})
            predictions = outputs[0][0]

            detections = []
            for pred in predictions.T:
                confidence = np.max(pred[4:])
                if confidence < YOLO_CONFIDENCE:
                    continue
                if np.argmax(pred[4:]) != 0:
                    continue

                xc, yc, pw, ph = pred[:4]
                x1 = int((xc - pw/2) * scale_x)
                y1 = int((yc - ph/2) * scale_y)
                x2 = int((xc + pw/2) * scale_x)
                y2 = int((yc + ph/2) * scale_y)

                box = [x1, y1, x2, y2]

                if is_valid_person_detection_worker(name, box, confidence, h, frame_id, frame):
                    detections.append({"box": box, "conf": float(confidence)})

            if len(detections) > 0:
                detections.sort(key=lambda x: x['conf'], reverse=True)
                filtered = []

                for d1 in detections:
                    keep = True
                    x1, y1, x2, y2 = d1['box']
                    area1 = (x2 - x1) * (y2 - y1)

                    for d2 in filtered:
                        xx1 = max(x1, d2['box'][0])
                        yy1 = max(y1, d2['box'][1])
                        xx2 = min(x2, d2['box'][2])
                        yy2 = min(y2, d2['box'][3])

                        if xx2 > xx1 and yy2 > yy1:
                            overlap = (xx2 - xx1) * (yy2 - yy1)
                            area2 = (d2['box'][2] - d2['box'][0]) * (d2['box'][3] - d2['box'][1])
                            union = area1 + area2 - overlap
                            iou = overlap / union if union > 0 else 0

                            if iou > yolo_iou:
                                keep = False
                                break

                    if keep:
                        filtered.append(d1)

                detections = filtered

            try:
                output_q.put_nowait((name, frame, detections, ts, capture_time))
            except Full:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Result queue full, dropping {name} detection")

        except Empty:
            continue
        except Exception as e:
            print(f"YOLO error: {e}")
            time.sleep(1)

def start_yolo_worker(task_q, result_q):
    global yolo_process
    if yolo_process and yolo_process.is_alive():
        return yolo_process
    yolo_process = multiprocessing.Process(
        target=yolo_worker_process,
        args=(task_q, result_q, MIN_PERSON_AREA, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO,
              ENABLE_AREA_FILTER, ENABLE_ASPECT_FILTER, ENABLE_DARK_FILTER,
              CAMERA_MIN_AREA, CAMERA_MAX_AREA, CAMERA_ASPECT_RATIOS,
              YOLO_IOU, DEBUG_LOG_INTERVAL),
        daemon=True
    )
    yolo_process.start()
    return yolo_process

def check_yolo_health(task_q, result_q):
    global yolo_process
    while True:
        time.sleep(YOLO_RESTART_DELAY)
        if yolo_process and not yolo_process.is_alive():
            print(f"⚠️ YOLO worker died! Restarting...")
            yolo_process.join(timeout=1)
            yolo_process = multiprocessing.Process(
                target=yolo_worker_process,
                args=(task_q, result_q, MIN_PERSON_AREA, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO,
                      ENABLE_AREA_FILTER, ENABLE_ASPECT_FILTER, ENABLE_DARK_FILTER,
                      CAMERA_MIN_AREA, CAMERA_MAX_AREA, CAMERA_ASPECT_RATIOS,
                      YOLO_IOU, DEBUG_LOG_INTERVAL),
                daemon=True
            )
            yolo_process.start()
            print(f"✅ YOLO worker restarted")

# ==========================================
# CAMERA STREAM
# ==========================================
class CameraStream:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.lock = threading.Lock()
        self.frame = None
        self.frame_time = 0
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        cap = None
        consecutive_failures = 0
        max_failures = 5
        while self.running:
            try:
                if cap is None:
                    cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not cap.isOpened():
                        raise Exception("Failed to open camera")
                    consecutive_failures = 0
                ret, frame = cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.frame_time = time.time()
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        cap.release()
                        cap = None
                        consecutive_failures = 0
                        time.sleep(2)
                        continue
                time.sleep(CAM_THREAD_SLEEP)
            except Exception as e:
                if cap:
                    cap.release()
                    cap = None
                time.sleep(2)

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.frame_time
        return None, 0

    def stop(self):
        self.running = False

# ==========================================
# SESSION WATCHDOG
# ==========================================
def session_watchdog():
    while True:
        time.sleep(WATCHDOG_CHECK)
        now = time.time()
        for name, state in camera_states.items():
            with state.lock:
                if state.state == SessionState.COMPLETED:
                    if now - state.last_reset_time > 5:
                        state.state = SessionState.IDLE
                    continue
                if state.state == SessionState.WAITING_RESET and state.last_waiting_start > 0:
                    if now - state.last_waiting_start > WATCHDOG_TIMEOUT:
                        print(f"⚠️ Watchdog: {name} stuck. Force resetting.")
                        state.state = SessionState.IDLE
                        state.count = 0
                        state.detection_id = None
                        state.active_session_id = None
                        state.last_processed_count = 0
                elif state.state == SessionState.ACTIVE and state.count > 0:
                    if now - state.last_activity > SESSION_TIMEOUT:
                        print(f"⏰ Session timeout: {name}")
                        state.state = SessionState.IDLE
                        state.count = 0
                        state.active_session_id = None
                        state.detection_id = None
                        state.last_processed_count = 0

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    start_webhook_server()
    task_q = multiprocessing.Queue(maxsize=TASK_QUEUE_SIZE)
    result_q = multiprocessing.Queue(maxsize=RESULT_QUEUE_SIZE)
    start_yolo_worker(task_q, result_q)
    threading.Thread(target=check_yolo_health, args=(task_q, result_q), daemon=True).start()
    streams = {n: CameraStream(n, cfg["cam_rtsp"]) for n, cfg in NODES.items()}
    time.sleep(2)

    print("=" * 60)
    print(f"CCTV DETECTOR v{VERSION}")
    print(f"ANALYSIS_INTERVAL: {ANALYSIS_INTERVAL}s")
    print(f"MAX_IMAGES: {MAX_IMAGES} per camera")
    print(f"UPLOAD_WORKERS: {UPLOAD_WORKERS}")
    print(f"UPLOAD_QUEUE_SIZE: {UPLOAD_QUEUE_SIZE}")
    print(f"UPLOAD_MAX_RETRIES: {UPLOAD_MAX_RETRIES}")
    print(f"UPLOAD_RETRY_DELAY_BASE: {UPLOAD_RETRY_DELAY_BASE}s")
    print(f"YOLO_AUTO_RESTART: Enabled")
    print(f"YOLO_CONFIDENCE: {YOLO_CONFIDENCE}")
    print(f"YOLO_INPUT_SIZE: {YOLO_INPUT_SIZE}")
    print(f"YOLO_IOU: {YOLO_IOU}")
    print(f"WEBHOOK_PORT: {WEBHOOK_PORT}")
    print(f"SESSION_TIMEOUT: {SESSION_TIMEOUT}s (idle sessions)")
    print(f"WATCHDOG: {WATCHDOG_CHECK}s check interval, {WATCHDOG_TIMEOUT}s timeout for stuck sessions")
    print(f"COOLDOWN: {COOLDOWN}s (unified cooldown)")
    print(f"RESET_DEDUP_WINDOW: {RESET_DEDUP_WINDOW}s (ignore duplicate resets)")
    print(f"MIN_PERSON_AREA: {MIN_PERSON_AREA}")
    print(f"MIN_ASPECT_RATIO: {MIN_ASPECT_RATIO}")
    print(f"DRAW_BOUNDING_BOXES: {DRAW_BOUNDING_BOXES}")
    print(f"TOP_EDGE_FILTER: {ENABLE_TOP_EDGE_FILTER} (margin={TOP_EDGE_MARGIN}px, high_conf={TOP_EDGE_HIGH_CONF})")
    print(f"AREA_FILTER: {ENABLE_AREA_FILTER}")
    print(f"ASPECT_FILTER: {ENABLE_ASPECT_FILTER}")
    print(f"DARK_FILTER: {ENABLE_DARK_FILTER} (threshold={DARKNESS_THRESHOLD}, ratio={DARK_PIXEL_RATIO:.0%}, min_conf={FILTER_CONFIDENCE})")
    print(f"MAX_AREA_FILTER: Enabled (per-camera thresholds)")
    print(f"DEBUG_LOG_INTERVAL: {DEBUG_LOG_INTERVAL}s")
    print(f"ENABLE_DEBUG_PRINTS: {ENABLE_DEBUG_PRINTS}")
    print(f"SAVE_IMAGE_INTERVAL: {SAVE_IMAGE_INTERVAL}s (rate-limited image saving)")
    print("EMAIL CONFIGURATION DEBUG:")
    print(f"  SMTP_HOST: {SMTP_HOST}")
    print(f"  SMTP_PORT: {SMTP_PORT}")
    print(f"  SMTP_USER: {'***SET***' if SMTP_USER else 'NOT SET'}")
    print(f"  SMTP_PASS: {'***SET***' if SMTP_PASS else 'NOT SET'}")
    print(f"  ALERT_TO: {ALERT_TO if ALERT_TO else 'NOT SET'}")
    print(f"  ENABLE_EMAIL_ALERTS: {ENABLE_EMAIL_ALERTS}")
    print("=" * 60)

    print("CONFIG LOADING DEBUG:")
    print(f"Config sections: {config.sections()}")

    print("\nCAMERA_MIN_AREA from config:")
    for camera, value in config["CAMERA_MIN_AREA"].items():
        print(f"  {camera} = {value}")

    print("\nCAMERA_MAX_AREA from config:")
    for camera, value in config["CAMERA_MAX_AREA"].items():
        print(f"  {camera} = {value}")

    print("\nCAMERA_ASPECT_RATIO from config:")
    for camera, value in config["CAMERA_ASPECT_RATIO"].items():
        print(f"  {camera} = {value}")

    print("=" * 60)

    threading.Thread(target=session_watchdog, daemon=True).start()

    def handle_results():
        while True:
            try:
                name, frame, detections, ts, capture_time = result_q.get(timeout=0.01)

                current_count = None
                detection_id = None

                now = time.time()
                state = camera_states[name]

                with state.lock:
                    state.last_result_time = now

                if not detections:
                    continue

                with state.lock:
                    cooldown_remaining = COOLDOWN - (now - state.last_reset_time) if state.last_reset_time > 0 else 0

                if cooldown_remaining > 0:
                    continue

                with state.lock:
                    if state.state == SessionState.IDLE:
                        state.state = SessionState.ACTIVE
                        state.count = 0
                        state.active_session_id = None
                        state.last_processed_count = 0

                    if not (state.state == SessionState.ACTIVE and state.count < MAX_IMAGES):
                        continue

                    if state.count == 0 and (now - state.last_upload < COOLDOWN):
                        continue

                    next_count = state.count + 1
                    if next_count <= state.last_processed_count:
                        if next_count == state.last_processed_count:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {name}: Skipping duplicate frame count {next_count} (already processed) [SID:{state.detection_id}]")
                        continue

                    if state.count == 0:
                        state.detection_id = str(uuid.uuid4())[:8]
                        state.active_session_id = state.detection_id
                        # Clear previous session frames
                        with state.session_frames_lock:
                            state.session_frames = []
                        print(f"[{ts}] 🆔 {name}: New detection session {state.detection_id}")

                    state.count += 1
                    state.last_processed_count = state.count
                    state.last_activity = capture_time if capture_time > 0 else now
                    current_count = state.count
                    detection_id = state.detection_id
                    conf = detections[0]['conf'] if detections else 0

                    box = detections[0]['box']
                    x1, y1, x2, y2 = box
                    width = int(box[2] - box[0])
                    height = int(box[3] - box[1])

                    print(f"[{ts}] ⚡ >>> {name}: {state.count}/{MAX_IMAGES} [SID:{detection_id}] conf={conf:.2f} box={width}x{height}px area={width*height}px {y1}px")

                    # Store frame for email (keep only top MAX_IMAGES by confidence)
                    with state.session_frames_lock:
                        state.session_frames.append({
                            'confidence': conf,
                            'frame': frame.copy(),
                            'timestamp': ts,
                            'frame_num': state.count
                        })
                        # Keep only top MAX_IMAGES (sorted by confidence)
                        state.session_frames.sort(key=lambda x: x['confidence'], reverse=True)
                        state.session_frames = state.session_frames[:MAX_IMAGES]

                if current_count is None or detection_id is None:
                    continue

                if frame is not None:
                    draw_and_upload(name, NODES[name]["rpi_url"], frame, detections, ts,
                                  current_count, MAX_IMAGES, detection_id)

                with state.lock:
                    state.last_upload = now

            except Empty:
                continue
            except Exception as e:
                print(f"Handler error: {e}")

    threading.Thread(target=handle_results, daemon=True).start()

    try:
        while True:
            now = time.time()
            for name in NODES:
                state = camera_states[name]
                with state.lock:
                    if state.state == SessionState.WAITING_RESET:
                        continue
                    skip = now - state.last_queued_time < ANALYSIS_INTERVAL
                if skip:
                    continue
                frame, frame_time = streams[name].get_frame()
                if frame is not None:
                    try:
                        task_q.put_nowait((name, frame, time.strftime('%Y-%m-%d %H:%M:%S'), frame_time))
                        with state.lock:
                            state.last_queued_time = now
                    except Full:
                        metrics.inc_dropped_frames()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nShutting down...")
        upload_executor.shutdown(wait=True)
        email_executor.shutdown(wait=True)
        if yolo_process:
            yolo_process.terminate()
            yolo_process.join(timeout=2)
        for s in streams.values():
            s.stop()
        sys.exit(0)
