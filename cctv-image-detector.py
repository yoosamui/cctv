#!/usr/bin/env python3
# ==============================================================================
# CCTV IMAGE DETECTOR - VERSION 3.24.6
# ==============================================================================
#
# IMPROVEMENTS in v3.24.6:
#   1. Fixed shared log dictionary bug (separate _last_maxarea_log_time for max area filter)
#   2. Ensured each filter has its own independent rate-limit timer for debug logging
#      - AREA (too small) uses _last_area_log_time
#      - MAX AREA (too large) uses _last_maxarea_log_time
#      - ASPECT RATIO uses _last_aspect_log_time
#      - TOP-EDGE rejection uses _last_topedge_log_time
#      - TOP-EDGE acceptance uses _last_topedge_accept_log_time
#   3. This prevents one filter's debug messages from suppressing another filter's messages
#
# IMPROVEMENTS in v3.24.5:
#   1. Fixed duplicate image annotation (removed unused annotated_frame variable)
#   2. Fixed silent exception swallowing (added error logging to all except blocks)
#
# IMPROVEMENTS in v3.24.4:
#   1. Added MAX AREA FILTER to reject large false positives (headlights, vehicles close to camera)
#   2. Added camera-specific CAMERA_MAX_AREA configuration
#   3. Added save_rejected_image support for area_too_large rejections
#
# AUTHOR: yoosamui
# DATE: 2026-06-01
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
from queue import Empty, Full
from urllib.parse import quote
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import logging
from collections import defaultdict

VERSION = "3.24.6"

# ==========================================
# SILENCE LOGS
# ==========================================
werkzeug_log = logging.getLogger('werkzeug')
werkzeug_log.disabled = True
logging.getLogger('werkzeug').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# ==========================================
# CONFIGURATION
# ==========================================
ANALYSIS_INTERVAL = 3               # controls how often YOLO analyzes a new frame from each camera.
MAX_IMAGES = 3                      # Number of frames to capture per detection session
COOLDOWN = 5.0                      # Unified cooldown: seconds to wait after session ends or reset before starting new one
CAM_THREAD_SLEEP = 0.05             # Seconds between camera frame capture attempts
YOLO_CONFIDENCE = 0.25              # Minimum confidence score (0-1) - lower = more sensitive but more false positives
YOLO_INPUT_SIZE = 480               # Resize frames to 480x480 pixels before YOLO inference
YOLO_IOU = 0.40                     # IoU threshold for Non-Maximum Suppression (overlap removal)
JPEG_QUALITY = 80                   # JPEG compression quality (1-100, higher = better quality but larger files)
WEBHOOK_PORT = 5001                 # Port for Flask webhook server (receives reset signals from recorder)
SESSION_TIMEOUT = 300               # Seconds (5 min) - force reset idle sessions with no activity
WATCHDOG_TIMEOUT = 300              # Seconds (5 min) - force reset sessions stuck in WAITING_RESET state
WATCHDOG_CHECK = 10                 # Seconds between watchdog checks for stuck sessions
RESET_DEDUP_WINDOW = 2              # Seconds to ignore duplicate reset signals from recorder
DRAW_BOUNDING_BOXES = True          # Set to True to draw bounding boxes on uploaded images

# ==========================================
# RATE LIMITED DEBUG LOGGING
# ==========================================
DEBUG_LOG_INTERVAL = 300.0     # Seconds between repeated debug messages
ENABLE_DEBUG_PRINTS = True     # set to False to disable debug logs

# ==========================================
# DEBUG - REJECTED IMAGES
# ==========================================
SAVE_REJECTED_IMAGES = False    # Save images rejected by filters for debugging
REJECTED_IMAGES_DIR = "/home/pi/cctv_rejected"
MAX_REJECTED_IMAGES = 100      # Max number of rejected images to keep (rotates)

# ==========================================
# PERSON VALIDATION FILTERS
# ==========================================
ENABLE_AREA_FILTER = True      # Too small? Probably not a person
ENABLE_ASPECT_FILTER = True    # Wrong shape? Probably not a person

# TOP-EDGE (GROUND) FILTER CONFIGURATION
TOP_EDGE_MARGIN = 20           # Pixels from top edge considered "airborne" (suspicious)
TOP_EDGE_HIGH_CONF = 0.75      # Minimum confidence required to keep top-edge detections
ENABLE_TOP_EDGE_FILTER = True  # Enable/disable the top-edge ground filter

# ==========================================
# GLOBAL DEFAULTS (used if camera not in below dictionaries)
# ==========================================
MIN_PERSON_AREA = 400
MIN_ASPECT_RATIO = 1.2
MAX_ASPECT_RATIO = 4.0

# ==========================================
# CAMERA-SPECIFIC ASPECT RATIO FILTERS
# ==========================================
CAMERA_ASPECT_RATIOS = {
    'Gate': (1.4, 4.0),
    'Center': (1.5, 4.0),
    'Entrance': (1.2, 4.0),
    'Garage': (1.2, 4.0),
    'Behind': (1.2, 4.0),
    'Left': (1.2, 4.0)
}

# ==========================================
# CAMERA-SPECIFIC MIN/MAX AREA THRESHOLDS
# ==========================================
CAMERA_MIN_AREA = {
    'Gate': 1100,
    'Center': 1100,
    'Entrance': 1100,
    'Garage': 1100,
    'Behind': 1100,
    'Left': 1100
}

CAMERA_MAX_AREA = {
    'Gate': 30000,
    'Center': 30000,
    'Entrance': 45000,
    'Garage': 45000,
    'Behind': 25000,
    'Left': 25000,
}

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

def save_rejected_image(camera_name, frame, box, confidence, reason, aspect_ratio=None, area=None, min_value=None, top_y=None, min_conf=None):
    """Save rejected detection image for debugging with clean filename format"""
    if not SAVE_REJECTED_IMAGES:
        return

    try:
        ensure_rejected_dir()

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        debug_frame = frame.copy()
        yellow = (0, 255, 255)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), yellow, 1)

        timestamp = time.strftime('%Y-%m-%d_%H%M%S')

        if reason == "area":
            filename = f"[{timestamp}] _[DEBUG] {camera_name}: Area too small ({area}px < {min_value}px) - box: {width}x{height}px conf={confidence:.2f} REJECTED.jpg"
        elif reason == "area_too_large":
            filename = f"[{timestamp}] _[DEBUG] {camera_name}: Area too large ({area}px > {min_value}px) - box: {width}x{height}px conf={confidence:.2f} REJECTED.jpg"
        elif reason == "aspect":
            filename = f"[{timestamp}] _[DEBUG] {camera_name}: Bad aspect ratio ({aspect_ratio:.2f}) - box: {width}x{height}px conf={confidence:.2f} REJECTED.jpg"
        elif reason == "top_edge":
            filename = f"[{timestamp}] _[DEBUG] {camera_name}: Top-edge rejection - y1={top_y}, conf={confidence:.2f}<{min_conf} - box: {width}x{height}px REJECTED.jpg"
        else:
            filename = f"[{timestamp}] _[DEBUG] {camera_name}: {reason} - box: {width}x{height}px conf={confidence:.2f} REJECTED.jpg"

        filename = filename.replace(" ", "_").replace(":", "-")
        filepath = os.path.join(REJECTED_IMAGES_DIR, filename)

        if reason == "area":
            text = f"REJECTED: Area too small ({area}px < {min_value}px)"
        elif reason == "area_too_large":
            text = f"REJECTED: Area too large ({area}px > {min_value}px)"
        elif reason == "aspect":
            text = f"REJECTED: Bad aspect ratio ({aspect_ratio:.2f})"
        elif reason == "top_edge":
            text = f"REJECTED: Top-edge (y1={top_y}, conf={confidence:.2f}<{min_conf})"
        else:
            text = f"REJECTED: {reason}"

        cv2.putText(debug_frame, text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1)
        cv2.putText(debug_frame, f"Conf: {confidence:.2f}", (x1, y2 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, yellow, 1)

        cv2.imwrite(filepath, debug_frame)

    except Exception as e:
        print(f"[ERROR] save_rejected_image failed for {camera_name}: {e}")

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

camera_states = {n: CameraState() for n in NODES}
upload_executor = ThreadPoolExecutor(max_workers=UPLOAD_WORKERS, thread_name_prefix="upload")

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
    # FIXED: Removed dead code - only use working_frame
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
# YOLO WORKER PROCESS
# ==========================================
def yolo_worker_process(input_q, output_q, min_person_area, min_aspect_ratio, max_aspect_ratio,
                        enable_area_filter, enable_aspect_filter, camera_min_area_dict,
                        camera_max_area_dict, camera_aspect_ratios_dict, yolo_iou, debug_log_interval):

    import threading
    # Separate timers for each filter type (prevents cross-filter suppression)
    _last_area_log_time = {}          # Area too small
    _last_maxarea_log_time = {}       # Area too large (headlights, vehicles)
    _last_aspect_log_time = {}        # Bad aspect ratio
    _last_topedge_log_time = {}       # Top-edge rejection
    _last_topedge_accept_log_time = {} # Top-edge acceptance
    _log_lock = threading.Lock()

    def is_valid_person_detection_worker(camera_name, box, confidence, image_height, frame_id=None, original_frame=None):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        min_area = camera_min_area_dict.get(camera_name, min_person_area)
        max_area = camera_max_area_dict.get(camera_name) if camera_max_area_dict else None
        min_ratio, max_ratio = camera_aspect_ratios_dict.get(camera_name, (min_aspect_ratio, max_aspect_ratio))

        # ==========================================
        # AREA FILTER (too small)
        # ==========================================
        if enable_area_filter and area < min_area:
            with _log_lock:
                now = time.time()
                last_time = _last_area_log_time.get(camera_name, 0)
                if now - last_time >= debug_log_interval and ENABLE_DEBUG_PRINTS:
                    _last_area_log_time[camera_name] = now
                    msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: Area too small ({area}px < {min_area}px) - box: {width}x{height}px REJECTED!"
                    print(msg)

            if original_frame is not None and SAVE_REJECTED_IMAGES:
                save_rejected_image(camera_name, original_frame, box, confidence, "area",
                                  area=area, min_value=min_area)
            return False

        # ==========================================
        # MAX AREA FILTER (too large - headlights, vehicles)
        # ==========================================
        if max_area is not None and area > max_area:
            with _log_lock:
                now = time.time()
                # FIXED: Use separate _last_maxarea_log_time dictionary
                last_time = _last_maxarea_log_time.get(camera_name, 0)
                if now - last_time >= debug_log_interval:
                    _last_maxarea_log_time[camera_name] = now
                    msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: Area too large ({area}px > {max_area}px) - box: {width}x{height}px conf={confidence:.2f} REJECTED!"
                    print(msg)

            if original_frame is not None and SAVE_REJECTED_IMAGES:
                save_rejected_image(camera_name, original_frame, box, confidence, "area_too_large",
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
                    save_rejected_image(camera_name, original_frame, box, confidence, "aspect",
                                      aspect_ratio=aspect_ratio)
                return False

        # ==========================================
        # TOP-EDGE (GROUND) FILTER
        # ==========================================
        if ENABLE_TOP_EDGE_FILTER and image_height is not None:
            top_y = y1
            if top_y <= TOP_EDGE_MARGIN:
                if confidence < TOP_EDGE_HIGH_CONF:
                    with _log_lock:
                        now = time.time()
                        last_time = _last_topedge_log_time.get(camera_name, 0)
                        if now - last_time >= debug_log_interval:
                            _last_topedge_log_time[camera_name] = now
                            msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: Top-edge rejection - y1={top_y}, conf={confidence:.2f}<{TOP_EDGE_HIGH_CONF} REJECTED!"
                            print(msg)

                    if original_frame is not None and SAVE_REJECTED_IMAGES:
                        save_rejected_image(camera_name, original_frame, box, confidence, "top_edge",
                                          top_y=top_y, min_conf=TOP_EDGE_HIGH_CONF)
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
              ENABLE_AREA_FILTER, ENABLE_ASPECT_FILTER, CAMERA_MIN_AREA, CAMERA_MAX_AREA, CAMERA_ASPECT_RATIOS,
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
                      ENABLE_AREA_FILTER, ENABLE_ASPECT_FILTER, CAMERA_MIN_AREA, CAMERA_MAX_AREA, CAMERA_ASPECT_RATIOS,
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
    print(f"MAX_AREA_FILTER: Enabled (per-camera thresholds)")
    print(f"DEBUG_LOG_INTERVAL: {DEBUG_LOG_INTERVAL}s")
    print(f"ENABLE_DEBUG_PRINTS: {ENABLE_DEBUG_PRINTS}")
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
                        print(f"[{ts}] 🆔 {name}: New detection session {state.detection_id}")

                    state.count += 1
                    state.last_processed_count = state.count
                    state.last_activity = capture_time if capture_time > 0 else now
                    current_count = state.count
                    detection_id = state.detection_id
                    conf = detections[0]['conf'] if detections else 0

                    box = detections[0]['box']
                    width = int(box[2] - box[0])
                    height = int(box[3] - box[1])

                    print(f"[{ts}] ⚡ {name}: {state.count}/{MAX_IMAGES} [SID:{detection_id}] conf={conf:.2f} box={width}x{height}px area={width*height}px")

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
        if yolo_process:
            yolo_process.terminate()
            yolo_process.join(timeout=2)
        for s in streams.values():
            s.stop()
        sys.exit(0)
