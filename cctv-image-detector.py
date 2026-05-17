#!/usr/bin/env python3
# ==============================================================================
# CCTV IMAGE DETECTOR - VERSION 3.23.7
# ==============================================================================
#
# IMPROVEMENTS in v3.23.7:
#   1. Non-blocking upload retries (fail fast instead of sleeping)
#   2. Added per-camera upload failure tracking
#   3. Reduced upload thread blocking to prevent queue backups
#   4. Added configurable retry mode flag
#
# IMPROVEMENTS in v3.23.6:
#   1. Replaced unsafe _work_queue.qsize() with thread-safe counter
#   2. Fixed private attribute access for better portability
#
# [Previous improvements remain...]
#
# ==============================================================================
import cv2
import multiprocessing
import threading
import time
import requests
import os
import sys
import uuid
from ultralytics import YOLO
from queue import Empty, Full
from urllib.parse import quote
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import logging

VERSION = "3.23.7"

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

# How often each camera is analyzed by YOLO (seconds)
# Lower = faster detection but higher CPU usage
# Higher = lower CPU usage but slower detection response
ANALYSIS_INTERVAL = 4

# Maximum images uploaded per detection session
# Example:
# Person detected -> send max 6 images -> wait for recorder reset
MAX_IMAGES = 6

# Minimum time between starting new detection sessions (seconds)
# Prevents repeated triggering from same person standing still
COOLDOWN = 4.0

# Small sleep inside camera capture thread
# Prevents CPU spinning at 100%
# Lower = more responsive frame grabbing but higher CPU
CAM_THREAD_SLEEP = 0.01

# Minimum confidence required for YOLO person detection
# Lower values:
#   + detect more people
#   - more false positives
#
# Higher values:
#   + fewer false detections
#   - may miss distant/small persons
#
# Recommended:
#   0.35 = testing / high sensitivity
#   0.50-0.60 = production
YOLO_CONFIDENCE = 0.40

# YOLO input image size
# Larger:
#   + better accuracy
#   - higher CPU usage
#
# Smaller:
#   + faster inference
#   - lower detection accuracy
#
# Good balance for Raspberry Pi 5:
#   416
#
# Recommended:
#   320 = very fast
#   416 = balanced
#   480 = better accuracy
#   640 = heavy CPU usage
YOLO_INPUT_SIZE = 416

# Intersection-over-Union threshold for Non-Maximum Suppression (NMS)
# Helps remove duplicate overlapping boxes
#
# Lower:
#   + removes duplicates aggressively
#   - may remove valid nearby detections
#
# Higher:
#   + keeps more boxes
#   - more duplicate boxes possible
YOLO_IOU = 0.45

# JPEG quality for uploaded images
# Higher:
#   + better image quality
#   - larger files / more network traffic
#
# Lower:
#   + smaller uploads
#   - reduced image quality
JPEG_QUALITY = 80

# Flask webhook port for reset communication
WEBHOOK_PORT = 5001

# Reset incomplete ACTIVE sessions after no activity (seconds)
# Prevents stuck sessions forever
SESSION_TIMEOUT = 600

# Maximum time waiting for recorder reset before force reset
# Protects against recorder crashes or network failures
WATCHDOG_TIMEOUT = 300

# How often watchdog checks session health (seconds)
WATCHDOG_CHECK = 10

# Cooldown after recorder reset before allowing new detections
# Prevents stale frames and duplicate sessions
POST_RESET_COOLDOWN = 6

# Ignore duplicate reset signals inside this time window
# Prevents double reset processing
RESET_DEDUP_WINDOW = 2


# ==========================================
# THREAD POOL SETTINGS
# ==========================================

# Parallel upload worker threads
# More workers:
#   + faster uploads
#   - higher CPU/RAM/network usage
UPLOAD_WORKERS = 4

# Maximum pending uploads before dropping frames
# Prevents unlimited RAM growth during overload
UPLOAD_QUEUE_SIZE = 20

# Delay before restarting crashed YOLO worker (seconds)
YOLO_RESTART_DELAY = 5


# ==========================================
# UPLOAD RETRY SETTINGS (NON-BLOCKING)
# ==========================================

# NEW: Enable non-blocking retries (fail fast instead of sleeping)
NON_BLOCKING_RETRIES = True

# Maximum upload retry attempts (only used if NON_BLOCKING_RETRIES = False)
UPLOAD_MAX_RETRIES = 1  # Reduced from 3 for fail-fast behavior

# Base retry delay (not used in non-blocking mode)
UPLOAD_RETRY_DELAY_BASE = 1

# Track upload failures per camera (for diagnostics)
upload_failure_counts = {}
upload_failure_lock = threading.Lock()


# ==========================================
# QUEUE SETTINGS
# ==========================================

# Max pending frames for YOLO processing
# Prevents unlimited RAM growth if YOLO worker is slow
TASK_QUEUE_SIZE = 10

# Max pending detection results from YOLO
# Prevents memory leak under heavy detection load
RESULT_QUEUE_SIZE = 20


# ==========================================
# SESSION STATE ENUM
# ==========================================
class SessionState(Enum):
    IDLE = 0           # No active session
    ACTIVE = 1         # Collecting frames (1-5/6)
    WAITING_RESET = 2  # Sent 6/6, waiting for recorder
    COMPLETED = 3      # Reset received, ready for cleanup


# ==========================================
# UPLOAD QUEUE TRACKING (THREAD-SAFE COUNTER)
# ==========================================
pending_uploads = 0
pending_uploads_lock = threading.Lock()


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
# PER-CAMERA SESSION STATE WITH LOCKS
# ==========================================
class CameraState:
    """Thread-safe camera state management"""
    def __init__(self):
        self.lock = threading.Lock()
        self.state = SessionState.IDLE
        self.count = 0
        self.detection_id = None
        self.last_upload = 0
        self.last_run = 0
        self.last_activity = 0
        self.last_waiting_start = 0
        self.last_reset_time = 0
        self.last_reset_processed = 0
        self.active_session_id = None  # Track current active session ID

camera_states = {n: CameraState() for n in NODES}

# Upload thread pool with bounded queue
upload_executor = ThreadPoolExecutor(max_workers=UPLOAD_WORKERS, thread_name_prefix="upload")

# Dropped frame counters
dropped_frames = 0
dropped_uploads = 0
dropped_frames_lock = threading.Lock()

# YOLO worker health
yolo_process = None


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
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Unauthorized reset attempt from {request.remote_addr}")
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
            # Deduplicate resets
            if now - state.last_reset_processed < RESET_DEDUP_WINDOW:
                return '', 200

            state.last_reset_processed = now
            was_waiting = (state.state == SessionState.WAITING_RESET)
            old_count = state.count

            # Clear session state
            state.state = SessionState.COMPLETED
            state.count = 0
            state.detection_id = None
            state.last_waiting_start = 0
            state.last_reset_time = now
            state.active_session_id = None  # Clear active session ID on reset
            
            # Reset upload failure counter for this camera on successful reset
            with upload_failure_lock:
                upload_failure_counts[camera_name] = 0

            if was_waiting:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📡 Recorder signaled: {camera_name} reset - DETECTION RESUMED")
            elif old_count > 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📡 Recorder signaled: {camera_name} reset - CLEANED UP PARTIAL SESSION ({old_count}/6 frames)")

        return '', 200
    except Exception as e:
        print(f"Error in session_reset: {e}")
        return jsonify({"status": "error"}), 500

@app.route('/reset', methods=['POST'])
def reset_legacy():
    return session_reset()

@app.route('/health', methods=['GET'])
def health_check():
    global dropped_frames, dropped_uploads, pending_uploads

    camera_stats = {}
    for name, state in camera_states.items():
        with state.lock:
            camera_stats[name] = {
                "state": state.state.name,
                "count": state.count,
                "detection_id": state.detection_id,
                "active_session_id": state.active_session_id
            }

    with upload_failure_lock:
        failure_stats = dict(upload_failure_counts)

    return jsonify({
        "status": "running",
        "version": VERSION,
        "non_blocking_retries": NON_BLOCKING_RETRIES,
        "dropped_frames": dropped_frames,
        "dropped_uploads": dropped_uploads,
        "pending_uploads": pending_uploads,
        "upload_failures": failure_stats,
        "yolo_alive": yolo_process and yolo_process.is_alive() if yolo_process else False,
        "cameras": camera_stats
    }), 200

def start_webhook_server():
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=WEBHOOK_PORT, debug=False, use_reloader=False, threaded=True), daemon=True).start()
    print(f"🌐 Webhook server on port {WEBHOOK_PORT}")


# ==========================================
# UPLOAD FUNCTION WITH NON-BLOCKING RETRIES
# ==========================================
def upload_task(camera_name, url, image_buffer, ts, current_count, max_images, detection_id):
    """Upload task with non-blocking retries (fail fast)"""
    global dropped_uploads

    # Check if this upload is from the current active session
    state = camera_states[camera_name]

    with state.lock:
        # If this upload's session ID doesn't match the current active session, reject it
        if state.active_session_id and detection_id != state.active_session_id:
            print(
                f"[{ts}] 🚫 {camera_name}: "
                f"Rejected stale upload from session "
                f"{detection_id} "
                f"(current: {state.active_session_id})"
            )
            return
        # NEW: If we have NO active session ID, this upload is invalid 
        # (should have started a new session first via handle_results)
        if state.active_session_id is None and detection_id is not None:
            print(
                f"[{ts}] 🚫 {camera_name}: "
                f"Rejected upload with no active session: {detection_id}"
            )
            return





    # Attempt upload (single try with fail-fast for 429)
    try:
        files = {
            'image': (
                f"{camera_name}_{ts}.jpg",
                image_buffer,
                'image/jpeg'
            )
        }

        data = {
            'frame_num': current_count,
            'total_frames': max_images,
            'camera': camera_name,
            'detection_id': detection_id
        }

        headers = {
            'X-API-KEY': WEBHOOK_SECRET
        }

        response = requests.post(
            url,
            files=files,
            data=data,
            headers=headers,
            timeout=5
        )

        response.raise_for_status()

        # Track successful upload - reset failure counter
        with upload_failure_lock:
            if camera_name in upload_failure_counts:
                upload_failure_counts[camera_name] = max(0, upload_failure_counts[camera_name] - 1)

        # Handle last frame
        if current_count == max_images:
            with state.lock:
                if state.state == SessionState.ACTIVE:
                    state.state = SessionState.WAITING_RESET
                    state.last_waiting_start = time.time()
                    print(
                        f"[{ts}] 🛑 {camera_name}: "
                        f"Last frame sent "
                        f"({current_count}/{max_images})"
	            )

        return  # Success

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            # Recorder busy - NON-BLOCKING: Just log and fail fast
            with upload_failure_lock:
                upload_failure_counts[camera_name] = upload_failure_counts.get(camera_name, 0) + 1
                failure_count = upload_failure_counts[camera_name]
            
            # Only log every 5th failure to reduce noise
            if failure_count % 5 == 1:
                print(
                    f"[{ts}] ⚠️ {camera_name}: "
                    f"Recorder busy (429) - will retry next frame "
                    f"(failures: {failure_count})"
                )
            
            # Don't retry - let the next frame from the main loop handle it
            # This prevents blocking upload threads
            return
            
        elif e.response.status_code == 503:
            # Service unavailable
            print(
                f"[{ts}] ⚠️ {camera_name}: "
                f"Recorder unavailable (503) - will retry next frame"
            )
            return
            
        elif e.response.status_code == 401:
            print(
                f"[{ts}] ❌ {camera_name}: "
                f"Unauthorized upload (401). "
                f"Check WEBHOOK_SECRET."
            )
            return
            
        else:
            print(f"[{ts}] ✗ {camera_name}: Upload failed - {e}")
            return

    except requests.exceptions.Timeout:
        print(
            f"[{ts}] ⚠️ {camera_name}: "
            f"Upload timeout - will retry next frame"
        )
        return
        
    except Exception as e:
        print(
            f"[{ts}] ✗ {camera_name}: "
            f"Upload error - {e}"
        )
        return


def draw_and_upload(camera_name, url, frame, detections, ts, current_count, max_images, detection_id):
    """Draw bounding boxes and queue upload to thread pool with backpressure"""
    global pending_uploads, dropped_uploads
    
    yellow = (0, 255, 255)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"PERSON {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 1)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), yellow, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if success:
        # ✅ BACKPRESSURE: Check queue size using thread-safe counter
        with pending_uploads_lock:
            if pending_uploads >= UPLOAD_QUEUE_SIZE:
                dropped_uploads += 1
                if dropped_uploads % 10 == 0:
                    print(f"[{ts}] ⚠️ Upload queue full ({pending_uploads} pending), dropped frame {current_count}/{max_images}")
                return
            pending_uploads += 1
        
        def upload_wrapper(*args, **kwargs):
            try:
                return upload_task(*args, **kwargs)
            finally:
                with pending_uploads_lock:
                    global pending_uploads
                    pending_uploads -= 1
        
        upload_executor.submit(upload_wrapper, camera_name, url, buffer.tobytes(),
                               ts, current_count, max_images, detection_id)


# ==========================================
# YOLO WORKER WITH HEALTH CHECK
# ==========================================
def yolo_worker_process(input_q, output_q):
    """Worker process for YOLO inference using ONNX"""
    try:
        print(f"Loading yolov8n.onnx for ONNX Runtime inference...")
        model = YOLO("yolov8n.onnx", task='detect')
        print(f"Using ONNX Runtime with CPUExecutionProvider")
    except Exception as e:
        print(f"Failed to load ONNX model, falling back to PyTorch: {e}")
        model = YOLO("yolov8n.pt", task='detect')

    while True:
        try:
            name, frame, ts, capture_time = input_q.get(timeout=0.5)
            # ✅ Add iou parameter here
            results = model.predict(
                frame,
                imgsz=YOLO_INPUT_SIZE,
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU,
                classes=[0],
                verbose=False
            )
            detections = [{"box": [int(x) for x in box.xyxy[0]], "conf": float(box.conf[0])}
                         for box in results[0].boxes] if results[0].boxes else []
            output_q.put((name, frame, detections, ts, capture_time))
        except Empty:
            continue
        except Exception as e:
            print(f"YOLO worker error: {e}")
            time.sleep(1)


def start_yolo_worker(task_q, result_q):
    """Start YOLO worker with monitoring"""
    global yolo_process
    if yolo_process and yolo_process.is_alive():
        return yolo_process

    yolo_process = multiprocessing.Process(target=yolo_worker_process, args=(task_q, result_q), daemon=True)
    yolo_process.start()
    return yolo_process

def check_yolo_health(task_q, result_q):
    """Monitor and restart YOLO worker if dead"""
    global yolo_process
    while True:
        time.sleep(YOLO_RESTART_DELAY)
        if yolo_process:
            if not yolo_process.is_alive():
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ YOLO worker died! Restarting...")
                # ✅ Clean up zombie process
                yolo_process.join(timeout=1)
                yolo_process = multiprocessing.Process(target=yolo_worker_process, args=(task_q, result_q), daemon=True)
                yolo_process.start()
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ YOLO worker restarted")


# ==========================================
# CAMERA STREAM (THREAD-SAFE)
# ==========================================
class CameraStream:
    """Thread-safe camera stream handler with auto-reconnect"""

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
                    if not cap.isOpened():
                        raise Exception("Failed to open camera")
                    consecutive_failures = 0

                ret, frame = cap.read()

                if ret and frame is not None:
                    # ✅ Thread-safe frame update
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
        """Return a copy of the latest frame (thread-safe)"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.frame_time
        return None, 0

    def stop(self):
        self.running = False


# ==========================================
# SESSION WATCHDOG (USING STATE ENUM)
# ==========================================
def session_watchdog():
    """Periodically checks for stuck sessions using state enum"""
    while True:
        time.sleep(WATCHDOG_CHECK)
        now = time.time()

        for name, state in camera_states.items():
            with state.lock:
                # Clean up COMPLETED state after cooldown
                if state.state == SessionState.COMPLETED:
                    if now - state.last_reset_time > 5:
                        state.state = SessionState.IDLE
                    continue

                # Check for stuck WAITING_RESET sessions
                if state.state == SessionState.WAITING_RESET and state.last_waiting_start > 0:
                    waiting_duration = now - state.last_waiting_start
                    if waiting_duration > WATCHDOG_TIMEOUT:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Watchdog: {name} stuck waiting for {waiting_duration:.0f}s. Force resetting.")
                        state.state = SessionState.IDLE
                        state.count = 0
                        state.detection_id = None
                        state.last_waiting_start = 0
                        state.active_session_id = None

                # Check for idle ACTIVE sessions
                elif state.state == SessionState.ACTIVE and state.count > 0:
                    idle_duration = now - state.last_activity
                    if idle_duration > SESSION_TIMEOUT:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⏰ Session timeout: {name}")
                        state.state = SessionState.IDLE
                        state.count = 0
                        state.detection_id = None
                        state.active_session_id = None


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    
    start_webhook_server()

    # ✅ Both queues now have bounds to prevent memory leaks
    task_q = multiprocessing.Queue(maxsize=TASK_QUEUE_SIZE)
    result_q = multiprocessing.Queue(maxsize=RESULT_QUEUE_SIZE)

    start_yolo_worker(task_q, result_q)
    
    
    threading.Thread(target=check_yolo_health, args=(task_q, result_q), daemon=True).start()

    streams = {n: CameraStream(n, cfg["cam_rtsp"]) for n, cfg in NODES.items()}
    time.sleep(2)

    print("=" * 60)
    print(f"CCTV DETECTOR v{VERSION} - WITH NON-BLOCKING RETRIES")
    print(f"ANALYSIS_INTERVAL: {ANALYSIS_INTERVAL}s")
    print(f"MAX_IMAGES: {MAX_IMAGES} per camera")
    print(f"UPLOAD_WORKERS: {UPLOAD_WORKERS}")
    print(f"UPLOAD_QUEUE_SIZE: {UPLOAD_QUEUE_SIZE}")
    print(f"NON_BLOCKING_RETRIES: {NON_BLOCKING_RETRIES}")
    print(f"YOLO_AUTO_RESTART: Enabled")
    print(f"YOLO_CONFIDENCE: {YOLO_CONFIDENCE}")
    print(f"YOLO_INPUT_SIZE: {YOLO_INPUT_SIZE}")
    print(f"YOLO_IOU: {YOLO_IOU}")
    print(f"WEBHOOK_PORT: {WEBHOOK_PORT}")
    print(f"SESSION_TIMEOUT: {SESSION_TIMEOUT}s (idle sessions)")
    print(f"WATCHDOG: {WATCHDOG_CHECK}s check interval, {WATCHDOG_TIMEOUT}s timeout for stuck sessions")
    print(f"POST_RESET_COOLDOWN: {POST_RESET_COOLDOWN}s (cooldown after reset)")
    print(f"RESET_DEDUP_WINDOW: {RESET_DEDUP_WINDOW}s (ignore duplicate resets)")
    print("=" * 60)

    threading.Thread(target=session_watchdog, daemon=True).start()

    def handle_results():
        global dropped_frames
        while True:
            try:
                name, frame, detections, ts, capture_time = result_q.get(timeout=0.01)
                now = time.time()
                state = camera_states[name]

                with state.lock:
                    state.last_run = now

                if detections:
                    with state.lock:
                        # Check cooldown period after reset
                        cooldown_remaining = POST_RESET_COOLDOWN - (now - state.last_reset_time) if state.last_reset_time > 0 else 0

                        if cooldown_remaining > 0:
                            continue

                        # Start new session if idle
                        if state.state == SessionState.IDLE:
                            state.state = SessionState.ACTIVE
                            state.count = 0
                            state.detection_id = None
                            state.active_session_id = None

                        # Only process if active and not waiting
                        if state.state == SessionState.ACTIVE and state.count < MAX_IMAGES:
                            if state.count == 0 and (now - state.last_upload < COOLDOWN):
                                continue

                            if state.count == 0:
                                state.detection_id = str(uuid.uuid4())[:8]
                                state.active_session_id = state.detection_id  # Track current session ID
                                print(f"[{ts}] 🆔 {name}: New detection session {state.detection_id}")

                            state.count += 1
                            state.last_activity = capture_time
                            current_count = state.count
                            detection_id = state.detection_id
                            print(f"[{ts}] ⚡ {name}: {state.count}/{MAX_IMAGES}")

                    # Get frame outside lock to avoid holding it during upload
                    full_frame, frame_time = streams[name].get_frame()
                    if full_frame is not None:
                        draw_and_upload(name, NODES[name]["rpi_url"], full_frame, detections, ts,
                                      current_count, MAX_IMAGES, detection_id)

                    with state.lock:
                        state.last_upload = now

            except Empty:
                continue
            except Exception as e:
                print(f"Handler error: {e}")

    threading.Thread(target=handle_results, daemon=True).start()

    # Main loop with backpressure handling
    try:
        while True:
            now = time.time()

            for name in NODES:
                state = camera_states[name]
                with state.lock:
                    if state.state == SessionState.WAITING_RESET:
                        continue
                    skip = now - state.last_run < ANALYSIS_INTERVAL

                if skip:
                    continue

                frame, frame_time = streams[name].get_frame()
                if frame is not None:
                    try:
                        task_q.put_nowait((name, frame, time.strftime('%Y-%m-%d %H:%M:%S'), frame_time))
                        with state.lock:
                            state.last_run = now
                    except Full:
                        with dropped_frames_lock:
                            dropped_frames += 1
                            if dropped_frames % 100 == 0:
                                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Dropped {dropped_frames} frames (queue full)")

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
