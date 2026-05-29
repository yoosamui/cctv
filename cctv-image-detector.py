#!/usr/bin/env python3
# ==============================================================================
# CCTV IMAGE DETECTOR - VERSION 3.23.9
# ==============================================================================
#
# IMPROVEMENTS in v3.23.9:
#   1. Fixed YOLO worker process to receive configuration values from main process
#   2. Removed hardcoded validation values in worker
#   3. Added camera-specific thresholds passed to worker
#   4. Lowered default MIN_PERSON_AREA to 1500 for better night detection
#   5. Lowered MIN_ASPECT_RATIO to 0.8 for high-mounted cameras
#
# AUTHOR: yoosamui
# DATE: 2026-05-29
# ==============================================================================
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
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import logging

VERSION = "3.23.9"

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
#controls how often YOLO analyzes a new frame from each camera.
ANALYSIS_INTERVAL = 2.5

MAX_IMAGES = 3
COOLDOWN = 4.0
CAM_THREAD_SLEEP = 0.05
YOLO_CONFIDENCE = 0.55
YOLO_INPUT_SIZE = 480
YOLO_IOU = 0.40
JPEG_QUALITY = 80
WEBHOOK_PORT = 5001
SESSION_TIMEOUT = 600
WATCHDOG_TIMEOUT = 300
WATCHDOG_CHECK = 10
POST_RESET_COOLDOWN = 6
RESET_DEDUP_WINDOW = 2

# ==========================================
# PERSON VALIDATION FILTERS
# ==========================================
ENABLE_AREA_FILTER = True
ENABLE_ASPECT_FILTER = True

# ==========================================
# GLOBAL DEFAULTS (used if camera not in below dictionaries)
# ==========================================
MIN_PERSON_AREA = 400  # (global fallback if camera not in CAMERA_MIN_AREA)
MIN_ASPECT_RATIO = 1.2 # (global fallback if camera not in CAMERA_ASPECT_RATIOS)
MAX_ASPECT_RATIO = 4.0

# ==========================================
# CAMERA-SPECIFIC ASPECT RATIO FILTERS
# ==========================================
# Aspect ratio = height / width
#
# Values > 1.0 = object is taller than wide (normal person standing)
# Values < 1.0 = object is wider than tall (person lying down, high camera angle, or false positive)
#
# Format: (min_ratio, max_ratio)
# - min_ratio: Minimum allowed height/width ratio (1.2 = person must be at least 20% taller than wide)
# - max_ratio: Maximum allowed height/width ratio (4.0 = very tall/thin person)
#
# Why 1.2? A standing person typically has height/width between 1.5 and 3.5
# Setting min to 1.2 filters out most false positives (shadows, cars, bushes)
# while still allowing people who are sitting, crouching, or at odd angles
#
CAMERA_ASPECT_RATIOS = {
    'Gate': (1.2, 4.0),
    'Center': (1.2, 4.0),
    'Entrance': (1.2, 4.0),
    'Garage': (1.2, 4.0),
    'Behind': (1.2, 4.0),
    'Left': (1.2, 4.0)
}

# ==========================================
# CAMERA-SPECIFIC MINIMUM AREA THRESHOLDS (pixels)
# ==========================================
# Minimum bounding box area required for a valid person detection
#
# How to calculate: width × height of the person in the frame
# Examples:
# - Person close to camera: 200×500 = 100,000 pixels
# - Person medium distance: 100×250 = 25,000 pixels
# - Person far away: 40×100 = 4,000 pixels
# - Person very far/partial: 20×50 = 1,000 pixels
#
# Lower values = detect smaller/distant people (but more false positives)
# Higher values = ignore smaller objects (but may miss distant people)
#
# These values (300-400) are optimized for night detection where bounding boxes are smaller
# Daytime can use higher values (1000-2000) if needed
#
CAMERA_MIN_AREA = {
    'Gate': 400,
    'Center': 350,
    'Entrance': 400,
    'Garage': 300,
    'Behind': 400,
    'Left': 350
}
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
# GLOBALS
# ==========================================
pending_uploads = 0
pending_uploads_lock = threading.Lock()
dropped_frames = 0
dropped_uploads = 0
dropped_frames_lock = threading.Lock()
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
        self.last_run = 0
        self.last_activity = 0
        self.last_waiting_start = 0
        self.last_reset_time = 0
        self.last_reset_processed = 0
        self.active_session_id = None

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
            state.state = SessionState.COMPLETED
            state.count = 0
            state.detection_id = None
            state.last_waiting_start = 0
            state.last_reset_time = now
            state.active_session_id = None
            if was_waiting:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📡 Recorder signaled: {camera_name} reset - DETECTION RESUMED")
            elif old_count > 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📡 Recorder signaled: {camera_name} reset - CLEANED UP PARTIAL SESSION ({old_count}/{MAX_IMAGES} frames)")

        return '', 200
    except Exception as e:
        print(f"Error in session_reset: {e}")
        return jsonify({"status": "error"}), 500

@app.route('/reset', methods=['POST'])
def reset_legacy():
    return session_reset()

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
    for attempt in range(max_retries):
        try:
            files = {'image': (f"{camera_name}_{ts}.jpg", image_buffer, 'image/jpeg')}
            data = {'frame_num': current_count, 'total_frames': max_images, 'camera': camera_name, 'detection_id': detection_id}
            headers = {'X-API-KEY': WEBHOOK_SECRET}
            response = requests.post(url, files=files, data=data, headers=headers, timeout=5)
            response.raise_for_status()
            if current_count == max_images:
                with state.lock:
                    if state.state == SessionState.ACTIVE:
                        state.state = SessionState.WAITING_RESET
                        state.last_waiting_start = time.time()
                print(f"[{ts}] 🛑 {camera_name}: Last frame sent ({current_count}/{max_images})")
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
    global pending_uploads, dropped_uploads
    yellow = (0, 255, 255)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"PERSON {d['conf']:.2f}"
       # cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 1)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), yellow, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if success:
        with pending_uploads_lock:
            if pending_uploads >= UPLOAD_QUEUE_SIZE:
                dropped_uploads += 1
                return
            pending_uploads += 1
        def upload_wrapper(*args, **kwargs):
            try:
                return upload_task(*args, **kwargs)
            finally:
                with pending_uploads_lock:
                    global pending_uploads
                    pending_uploads -= 1
        upload_executor.submit(upload_wrapper, camera_name, url, buffer.tobytes(), ts, current_count, max_images, detection_id)

# ==========================================
# YOLO WORKER PROCESS (FIXED)
# ==========================================
def yolo_worker_process(input_q, output_q, min_person_area, min_aspect_ratio, max_aspect_ratio,
                        enable_area_filter, enable_aspect_filter, camera_min_area_dict,
                        camera_aspect_ratios_dict):

    # Track last logged frame per camera (for duplicate suppression)
    last_logged_frame = {}

    def is_valid_person_detection_worker(camera_name, box, confidence, frame_id=None):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        min_area = camera_min_area_dict.get(camera_name, min_person_area)
        min_ratio, max_ratio = camera_aspect_ratios_dict.get(camera_name, (min_aspect_ratio, max_aspect_ratio))

        # AREA FILTER
        if enable_area_filter and area < min_area:
            if frame_id and last_logged_frame.get(camera_name) != frame_id:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: Area too small ({area}px < {min_area}px)")
                last_logged_frame[camera_name] = frame_id
            return False

        # ASPECT RATIO FILTER
        if enable_aspect_filter and width > 0:
            aspect_ratio = height / width
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                if frame_id and last_logged_frame.get(camera_name) != frame_id:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: Bad aspect ratio ({aspect_ratio:.2f})")
                    last_logged_frame[camera_name] = frame_id
                return False

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

            # Create unique frame ID for duplicate suppression
            frame_id = f"{name}_{capture_time}"

            h, w = frame.shape[:2]
            scale_x = w / YOLO_INPUT_SIZE
            scale_y = h / YOLO_INPUT_SIZE

            # Preprocessing
            img = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1).reshape(1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)

            # Inference
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

                # Pass frame_id to validation function
                if is_valid_person_detection_worker(name, box, confidence, frame_id):
                    detections.append({"box": box, "conf": float(confidence)})

            # ==========================================
            # NMS - Merge overlapping boxes (ALWAYS RUN)
            # ==========================================
            if len(detections) > 0:
                # Sort by confidence (highest first)
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
                            # If >50% overlap, keep only the one with higher confidence
                            if overlap / area1 > 0.5:
                                keep = False
                                break
                    
                    if keep:
                        filtered.append(d1)
                
                detections = filtered

            output_q.put((name, frame, detections, ts, capture_time))

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
              ENABLE_AREA_FILTER, ENABLE_ASPECT_FILTER, CAMERA_MIN_AREA, CAMERA_ASPECT_RATIOS),
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
                      ENABLE_AREA_FILTER, ENABLE_ASPECT_FILTER, CAMERA_MIN_AREA, CAMERA_ASPECT_RATIOS),
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
                elif state.state == SessionState.ACTIVE and state.count > 0:
                    if now - state.last_activity > SESSION_TIMEOUT:
                        print(f"⏰ Session timeout: {name}")
                        state.state = SessionState.IDLE
                        state.count = 0
                        state.active_session_id = None

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
    print(f"POST_RESET_COOLDOWN: {POST_RESET_COOLDOWN}s (cooldown after reset)")
    print(f"RESET_DEDUP_WINDOW: {RESET_DEDUP_WINDOW}s (ignore duplicate resets)")
    print(f"MIN_PERSON_AREA: {MIN_PERSON_AREA}")
    print(f"MIN_ASPECT_RATIO: {MIN_ASPECT_RATIO}")
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
                        cooldown_remaining = POST_RESET_COOLDOWN - (now - state.last_reset_time) if state.last_reset_time > 0 else 0
                        if cooldown_remaining > 0:
                            continue
                        if state.state == SessionState.IDLE:
                            state.state = SessionState.ACTIVE
                            state.count = 0
                            state.active_session_id = None
                        if state.state == SessionState.ACTIVE and state.count < MAX_IMAGES:
                            if state.count == 0 and (now - state.last_upload < COOLDOWN):
                                continue
                            if state.count == 0:
                                state.detection_id = str(uuid.uuid4())[:8]
                                state.active_session_id = state.detection_id
                                print(f"[{ts}] 🆔 {name}: New detection session {state.detection_id}")
                            state.count += 1
                            state.last_activity = capture_time
                            current_count = state.count
                            detection_id = state.detection_id
                            print(f"[{ts}] ⚡ {name}: {state.count}/{MAX_IMAGES}")

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
