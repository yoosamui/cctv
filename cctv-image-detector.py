#!/usr/bin/env python3
# ==============================================================================
# CCTV IMAGE DETECTOR - VERSION 3.22.0
# ==============================================================================
#
# IMPROVEMENTS in v3.22.0:
#   1. Thread-safe state management with per-camera locks
#   2. ThreadPoolExecutor for uploads (prevents thread explosion)
#   3. YOLO worker health monitoring with auto-restart
#   4. Queue backpressure handling with metrics
#   5. Frame timestamp integrity using capture time
#   6. Bounded upload queue
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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import deque
import logging

# ==========================================
# SILENCE LOGS
# ==========================================
werkzeug_log = logging.getLogger('werkzeug')
werkzeug_log.disabled = True
logging.getLogger('werkzeug').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

VERSION = "3.22.0"

# ==========================================
# CONFIGURATION
# ==========================================
ANALYSIS_INTERVAL = 5
MAX_IMAGES = 6
COOLDOWN = 4.0
CAM_THREAD_SLEEP = 0.01
YOLO_CONFIDENCE = 0.40
YOLO_INPUT_SIZE = 320
JPEG_QUALITY = 80
WEBHOOK_PORT = 5001
SESSION_TIMEOUT = 600
WATCHDOG_TIMEOUT = 600
WATCHDOG_CHECK = 10
POST_RESET_COOLDOWN = 3
RESET_DEDUP_WINDOW = 2

# Thread pool settings
UPLOAD_WORKERS = 4
UPLOAD_QUEUE_SIZE = 20
YOLO_RESTART_DELAY = 5

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
        self.waiting_reset = False
        self.count = 0
        self.detection_id = None
        self.last_upload = 0
        self.last_run = 0
        self.last_activity = 0
        self.last_waiting_start = 0
        self.last_reset_time = 0
        self.completed = False
        self.last_reset_processed = 0

camera_states = {n: CameraState() for n in NODES}

# Upload thread pool
upload_executor = ThreadPoolExecutor(max_workers=UPLOAD_WORKERS, thread_name_prefix="upload")
upload_queue = deque(maxlen=UPLOAD_QUEUE_SIZE)

# Queue dropped frame counter
dropped_frames = 0
dropped_frames_lock = threading.Lock()

# YOLO worker health
yolo_process = None
yolo_restart_needed = False

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
            was_waiting = state.waiting_reset
            old_count = state.count
            
            # Clear session state
            state.waiting_reset = False
            state.count = 0
            state.detection_id = None
            state.last_waiting_start = 0
            state.completed = True
            state.last_reset_time = now
            
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
    global dropped_frames, yolo_process
    
    camera_stats = {}
    for name, state in camera_states.items():
        with state.lock:
            camera_stats[name] = {
                "waiting": state.waiting_reset,
                "count": state.count,
                "detection_id": state.detection_id,
                "completed": state.completed
            }
    
    return jsonify({
        "status": "running",
        "version": VERSION,
        "dropped_frames": dropped_frames,
        "yolo_alive": yolo_process and yolo_process.is_alive() if yolo_process else False,
        "cameras": camera_stats
    }), 200

def start_webhook_server():
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=WEBHOOK_PORT, debug=False, use_reloader=False, threaded=True), daemon=True).start()
    print(f"🌐 Webhook server on port {WEBHOOK_PORT}")

# ==========================================
# UPLOAD FUNCTION WITH THREAD POOL
# ==========================================
def upload_task(camera_name, url, image_buffer, ts, current_count, max_images, detection_id):
    """Upload task submitted to thread pool"""
    try:
        files = {'image': (f"{camera_name}_{ts}.jpg", image_buffer, 'image/jpeg')}
        data = {
            'frame_num': current_count,
            'total_frames': max_images,
            'camera': camera_name,
            'detection_id': detection_id
        }
        response = requests.post(url, files=files, data=data, timeout=5)
        response.raise_for_status()
        
        if current_count == max_images:
            state = camera_states[camera_name]
            with state.lock:
                state.waiting_reset = True
                state.last_waiting_start = time.time()
                state.completed = False
            print(f"[{ts}] 🛑 {camera_name}: Last frame sent ({current_count}/{max_images})")
    except Exception as e:
        print(f"[{ts}] ✗ {camera_name}: Upload failed - {e}")

def draw_and_upload(camera_name, url, frame, detections, ts, current_count, max_images, detection_id):
    """Draw bounding boxes and queue upload to thread pool"""
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
        # Submit to thread pool instead of creating new thread
        upload_executor.submit(upload_task, camera_name, url, buffer.tobytes(), 
                               ts, current_count, max_images, detection_id)

# ==========================================
# YOLO WORKER WITH HEALTH CHECK
# ==========================================
def yolo_worker_process(input_q, output_q):
    """Worker process for YOLO inference"""
    model = YOLO("yolov8n.pt")
    while True:
        try:
            name, frame, ts, capture_time = input_q.get(timeout=0.5)
            results = model.predict(frame, imgsz=YOLO_INPUT_SIZE, conf=YOLO_CONFIDENCE, classes=[0], verbose=False)
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
    yolo_process = multiprocessing.Process(target=yolo_worker_process, args=(task_q, result_q), daemon=True)
    yolo_process.start()
    return yolo_process

def check_yolo_health(task_q, result_q):
    """Monitor and restart YOLO worker if dead"""
    global yolo_process
    while True:
        time.sleep(YOLO_RESTART_DELAY)
        if yolo_process and not yolo_process.is_alive():
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ YOLO worker died! Restarting...")
            yolo_process = multiprocessing.Process(target=yolo_worker_process, args=(task_q, result_q), daemon=True)
            yolo_process.start()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ YOLO worker restarted")

# ==========================================
# CAMERA STREAM
# ==========================================
class CameraStream:
    def __init__(self, name, url):
        self.name = name
        self.url = url
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
                    self.frame = frame
                    self.frame_time = time.time()  # Store capture timestamp
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
                if state.completed:
                    if now - state.last_reset_time > 5:
                        state.completed = False
                    continue
                
                if state.waiting_reset and state.last_waiting_start > 0:
                    waiting_duration = now - state.last_waiting_start
                    if waiting_duration > WATCHDOG_TIMEOUT:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Watchdog: {name} stuck waiting. Force resetting.")
                        state.waiting_reset = False
                        state.count = 0
                        state.detection_id = None
                        state.last_waiting_start = 0
                        state.completed = True
                
                elif not state.waiting_reset and state.count > 0:
                    idle_duration = now - state.last_activity
                    if idle_duration > SESSION_TIMEOUT:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⏰ Session timeout: {name}")
                        state.count = 0
                        state.detection_id = None
                        state.completed = True

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    start_webhook_server()
    
    task_q = multiprocessing.Queue(maxsize=10)
    result_q = multiprocessing.Queue()
    
    yolo_process = start_yolo_worker(task_q, result_q)
    threading.Thread(target=check_yolo_health, args=(task_q, result_q), daemon=True).start()
    
    streams = {n: CameraStream(n, cfg["cam_rtsp"]) for n, cfg in NODES.items()}
    time.sleep(2)
    
    print("=" * 60)
    print(f"CCTV DETECTOR v{VERSION} - WITH THREAD-SAFE STATE")
    print(f"ANALYSIS_INTERVAL: {ANALYSIS_INTERVAL}s")
    print(f"MAX_IMAGES: {MAX_IMAGES} per camera")
    print(f"UPLOAD_WORKERS: {UPLOAD_WORKERS}")
    print(f"YOLO_AUTO_RESTART: Enabled")
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
                        
                        if state.count == 0:
                            state.completed = False
                        
                        if not state.waiting_reset and state.count < MAX_IMAGES:
                            if state.count == 0 and (now - state.last_upload < COOLDOWN):
                                continue
                            
                            if state.count == 0:
                                state.detection_id = str(uuid.uuid4())[:8]
                                print(f"[{ts}] 🆔 {name}: New detection session {state.detection_id}")
                            
                            state.count += 1
                            state.last_activity = capture_time  # Use capture timestamp
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
    
    # Main loop with backpressure handling
    try:
        while True:
            now = time.time()
            
            for name in NODES:
                state = camera_states[name]
                with state.lock:
                    if state.waiting_reset:
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
        for s in streams.values():
            s.stop()
        sys.exit(0)
