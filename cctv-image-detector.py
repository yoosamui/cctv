#!/usr/bin/env python3
# ==============================================================================
# CCTV IMAGE DETECTOR - VERSION 3.21.0
# ==============================================================================
#
# WHAT THIS DOES:
#   Detects people in 6 CCTV camera streams using YOLOv8
#   Sends annotated images to Raspberry Pi recorders
#   Receives reset signals when recorders finish saving
#
# HOW IT WORKS:
#   1. CameraStream threads capture frames from RTSP streams
#   2. YOLO worker process detects people (parallel inference)
#   3. When person detected -> upload 6 frames to RPi recorder
#   4. After 6th frame -> wait for recorder to signal completion
#   5. Recorder signals back via HTTP POST -> reset session
#
# ARCHITECTURE:
#   [6 Cameras] -> [Detector RPI 5] -> [6 RPis 4B] -> [Save Images]
#                      ↑                      │
#                      └─── Reset Signal ─────┘
#                      └───> send images ─────┘
#
# SETTINGS (Adjust these based on your needs):
#   ANALYSIS_INTERVAL = 2.5   Seconds between camera checks (1-5)
#   MAX_IMAGES = 6            Frames per detection event (3-10)
#   COOLDOWN = 4.0            Seconds between frames (2-6)
#   YOLO_CONFIDENCE = 0.40    Detection confidence (0.25 low, 0.50 high)
#   YOLO_INPUT_SIZE = 320     Image size for AI (160 fast, 320 balanced, 640 accurate)
#   SESSION_TIMEOUT = 30      Seconds to wait for recorder reset before forcing reset
# 
# DEPENDENCIES:
#   pip install ultralytics opencv-python flask requests python-dotenv
#
# USAGE:
#   python3 cctv_detector.py
#
# AUTHOR: yoosamui
# DATE: 2026-05-13
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
from queue import Empty
from urllib.parse import quote
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# ==========================================
# COMPLETELY SILENCE FLASK ACCESS LOGS
# ==========================================
import logging

# Disable all Werkzeug/Flask logging
werkzeug_log = logging.getLogger('werkzeug')
werkzeug_log.disabled = True

# Or alternatively, set to critical (only shows fatal errors)
logging.getLogger('werkzeug').setLevel(logging.CRITICAL)

# Also silence requests library logs
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

VERSION = "3.21.0"

# Suppress FFmpeg warnings
#os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
#os.environ['FFMPEG_LOG_LEVEL'] = 'panic'

# ==========================================
# CONFIGURATION
#
# SESSION_TIMEOUT = 600 # important
# You were moving, so each camera only saw you for a moment - just long enough for 1 frame before you walked out of frame.
# This is NORMAL and EXPECTED behavior!
#
# The system is working correctly:
#
# -  You walk past a camera → YOLO detects you
# -  Session starts (1/6 frames)
# -  You leave the frame → No more detections
# -  Watchdog waits 5 minutes, then cleans up the incomplete session
#
# The watchdog messages are NOT errors - they're information!
#
# ==========================================
ANALYSIS_INTERVAL = 5        # How often to check cameras (seconds)
MAX_IMAGES = 6               # Maximum images per detection event
COOLDOWN = 4.0               # Seconds between frames in a session
CAM_THREAD_SLEEP = 0.01      # Camera capture rate (seconds)
YOLO_CONFIDENCE = 0.40       # Detection confidence threshold
YOLO_INPUT_SIZE = 320        # Image size for YOLO (pixels)
JPEG_QUALITY = 80            # Image quality (0-100)
WEBHOOK_PORT = 5001          # Port for receiving reset signals

			     # This will prevent the watchdog from resetting sessions that are legitimately waiting for the recorder's 5-minute segment to complete.
SESSION_TIMEOUT = 600        # Seconds of inactivity before session timeout. 5 minutes - enough time for multiple detection waves

WATCHDOG_TIMEOUT = 600       # 20 minutes timeout for stuck waiting sessions
WATCHDOG_CHECK = 10          # Check watchdog every 10 seconds
POST_RESET_COOLDOWN = 3      # Seconds to wait after reset before allowing new detections
RESET_DEDUP_WINDOW = 2       # Ignore duplicate resets within 2 seconds

# ==========================================
# AUTHENTICATION
# ==========================================
load_dotenv("/etc/cctv/credentials.env")
CAM_PASS = os.getenv("CAM_PASS")
if not CAM_PASS:
    print("ERROR: CAM_PASS not found!")
    sys.exit(1)
password = quote(CAM_PASS)

# Load webhook secret (optional, with default for testing)
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
session_waiting_reset = {n: False for n in NODES}  # Camera is waiting for recorder reset
session_count = {n: 0 for n in NODES}
session_detection_id = {n: None for n in NODES}  # Unique ID for each detection session
last_upload = {n: 0 for n in NODES}
last_run = {n: 0 for n in NODES}
last_activity = {n: 0 for n in NODES}  # Track last detection activity
last_waiting_start = {n: 0 for n in NODES}  # Track when waiting state began
last_reset_time = {n: 0 for n in NODES}  # Track when reset was last received
session_completed = {n: False for n in NODES}  # Track if session was completed by recorder
last_reset_processed = {n: 0 for n in NODES}  # Track last reset time for deduplication

# ==========================================
# FLASK WEBHOOK SERVER WITH AUTH
# ==========================================
app = Flask(__name__)

def verify_auth():
    """Verify the request has a valid API key"""
    api_key = request.headers.get('X-API-KEY')
    if not api_key or api_key != WEBHOOK_SECRET:
        return False
    return True

@app.route('/session-reset', methods=['POST'])
def session_reset():
    """Called by RPi recorder when a session is complete"""
    # Verify authentication
    if not verify_auth():
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Unauthorized reset attempt from {request.remote_addr}")
        return jsonify({"status": "unauthorized", "error": "Invalid API key"}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "error": "No JSON data"}), 400
            
        camera_name = data.get('camera')
        
        if camera_name and camera_name in session_waiting_reset:
            now = time.time()
            
            # ✅ IGNORE duplicate resets within RESET_DEDUP_WINDOW seconds
            last_time = last_reset_processed.get(camera_name, 0)
            if now - last_time < RESET_DEDUP_WINDOW:
                # Silent ignore - duplicate reset
                return '', 200
            
            # Update last reset time for this camera
            last_reset_processed[camera_name] = now
            
            was_waiting = session_waiting_reset[camera_name]
            old_count = session_count[camera_name]  # Store before clearing
            
            # Clear ALL session state for this camera
            session_waiting_reset[camera_name] = False
            session_count[camera_name] = 0
            session_detection_id[camera_name] = None
            last_waiting_start[camera_name] = 0
            session_completed[camera_name] = True
            last_reset_time[camera_name] = now
            
            if was_waiting:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📡 Recorder signaled: {camera_name} reset - DETECTION RESUMED (cooldown: {POST_RESET_COOLDOWN}s)")
            else:
                # Only log partial session cleanup if there were frames
                if old_count > 0:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📡 Recorder signaled: {camera_name} reset - CLEANED UP PARTIAL SESSION ({old_count}/6 frames)")
                # Silent ignore for completely idle cameras (no log spam)
            
            return '', 200
        
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        print(f"Error in session_reset: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_legacy():
    """Legacy endpoint for compatibility with recorders calling /reset"""
    return session_reset()

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint (no auth required)"""
    return jsonify({
        "status": "running",
        "version": VERSION,
        "cameras": {
            name: {
                "waiting": session_waiting_reset[name],
                "count": session_count[name],
                "detection_id": session_detection_id[name],
                "last_activity": last_activity[name],
                "last_upload": last_upload[name],
                "waiting_seconds": int(time.time() - last_waiting_start[name]) if session_waiting_reset[name] else 0,
                "cooldown_remaining": max(0, POST_RESET_COOLDOWN - int(time.time() - last_reset_time[name])) if last_reset_time[name] > 0 else 0,
                "completed": session_completed[name]
            } for name in NODES
        }
    }), 200

def start_webhook_server():
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=WEBHOOK_PORT, debug=False, use_reloader=False), daemon=True).start()
    print(f"🌐 Webhook server on port {WEBHOOK_PORT}")
    print(f"🔐 Authentication: {'Enabled' if WEBHOOK_SECRET != 'change-this-default-secret' else 'WARNING: Using default secret!'}")

# ==========================================
# UPLOAD FUNCTION
# ==========================================
def draw_and_upload(camera_name, url, frame, detections, ts, current_count, max_images, detection_id):
    """Draw bounding boxes and upload frame to recorder"""
    yellow = (0, 255, 255)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"PERSON {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 1)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), yellow, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Retry logic for upload failures
    max_retries = 3
    for attempt in range(max_retries):
        try:
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if success:
                files = {'image': (f"{camera_name}_{ts}.jpg", buffer.tobytes(), 'image/jpeg')}
                data = {
                    'frame_num': current_count,
                    'total_frames': max_images,
                    'camera': camera_name,
                    'detection_id': detection_id
                }
                response = requests.post(url, files=files, data=data, timeout=5)
                response.raise_for_status()
                
                if current_count == max_images:
                    print(f"[{ts}] 🛑 {camera_name}: Last frame sent ({current_count}/{max_images}) - waiting for recorder reset...")
                    session_waiting_reset[camera_name] = True
                    last_waiting_start[camera_name] = time.time()  # Track when waiting started
                    session_completed[camera_name] = False  # Reset completed flag for new session
                return  # Success, exit function
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503 and attempt < max_retries - 1:
                print(f"[{ts}] ⚠️ {camera_name}: 503 error, retrying in 1s... (attempt {attempt+1}/{max_retries})")
                time.sleep(1)
            else:
                print(f"[{ts}] ✗ {camera_name}: Upload failed - {e}")
                break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[{ts}] ⚠️ {camera_name}: Upload error, retrying in 1s... (attempt {attempt+1}/{max_retries})")
                time.sleep(1)
            else:
                print(f"[{ts}] ✗ {camera_name}: Upload failed - {e}")
                break

# ==========================================
# YOLO WORKER
# ==========================================
def yolo_worker(input_q, output_q):
    """Worker process for YOLO inference"""
    model = YOLO("yolov8n.pt")
    while True:
        try:
            name, frame, ts = input_q.get(timeout=0.05)
            results = model.predict(frame, imgsz=YOLO_INPUT_SIZE, conf=YOLO_CONFIDENCE, classes=[0], verbose=False)
            detections = [{"box": [int(x) for x in box.xyxy[0]], "conf": float(box.conf[0])} for box in results[0].boxes] if results[0].boxes else []
            output_q.put((name, frame, detections, ts))
        except Empty:
            continue
        except Exception as e:
            print(f"YOLO error: {e}")

# ==========================================
# CAMERA STREAM WITH RECONNECT LOGIC
# ==========================================
class CameraStream:
    """Thread-safe camera stream handler with auto-reconnect"""
    
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.frame = None
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
        return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.running = False

# ==========================================
# SESSION WATCHDOG THREAD
# ==========================================
def session_watchdog():
    """
    Periodically checks for stuck waiting sessions and idle sessions.
    Runs independently of frame processing.
    
    FIXED: Now properly respects sessions that were already reset by recorder
    """
    while True:
        time.sleep(WATCHDOG_CHECK)
        now = time.time()
        
        for name in NODES:
            # Skip if session was already completed by recorder
            if session_completed[name]:
                # Reset the completed flag after a short time to allow new sessions
                if now - last_reset_time[name] > 5:  # Clear completed flag after 5 seconds
                    session_completed[name] = False
                continue
            
            # Check for stuck waiting sessions (recorder never responded)
            if session_waiting_reset[name] and last_waiting_start[name] > 0:
                waiting_duration = now - last_waiting_start[name]
                if waiting_duration > WATCHDOG_TIMEOUT:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Watchdog: {name} stuck waiting for {waiting_duration:.0f}s (timeout: {WATCHDOG_TIMEOUT}s). Force resetting.")
                    session_waiting_reset[name] = False
                    session_count[name] = 0
                    session_detection_id[name] = None
                    last_waiting_start[name] = 0
                    session_completed[name] = True
            
            # Check for idle sessions but ONLY if NOT waiting for reset
            elif not session_waiting_reset[name] and session_count[name] > 0:
                idle_duration = now - last_activity[name]
                if idle_duration > SESSION_TIMEOUT:
                    # Only log timeout if session wasn't already completed
                    if not session_completed[name]:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⏰ Session timeout: {name} - no activity for {idle_duration:.0f}s (timeout: {SESSION_TIMEOUT}s). Resetting.")
                    session_count[name] = 0
                    session_detection_id[name] = None
                    session_completed[name] = True

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    start_webhook_server()
    
    task_q = multiprocessing.Queue(maxsize=10)
    result_q = multiprocessing.Queue()
    multiprocessing.Process(target=yolo_worker, args=(task_q, result_q), daemon=True).start()
    
    streams = {n: CameraStream(n, cfg["cam_rtsp"]) for n, cfg in NODES.items()}
    time.sleep(2)
    
    print("=" * 60)
    print(f"CCTV DETECTOR v{VERSION} - WITH DETECTION ID SESSION TRACKING")
    print(f"ANALYSIS_INTERVAL: {ANALYSIS_INTERVAL}s.")
    print(f"YOLO_CONFIDENCE: {YOLO_CONFIDENCE}")
    print(f"MAX_IMAGES: {MAX_IMAGES} per camera")
    print(f"WEBHOOK_PORT: {WEBHOOK_PORT}")
    print(f"SESSION_TIMEOUT: {SESSION_TIMEOUT}s (idle sessions)")
    print(f"WATCHDOG: {WATCHDOG_CHECK}s check interval, {WATCHDOG_TIMEOUT}s timeout for stuck sessions")
    print(f"POST_RESET_COOLDOWN: {POST_RESET_COOLDOWN}s (cooldown after reset)")
    print(f"RESET_DEDUP_WINDOW: {RESET_DEDUP_WINDOW}s (ignore duplicate resets)")
    print("=" * 60)
    
    # Start watchdog thread
    threading.Thread(target=session_watchdog, daemon=True).start()
    
    # Result handler thread
    def handle_results():
        while True:
            try:
                name, frame, detections, ts = result_q.get(timeout=0.01)
                now = time.time()
                
                # ALWAYS update last_run when we get a result
                last_run[name] = now
                
                if detections:
                    # Check if we're in post-reset cooldown
                    cooldown_remaining = POST_RESET_COOLDOWN - (now - last_reset_time[name]) if last_reset_time[name] > 0 else 0
                    
                    if cooldown_remaining > 0:
                        # Skip processing during cooldown
                        continue
                    
                    # Reset completed flag when starting new detection
                    if session_count[name] == 0:
                        session_completed[name] = False
                    
                    # Only process if not waiting for reset
                    if not session_waiting_reset[name] and session_count[name] < MAX_IMAGES:
                        if session_count[name] == 0 and (now - last_upload[name] < COOLDOWN):
                            continue
                        
                        # Generate new detection_id for first frame of session
                        if session_count[name] == 0:
                            session_detection_id[name] = str(uuid.uuid4())[:8]  # Short unique ID
                            print(f"[{ts}] 🆔 {name}: New detection session {session_detection_id[name]}")
                        
                        session_count[name] += 1
                        last_activity[name] = now  # Update activity timestamp
                        print(f"[{ts}] ⚡ {name}: {session_count[name]}/{MAX_IMAGES}")
                        
                        full_frame = streams[name].get_frame()
                        if full_frame is not None:
                            threading.Thread(
                                target=draw_and_upload,
                                args=(name, NODES[name]["rpi_url"], full_frame, detections, ts,
                                      session_count[name], MAX_IMAGES, session_detection_id[name]),
                                daemon=True
                            ).start()
                        last_upload[name] = now
                # No else block needed - last_run is always updated above
                        
            except Empty:
                continue
            except Exception as e:
                print(f"Handler error: {e}")
    
    threading.Thread(target=handle_results, daemon=True).start()
    
    # Main loop - EACH CAMERA INDEPENDENT
    try:
        while True:
            now = time.time()
            
            for name in NODES:
                # ONLY skip this specific camera if it's waiting for reset
                if session_waiting_reset[name]:
                    continue  # This camera is blocked, but others work fine
                
                if now - last_run[name] >= ANALYSIS_INTERVAL:
                    frame = streams[name].get_frame()
                    if frame is not None:
                        try:
                            task_q.put_nowait((name, frame, time.strftime('%Y-%m-%d %H:%M:%S')))
                            last_run[name] = now
                        except:
                            pass
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        for s in streams.values():
            s.stop()
        sys.exit(0)
