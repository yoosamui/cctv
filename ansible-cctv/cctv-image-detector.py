# ==========================================
# CCTV IMAGE DETECTOR - VERSION 3.17.1
# FIXED: Non-blocking per-camera sessions
# ==========================================

import cv2
import multiprocessing
import threading
import time
import requests
import os
import sys
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

VERSION = "3.17.1"

# Suppress FFmpeg warnings
#os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
#os.environ['FFMPEG_LOG_LEVEL'] = 'panic'

# ==========================================
# CONFIGURATION
# ==========================================
ANALYSIS_INTERVAL = 2.5
MAX_IMAGES = 6
COOLDOWN = 4.0
CAM_THREAD_SLEEP = 0.01
YOLO_CONFIDENCE = 0.40
YOLO_INPUT_SIZE = 320
JPEG_QUALITY = 80
WEBHOOK_PORT = 5001

# ==========================================
# AUTHENTICATION
# ==========================================
load_dotenv("/etc/cctv/credentials.env")
CAM_PASS = os.getenv("CAM_PASS")
if not CAM_PASS:
    print("ERROR: CAM_PASS not found!")
    sys.exit(1)
password = quote(CAM_PASS)

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
last_upload = {n: 0 for n in NODES}
last_run = {n: 0 for n in NODES}

# ==========================================
# FLASK WEBHOOK SERVER
# ==========================================
app = Flask(__name__)

"""
@app.route('/session-reset', methods=['POST'])
def session_reset():
# Called by RPi recorder when a session is complete
    try:
        data = request.get_json()
        camera_name = data.get('camera')
        
        if camera_name and camera_name in session_waiting_reset:
            session_waiting_reset[camera_name] = False
            session_count[camera_name] = 0
            print(f"[{time.strftime('%H:%M:%S')}] 📡 Recorder signaled: {camera_name} reset - DETECTION RESUMED")
            return jsonify({"status": "ok"}), 200
        
        return jsonify({"status": "error"}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error"}), 500
"""
# To see only relevant resets
@app.route('/session-reset', methods=['POST'])
def session_reset():
    """Called by RPi recorder when a session is complete"""
    try:
        data = request.get_json()
        camera_name = data.get('camera')
        
        if camera_name and camera_name in session_waiting_reset:
            if session_waiting_reset[camera_name]:  # Only log if actually waiting
                session_waiting_reset[camera_name] = False
                session_count[camera_name] = 0
                print(f"[{time.strftime('%H:%M:%S')}] 📡 Recorder signaled: {camera_name} reset - DETECTION RESUMED")
                return '', 200  # Return empty response, no logging

            else:
                # Silent ignore - camera wasn't waiting
                # pass
                return '', 200  # Return empty response, no logging
            #return jsonify({"status": "ok"}), 200
        
        # Silently ignore resets for cameras not waiting
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error"}), 500


def start_webhook_server():
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=WEBHOOK_PORT, debug=False, use_reloader=False), daemon=True).start()
    print(f"🌐 Webhook server on port {WEBHOOK_PORT}")

# ==========================================
# UPLOAD FUNCTION
# ==========================================
def draw_and_upload(camera_name, url, frame, detections, ts, current_count, max_images):
    yellow = (0, 255, 255)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"PERSON {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 1)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), yellow, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    timestamp = time.strftime('%H:%M:%S')
   # cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
   # cv2.putText(frame, f"Frame {current_count}/{max_images}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    try:
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if success:
            files = {'image': (f"{camera_name}_{ts}.jpg", buffer.tobytes(), 'image/jpeg')}
            data = {'frame_num': current_count, 'total_frames': max_images, 'camera': camera_name}
            response = requests.post(url, files=files, data=data, timeout=5)
            response.raise_for_status()
            
            if current_count == max_images:
                print(f"[{ts}] 🛑 {camera_name}: Last frame sent ({current_count}/{max_images}) - waiting for recorder reset...")
                session_waiting_reset[camera_name] = True  # Mark as waiting AFTER last frame sent
                
    except Exception as e:
        print(f"[{ts}] ✗ {camera_name}: Upload failed - {e}")

# ==========================================
# YOLO WORKER
# ==========================================
def yolo_worker(input_q, output_q):
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
# CAMERA STREAM
# ==========================================
class CameraStream:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.frame = None
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()
    
    def update(self):
        cap = None
        while self.running:
            try:
                if cap is None:
                    cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        raise Exception("Failed to open")
                
                ret, frame = cap.read()
                if ret:
                    self.frame = frame
                time.sleep(CAM_THREAD_SLEEP)
            except:
                if cap:
                    cap.release()
                    cap = None
                time.sleep(2)
    
    def get_frame(self):
        return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.running = False

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
    print(f"CCTV DETECTOR v{VERSION} - NON-BLOCKING SESSIONS")
    print(f"ANALYSIS_INTERVAL: {ANALYSIS_INTERVAL} seconds.")
    print(f"YOLO_CONFIDENCE: {YOLO_CONFIDENCE} per camera")
    print(f"MAX_IMAGES: {MAX_IMAGES} per camera")
    print(f"WEBHOOK_PORT : {WEBHOOK_PORT} ")
    print("=" * 60)
    
    # Result handler thread
    def handle_results():
        while True:
            try:
                name, frame, detections, ts = result_q.get(timeout=0.01)
                now = time.time()
                
                if detections:
                    # Only process if not waiting for reset
                    if not session_waiting_reset[name] and session_count[name] < MAX_IMAGES:
                        if session_count[name] == 0 and (now - last_upload[name] < COOLDOWN):
                            continue
                        
                        session_count[name] += 1
                        print(f"[{ts}] ⚡ {name}: {session_count[name]}/{MAX_IMAGES}")
                        
                        full_frame = streams[name].get_frame()
                        if full_frame is not None:
                            threading.Thread(
                                target=draw_and_upload,
                                args=(name, NODES[name]["rpi_url"], full_frame, detections, ts,
                                      session_count[name], MAX_IMAGES),
                                daemon=True
                            ).start()
                        last_upload[name] = now
                else:
                    # Reset if no activity for 30 seconds
                    if session_count[name] > 0 and (now - last_run[name] > 30):
                        print(f"[{ts}] ⏰ {name}: Session timeout, resetting")
                        session_count[name] = 0
                        session_waiting_reset[name] = False
                        
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
                            task_q.put_nowait((name, frame, time.strftime('%H:%M:%S')))
                            last_run[name] = now
                        except:
                            pass
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        for s in streams.values():
            s.stop()
        sys.exit(0)
