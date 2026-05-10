import cv2
import threading
import time
import os
import requests
import psutil
from ultralytics import YOLO
from queue import Queue, Empty
from urllib.parse import quote
from dotenv import load_dotenv
import datetime
from threading import Lock

# VERSION 2.6 - Optimized Performance + Professional Annotations
load_dotenv("/etc/cctv/credentials.env")
CAM_PASS = os.getenv("CAM_PASS")

if not CAM_PASS:
    print("[ERROR] CAM_PASS not found")
    exit(1)

password = quote(CAM_PASS)
# Optimize RTSP transport for lower latency
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

# --- CONFIGURATION ---
NODES = {
    "Gate":     {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.99:554/Streaming/channels/102", "rpi_upload_url": "http://192.168.1.14:5000/upload"},
    "Center":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.82:554/Streaming/channels/102", "rpi_upload_url": "http://192.168.1.13:5000/upload"},
    "Entrance": {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.89:554/Streaming/channels/102", "rpi_upload_url": "http://192.168.1.15:5000/upload"},
    "Garage":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.81:554/Streaming/channels/102", "rpi_upload_url": "http://192.168.1.16:5000/upload"},
    "Behind":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.92:554/Streaming/channels/102", "rpi_upload_url": "http://192.168.1.17:5000/upload"},
    "Left":     {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.93:554/Streaming/channels/102", "rpi_upload_url": "http://192.168.1.18:5000/upload"}
}

# 320 is the "sweet spot" for speed on a 4-core CPU
YOLO_CONFIG = {"imgsz": 320, "conf": 0.35, "classes": [0], "verbose": False}
ANALYSIS_INTERVAL = 5.0  
ALERT_COOLDOWN = 5.0  
HEARTBEAT_INTERVAL = 3.0 

# --- STATE MANAGEMENT ---
frame_queue = {name: Queue(maxsize=1) for name in NODES}
state_locks = {name: Lock() for name in NODES}
last_alert = {name: 0 for name in NODES}
last_yolo_run = {name: 0 for name in NODES}
yolo_counter = {name: 0 for name in NODES}
scan_state = {name: "Idle" for name in NODES}
last_detect_time = {name: 0 for name in NODES}
pause_until = {name: 0 for name in NODES} 
current_limit_val = {name: 0 for name in NODES}

def update_state_safe(camera_name, state_key, value):
    with state_locks[camera_name]:
        globals()[state_key][camera_name] = value

def get_state_safe(camera_name, state_key):
    with state_locks[camera_name]:
        return globals()[state_key][camera_name]

# --- IMAGE UPLOAD ---
def send_to_rpi(camera_name, url, frame, timestamp_str):
    if not url: return
    print(f"[{timestamp_str}] [SENDING] UPLOADING DETECTION ({camera_name})")
    
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success: return

    try:
        files = {'image': (f"{camera_name}.jpg", buffer.tobytes(), 'image/jpeg')}
        response = requests.post(url, files=files, timeout=(2, 10))
        
        if response.status_code == 200:
            print(f"[{timestamp_str}] ....................................... SUCCESS")
            update_state_safe(camera_name, "scan_state", "Idle")
            update_state_safe(camera_name, "pause_until", 0)
        
        elif response.status_code == 429:
            data = response.json()
            limit_val = abs(data.get("duration_secs", 30))
            print(f"[{timestamp_str}] [!!] HTTP 429 RECEIVED FROM {camera_name.upper()}")
            print(f"[{timestamp_str}] [!!] PAUSE_LMT ACTIVATED FOR: {limit_val} SECONDS")
            
            update_state_safe(camera_name, "current_limit_val", limit_val)
            update_state_safe(camera_name, "pause_until", time.time() + float(limit_val))
            update_state_safe(camera_name, "scan_state", "PAUSE_LMT")

    except Exception as e:
        print(f"[{timestamp_str}] [ERROR] Network failure for {camera_name}: {e}")

# --- RTSP STREAMING ---
class CameraStream:
    def __init__(self, name, url):
        self.name, self.url, self.stopped, self.cap = name, url, False, None
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                time.sleep(2)
                continue
            ret, frame = self.cap.read()
            if ret:
                # Buffer management: Discard old frame to prevent lag
                if not frame_queue[self.name].empty():
                    try: frame_queue[self.name].get_nowait()
                    except: pass
                frame_queue[self.name].put(frame)
            time.sleep(0.01)

# --- AI ENGINE ---
model = YOLO("yolov8n.pt") 
for name, cfg in NODES.items():
    CameraStream(name, cfg["cam_rtsp"])

last_heartbeat = time.time()

try:
    while True:
        current_time = time.time()

        for name in NODES:
            # 1. LIMIT CHECK
            until = get_state_safe(name, "pause_until")
            if until > 0:
                if current_time < until:
                    continue
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] [INFO] {name} Limit period ended. Resuming...")
                    update_state_safe(name, "pause_until", 0)
                    update_state_safe(name, "current_limit_val", 0)
                    update_state_safe(name, "scan_state", "Idle")

            # 2. ANALYSIS TIMING
            last_run = get_state_safe(name, "last_yolo_run")
            if current_time - last_run >= ANALYSIS_INTERVAL:
                # CRITICAL: Flush queue to get the REAL-TIME frame
                raw_frame = None
                while not frame_queue[name].empty():
                    raw_frame = frame_queue[name].get_nowait()
                
                if raw_frame is None:
                    continue

                timestamp_str = time.strftime('%H:%M:%S')
                print(f"[{timestamp_str}] [YOLO] Analyzing {name}...")
                
                update_state_safe(name, "last_yolo_run", current_time)
                with state_locks[name]: yolo_counter[name] += 1
                
                # PREDICT
                results = model.predict(
                    raw_frame, 
                    imgsz=YOLO_CONFIG["imgsz"], 
                    conf=YOLO_CONFIG["conf"], 
                    classes=YOLO_CONFIG["classes"], 
                    verbose=False
                )

                if results[0].boxes:
                    update_state_safe(name, "last_detect_time", current_time)
                    print(f"[{timestamp_str}] ⚡ [DETECT] PERSON in {name.upper()} FOUND")
                    
                    # ANNOTATIONS
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        label = f"PERSON {conf:.2f}"

                        # Draw Box (Yellow)
                        cv2.rectangle(raw_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                        
                        # Draw Title Header (Yellow)
                        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        cv2.rectangle(raw_frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), (0, 255, 255), -1)
                        
                        # Draw Label Text (Black)
                        cv2.putText(raw_frame, label, (x1 + 2, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
                    # UPLOAD LOGIC
                    if current_time - get_state_safe(name, "last_alert") >= ALERT_COOLDOWN:
                        send_to_rpi(name, NODES[name]["rpi_upload_url"], raw_frame, timestamp_str)
                        update_state_safe(name, "last_alert", current_time)

        # 3. CONSOLE REPORT
        if current_time - last_heartbeat >= HEARTBEAT_INTERVAL:
            print(f"[{time.strftime('%H:%M:%S')}] [HEARTBEAT] CPU={psutil.cpu_percent()}%")
            for n in NODES:
                st = get_state_safe(n, "scan_state")
                l_v = get_state_safe(n, "current_limit_val")
                display = f"PAUSE({l_v}s)" if st == "PAUSE_LMT" else st
                last_s = int(current_time-get_state_safe(n,'last_detect_time')) if get_state_safe(n,'last_detect_time') > 0 else '--'
                print(f"  - {n:10s} {display:14s} Last Detect: {last_s}s")
                update_state_safe(n, "yolo_counter", 0)
            last_heartbeat = current_time

        time.sleep(0.005)

except KeyboardInterrupt:
    print("\n[STOPPING] Shutting down camera streams...")
