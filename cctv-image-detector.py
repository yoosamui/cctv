import cv2
import threading
import time
import os
import requests
import psutil
from ultralytics import YOLO
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from dotenv import load_dotenv

# --- CREDENTIALS LOADING ---
# Ensure this path is correct for your environment
load_dotenv("/etc/cctv/credentials.env")
CAM_PASS = os.getenv("CAM_PASS")
# quote() handles special characters in passwords for the RTSP string
password = quote(CAM_PASS) if CAM_PASS else "password"

# --- NETWORK OPTIMIZATION ---
# Forces TCP to prevent frame corruption/smearing
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

# --- STATE MANAGEMENT ---
frame_queue = {name: Queue(maxsize=1) for name in NODES}
executor = ThreadPoolExecutor(max_workers=len(NODES)) # One worker per camera

last_alert = {name: 0 for name in NODES}
last_yolo_run = {name: 0 for name in NODES}
yolo_counter = {name: 0 for name in NODES}
scan_state = {name: "Idle" for name in NODES}
last_detect_time = {name: 0 for name in NODES} 

# --- IMAGE UPLOAD LOGIC ---
def send_to_rpi(camera_name, url, frame, timestamp):
    if not url: return

    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        print(f"[ERROR] {camera_name} encoding failed")
        return

    filename = f"{camera_name}_{timestamp}.jpg"
    try:
        files = {'image': (filename, buffer.tobytes(), 'image/jpeg')}
        # Separate connection and read timeouts
        response = requests.post(url, files=files, timeout=(2, 10))
        if response.status_code == 200:
            print(f"[{time.strftime('%H:%M:%S')}] .........................................>> [UPLOADED] {filename}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] [HTTP ERROR] {camera_name}: {response.status_code}")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [NETWORK ERROR] {camera_name}: {e}")

# --- CAMERA STREAMING CLASS ---
class CameraStream:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while not self.stopped:
            ret, frame = cap.read()
            if ret:
                # Flush the queue to ensure only the latest frame is stored
                if frame_queue[self.name].full():
                    try:
                        frame_queue[self.name].get_nowait()
                    except Empty:
                        pass
                frame_queue[self.name].put(frame)
            else:
                print(f"[WARN] {self.name} connection lost. Reconnecting...")
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            time.sleep(0.01)

# --- INITIALIZE ENGINE ---
print(f"[{time.strftime('%H:%M:%S')}] [INFO] Starting AI Engine...")
model = YOLO("yolov8n.pt")

for name, cfg in NODES.items():
    CameraStream(name, cfg["cam_rtsp"])

last_heartbeat = time.time()
print(f"[{time.strftime('%H:%M:%S')}] 🚀 ENGINE LIVE - MONITORING {len(NODES)} NODES")

# --- MAIN AI LOOP ---
try:
    while True:
        for name in NODES:
            # 1. DRAIN QUEUE TO GET FRESH FRAME
            raw_frame = None
            try:
                while not frame_queue[name].empty():
                    raw_frame = frame_queue[name].get_nowait()
            except Empty:
                continue

            if raw_frame is None:
                continue

            # 2. RATE LIMIT (Analyze every 5 seconds per camera)
            now = time.time()
            if now - last_yolo_run[name] < 5.0:
                continue

            # 3. START ANALYSIS
            last_yolo_run[name] = now
            yolo_counter[name] += 1 # Count every attempt for real FPS status
            ts = time.strftime('%H:%M:%S')
            
            print(f"[{ts}] [YOLO] Analyzing {name}...")
            scan_state[name] = "Scanning"
            
            results = model.predict(raw_frame, imgsz=416, conf=0.40, classes=[0], verbose=False)

            if results[0].boxes:
                last_detect_time[name] = now
                print(f"[{ts}] .........................................>> [DETECT] {name} - PERSON FOUND")

                annotated_frame = raw_frame.copy()
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Drawing logic
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    label = f"PERSON {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    y_text = max(y1 - 5, 15)
                    cv2.rectangle(annotated_frame, (x1, y_text - h - 5), (x1 + w, y_text), (0, 255, 255), -1)
                    cv2.putText(annotated_frame, label, (x1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                # 4. ALERTING WITH COOLDOWN
                if now - last_alert[name] > 5.0:
                    target_url = NODES[name].get("rpi_upload_url")
                    executor.submit(send_to_rpi, name, target_url, annotated_frame, ts)
                    last_alert[name] = now

            scan_state[name] = "Idle"

        # --- HEARTBEAT STATUS REPORT ---
        if time.time() - last_heartbeat > 6:
            ts = time.strftime('%H:%M:%S')
            cpu = psutil.cpu_percent()
            print(f"\n[{ts}] [HEARTBEAT] CPU={cpu:.1f}%")

            for name in NODES:
                # Actual processed frames per second
                fps = yolo_counter[name] / 6.0
                qsize = frame_queue[name].qsize()
                
                if last_detect_time[name] > 0:
                    seconds_ago = int(time.time() - last_detect_time[name])
                    last_seen = f"{seconds_ago}s ago"
                else:
                    last_seen = "--"

                print(f"  - {name:10s} FPS={fps:.2f} Q={qsize} {scan_state[name]:10s} Last={last_seen}")
                yolo_counter[name] = 0 # Reset for next 6s window

            last_heartbeat = time.time()
        
        time.sleep(0.01) # Small sleep to prevent CPU pegging

except KeyboardInterrupt:
    print("\n[INFO] Engine stopping...")
