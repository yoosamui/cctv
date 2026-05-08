import cv2
import threading
import time
import os
import requests
import psutil
from ultralytics import YOLO
from queue import Queue, Empty, Full
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from dotenv import load_dotenv
import datetime
from threading import Lock

# --- CREDENTIALS LOADING ---
# Ensure this path is correct for your environment
load_dotenv("/etc/cctv/credentials.env")
CAM_PASS = os.getenv("CAM_PASS")

# Handle missing credentials properly
if not CAM_PASS:
    print("[ERROR] CAM_PASS not found in environment variables")
    print("[ERROR] Please check /etc/cctv/credentials.env file")
    exit(1)

# quote() handles special characters in passwords for the RTSP string
password = quote(CAM_PASS)

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

# --- MODEL CONFIGURATION ---
YOLO_CONFIG = {
    "imgsz": 416,
    "conf": 0.40,
    "classes": [0],  # Person class only
    "verbose": False
}

# --- RATE LIMIT CONFIGURATION ---
ANALYSIS_INTERVAL = 5.0  # seconds between YOLO analyses per camera
ALERT_COOLDOWN = 5.0  # seconds between alerts per camera
HEARTBEAT_INTERVAL = 6.0  # seconds between status reports

# --- STATE MANAGEMENT ---
frame_queue = {name: Queue(maxsize=1) for name in NODES}
executor = ThreadPoolExecutor(max_workers=len(NODES))

# Thread-safe state dictionaries with locks
state_locks = {name: Lock() for name in NODES}

# Protected state variables
last_alert = {name: 0 for name in NODES}
last_yolo_run = {name: 0 for name in NODES}
yolo_counter = {name: 0 for name in NODES}
scan_state = {name: "Idle" for name in NODES}
last_detect_time = {name: 0 for name in NODES}

# Track active camera streams
active_streams = {}

# --- IMAGE UPLOAD LOGIC ---
def send_to_rpi(camera_name, url, frame, timestamp_str):
    """Upload annotated frame to Raspberry Pi with retry logic"""
    if not url:
        return

    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        print(f"[ERROR] {camera_name} encoding failed")
        return

    # Use microseconds to avoid filename collisions
    timestamp_full = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    filename = f"{camera_name}_{timestamp_full}.jpg"
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            files = {'image': (filename, buffer.tobytes(), 'image/jpeg')}
            # Separate connection and read timeouts
            response = requests.post(url, files=files, timeout=(2, 10))
            if response.status_code == 200:
                print(f"[{timestamp_str}] .........................................>> [UPLOADED] {filename}")
                return
            else:
                print(f"[{timestamp_str}] [HTTP ERROR] {camera_name}: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        except requests.exceptions.Timeout:
            print(f"[{timestamp_str}] [TIMEOUT] {camera_name} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            print(f"[{timestamp_str}] [NETWORK ERROR] {camera_name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

# --- CAMERA STREAMING CLASS ---
class CameraStream:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.stopped = False
        self.cap = None
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def reconnect(self):
        """Reconnect camera stream with proper cleanup"""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        time.sleep(5)
        
        try:
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[INFO] {self.name} reconnected successfully")
        except Exception as e:
            print(f"[ERROR] {self.name} failed to reconnect: {e}")

    def update(self):
        """Main capture loop - runs in separate thread"""
        self.reconnect()  # Initial connection
        
        consecutive_errors = 0
        
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                print(f"[WARN] {self.name} camera not available. Reconnecting...")
                self.reconnect()
                continue
            
            try:
                ret, frame = self.cap.read()
                
                if ret:
                    consecutive_errors = 0
                    # Thread-safe queue update
                    q = frame_queue[self.name]
                    try:
                        # Try to put the new frame
                        q.put_nowait(frame)
                    except Full:
                        # Queue is full, discard old frame and add new one
                        try:
                            q.get_nowait()  # Remove oldest
                            q.put_nowait(frame)  # Add newest
                        except Empty:
                            # Should not happen after Full, but handle gracefully
                            try:
                                q.put_nowait(frame)
                            except Full:
                                pass
                else:
                    consecutive_errors += 1
                    print(f"[WARN] {self.name} failed to read frame (error {consecutive_errors})")

                    if consecutive_errors >= 5:
                        print(f"[ERROR] {self.name} too many errors, reconnecting...")
                        self.reconnect()
                        consecutive_errors = 0

            except Exception as e:
                print(f"[ERROR] {self.name} unexpected error in capture: {e}")
                self.reconnect()

            # Small sleep to prevent CPU overuse
            time.sleep(0.01)

    def stop(self):
        """Clean shutdown of camera stream"""
        self.stopped = True
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

# --- HELPER FUNCTIONS ---
def draw_detection(frame, box, conf):
    """Draw bounding box and label on frame"""
    x1, y1, x2, y2 = box
    annotated_frame = frame.copy()

    # Draw rectangle
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Draw label background and text
    label = f"PERSON {conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    (w, h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    y_text = max(y1 - 10, h + 10)
    
    cv2.rectangle(annotated_frame, 
                  (x1, y_text - h - 5), 
                  (x1 + w, y_text + baseline), 
                  (0, 255, 255), 
                  -1)
    cv2.putText(annotated_frame, 
                label, 
                (x1, y_text), 
                font, 
                font_scale, 
                (0, 0, 0), 
                thickness)
    
    return annotated_frame

def update_state_safe(camera_name, state_key, value):
    """Thread-safe state update"""
    with state_locks[camera_name]:
        if state_key == "last_alert":
            last_alert[camera_name] = value
        elif state_key == "last_yolo_run":
            last_yolo_run[camera_name] = value
        elif state_key == "yolo_counter":
            yolo_counter[camera_name] = value
        elif state_key == "scan_state":
            scan_state[camera_name] = value
        elif state_key == "last_detect_time":
            last_detect_time[camera_name] = value

def get_state_safe(camera_name, state_key):
    """Thread-safe state retrieval"""
    with state_locks[camera_name]:
        if state_key == "last_alert":
            return last_alert[camera_name]
        elif state_key == "last_yolo_run":
            return last_yolo_run[camera_name]
        elif state_key == "yolo_counter":
            return yolo_counter[camera_name]
        elif state_key == "scan_state":
            return scan_state[camera_name]
        elif state_key == "last_detect_time":
            return last_detect_time[camera_name]
    return None

def increment_counter_safe(camera_name):
    """Thread-safe counter increment"""
    with state_locks[camera_name]:
        yolo_counter[camera_name] += 1

def reset_counters():
    """Reset all YOLO counters for heartbeat calculation"""
    for name in NODES:
        with state_locks[name]:
            yolo_counter[name] = 0

# --- INITIALIZE ENGINE ---
print(f"[{time.strftime('%H:%M:%S')}] [INFO] Starting AI Engine...")
print(f"[{time.strftime('%H:%M:%S')}] [INFO] Loading YOLO model...")

try:
    model = YOLO("yolov8n.pt")
    print(f"[{time.strftime('%H:%M:%S')}] [INFO] YOLO model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model: {e}")
    exit(1)

# Start camera streams
for name, cfg in NODES.items():
    print(f"[{time.strftime('%H:%M:%S')}] [INFO] Starting camera stream: {name}")
    active_streams[name] = CameraStream(name, cfg["cam_rtsp"])

last_heartbeat = time.time()
print(f"[{time.strftime('%H:%M:%S')}] 🚀 ENGINE LIVE - MONITORING {len(NODES)} NODES")
print(f"[{time.strftime('%H:%M:%S')}] [CONFIG] Analysis interval: {ANALYSIS_INTERVAL}s, Alert cooldown: {ALERT_COOLDOWN}s")

# --- MAIN AI LOOP ---
try:
    while True:
        current_time = time.time()
        
        for name in NODES:
            # 1. GET LATEST FRAME FROM QUEUE
            raw_frame = None
            try:
                # Get the most recent frame (queue only holds 1 frame max)
                raw_frame = frame_queue[name].get_nowait()
            except Empty:
                continue

            if raw_frame is None:
                continue

            # 2. CHECK RATE LIMIT
            last_run = get_state_safe(name, "last_yolo_run")
            if current_time - last_run < ANALYSIS_INTERVAL:
                continue

            # 3. UPDATE STATE AND START ANALYSIS
            update_state_safe(name, "last_yolo_run", current_time)
            increment_counter_safe(name)
            update_state_safe(name, "scan_state", "Scanning")
            
            timestamp_str = time.strftime('%H:%M:%S')
            print(f"[{timestamp_str}] [YOLO] Analyzing {name}...")
            
            # 4. RUN YOLO INFERENCE WITH ERROR HANDLING
            try:
                results = model.predict(
                    raw_frame, 
                    imgsz=YOLO_CONFIG["imgsz"], 
                    conf=YOLO_CONFIG["conf"], 
                    classes=YOLO_CONFIG["classes"], 
                    verbose=YOLO_CONFIG["verbose"]
                )
                
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    update_state_safe(name, "last_detect_time", current_time)
                    print(f"[{timestamp_str}] .........................................>> [DETECT] {name} - PERSON FOUND ({len(results[0].boxes)} person(s))")
                    
                    # Annotate frame with all detections
                    annotated_frame = raw_frame.copy()
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Draw each detection
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        annotated_frame = draw_detection(annotated_frame, (x1, y1, x2, y2), conf)
                    
                    # 5. CHECK ALERT COOLDOWN
                    last_alert_time = get_state_safe(name, "last_alert")
                    if current_time - last_alert_time >= ALERT_COOLDOWN:
                        target_url = NODES[name].get("rpi_upload_url")
                        if target_url:
                            executor.submit(send_to_rpi, name, target_url, annotated_frame, timestamp_str)
                            update_state_safe(name, "last_alert", current_time)
                        else:
                            print(f"[WARN] {name} has no upload URL configured")
                
            except Exception as e:
                print(f"[ERROR] YOLO prediction failed for {name}: {e}")
            
            # Update state after analysis
            update_state_safe(name, "scan_state", "Idle")

        # --- HEARTBEAT STATUS REPORT ---
        if current_time - last_heartbeat >= HEARTBEAT_INTERVAL:
            timestamp_str = time.strftime('%H:%M:%S')
            
            try:
                cpu_percent = psutil.cpu_percent(interval=0.5)
                memory = psutil.virtual_memory()
                print(f"\n[{timestamp_str}] [HEARTBEAT] CPU={cpu_percent:.1f}% | RAM={memory.percent:.1f}% | Active Streams={len(active_streams)}")
                
                for name in NODES:
                    # Get current stats safely
                    with state_locks[name]:
                        fps = yolo_counter[name] / HEARTBEAT_INTERVAL
                        qsize = frame_queue[name].qsize()
                        state = scan_state[name]
                        last_detect = last_detect_time[name]
                    
                    if last_detect > 0:
                        seconds_ago = int(current_time - last_detect)
                        last_seen = f"{seconds_ago}s ago"
                    else:
                        last_seen = "--"
                    
                    # Show queue health indicator
                    queue_status = "✓" if qsize == 1 else "⚠" if qsize == 0 else "!"
                    
                    print(f"  - {name:10s} FPS={fps:.2f} {queue_status} Q={qsize} {state:10s} Last={last_seen}")
                
                # Reset counters for next interval
                reset_counters()
                
            except Exception as e:
                print(f"[ERROR] Heartbeat monitoring failed: {e}")
            
            last_heartbeat = current_time
        
        # Small sleep to prevent CPU pegging (adjustable)
        time.sleep(0.005)

except KeyboardInterrupt:
    print("\n[INFO] Engine stopping gracefully...")
    
    # Clean shutdown
    print("[INFO] Stopping camera streams...")
    for name, stream in active_streams.items():
        stream.stop()
    
    print("[INFO] Shutting down thread pool...")
    executor.shutdown(wait=True, timeout=5.0)
    
    print("[INFO] Engine stopped successfully")

except Exception as e:
    print(f"\n[FATAL ERROR] Unexpected error in main loop: {e}")
    import traceback
    traceback.print_exc()
    
    # Attempt cleanup
    for name, stream in active_streams.items():
        stream.stop()
    executor.shutdown(wait=False)
