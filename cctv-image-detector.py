import cv2
import threading
import time
import os
import requests
import psutil
from ultralytics import YOLO
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# --- NETWORK OPTIMIZATION ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

# --- CONFIG ---
NODES = {
    "Gate": {
        "cam_rtsp": "rtsp://admin:master%2131416Pi@192.168.1.99:554/Streaming/channels/102",
        "rpi_upload_url": "http://192.168.1.14:5000/upload"
    },
    "Center": {
        "cam_rtsp": "rtsp://admin:master%2131416Pi@192.168.1.82:554/Streaming/channels/102",
        "rpi_upload_url": "http://192.168.1.13:5000/upload"
    },
    "Entrance": {
        "cam_rtsp": "rtsp://admin:master%2131416Pi@192.168.1.89:554/Streaming/channels/102",
        "rpi_upload_url": "http://192.168.1.15:5000/upload"
    }
}

# --- FRAME STORAGE ---
frame_queue = {name: Queue(maxsize=1) for name in NODES}

# --- THREAD POOL ---
executor = ThreadPoolExecutor(max_workers=2)

# --- STATE ---
last_alert = {name: 0 for name in NODES}
last_yolo_run = {name: 0 for name in NODES}
yolo_counter = {name: 0 for name in NODES}
scan_state = {name: "Idle" for name in NODES}
last_detect_time = {name: None for name in NODES}

# --- UPLOAD ---
def send_to_rpi(camera_name, frame, timestamp):
    url = NODES[camera_name].get("rpi_upload_url")
    if not url:
        return

    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        print(f"[ERROR] {camera_name} encode failed")
        return

    filename = f"{camera_name}_{timestamp}.jpg"

    try:
        files = {'image': (filename, buffer.tobytes(), 'image/jpeg')}
        response = requests.post(url, files=files, timeout=3)

        if response.status_code == 200:
            print(f"[{time.strftime('%H:%M:%S')}] [SENT..............................] {filename}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] [ERROR] {filename} ({response.status_code})")

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [ERROR] Upload failed ({camera_name}): {e}")

# --- CAMERA THREAD ---
class CameraStream:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while True:
            ret, frame = cap.read()

            if ret:
                if frame_queue[self.name].full():
                    try:
                        frame_queue[self.name].get_nowait()
                    except:
                        pass
                frame_queue[self.name].put(frame)
            else:
                print(f"[WARN] {self.name} reconnecting...")
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

            time.sleep(0.01)

# --- START ---
print(f"[{time.strftime('%H:%M:%S')}] [INFO] Starting AI Engine...")

model = YOLO("yolov8n.pt")

for name, cfg in NODES.items():
    CameraStream(name, cfg["cam_rtsp"])

last_heartbeat = time.time()

print(f"[{time.strftime('%H:%M:%S')}] 🚀 ENGINE LIVE")

# --- MAIN LOOP ---
try:
    while True:
        for name in NODES:

            try:
                raw_frame = frame_queue[name].get_nowait()
            except:
                continue

            now = time.time()

            # --- YOLO every 5 seconds ---
            if now - last_yolo_run[name] < 5.0:
                continue

            last_yolo_run[name] = now
            ts = time.strftime('%H:%M:%S')

            scan_state[name] = "Scanning..."
            yolo_counter[name] += 1

            print(f"[{ts}] [YOLO] Processing {name}")

            # --- DETECTION ---
            results = model.predict(
                raw_frame,
                imgsz=416,
                conf=0.35,
                classes=[0],
                verbose=False
            )

            if results[0].boxes:
                print(f"[{ts}] [DETECT] {name} - PERSON")
                last_detect_time[name] = now

                #  CUSTOM DRAW (YELLOW BOX + BLACK TEXT)
                annotated_frame = raw_frame.copy()

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    box_color = (0, 255, 255)   # yellow
                    text_color = (0, 0, 0)      # black
                    bg_color = (0, 255, 255)    # yellow

                    label = f"PERSON {conf:.2f}"

                    # Draw box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 1)

                    # Text size
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                    # Prevent text going outside frame
                    y_text = max(y1 - 5, 15)

                    # Background rectangle
                    cv2.rectangle(annotated_frame, (x1, y_text - h - 5), (x1 + w, y_text), bg_color, -1)

                    # Text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y_text - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        1
                    )

                if now - last_alert[name] > 5.0:
                    executor.submit(send_to_rpi, name, annotated_frame, ts)
                    last_alert[name] = now

            scan_state[name] = "Idle"

        # --- HEARTBEAT ---
        if time.time() - last_heartbeat > 6:
            ts = time.strftime('%H:%M:%S')
            cpu = psutil.cpu_percent(interval=None)

            print(f"\n[{ts}] [HEARTBEAT] CPU={cpu:.1f}%")

            for name in NODES:
                fps = yolo_counter[name] / 6.0
                qsize = frame_queue[name].qsize()

                if last_detect_time[name]:
                    seconds_ago = int(time.time() - last_detect_time[name])
                    last_seen = f"{seconds_ago}s ago"
                else:
                    last_seen = "--"

                print(f"  - {name:10s} FPS={fps:.2f} Q={qsize} {scan_state[name]:10s} Last={last_seen}")

                yolo_counter[name] = 0

            last_heartbeat = time.time()

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n[INFO] Shutting down...")


















