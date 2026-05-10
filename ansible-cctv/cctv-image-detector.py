# ==========================================
# CCTV IMAGE DETECTOR - VERSION 3.16.1
# FIXED: CPU SPIKE & LOGGING CRASH
# ==========================================

import cv2
import multiprocessing
import threading
import time
import datetime
import requests
import os
from ultralytics import YOLO
from queue import Empty
from urllib.parse import quote
from dotenv import load_dotenv

VERSION = "3.16.1"

# This tells FFmpeg to only show "Panic" errors and hide "Errors" or "Warnings"
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['FFMPEG_LOG_LEVEL'] = 'panic'


# --- CONFIG & AUTH ---
load_dotenv("/etc/cctv/credentials.env")
CAM_PASS = os.getenv("CAM_PASS")
password = quote(CAM_PASS)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

NODES = {
    "Gate":     {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.99:554/Streaming/channels/102", "rpi_url": "http://192.168.1.14:5000/upload"},
    "Center":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.82:554/Streaming/channels/102", "rpi_url": "http://192.168.1.13:5000/upload"},
    "Entrance": {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.89:554/Streaming/channels/102", "rpi_url": "http://192.168.1.15:5000/upload"},
    "Garage":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.81:554/Streaming/channels/102", "rpi_url": "http://192.168.1.16:5000/upload"},
    "Behind":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.92:554/Streaming/channels/102", "rpi_url": "http://192.168.1.17:5000/upload"},
    "Left":     {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.93:554/Streaming/channels/102", "rpi_url": "http://192.168.1.18:5000/upload"}
}

# --- PERFORMANCE TUNING ---
ANALYSIS_INTERVAL = 2.5  # Check every 2.5s (Massive CPU saving)
MAX_IMAGES = 3           # Matches RPi Limit
COOLDOWN = 4.0           # Spacing between images

def draw_and_upload(camera_name, url, frame, detections, ts):
    """Draws boxes and uploads in a background thread."""
    #print(f"[{ts}] Detections: {len(detections)}")

    yellow = (0, 255, 255)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"PERSON {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), yellow, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    try:
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            requests.post(url, files={'image': (f"{camera_name}.jpg", buffer.tobytes(), 'image/jpeg')}, timeout=5)
    except:
        pass

def yolo_worker(input_q, output_q):
    """AI Process: Pure math, no drawing."""
    model = YOLO("yolov8n.pt")
    while True:
        try:
            name, frame, ts = input_q.get()
            results = model.predict(frame, imgsz=320, conf=0.40, classes=[0], verbose=False)
            detections = [{"box": [int(x) for x in box.xyxy[0]], "conf": float(box.conf[0])} for box in results[0].boxes] if results[0].boxes else []
            output_q.put((name, frame, detections, ts))
        except: pass

class CameraStream:
    def __init__(self, name, url):
        self.name, self.url, self.frame = name, url, None
        threading.Thread(target=self.update, daemon=True).start()
    def update(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        while True:
            ret, frame = cap.read()
            if ret: self.frame = frame
            else:
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            time.sleep(0.1)

if __name__ == "__main__":
    task_q = multiprocessing.Queue(maxsize=10)
    result_q = multiprocessing.Queue()
    multiprocessing.Process(target=yolo_worker, args=(task_q, result_q), daemon=True).start()
    
    streams = {n: CameraStream(n, cfg["cam_rtsp"]) for n, cfg in NODES.items()}
    last_run, last_upload, counts = {n:0 for n in NODES}, {n:0 for n in NODES}, {n:0 for n in NODES}
    
    print(f"--- yoosamui cctv-image-detector SYSTEM {VERSION} STARTED (STABLE ) ---")

    while True:
        now = time.time()
        for name in NODES:
            if now - last_run[name] >= ANALYSIS_INTERVAL:
                if streams[name].frame is not None:
                    try:
                        task_q.put_nowait((name, streams[name].frame.copy(), time.strftime('%H:%M:%S')))
                        last_run[name] = now
                    except: pass

        while not result_q.empty():
            res_name, res_frame, detections, res_ts = result_q.get_nowait()
            if detections:
                if counts[res_name] < MAX_IMAGES and (now - last_upload[res_name] >= COOLDOWN):
                    print(f"[{res_ts}] ⚡ {res_name}: Sending Detection {counts[res_name]+1}/{MAX_IMAGES}")
                    threading.Thread(target=draw_and_upload, args=(res_name, NODES[res_name]["rpi_url"], res_frame.copy(), detections, res_ts)).start()
                    last_upload[res_name] = now
                    counts[res_name] += 1
            else:
                # Reset counter if camera is clear for 30s
                if now - last_run[res_name] > 30:
                    counts[res_name] = 0
        
        time.sleep(0.1)

