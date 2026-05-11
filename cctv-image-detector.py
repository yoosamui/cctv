# ==========================================
# CCTV IMAGE DETECTOR - VERSION 3.18-optimized
# SMART-THROTTLE: LOW CPU + FAST CAPTURE
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

load_dotenv("/etc/cctv/credentials.env")
CAM_PASS = os.getenv("CAM_PASS")
password = quote(CAM_PASS)

NODES = {
    "Gate":     {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.99:554/Streaming/channels/102", "rpi_url": "http://192.168.1.14:5000/upload"},
    "Center":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.82:554/Streaming/channels/102", "rpi_url": "http://192.168.1.13:5000/upload"},
    "Entrance": {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.89:554/Streaming/channels/102", "rpi_url": "http://192.168.1.15:5000/upload"},
    "Garage":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.81:554/Streaming/channels/102", "rpi_url": "http://192.168.1.16:5000/upload"},
    "Behind":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.92:554/Streaming/channels/102", "rpi_url": "http://192.168.1.17:5000/upload"},
    "Left":     {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.93:554/Streaming/channels/102", "rpi_url": "http://192.168.1.18:5000/upload"}
}

# --- DYNAMIC TUNING ---
IDLE_INTERVAL = 3.0      # Check every 3s when quiet
ACTIVE_INTERVAL = 3.0    # Check every 1s when person detected
MAX_IMAGES = 6           # Capture 6 images per event session
COOLDOWN = 6.0           # Wait 6s after a session before re-triggering

def draw_and_upload(camera_name, url, frame, detections, ts, count):
    # Log updated to show progress out of 6
    print(f"[{ts}] ⚡ {camera_name:<10} | Found: {len(detections)} | Capture: {count}/{MAX_IMAGES}")
    yellow = (0, 255, 255)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"PERSON {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 1)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), yellow, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    try:
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            requests.post(url, files={'image': (f"{camera_name}.jpg", buffer.tobytes(), 'image/jpeg')}, timeout=5)
    except: pass

def yolo_worker(input_q, output_q):
    model = YOLO("yolov8n.pt") # Using Nano for speed/stability
    while True:
        try:
            name, frame, ts = input_q.get()
            # imgsz=320 is the sweet spot for your current setup
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
            if ret: 
                self.frame = frame
            else:
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            time.sleep(0.01) # Faster update to keep buffer fresh

if __name__ == "__main__":
    task_q = multiprocessing.Queue(maxsize=10)
    result_q = multiprocessing.Queue()
    multiprocessing.Process(target=yolo_worker, args=(task_q, result_q), daemon=True).start()
    
    streams = {n: CameraStream(n, cfg["cam_rtsp"]) for n, cfg in NODES.items()}
    last_run, last_upload, counts = {n:0 for n in NODES}, {n:0 for n in NODES}, {n:0 for n in NODES}
    
    print(f"--- SYSTEM 3.18 ONLINE (SMART-THROTTLE) ---")
    print(f"Settings: IDLE={IDLE_INTERVAL}s, ACTIVE={ACTIVE_INTERVAL}s, MAX_IMG={MAX_IMAGES}")

    while True:
        now = time.time()
        for name in NODES:
            # 1. Determine the interval
            # If we are currently capturing a session, move fast. 
            # If we just finished a session, wait for COOLDOWN.
            if 0 < counts[name] < MAX_IMAGES:
                current_interval = ACTIVE_INTERVAL
            else:
                current_interval = IDLE_INTERVAL

            # 2. Check if it's time to run detection
            if now - last_run[name] >= current_interval:
                if streams[name].frame is not None:
                    try:
                        task_q.put_nowait((name, streams[name].frame.copy(), time.strftime('%H:%M:%S')))
                        last_run[name] = now
                    except: pass

        # 3. Handle results
        while not result_q.empty():
            res_name, res_frame, detections, res_ts = result_q.get_nowait()
            
            if detections:
                # Check if we are within the session limit and outside the cooldown
                if counts[res_name] < MAX_IMAGES and (now - last_upload[res_name] >= 0.8): # Slight buffer between active frames
                    
                    # Prevent starting a new session if we are in COOLDOWN
                    if counts[res_name] == 0 and (now - last_upload[res_name] < COOLDOWN):
                        continue
                        
                    counts[res_name] += 1
                    last_upload[res_name] = now
                    threading.Thread(target=draw_and_upload, args=(res_name, NODES[res_name]["rpi_url"], res_frame.copy(), detections, res_ts, counts[res_name])).start()
            else:
                # If no person is seen for 15 seconds, reset count to allow a new session later
                if counts[res_name] >= MAX_IMAGES or (now - last_run[res_name] > 15):
                    if now - last_upload[res_name] > COOLDOWN:
                        counts[res_name] = 0

        time.sleep(0.1)
