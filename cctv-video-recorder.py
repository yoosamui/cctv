import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import cv2
import requests
import time
from datetime import datetime
from ultralytics import YOLO
from threading import Thread
import queue

# --- CONFIG ---
CAMERAS = {
    "Gate": {
        "url": "rtsp://admin:master!31416Pi@192.168.1.99:554/Streaming/channels/102",
        "pi_endpoint": "http://192.168.1.14:5000/upload"
    },
}

model = YOLO("yolov8n.pt") 
model.to('cpu')

# Image every 3 seconds
COOLDOWN = 3 
last_sent = {name: 0 for name in CAMERAS}

# Keep video capture objects open
captures = {}

print("AI Master Brain: Yellow Boxes | 3s Interval | Fixed Naming")

def process_frame(name, config, frame):
    """Process a single frame for detections"""
    results = model(frame, classes=[0], conf=0.5, verbose=False, device='cpu')
    
    if len(results[0].boxes) > 0:
        yellow = (0, 255, 255)
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 2)
            label = f"PERSON {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 2)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] PERSON on {name} - Sending Image")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{name}_{timestamp}_PERSON.jpg"
        _, img_encoded = cv2.imencode('.jpg', frame)
        
        try:
            requests.post(
                config['pi_endpoint'], 
                files={"image": (filename, img_encoded.tobytes(), 'image/jpeg')},
                timeout=2
            )
            return True
        except Exception as e:
            print(f"Failed to send: {e}")
            return False
    return False

# Initialize captures
for name, config in CAMERAS.items():
    cap = cv2.VideoCapture(config['url'])
    if cap.isOpened():
        captures[name] = cap
        print(f"Connected to {name}")
    else:
        print(f"Failed to connect to {name}")

try:
    while True:
        for name, config in CAMERAS.items():
            cap = captures.get(name)
            if not cap or not cap.isOpened():
                # Try to reconnect
                print(f"Reconnecting to {name}...")
                cap = cv2.VideoCapture(config['url'])
                if cap.isOpened():
                    captures[name] = cap
                else:
                    continue
            
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read from {name}")
                continue
            
            now = time.time()
            if now - last_sent[name] > COOLDOWN:
                if process_frame(name, config, frame):
                    last_sent[name] = now
        
        time.sleep(0.1)  # Small delay to prevent CPU spinning
        
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    # Clean up
    for cap in captures.values():
        cap.release()
    print("Cleanup complete")
