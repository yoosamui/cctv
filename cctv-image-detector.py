import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import cv2
import requests
import time
from datetime import datetime
from ultralytics import YOLO

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

print("AI Master Brain: Yellow Boxes | 3s Interval | Fixed Naming")

while True:
    for name, config in CAMERAS.items():
        try:
            cap = cv2.VideoCapture(config['url'])
            success, frame = cap.read()
            cap.release() 

            if success:
                now = time.time()
                if now - last_sent[name] > COOLDOWN:
                    results = model(frame, classes=[0], conf=0.5, verbose=False, device='cpu')
                    
                    if len(results[0].boxes) > 0:
                        # --- MANUAL YELLOW BOX DRAWING ---
                        # BGR for Yellow is (0, 255, 255)
                        yellow = (0, 255, 255)
                        
                        for box in results[0].boxes:
                            # Get coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            # Draw the rectangle
                            cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 2)
                            
                            # Add label
                            label = f"PERSON {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 2)

                        print(f"[{datetime.now().strftime('%H:%M:%S')}] PERSON on {name} - Sending Image")
                        
                        # --- FILENAME: Gate_2026-05-03_11-50-10_PERSON.jpg ---
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        filename = f"{name}_{timestamp}_PERSON.jpg"
                        
                        _, img_encoded = cv2.imencode('.jpg', frame)
                        
                        try:
                            requests.post(
                                config['pi_endpoint'], 
                                files={"image": (filename, img_encoded.tobytes(), 'image/jpeg')},
                                timeout=2
                            )
                            last_sent[name] = now 
                        except Exception as e:
                            print(f"Failed to send: {e}")

        except Exception as e:
            print(f"Error: {e}")
    
    time.sleep(0.1)
