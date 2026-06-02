# 🛡️ Raspberry Pi Multi-Camera CCTV Recorder & Person Detector

An intelligent, decentralized, edge-AI CCTV detection system optimized for low-resource hardware. It continuously captures 24/7 video feeds via RTSP and triggers intelligent, filtered YOLOv8 person detection completely locally—no cloud costs, no privacy concerns.

---

## 🏗️ Architecture Overview

The system splits the workload across a cluster of Raspberry Pis. 6 IP cameras record 24/7 continuously to dedicated storage, while a Raspberry Pi 5 handles real-time machine learning inference, communicating asynchronously back to the recorders via webhooks.

```text
       [STORAGE RECORDER NETWORK]
       
 (Center)  (Gate)  (Entrance) (Garage)  (Behind)   (Left)    [Detector]
  [RPi4]    [RPi4]    [RPi4]    [RPi4]    [RPi4]    [RPi4]     *[RPi5]
    |         |         |         |         |         |          |
   CAM       CAM       CAM       CAM       CAM       CAM      ALL CAMS

```

* **Continuous Recording:** Six Raspberry Pi 4B nodes continuously save 5-minute video segments 24/7 to ensure zero data loss.
* **Smart Detection:** A central Raspberry Pi 5 pulls live frames, processes them through an optimized YOLOv8 engine, and pushes annotated JPEG alerts back to the respective RPi4 recorder when a person is detected.

---

## 🚀 Features

### 1. Multi-Camera Real-Time Detection with Smart Filtering

Processes up to 6 cameras simultaneously, running YOLOv8n person detection on each frame at configurable intervals (default: every 2.5 seconds). To eliminate false positives from weather, animals, and shadows, it uses a powerful three-layer filtering system:

| Filter | Purpose | Example |
| --- | --- | --- |
| **Area Filter** | Rejects detections that are too small (distant) or too large (headlights, vehicles close to camera). | A person at 50m might be 20x50px (1,000px) → **Rejected**;<br>

<br>A headlight flash at 27,000px → **Rejected** |
| **Aspect Ratio Filter** | Rejects detections with the wrong shape (cars, shadows, pets). | Car: 300x200px (ratio 0.67) → **Rejected**;<br>

<br>Person standing: 100x250px (ratio 2.5) → **Accepted** |
| **Top-Edge Filter** | Rejects "airborne" false positives near the top of the frame (birds, drones, partial figures). | Bird at y=5px with low confidence → **Rejected** |

> 💡 **Per-Camera Customization:** Each camera has independent thresholds, allowing fine-tuning per location (e.g., *Garage* accepts close-ups; *Center* strictly rejects cars).

### 2. Asynchronous Session Management & Webhooks

The system manages complete detection sessions and communicates seamlessly with individual recorder services to prevent latency build-up or memory leakage.

* **Session Isolation:** Generates a unique Session ID (SID) when a person enters a frame.
* **Frame Burst:** Captures up to 3 frames at ~3-second intervals (`1/3` ➔ `2/3` ➔ `3/3`) and streams them via HTTP POST.
* **Webhook Resets:** Waits for a `POST /session-reset` confirmation from the recorder node before clearing the cooldown state.

| Feature | Benefit |
| --- | --- |
| **Async Communication** | Detector uses an upload queue; never blocks the primary video capture thread. |
| **Partial Session Cleanup** | If a person leaves early, the session auto-cleans without leaving ghost alerts. |
| **Rate-Limited Deduplication** | Ignores duplicate reset signals within a 2-second window to prevent race conditions. |
| **Watchdog Timers** | Force-resets stuck sessions automatically after 5 minutes. |

### 3. Hardware-Optimized for Raspberry Pi

Running YOLOv8n on edge devices requires aggressive performance optimizations:

* **`YOLO_INPUT_SIZE = 480`:** Reduces total pixels processed by 75% compared to standard 960px streams.
* **Non-blocking Queues:** `put_nowait` ensures an unstable RTSP stream cannot deadlock the system core.
* **Isolated Processing:** YOLO inference runs on a dedicated separate multiprocessing subprocess.
* **Smart Debugging (v3.24+):** Automatically dumps rejected frames into `/home/pi/cctv_rejected` with the rejection reason explicitly appended to the filename.

#### Production Logging Example:

```text
[2026-06-01 22:55:32] 🆔 Garage: New detection session ae3a2d65
[2026-06-01 22:55:32] ⚡ Garage: 1/3 [SID:ae3a2d65] conf=0.70 box=44x118px area=5192px
[2026-06-01 22:55:35] ⚡ Garage: 2/3 [SID:ae3a2d65] conf=0.70 box=76x189px area=14364px
[2026-06-01 22:55:38] ⚡ Garage: 3/3 [SID:ae3a2d65] conf=0.59 box=104x258px area=26832px
[2026-06-01 22:55:38] 🛑 Garage: Last frame sent (3/3) [SID:ae3a2d65]
❌ [DEBUG] Garage: Area too large (40261px > 30000px) - box: 163x247px conf=0.66 REJECTED!

```

---

## 📦 Tech Stack

| Component | Technology |
| --- | --- |
| **Detection Engine** | YOLOv8n with ONNX Runtime |
| **Video Capture** | OpenCV with FFmpeg (RTSP over TCP) |
| **Web Server** | Flask (Webhook Receiver Engine) |
| **Concurrency** | Multiprocessing (YOLO), Threading (Streams), ThreadPoolExecutor (Uploads) |
| **Communication** | REST API + Webhooks |
| **Hardware Targets** | Raspberry Pi 5 (Detector), Raspberry Pi 4B (Recorders) |

---

## 🛠️ Getting Started

### Prerequisites

#### 1. System Libraries

Your Raspberry Pi OS requires system-level libraries for video encoding, array manipulation, and neural math acceleration.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-dev python3-venv
sudo apt install -y libopenblas-dev libatlas-base-dev libopenjp2-7
sudo apt install -y libopencv-dev libssl-dev libffi-dev ffmpeg

```

#### 2. Network Configurations

* **IP Cameras:** Ensure your cameras expose an accessible standard `rtsp://` stream.
* **Network Stability:** A **wired Ethernet connection** for the Raspberry Pis is highly recommended over Wi-Fi to handle 6 concurrent high-bandwidth RTSP feeds.

---

### Installation & Setup

#### 1. Clone the Repository

```bash
cd /home/pi
git clone [https://github.com/yoosamui/cctv.git](https://github.com/yoosamui/cctv.git)
cd cctv

```

#### 2. Configure Environment & Virtual Workspace

Create a Python virtual environment to protect system packages and pin dependencies to ensure ARM stability:

```bash
python3 -m venv ~/cctv_env
source ~/cctv_env/bin/activate

pip install --upgrade pip
pip install numpy==1.26.4
pip install opencv-python onnxruntime ultralytics flask requests python-dotenv

```

#### 3. Model Weights Deployment

The script expects a dedicated, resource-light `yolov8n.onnx` file inside its active working directory. You can export one directly from your environment:

```bash
yolo export model=yolov8n.pt format=onnx imgsz=480

```

Make sure the resulting file `yolov8n.onnx` is placed inside your cloned `/home/pi/cctv` directory.

#### 4. Environment Variables Setup

The software securely fetches passwords from a protected system file. Create the configuration path:

```bash
sudo mkdir -p /etc/cctv
sudo nano /etc/cctv/credentials.env

```

Add your secure local credentials to the file:

```env
CAM_PASS="your_camera_password_here"
WEBHOOK_SECRET="your_secure_webhook_key"

```

---

### 🧪 Verification Checklist

Verify your setup is functional by checking dependencies and model parameters before initialization:

1. **Verify Packages:** `pip list | grep -E "onnxruntime|opencv-python|ultralytics|flask"`
2. **Verify Model:** `ls -la yolov8n.onnx`
3. **Verify Environment Permissions:** `cat /etc/cctv/credentials.env`
4. **Run Live Library Test:**
```bash
python3 -c "import cv2, onnxruntime, flask; print('✅ System dependencies verification: OK')"

```



---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

```

```






