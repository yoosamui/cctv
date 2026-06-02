# Raspberry Pi Multi-Camera CCTV Recorder & Person Detector

An intelligent, decentralized, edge-AI CCTV detection system optimized for low-resource hardware. It continuously captures 24/7 video feeds via RTSP and triggers intelligent, filtered YOLOv8 person detection completely locally—no cloud costs, no privacy concerns.

---

## Architecture Overview Eample

The system splits the workload across a cluster of Raspberry Pis. 6 IP cameras record 24/7 continuously to dedicated storage, while a Raspberry Pi 5 handles real-time machine learning inference, communicating asynchronously back to the recorders via webhooks.

```text

          IP CAMERAS
      /     |     |     \
     /      |     |      \

+---------+ +---------+ +---------+
| RPi4 #1 | | RPi4 #2 | | RPi4 #3 |
| Recorder| | Recorder| | Recorder|
+---------+ +---------+ +---------+

        HTTP/Webhooks

              ↓

      +----------------+
      | Raspberry Pi 5 |
      | YOLO Detector  |
      +----------------+

           RTSP Pull

```

* **Continuous Recording:** Six Raspberry Pi 4B nodes continuously save 5-minute video segments 24/7 to ensure zero data loss.
* **Smart Detection:** A central Raspberry Pi 5 pulls live frames, processes them through an optimized YOLOv8 engine, and pushes annotated JPEG alerts back to the respective RPi4 recorder when a person is detected.


* **Example Detections**

<img width="1151" height="236" alt="image" src="https://github.com/user-attachments/assets/974b7523-e794-4ac5-9ae0-bf3148693916" />



---

## Features

### 1. Multi-Camera Real-Time Detection with Smart Filtering

Processes up to 6 cameras simultaneously, running YOLOv8n person detection on each frame at configurable intervals (default: every 3 seconds). To eliminate false positives from weather, animals, and shadows, it uses a powerful three-layer filtering system:

Here's the complete description of the **three-layer filtering system**:

* **Example rejected False Positive**

<img width="1096" height="309" alt="image" src="https://github.com/user-attachments/assets/18ed7f4b-d03d-4862-adb8-0b6a31320837" />

---

## Three-Layer Filtering System

To eliminate false positives from weather, animals, shadows, and vehicle headlights, the system uses a powerful three-layer filtering system. Each camera can have **independent thresholds** tuned to its specific location and angle.

## Low-Latency Frame Strategy
The detector never queues video frames.
Each camera thread always keeps only the most recent frame.

Benefits:

- Constant memory usage
- No backlog growth
- No increasing detection delay
- Real-time behavior even under heavy CPU load
---

### Filter 1: Area Filter

**Purpose:** Rejects detections that are too small (distant people, pets, birds) or too large (headlights, vehicles very close to camera, large shadows).

**How it works:**
- Calculates bounding box area: `width × height` (pixels)
- Compares against `CAMERA_MIN_AREA` and `CAMERA_MAX_AREA` thresholds

**Example:**

| Scenario | Box Size | Area | Result | Reason |
|----------|----------|------|--------|--------|
| Person at 50m | 20×50px | 1,000px | ❌ REJECTED | Too small (distant) |
| Person at 10m | 60×150px | 9,000px | ✅ ACCEPTED | Normal range |
| Person very close | 160×250px | 40,000px | ✅ ACCEPTED | Normal (close-up) |
| Vehicle headlight | 170×240px | 40,800px | ❌ REJECTED | Too large (false positive) |
| Cat at 5m | 40×40px | 1,600px | ❌ REJECTED | Too small (animal) |

**Configuration example:**
```python 

# ==========================================
# CAMERA-SPECIFIC MIN/MAX AREA THRESHOLDS
# ==========================================
CAMERA_MIN_AREA = {
    'Gate': 780,          # Reject anything smaller than 780px
    'Center': 780,
    'Entrance': 780,
    'Garage': 780,
    'Behind': 780,
    'Left': 780
}

CAMERA_MAX_AREA = {
    'Gate': 30000,          # Reject anything larger than 30,000px
    'Center': 30000,
    'Entrance': 45000,      # Allow closer people (up to 45,000px)
    'Garage': 45000,
    'Behind': 25000,
    'Left': 25000,
}




```

---

### Filter 2: Aspect Ratio Filter

**Purpose:** Rejects detections with the wrong shape. Real people are typically taller than wide (aspect ratio > 1.2), while false positives like cars, shadows, and animals are often wider than tall (aspect ratio < 1.0).

**How it works:**
- Calculates aspect ratio: `height ÷ width`
- Compares against `CAMERA_ASPECT_RATIOS (min_ratio, max_ratio)`

**Example:**

| Object | Box Size | Aspect Ratio | Result | Reason |
|--------|----------|--------------|--------|--------|
| Standing person | 100×250px | 2.5 | ✅ ACCEPTED | Taller than wide |
| Sitting person | 150×180px | 1.2 | ✅ ACCEPTED | Slightly taller |
| Car (side view) | 300×120px | 0.4 | ❌ REJECTED | Much wider than tall |
| Shadow | 200×80px | 0.4 | ❌ REJECTED | Much wider than tall |
| Dog (side view) | 150×100px | 0.67 | ❌ REJECTED | Wider than tall |
| Person crouching | 120×110px | 0.92 | ❌ REJECTED | Too wide (depends on camera) |

**Configuration example:**
```python

# ==========================================
# CAMERA-SPECIFIC ASPECT RATIO FILTERS
# ==========================================
CAMERA_ASPECT_RATIOS = {
    'Gate': (1.2, 4.0),
    'Center': (1.2, 4.0),
    'Entrance': (1.2, 4.0),
    'Garage': (1.2, 4.0),
    'Behind': (1.2, 4.0),
    'Left': (1.2, 4.0)
}

```

**Special case - Left camera:** Mounted high, looking down. People appear wider than tall (ratio ~0.85). Minimum set to 0.85 to accept real people while still rejecting cars (0.4-0.6).

---

### Filter 3: Top-Edge (Ground) Filter

**Purpose:** Rejects "airborne" false positives that appear near the top edge of the frame, such as birds, drones, insects, or partial people walking into the top of frame.

**How it works:**
- Checks if the top of bounding box (`y1`) is within `TOP_EDGE_MARGIN` pixels of the top edge
- If yes, requires confidence ≥ `TOP_EDGE_HIGH_CONF` to accept

**Example:**

| Scenario | y1 position | Confidence | Result | Reason |
|----------|-------------|------------|--------|--------|
| Bird flying | y1 = 5px | 0.45 | ❌ REJECTED | Top-edge + low confidence |
| Bird flying | y1 = 5px | 0.85 | ✅ ACCEPTED | Top-edge + high confidence (rare) |
| Person entering from top | y1 = 3px | 0.62 | ❌ REJECTED | Partial person (wait for full body) |
| Normal person | y1 = 150px | 0.70 | ✅ ACCEPTED | Not near top edge |

**Configuration:**
```python
TOP_EDGE_MARGIN = 30           # Consider any detection with y1 <= 30px as "airborne"
TOP_EDGE_HIGH_CONF = 0.75      # Require 75% confidence to keep top-edge detections
```

**Why this matters:** Prevents false alerts from birds, drones, leaves, or insects flying close to the camera, while still allowing real people on elevated platforms (balconies, stairs) if they have high confidence.

---

## Filter Interaction Example

When a detection occurs, filters run in this order:

```
1. AREA FILTER (too small?) → REJECT if area < min_area
2. AREA FILTER (too large?) → REJECT if area > max_area  
3. ASPECT RATIO FILTER → REJECT if ratio outside [min, max]
4. TOP-EDGE FILTER → REJECT if y1 <= margin AND confidence < high_conf
5. ALL FILTERS PASSED → Detection accepted → Session starts
```

**Real example from the log:**
```
[2026-06-01 22:52:42] ❌ [DEBUG] Garage: Area too large (40261px > 30000px) - box: 163x247px conf=0.66 REJECTED!
```
- Area filter caught this false positive (headlight/vehicle)
- Aspect ratio (1.52) would have passed
- Top-edge (y1 not near top) would have passed
- **Area filter correctly rejected it**

---

## Per-Camera Tuning Philosophy

| Camera       | Min Area | Max Area | Aspect Ratio | Why |
|--------------|----------|----------|--------------|-----|
| **Gate**     | 450      | 30,000   | (1.4, 4.0)   | Strict - expects standing people |
| **Center**   | 500      | 30,000   | (1.5, 4.0)   | Very strict - rejects most false positives |
| **Entrance** | 300      | 25,000   | (1.2, 4.0)   | Balanced - allows sitting people |
| **Garage**   | 300      | 45,000   | (1.2, 4.0)   | Permissive - allows close-up people |
| **Behind**   | 400      | 25,000   | (0.6, 4.0)   | High camera angle - people appear wider |
| **Left**     | 500      | 25,000   | (0.85, 4.0)  | High camera angle - allows sitting/crouching |

---

## Debugging Features

When a detection is rejected, the system logs the exact reason with box dimensions:

```
❌ [DEBUG] Garage: Area too large (40261px > 30000px) - box: 163x247px conf=0.66 REJECTED!
❌ [DEBUG] Center: Bad aspect ratio (0.38) - box: 312x118px conf=0.58 REJECTED!
❌ [DEBUG] Left: Top-edge rejection - y1=5px, conf=0.45<0.75 REJECTED!
```

Optionally, rejected images can be saved to disk for visual inspection:
```
/home/pi/cctv_rejected/[2026-06-01_191651]__[DEBUG]_Left-_Bad_aspect_ratio_(0.60)-_REJECTED.jpg
```

This allows fine-tuning of thresholds based on actual false positives in your specific environment. 

### 2. Asynchronous Session Management & Webhooks

The system manages complete detection sessions and communicates seamlessly with individual recorder services to prevent latency build-up or memory leakage.

* **Session Isolation:** Generates a unique Session ID (SID) when a person enters a frame.
* **Frame Burst:** Captures up to 3 frames at ~3-second intervals (`1/3` ➔ `2/3` ➔ `3/3`) and streams them via HTTP POST.
* **Webhook Resets:** Waits for a `POST /session-reset` confirmation from the recorder node before clearing the cooldown state.

| Feature                        | Benefit |
| -------------------------------| -------------------------------------------------------------------------------------|
| **Async Communication**        | Detector uses an upload queue; never blocks the primary video capture thread.        |
| **Partial Session Cleanup**    | If a person leaves early, the session auto-cleans without leaving ghost alerts.      |
| **Rate-Limited Deduplication** | Ignores duplicate reset signals within a 2-second window to prevent race conditions. |
| **Watchdog Timers**            | Force-resets stuck sessions automatically after 5 minutes.                           |

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

## Tech Stack

| Component              | Technology                                                                |
| ---------------------- | ------------------------------------------------------------------------- |
| **Detection Engine**   | YOLOv8n with ONNX Runtime                                                 |
| **Video Capture**      | OpenCV with FFmpeg (RTSP over TCP)                                        |
| **Web Server**         | Flask (Webhook Receiver Engine)                                           |
| **Concurrency**        | Multiprocessing (YOLO), Threading (Streams), ThreadPoolExecutor (Uploads) |
| **Communication**      | REST API + Webhooks                                                       |
| **Hardware Targets**   | Raspberry Pi 5 (Detector), Raspberry Pi 4B (Recorders)                    |


## Session State Machine Explained 

The code has a fairly sophisticated state machine:
The Session State Machine manages the lifecycle of each person detection from first sighting to completion. 
Each camera has its own independent state machine.

* **State Diagram**
```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
    ┌─────────┐   Person    ┌──────────┐   Frame 3/3   ┌───────────────┐
    │  IDLE   │ ──────────► │  ACTIVE  │ ────────────► │ WAITING_RESET │
    └─────────┘             └──────────┘               └───────────────┘
         ▲                       │                            │
         │                       │                            │
         │                   No more                          │
         │                   frames                           │ Recorder
         │                   (timeout)                        │ sends reset
         │                       │                            │
         │                       ▼                            ▼
         │                  ┌────────────┐              ┌─────────────┐
         └──────────────────│  COMPLETED │◄─────────────│   RESET     │
                            └────────────┘   (auto)     │  RECEIVED   │
                                                        └─────────────┘
```
**State 1: IDLE**

Meaning: Camera is ready and waiting for a person to be detected.

Conditions to enter:
    System startup
    After a session is completed and reset
    After watchdog timeout or force reset

Behavior:
    No active detection session
    detection_id = None
    count = 0
    Ready to start a new session

**State 2: ACTIVE**

Meaning: A person has been detected and the system is capturing frames (1/3, 2/3, or 3/3).

Conditions to enter:

    Detection passes all filters (area, aspect ratio, top-edge)
    Camera state is IDLE
    Cooldown period has passed since last session

Behavior:

    Generates unique detection_id (e.g., a8dca7ee) when first frame arrives
    Increments count for each valid detection frame (1 → 2 → 3)
    Updates last_activity timestamp after each frame
    Sends each frame to recorder via draw_and_upload()
    If count == MAX_IMAGES (3), transitions to WAITING_RESET

**Example log:**
                                                     
```bash

[22:55:32] 🆔 Garage: New detection session ae3a2d65
[22:55:32] ⚡ Garage: 1/3 [SID:ae3a2d65] conf=0.70 box=44x118px
[22:55:35] ⚡ Garage: 2/3 [SID:ae3a2d65] conf=0.70 box=76x189px
[22:55:38] ⚡ Garage: 3/3 [SID:ae3a2d65] conf=0.59 box=104x258px
[22:55:38] 🛑 Garage: Last frame sent (3/3) [SID:ae3a2d65]
```
**State 3: WAITING_RESET**

Meaning: All 3 frames have been sent to the recorder. The system is waiting for confirmation
(reset signal) before allowing new detections.

Conditions to enter:
    count == MAX_IMAGES (3)
    Successfully sent the last frame to recorder
    State was ACTIVE

Behavior:
    No new detections accepted
    last_waiting_start timestamp recorded
    Waits for webhook call: POST /session-reset
    Watchdog will force reset if stuck for WATCHDOG_TIMEOUT (300 seconds / 5 minutes)

Why this state exists:

    Prevents overlapping sessions
    Ensures recorder has processed all frames
    Allows recorder to control when next detection starts

**State 4: COMPLETED**

Meaning: Recorder has acknowledged receipt of the session (sent reset signal).

Conditions to enter:

    Recorder calls /session-reset webhook
    State was WAITING_RESET

Behavior:

    Session is marked as complete
    last_reset_time updated
    Auto-transitions to IDLE after 5 seconds (cleanup delay)

**Log example:**
```
[22:58:42] 📡 Recorder signaled: Garage reset - DETECTION RESUMED [SID:ae3a2d65]

```
**State Transitions Summary**

| From                   | To	                    | Trigger
|------------------------|------------------------|------------------------------------------------------------|
| IDLE                   | ACTIVE		| Valid person detection + cooldown passed                   |
| ACTIVE                 | WAITING_RESET	| 3 frames sent to recorder                                  |
| WAITING_RESET	     | COMPLETED		| Recorder sends /session-reset webhook                      | 
| COMPLETED	     | IDLE		| 5-second auto-cleanup                                      |
| ACTIVE		     | IDLE		| Session timeout (no activity for SESSION_TIMEOUT seconds)  |
| WAITING_RESET	     | IDLE		| Watchdog force reset (stuck for WATCHDOG_TIMEOUT seconds)  |



| Component              | Technology                                                                |
| ---------------------- | ------------------------------------------------------------------------- |
| **Detection Engine**   | YOLOv8n with ONNX Runtime                                                 |
| **Video Capture**      | OpenCV with FFmpeg (RTSP over TCP)                                        |
| **Web Server**         | Flask (Webhook Receiver Engine)                                           |
| **Concurrency**        | Multiprocessing (YOLO), Threading (Streams), ThreadPoolExecutor (Uploads) |
| **Communication**      | REST API + Webhooks                                                       |
| **Hardware Targets**   | Raspberry Pi 5 (Detector), Raspberry Pi 4B (Recorders)                    |




This is actually one of the strongest parts of the system.

### Session ID Management

Each session has a unique 8-character hexadecimal ID (e.g., ae3a2d65), generated when first 
frame is detected.

Session ID is used for:
    Tracing all frames belonging to the same person
    Matching reset signals from recorder
    Debugging and log correlation

Session ID lifecycle:
    Created in ACTIVE state (first frame)
    Stored in state.detection_id and state.active_session_id
    Included in every upload to recorder
    Cleared when session completes or times out

**Important Timers**
```
| Timer                 | Value               | Purpose                                                                 |
|-----------------------|---------------------|-------------------------------------------------------------------------|
| `COOLDOWN`            | 5.0 seconds         | Wait after a session ends before starting a new one (prevents rapid re-triggering). |
| `SESSION_TIMEOUT`     | 300 seconds (5 min) | Force reset if no frames are received while in the `ACTIVE` state.      |
| `WATCHDOG_TIMEOUT`    | 300 seconds (5 min) | Force reset if the system is stuck in the `WAITING_RESET` state.        |
| `RESET_DEDUP_WINDOW`  | 2 seconds           | Ignore duplicate reset signals.                                         |
| `POST_RESET_COOLDOWN` | Unified into `COOLDOWN` | Legacy setting; now replaced by `COOLDOWN`.                         |
```

**Example: Complete Walk-through**
```
A person walks past the Garage camera:
Time	Event	                    State Change	          Details
0s	Person detected	          IDLE → ACTIVE	          New session ae3a2d65, frame 1/3 sent
3s	Person still visible	ACTIVE	                    Frame 2/3 sent
6s	Person still visible	ACTIVE	                    Frame 3/3 sent → State → WAITING_RESET
6s - 60s	Recorder processes images	WAITING_RESET	          System waits for webhook
60s	Recorder sends reset	WAITING_RESET → COMPLETED	Webhook received
65s	Auto-cleanup	          COMPLETED → IDLE	          Ready for next detection
```

**What happens if the recorder never sends reset?**

After WATCHDOG_TIMEOUT (300 seconds / 5 minutes), the watchdog forces a reset:
```
⚠️ Watchdog: Garage stuck. Force resetting.

```
This transitions WAITING_RESET → IDLE, freeing the camera for new detections.

**Race Condition Protection**
The system includes safeguards against timing issues:
```
Protection	Method
Duplicate resets	RESET_DEDUP_WINDOW ignores resets within 2 seconds
Upload race	Re-checks session ID inside lock after HTTP request
Stale frames	Validates detection_id matches current session before state change
```
**Why This Design?**
```
Requirement	          Solution
Avoid false positives	Requires 3 consecutive detections of same person
Handle people leaving early	Partial session cleanup (1/3 or 2/3 frames)
Allow asynchronous processing	WAITING_RESET state decouples detection from recorder
Prevent session mixing	Unique SID per session
Recover from failures	Watchdog timers force reset stuck sessions
Debuggability	          Every state change logged with SID
```

This state machine ensures reliable, traceable, and robust person detection across multiple cameras.




### Hardware Performance Numbers
<img width="928" height="453" alt="image" src="https://github.com/user-attachments/assets/476da0e4-90c2-4bfe-8f0e-2ccff9984515" />
Key Takeaway: The single most important factor is your hardware. If you are using a Raspberry Pi 5, your current setup is likely already running much faster than the 200ms range due to the Pi 5's significantly improved CPU and RAM bandwidth.


* **Performance Estimates for Your Setup**
Hardware	       Model & Format	            Input Size	       Reported Inference Time
Raspberry Pi 4	YOLOv8n (Optimized)              640px	              ~170 ms
Raspberry Pi 4	YOLOv8n (Standard ONNX)          640px	              ~173 ms
Raspberry Pi 4	YOLOv8n (Standard)               640px	              ~240 ms
Raspberry Pi 5	YOLOv8n (ONNX + Optimization)    480px (Estimate)	~78 ms
Raspberry Pi 5	YOLOv8n (ONNX)                   640px	              ~85 ms        

The default specific configuration is well-tuned for performance:

Input Size (480px): You are using YOLO_INPUT_SIZE = 480. This is significantly smaller than the standard 640px used in many benchmarks. 
Since the model processes fewer pixels, your inference time is likely lower than the standard benchmarks listed above.

Model Format (ONNX): ONNX Runtime is highly optimized for ARM CPUs like the one in the Pi, making it generally much faster than running the raw PyTorch model

Multi-tasking Overhead: Remember that your total latency includes not just inference, but also:

    Pre-processing: Resizing the image (which you have optimized).
    Post-processing: NMS (Non-Maximum Suppression) to filter overlapping boxes 
    Capture Time: Fetching the frame from the RTSP stream.




## Getting Started

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

Ultralytics is required only to export YOLOv8 models to ONNX.
The detector itself runs entirely on ONNX Runtime.
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

### Verification Checklist

Verify your setup is functional by checking dependencies and model parameters before initialization:

1. **Verify Packages:** `pip list | grep -E "onnxruntime|opencv-python|ultralytics|flask"`
2. **Verify Model:** `ls -la yolov8n.onnx`
3. **Verify Environment Permissions:** `cat /etc/cctv/credentials.env`
4. **Run Live Library Test:**
```bash
python3 -c "import cv2, onnxruntime, flask; print('✅ System dependencies verification: OK')"

```



---
### Footage

<img width="652" height="514" alt="image" src="https://github.com/user-attachments/assets/40604a3e-3ece-4447-b4f3-6e8b7bb77040" />



## License

Distributed under the MIT License. See `LICENSE` for more information.
current version: 3.24.6

```
TO BE CONTINUE
```






