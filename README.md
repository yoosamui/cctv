

# Project Status

**Current Version:** `3.25.5, 1.8.7`

> This project is actively under development. Features, APIs, configuration parameters, and documentation may change between releases.

# Raspberry Pi Multi-Camera CCTV Recorder & Person Detector

An intelligent, decentralized edge-AI CCTV system that runs entirely on local
hardware — no cloud, no subscriptions, no third-party video storage.

Recorder nodes continuously capture RTSP camera streams to disk in fixed-length
segments. A separate detector node runs YOLOv8 person detection on the same
streams in real time, and the two halves coordinate over authenticated HTTP
webhooks so that exactly **one alert email is sent per person, per recording
segment** — with aggressive false-positive filtering in between.

| Item              | Value                            |
| ----------------- | -------------------------------- |
| Recorder version  | `1.8.7`                          |
| Detector version  | `3.25.5`                         |
| Detector model    | YOLOv8n (ONNX Runtime, CPU)      |
| Detector hardware | Raspberry Pi 5 (1 node, all cams)|
| Recorder hardware | Raspberry Pi 4B (1 node per cam) |
| Cameras tested    | 6                                |
| Recording mode    | Continuous 24/7, MP4 segments    |
| Alert transport   | HTTP webhooks + SMTP email       |
| Storage           | Local (network share)            |
| Cloud required    | No                               |

---

## Architecture

```text
        IP CAMERAS (RTSP)
       /    |    |    |    \
      /     |    |    |     \
  ┌─────────────────────────────────┐        ┌──────────────────────────┐
  │  RECORDER NODES (RPi 4B × N)    │        │  DETECTOR NODE (RPi 5)   │
  │  cctv-video-recorder.py         │        │  cctv-image-detector.py  │
  │                                 │        │                          │
  │  • ffmpeg → 5-min MP4 segments  │        │  • 1 CameraStream/camera │
  │  • stages in /tmp, moves to     │        │  • YOLOv8n worker (proc) │
  │    network share                │        │  • false-positive filters│
  │  • Flask :5000                  │        │  • Flask :5001           │
  │      /upload  /health           │        │      /session-reset      │
  │      /reset   /status           │        │      /health             │
  └─────────────────────────────────┘        └──────────────────────────┘
            ▲    detection frames (POST /upload)     │
            │ ◀──────────────────────────────────────┘
            │
            └──────────────────────────────────────▶  POST /session-reset
                 (end of each segment → fire email)
```

Both halves are independent processes connected only by HTTP. Either can restart
without corrupting the other's state.

### How a detection flows through the system

1. The **detector** grabs a fresh frame from each camera every `ANALYSIS_INTERVAL`
   seconds and hands it to a separate **YOLO worker process**.
2. The worker runs YOLOv8n inference, then applies the filter chain
   (aspect → high-confidence bypass → dark-pixel → min-area → max-area →
   top-edge). Surviving boxes are returned; rejected ones may be saved/emailed
   for tuning.
3. On the first valid detection a **session** opens (random 8-char `detection_id`).
   The detector annotates each frame and `POST`s it to the recorder's `/upload`,
   up to `MAX_IMAGES` frames, keeping the top frames by confidence for the alert.
4. After the last frame the camera **parks in `WAITING_RESET`** and the captured
   frames become a `PendingEmail` — it will not open a new session until the
   recorder signals the end of the segment. This is what guarantees one email per
   person per segment even if the person stands still for minutes.
5. The **recorder** finishes its current MP4 segment, moves it to the share, and
   `POST`s `/session-reset` to the detector.
6. The detector fires the pending alert email and resumes detection on that camera.

If the recorder's reset is ever missed (crash, network blip), two watchdogs make
sure nothing is lost: `pending_email_watchdog` sends the held email after
`PENDING_EMAIL_TIMEOUT` (360 s), and `session_watchdog` force-resets a camera
stuck in `WAITING_RESET` after `WATCHDOG_TIMEOUT`.

---

## Components

### `cctv-video-recorder.py` (recorder — one process per camera)

- Records a single RTSP camera to back-to-back MP4 segments with `ffmpeg`
  (`-c:v copy`, video only, `-rtsp_transport tcp`, `SEGMENT_DURATION` seconds each).
- The RTSP socket-timeout flag is **auto-detected** at startup (`-stimeout` on
  ffmpeg < 5.0, `-timeout` on ≥ 5.0) so an OS/ffmpeg upgrade can't silently break
  recording.
- Segments are written to `TEMP_DIR` (staging), then moved to
  `BASE_DIR/YYYY-MM-DD/<camera>/` by a background thread pool; the move triggers
  the `/session-reset` webhook to the detector.
- Hosts a Flask API (default port `5000`) that receives detection images from the
  detector and exposes health/status.
- Two cleanup jobs run on a schedule (`CLEANUP_INTERVAL_HOURS`): delete date
  folders older than `RETENTION_DAYS`, and sweep orphaned staging files left by
  failed moves (e.g. share unmounted) so `/tmp` can't fill up segment-by-segment.

**HTTP API**

| Method | Path             | Auth (`X-API-KEY`) | Purpose                                              |
| ------ | ---------------- | ------------------ | ---------------------------------------------------- |
| POST   | `/upload`        | required           | Receive a detection frame; enforces `MAX_IMAGES_PER_SESSION`. Returns `200` (not `429`) when the session is full so the detector stops retrying. |
| GET    | `/health`        | none               | Liveness + session counters + webhook-configured flag |
| GET    | `/status`        | none               | Current segment, dirs, session state                 |
| POST   | `/reset`         | none               | Manually clear the session counter                   |

**Run it**

```bash
python3 cctv-video-recorder.py --name Gate --url "rtsp://user:pass@192.168.1.99:554/Streaming/channels/102"
```

### `cctv-image-detector.py` (detector — one process, all cameras)

- Maintains a background `CameraStream` thread per camera (in the `NODES` dict),
  always holding the latest frame.
- A main loop queues one frame per camera every `ANALYSIS_INTERVAL` into a
  multiprocessing queue feeding a dedicated **YOLO worker process** (YOLOv8n via
  ONNX Runtime, CPU). Running YOLO in its own process keeps inference off the
  capture/Flask threads and lets a `check_yolo_health` thread restart it if it dies.
- Applies a layered false-positive filter chain with **per-camera** thresholds and
  **day/night overrides** (the worker reloads its own config on the day↔night
  transition, since it's a separate process).
- Builds detection sessions, uploads annotated frames to each camera's recorder,
  and manages the `IDLE → ACTIVE → WAITING_RESET → COMPLETED → IDLE` session
  state machine described above.
- Sends alert emails over SMTP — one **session email** with the best frames per
  detection, plus optional **rejection emails** for tuning the filters.

**Filter chain** (in `is_valid_person_detection_worker`)

| Order | Filter            | Notes                                                                 |
| ----- | ----------------- | --------------------------------------------------------------------- |
| 1     | Aspect ratio      | **Always applied**, even for high-confidence boxes — a person is taller than wide; rejects confident-but-wrong wide boxes (vehicles). |
| 2     | High-conf bypass  | `confidence ≥ FILTER_CONFIDENCE` skips the remaining filters (shape already checked). |
| 3     | Dark pixel        | Rejects boxes that are mostly dark (shadows/noise).                   |
| 4     | Min area          | Per-camera `CAMERA_MIN_AREA`.                                         |
| 5     | Max area          | Per-camera `CAMERA_MAX_AREA` — rejects headlights, close-up objects.  |
| 6     | Top edge          | Rejects boxes hugging the top edge unless very confident.             |

**HTTP API** (default port `5001`)

| Method | Path             | Auth (`X-API-KEY`) | Purpose                                          |
| ------ | ---------------- | ------------------ | ------------------------------------------------ |
| POST   | `/session-reset` | required           | Recorder signals end of segment → fire pending email, resume detection |
| POST   | `/reset`         | required           | Deprecated alias for `/session-reset`            |
| GET    | `/health`        | none               | Per-camera state/count + pending-email count     |

**Run it**

```bash
python3 cctv-image-detector.py
```

---

## Installation and Configuration 

### Recorder — `config.ini` (next to the script)
the recorder (cctv-video-recorder.py) in the heard of the cctv system.

**step-1**

```text
 ssh in to your Rasberry pi terminal.
 make sure you have git,ffmpeg are install. if not
 $ sudo apt update
 $ sudo apt install git, ffmpeg
 
 $ cd /home/pi
 $ git clone https://github.com/yoosamui/cctv.git
 $ cd cctv
 $ keep only files we need.
 $ rm -drf cctv-image* image_detector_config/ ansible-cctv/

 # credentials
 $ sudo mkdir -p /etc/cctv
 $ cd /etc/cctv
 $ cp /home/pi/cctv/credentials/recorder/credentials.env .

 # add you credentials and save.
 $ nano credentials.env

 # create the camera rtsp
 $ nano Gate.conf
 $ add this: URL=rtsp://user:pass@<IP>:554/Streaming/channels/101
 $ save the file 
 
 # configure the recorder
 $ cd /home/pi/cctv
 $ nano config.ini. 
 # made you changes and save

```

**config.ini**

```ini
[STORAGE]
base_dir = /media/share/cameras/cctv-storage   ; final recordings
temp_dir = /tmp/cctv_staging                    ; staging before move
retention_days = 14                             ; delete older date folders
cleanup_interval_hours = 6

[RECORDING]
segment_duration = 300                          ; seconds per MP4 segment
max_images_per_session = 3                      ; cap on detection frames

[NETWORK]
flask_port = 5000
detector_webhook_url = http://192.168.1.19:5001/session-reset   ; optional
```
**NOTE:** detector_webhook_url = <IP OF THE IMAGE_DETECTOR ON THE RPI5> 


After this the recorder is ready to run.
We will use a systemd service template for this.

```text
$ cd /home/pi/cctv
$ sudo cp cctv-video-recorder@.service /etc/systemd/system

$ sudo systemctl restart cctv-video-recorder@Gate.service
$ sudo systemctl status cctv-video-recorder@Gate.service

# journal log 
$ journalctl -u cctv-video-recorder@Garage.service  -f -o cat

Time      Event       Details
16:02:59  Reset       Garage session confirmed
16:07:59  Recording   2026-06-07_16-07-59_Garage.mp4
16:08:00  Moved       2026-06-07_16-02-59_Garage.mp4
16:08:00  Reset       Garage session confirmed
16:13:00  Recording   2026-06-07_16-13-00_Garage.mp4
16:13:00  Moved       2026-06-07_16-07-59_Garage.mp4
16:13:01  Reset       Garage session confirmed
16:18:01  Recording   2026-06-07_16-18-01_Garage.mp4
16:18:01  Moved       2026-06-07_16-13-00_Garage.mp4
16:18:01  Reset       Garage session confirmed

finish. 
```



### Detector — `/etc/cctv/config.ini`

Key sections (see the shipped file for the full annotated set):

- `[GENERAL]` — `ANALYSIS_INTERVAL`, `MAX_IMAGES`, `COOLDOWN`, `WEBHOOK_PORT`.
- `[YOLO]` — `CONFIDENCE`, `INPUT_SIZE`, `IOU`, `RESTART_DELAY`.
- `[FILTERS]` — enable flags + thresholds; `FILTER_CONFIDENCE` is the
  high-confidence bypass cutoff.
- `[SESSION]` — `WATCHDOG_TIMEOUT`, `WATCHDOG_CHECK`, `RESET_DEDUP_WINDOW`
  (`SESSION_TIMEOUT` is loaded but no longer used on the hot path).
- `[UPLOAD]` — worker/queue sizing and retry policy.
- `[DEBUG]` — rejected-image saving for filter tuning.
- `[CAMERA_MIN_AREA]`, `[CAMERA_MAX_AREA]`, `[CAMERA_ASPECT_RATIO]`,
  `[TOP_EDGE_CONFIG]` — per-camera thresholds, with `[NIGHT_*]` overrides applied
  outside the `07:00–19:00` daytime window.

### Credentials — `/etc/cctv/credentials.env`

Loaded with `python-dotenv` on both nodes:

```bash
WEBHOOK_SECRET="shared-secret-used-for-X-API-KEY-auth"

# Detector only:
CAM_PASS="rtsp-camera-password"
SMTP_HOST="smtp.gmail.com"
SMTP_PORT="587"
SMTP_USER="you@example.com"
SMTP_PASS="app-password"
ALERT_TO="alerts@example.com"
```

`WEBHOOK_SECRET` must match on both nodes — it authenticates the `/upload` and
`/session-reset` calls between them. Email alerts are enabled only when
`SMTP_USER`, `SMTP_PASS`, and `ALERT_TO` are all set.

### Detector camera map

Cameras live in the `NODES` dict in `cctv-image-detector.py` — each entry maps a
camera name to its RTSP source (`cam_rtsp`) and the recorder upload endpoint
(`rpi_url`). Camera names here must match the `--name` passed to the recorder and
the keys used in the per-camera config sections.

---

## Requirements

- Python 3
- `ffmpeg` (recorder)
- Python packages: `opencv-python`, `onnxruntime`, `numpy`, `flask`, `requests`,
  `python-dotenv`
- `yolov8n.onnx` in the detector's working directory

---

## Deployment (systemd)

Both nodes ship templated/standard unit files:

- **Recorder** — `cctv-video-recorder@.service` (instanced per camera):
  ```bash
  sudo systemctl enable --now cctv-video-recorder@Gate.service
  ```
  Reads `/etc/cctv/<camera>.conf` for the `URL` env var and runs
  `--name <camera> --url ${URL}`.

- **Detector** — `cctv-image-detector.service`:
  ```bash
  sudo systemctl enable --now cctv-image-detector.service
  ```

Logs go to the systemd journal:

```bash
journalctl -u cctv-image-detector.service -f
journalctl -u cctv-video-recorder@Gate.service -f
```

---

## Recent changes

**Recorder 1.8.7**
- Auto-detect the ffmpeg RTSP timeout flag (`-stimeout` vs `-timeout`).
- Sweep orphaned staging files from `TEMP_DIR`.
- Close a `session_image_count` race (slot reserved under one lock, rolled back on
  failure) so concurrent uploads can't exceed `MAX_IMAGES_PER_SESSION`.
- `DETECTOR_WEBHOOK_URL` configurable via `[NETWORK] detector_webhook_url`.

**Detector 3.25.4**
- One session per recording segment: a stationary person no longer spawns a new
  session every few seconds (camera parks in `WAITING_RESET` until `/session-reset`).
- Fixes pending-email loss — only one session can complete per reset cycle.
- YOLO worker reloads its own day/night config on transition.
- Removed a hardcoded thread-pool block that overrode `config.ini`.

---

## License

See [LICENSE](LICENSE).

////////////////////////////////////

# Project Status

**Current Version:** `v3.24.6`

> This project is actively under development. Features, APIs, configuration parameters, and documentation may change between releases.

---

# Raspberry Pi Multi-Camera CCTV Recorder & Person Detector

An intelligent, decentralized edge-AI CCTV detection system optimized for low-resource hardware.

The platform continuously records RTSP camera streams 24/7 while performing real-time person detection using YOLOv8 entirely on local hardware.

### Key Benefits

* 🔒 100% local processing (no cloud services required)
* 🎥 Continuous 24/7 recording
* 👤 Real-time person detection
* 🚫 Advanced false-positive filtering
* ⚡ Optimized for Raspberry Pi hardware
* 🔄 Asynchronous recorder/detector architecture
* 📡 Multi-camera support (6 cameras tested)

No subscription fees. No cloud dependency. No privacy concerns.

---

## At a Glance

| Item              | Value                      |
| ----------------- | -------------------------- |
| Cameras Tested    | 6                          |
| Detector          | YOLOv8n (ONNX Runtime)     |
| Detector Hardware | Raspberry Pi 5             |
| Recorder Hardware | Raspberry Pi 4B            |
| Recording Mode    | Continuous 24/7            |
| Detection Mode    | Real-Time Person Detection |
| Alert Transport   | HTTP Webhooks              |
| Storage           | Local                      |
| Cloud Required    | No                         |

---

## Architecture Overview

The system distributes workloads across a cluster of Raspberry Pis.

Six Raspberry Pi 4B recorder nodes continuously save 5-minute video segments from IP cameras, while a Raspberry Pi 5 performs real-time AI inference and communicates asynchronously back to the recorders via webhooks.

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

### Recording Layer

* Continuous 24/7 recording
* 5-minute video segment rotation
* Independent storage on each recorder node
* No single point of failure

### Detection Layer

* Centralized YOLOv8n inference
* Real-time RTSP frame acquisition
* Advanced false-positive filtering
* Session-based event tracking
* Annotated JPEG alert generation

**Example Detection**

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

| Timer | Value | Purpose |
|--------|--------|----------|
| `COOLDOWN` | 5 seconds | Wait after a session ends before allowing a new session. |
| `SESSION_TIMEOUT` | 300 seconds (5 min) | Force reset if no frames are received while in `ACTIVE`. |
| `WATCHDOG_TIMEOUT` | 300 seconds (5 min) | Force reset if stuck in `WAITING_RESET`. |
| `RESET_DEDUP_WINDOW` | 2 seconds | Ignore duplicate reset signals. |
| `POST_RESET_COOLDOWN` | Merged into `COOLDOWN` | Legacy parameter retained for documentation purposes. |

**Example: Complete Walk-through**

A person walks past the Garage camera:

| Time | Event | State Change | Details |
|------|-------|-------------|---------|
| 0s | Person detected | `IDLE → ACTIVE` | New session `ae3a2d65`, frame `1/3` sent |
| 3s | Person still visible | `ACTIVE` | Frame `2/3` sent |
| 6s | Person still visible | `ACTIVE → WAITING_RESET` | Frame `3/3` sent |
| 6s - 60s | Recorder processes images | `WAITING_RESET` | System waits for webhook |
| 60s | Recorder sends reset | `WAITING_RESET → COMPLETED` | Webhook received |
| 65s | Auto-cleanup | `COMPLETED → IDLE` | Ready for next detection |


**What happens if the recorder never sends reset?**

After WATCHDOG_TIMEOUT (300 seconds / 5 minutes), the watchdog forces a reset:
```
⚠️ Watchdog: Garage stuck. Force resetting.

```
This transitions WAITING_RESET → IDLE, freeing the camera for new detections.

**Race Condition Protection**
The system includes safeguards against timing issues:
| Protection | Method |
|------------|--------|
| Duplicate resets | `RESET_DEDUP_WINDOW` ignores resets received within 2 seconds. |
| Upload race | Re-checks the session ID inside the lock after the HTTP request completes. |
| Stale frames | Validates that `detection_id` matches the current active session before any state change. |


**Why This Design?**

| Requirement | Solution |
|-------------|----------|
| Avoid false positives | Requires 3 consecutive detections of the same person. |
| Handle people leaving early | Partial session cleanup (1/3 or 2/3 frames). |
| Allow asynchronous processing | `WAITING_RESET` state decouples detection from recorder processing. |
| Prevent session mixing | Unique Session ID (SID) assigned to every session. |
| Recover from failures | Watchdog timers automatically reset stuck sessions. |
| Debuggability | Every state transition is logged with the associated SID. |

This state machine ensures reliable, traceable, and robust person detection across multiple cameras.


### Hardware Performance Numbers
## System Performance Snapshot

| Metric | Value | Notes |
|--------|-------|------|
| CPU Temperature | `60.4°C` | Safe operating temperature for Raspberry Pi 5 |
| Total CPU Usage | `68.6%` of one core | Combined usage from detector processes |
| Total Memory Usage | `4%` | Very low RAM utilization |
| System Uptime | `5 days, 22 hours` | Stable long-term operation |
| Load Average (1m) | `0.72` | Low system load |
| Load Average (5m) | `1.75` | Moderate sustained load |
| Load Average (15m) | `1.92` | Stable multi-process workload |
| Total RAM | `8062.4 MiB` | Raspberry Pi 5 (8GB) |
| Free RAM | `5901.2 MiB` | Large memory headroom remaining |
| Used RAM | `562.0 MiB` | Actual application memory usage |
| Buff/Cache | `1712.6 MiB` | Linux filesystem cache |
| Available RAM | `7500.3 MiB` | Memory still available to applications |
| Swap Usage | `0.0 MiB / 2048 MiB` | No swap pressure |
| Total Tasks | `177` | System processes |
| Running Tasks | `1` | Most services idle/waiting |
| Sleeping Tasks | `176` | Normal Linux behavior |

---

## Detector Process Usage

| PID | Process | CPU Usage | RAM Usage | Resident Memory | Runtime |
|-----|---------|-----------|-----------|----------------|---------|
| `1678671` | `python3` | `51.7%` | `1.6%` | `129072 KiB` | `1:17.53` |
| `1678663` | `python3` | `19.0%` | `2.2%` | `184240 KiB` | `0:30.27` |

---

## Key Takeaways

- ✅ CPU temperature remains within safe limits
- ✅ Memory usage is extremely low for a 6-camera AI system
- ✅ No swap usage indicates healthy RAM availability
- ✅ System load remains stable during inference
- ✅ Suitable for continuous 24/7 operation
- ✅ Enough remaining resources for additional services or cameras

The single most important factor is your hardware. If you are using a Raspberry Pi 5, the current setup 
is likely already running much faster than the 200ms range due to the Pi 5's significantly improved CPU and RAM bandwidth.


* **Performance Estimates**

| Hardware           | Model & Format                    | Input Size | Inference Time |
| ------------------ | --------------------------------- | ---------- | -------------- |
| Raspberry Pi 4     | YOLOv8n (Optimized)               | 640px      | ~170 ms        |
| Raspberry Pi 4     | YOLOv8n (Standard ONNX)           | 640px      | ~173 ms        |
| Raspberry Pi 4     | YOLOv8n (Standard)                | 640px      | ~240 ms        |
| **Raspberry Pi 5** | **YOLOv8n (ONNX + Optimization)** | **480px**  | **~78 ms**     |
| Raspberry Pi 5     | YOLOv8n (ONNX)                    | 640px      | ~85 ms         |


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


## Roadmap

Planned future improvements:

- Coral TPU support
- ONNX model benchmarking suite
- Web dashboard for live camera status
- Historical event search
- Automatic recorder health monitoring
- Docker deployment support
- Multi-site federation support

Contributions and suggestions are welcome.







