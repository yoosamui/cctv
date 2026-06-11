# Raspberry Pi Multi-Camera CCTV Recorder & Person Detector

**Project Status:** Current Version:`detector 3.26.1,  recorder 1.8.7`

> This project is actively under development. Features, APIs, configuration parameters, and documentation may change between releases.


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
| Detector version  | `3.26.1`                         |
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
| 6     | Top edge          | Rejects boxes hugging the top edge unless very confident. Per-camera `(margin, high_conf)` from `[TOP_EDGE_CONFIG]`, falling back to the global `[FILTERS]` values. |
| 7     | exclude zones     | Per-camera exclusion zones — detections whose box is mostly inside one of |



**HTTP API** (default port `5001`)

| Method | Path             | Auth (`X-API-KEY`) | Purpose                                          |
| ------ | ---------------- | ------------------ | ------------------------------------------------ |
| POST   | `/session-reset` | required           | Recorder signals end of segment → fire pending email, resume detection |
| POST   | `/reset`         | required           | Deprecated alias for `/session-reset`            |
| GET    | `/health`        | none               | Per-camera state/count + pending-email count     |

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
journalctl -u cctv-image-detector.service -f -o cat
journalctl -u cctv-video-recorder@Gate.service -f -o cat
```
---

## Installation and Configuration 

### Recorder — `/etc/cctv/config.ini` 
The cctv-video-recorder.py is the core engine of the CCTV system - 
a rock-solid video recording module that works independently without requiring the detector.

**step-1**

```text
 ssh in to your Rasberry pi terminal.
 make sure git and ffmpeg are installed. if not:
 $ sudo apt update
 $ sudo apt install git ffmpeg
 
 $ cd /home/pi
 $ git clone https://github.com/yoosamui/cctv.git
 $ cd cctv
 $ keep only files we need.
 $ rm -drf cctv-image* image_detector_config/ ansible-cctv/

 credentials
 $ sudo mkdir -p /etc/cctv
 $ cd /etc/cctv
 $ mv /home/pi/cctv/credentials.env .
 $ mv /home/pi/cctv/config.ini .


 add you credentials and save.
 $ nano credentials.env

 create the camera rtsp
 $ nano Gate.conf
 $ add this: URL=rtsp://user:pass@<IP>:554/Streaming/channels/101
 $ save the file 
 
 configure the recorder
 $ cd /home/pi/cctv
 $ nano config.ini
 make your changes and save

```

**/etc/cctv/config.ini**

```ini

open this file and modify the settings.
Modify settings reuires a service restart:

$ sudo nano /etc/cctv/config.ini



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
**NOTE:** set `detector_webhook_url` to the detector (RPi 5) address — a full URL, e.g. `http://<RPI5_IP>:5001/session-reset`.


After this the recorder is ready to run.
We will use a systemd service template for this.

```text
$ cd /home/pi/cctv
$ sudo cp cctv-video-recorder@.service /etc/systemd/system

$ sudo systemctl restart cctv-video-recorder@Gate.service
$ sudo systemctl enable --now cctv-video-recorder@Gate.service
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

### Detector


**`/etc/cctv/config.ini`**

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

**Credentials — `/etc/cctv/credentials.env`**

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

> `WEBHOOK_SECRET` must match on both nodes — it authenticates the `/upload` and
> `/session-reset` calls between them. Email alerts are enabled only when
> `SMTP_USER`, `SMTP_PASS`, and `ALERT_TO` are all set.

### Detector camera map

Cameras live in the `NODES` dict in `cctv-image-detector.py` — each entry maps a
camera name to its RTSP source (`cam_rtsp`) and the recorder upload endpoint
(`rpi_url`). Camera names here must match the `--name` passed to the recorder and
the keys used in the per-camera config sections.

Note the detector's `cam_rtsp` uses each camera's **sub-stream**
(`/Streaming/channels/102`, lower resolution — lighter for inference), while the
recorder records the **main stream** (`/Streaming/channels/101`, full quality).
This split is intentional.


### Camera Recorder Configuration

> Before starting the system, update the following configuration values:

> 1. **Set the `rpi_url` node** for each recorder camera.
> 2. **Set the `cam_rtsp` node** in the `cctv-image-detector.py` file.
> 3. Replace the default username (`yoo`) with the appropriate camera username.
> 4. Configure the camera password in:

   ```bash

   /etc/cctv/credentials.env
   CAM_PASS=<your_camera_password>

   ```


**The `NODES` dict in `cctv-image-detector.py`:**
~~~

# ==========================================
# CAMERA NODES
# ==========================================
NODES = {
    "Gate":     {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.99:554/Streaming/channels/102", "rpi_url": "http://192.168.1.14:5000/upload"},
    "Center":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.82:554/Streaming/channels/102", "rpi_url": "http://192.168.1.13:5000/upload"},
    "Entrance": {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.89:554/Streaming/channels/102", "rpi_url": "http://192.168.1.15:5000/upload"},
    "Garage":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.81:554/Streaming/channels/102", "rpi_url": "http://192.168.1.16:5000/upload"},
    "Behind":   {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.92:554/Streaming/channels/102", "rpi_url": "http://192.168.1.17:5000/upload"},
    "Left":     {"cam_rtsp": f"rtsp://yoo:{password}@192.168.1.93:554/Streaming/channels/102", "rpi_url": "http://192.168.1.18:5000/upload"}
}


~~~


## Detector installation

**Step-1**
```text

make sure git and ffmpeg are installed. if not:
 $ sudo apt update
 $ sudo apt install git ffmpeg
 
 $ cd /home/pi
 $ git clone https://github.com/yoosamui/cctv.git
 $ cd cctv
 $ keep only files we need.
 $ rm -drf cctv-video* ansible-cctv/ config.ini

 $ sudo mkdir -p /etc/cctv
 $ cd /etc/cctv
 $ cp /home/pi/cctv/image_detector_config/*.* .

you have created the configuration files
config.ini  credentials.env
take your time and make the changes you need and save the changes.
```


The detector is now ready; we can start it now.
For this we also use a systemd service unit.

**Step-2**

```text
$ cd /home/pi/cctv
$ sudo cp cctv-image-detector.service /etc/systemd/system

$ sudo systemctl restart cctv-image-detector.service
$ sudo systemctl status cctv-image-detector.service
$ sudo systemctl enable cctv-image-detector.service

journal log 
$ journalctl -u cctv-image-detector.service  -f -o cat

[2026-06-01 22:55:32] 🆔 Garage: New detection session ae3a2d65
[2026-06-01 22:55:32] ⚡ Garage: 1/3 [SID:ae3a2d65] conf=0.70 box=44x118px area=5192px
[2026-06-01 22:55:35] ⚡ Garage: 2/3 [SID:ae3a2d65] conf=0.70 box=76x189px area=14364px
[2026-06-01 22:55:38] ⚡ Garage: 3/3 [SID:ae3a2d65] conf=0.59 box=104x258px area=26832px
[2026-06-01 22:55:38] 🛑 Garage: Last frame sent (3/3) [SID:ae3a2d65]
❌ [DEBUG] Garage: Area too large (40261px > 30000px) - box: 163x247px conf=0.66 REJECTED!

finish.
```
## Hardware Performance Numbers
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






---


####  Configure Environment & Virtual Workspace

Create a Python virtual environment to protect system packages and pin dependencies to ensure ARM stability:

```bash
python3 -m venv ~/cctv_env
source ~/cctv_env/bin/activate

pip install --upgrade pip
pip install numpy==1.26.4
pip install opencv-python onnxruntime ultralytics flask requests python-dotenv
```

Ultralytics is required only to export YOLOv8 models to ONNX.
The detector itself runs entirely on ONNX Runtime.

####  Model Weights Deployment

The script expects a dedicated, resource-light `yolov8n.onnx` file inside its active working directory. You can export one directly from your environment:

```bash
yolo export model=yolov8n.pt format=onnx imgsz=480

```

Make sure the resulting file `yolov8n.onnx` is placed inside your cloned `/home/pi/cctv` directory.

####  Environment Variables Setup

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

**Example Detection**

<img width="1151" height="236" alt="image" src="https://github.com/user-attachments/assets/974b7523-e794-4ac5-9ae0-bf3148693916" />

---
**Example rejected False Positive**

<img width="1096" height="309" alt="image" src="https://github.com/user-attachments/assets/18ed7f4b-d03d-4862-adb8-0b6a31320837" />

---
---
### Footage


| Filename | Size | Type |
|----------|------|------|
| 2026-06-02_14-13-04_Gate.mp4 | 54.8 MB | Video |
| 2026-06-02_14-18-06_Gate.mp4 | 62.6 MB | Video |
| 2026-06-02_14-23-07_Gate.mp4 | 60.5 MB | Video |
| 2026-06-02_14-23-07_Gate_DETECTION_9b703362_2026-06-02_14-25-55-226.jpg | 55.5 kB | Image |
| 2026-06-02_14-23-07_Gate_DETECTION_9b703362_2026-06-02_14-26-10-245.jpg | 56.2 kB | Image |
| 2026-06-02_14-23-07_Gate_DETECTION_9b703362_2026-06-02_14-26-13-244.jpg | 56.3 kB | Image |
| 2026-06-02_14-28-07_Gate.mp4 | 58.1 MB | Video |
| 2026-06-02_14-28-07_Gate_DETECTION_880bcde7_2026-06-02_14-28-15-395.jpg | 55.8 kB | Image |
| 2026-06-02_14-28-07_Gate_DETECTION_880bcde7_2026-06-02_14-28-33-431.jpg | 58.9 kB | Image |
| 2026-06-02_14-28-07_Gate_DETECTION_880bcde7_2026-06-02_14-28-36-437.jpg | 57.1 kB | Image |
| 2026-06-02_14-33-09_Gate.mp4 | 54.8 MB | Video |
| 2026-06-02_14-33-09_Gate.mp4 | 51.4 MB | Video |
| 2026-06-02_14-43-11_Gate.mp4 | 61.5 MB | Video |
| 2026-06-02_14-48-12_Gate.mp4 | 67.0 MB | Video |
| 2026-06-02_14-53-13_Gate.mp4 | 67.2 MB | Video |
| 2026-06-02_14-58-14_Gate.mp4 | 67.2 MB | Video |
| 2026-06-02_15-03-15_Gate.mp4 | 66.2 MB | Video |
| 2026-06-02_15-08-16_Gate.mp4 | 63.9 MB | Video |
| 2026-06-02_15-13-17_Gate.mp4 | 64.8 MB | Video |
| 2026-06-02_15-13-17_Gate_DETECTION_836a884a_2026-06-02_15-15-28-183.jpg | 49.3 kB | Image |
| 2026-06-02_15-13-17_Gate_DETECTION_836a884a_2026-06-02_15-15-37-222.jpg | 53.5 kB | Image |
| 2026-06-02_15-13-17_Gate_DETECTION_836a884a_2026-06-02_15-15-40-199.jpg | 53.6 kB | Image |


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

---

## Recent changes

**Recorder 1.8.7**
- Auto-detect the ffmpeg RTSP timeout flag (`-stimeout` vs `-timeout`).
- Sweep orphaned staging files from `TEMP_DIR`.
- Close a `session_image_count` race (slot reserved under one lock, rolled back on
  failure) so concurrent uploads can't exceed `MAX_IMAGES_PER_SESSION`.
- `DETECTOR_WEBHOOK_URL` configurable via `[NETWORK] detector_webhook_url`.
 
**Detector 3.25.5**
- Top-edge filter is now per-camera: `[TOP_EDGE_CONFIG]` (margin,high_conf) is
  loaded via `load_camera_config()` with day/night overrides and passed to the
  YOLO worker, instead of the single global pair. Cameras not listed fall back to
  the global `[FILTERS]` `TOP_EDGE_MARGIN` / `TOP_EDGE_HIGH_CONF`.

**Detector 3.25.4**
- One session per recording segment: a stationary person no longer spawns a new
  session every few seconds (camera parks in `WAITING_RESET` until `/session-reset`).
- Fixes pending-email loss — only one session can complete per reset cycle.
- YOLO worker reloads its own day/night config on transition.
- Removed a hardcoded thread-pool block that overrode `config.ini`.

---



## Contributing
Pull Requests welcome! Feel free to report issues.


## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

**Current version:** detector 3.26.1, recorder 1.8.7


