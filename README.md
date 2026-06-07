# Raspberry Pi Multi-Camera CCTV Recorder & Person Detector

**Project Status:** Current Version:`detector 3.25.5,  recorder 1.8.7`

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

 credentials
 $ sudo mkdir -p /etc/cctv
 $ cd /etc/cctv
 $ cp /home/pi/cctv/credentials/recorder/credentials.env .

 add you credentials and save.
 $ nano credentials.env

 create the camera rtsp
 $ nano Gate.conf
 $ add this: URL=rtsp://user:pass@<IP>:554/Streaming/channels/101
 $ save the file 
 
 configure the recorder
 $ cd /home/pi/cctv
 $ nano config.ini. 
 made you changes and save

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
**NOTE:** detector_webhook_url = IP_OF_THE_IMAGE_DETECTOR_RPI5 


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
  `/session-reset` calls between them. Email alerts are enabled only when
  `SMTP_USER`, `SMTP_PASS`, and `ALERT_TO` are all set.

### Detector camera map

Cameras live in the `NODES` dict in `cctv-image-detector.py` — each entry maps a
camera name to its RTSP source (`cam_rtsp`) and the recorder upload endpoint
(`rpi_url`). Camera names here must match the `--name` passed to the recorder and
the keys used in the per-camera config sections.

> YOU NEED TO SET THE NODES rpi_url for all recorder cameras.
  YOU ALSO NEED TO SET THE NODES cam_rtsp 
  in the cctv-image-detector.py file.

**rows 530 -542**
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

make sure you have git,ffmpeg are install. if not
 $ sudo apt update
 $ sudo apt install git, ffmpeg
 
 $ cd /home/pi
 $ git clone https://github.com/yoosamui/cctv.git
 $ cd cctv
 $ keep only files we need.
 $ rm -drf cctv-video ansible-cctv/ config.ini

 $ sudo mkdir -p /etc/cctv
 $ cd /etc/cctv
 $ cp /home/pi/cctv/image_detector_config/*.* .

you have create the configuration files
config.ini  credentials.env
take your time and made the chnages you need and save the changes.
```


The detetor is now ready  we can start it now.
For this we also use a systemd service unit.

**Step-2**

```text
$ cd /home/pi/cctv
$ sudo cp cctv-image-detector.service /etc/systemd/system

$ sudo systemctl restart cctv-image-detector.service
$ sudo systemctl status cctv-image-detector.service
$ sudo systemctl renable cctv-image-detector.service

journal log 
$ journalctl -u cctv-image-detector.service  -f -o cat
+----------+----------+----------------+--------------------------+
| Time     | Camera   | Event          | Details                  |
+----------+----------+----------------+--------------------------+
| 17:26:54 | Gate     | Recorder Reset | No pending email         |
| 17:27:52 | Entrance | New Session    | bc834c7c                 |
| 17:27:53 | Center   | New Session    | 19b5ef05                 |
| 17:27:56 | Center   | REJECTED       | Bad aspect ratio (0.38)  |
| 17:28:00 | Center   | Email Sent     | Rejection notice         |
| 17:28:10 | Garage   | Recorder Reset | No pending email         |
| 17:28:12 | Gate     | New Session    | 095599eb                 |
| 17:29:02 | Center   | Idle Timeout   | Cleanup (1 frame saved)  |
| 17:29:38 | Center   | New Session    | 42117b4c                 |
| 17:30:10 | Entrance | Recorder Reset | Email sent (bc834c7c)    |
| 17:30:10 | Center   | Recorder Reset | Email sent (42117b4c)    |
| 17:30:18 | Behind   | Recorder Reset | No pending email         |
| 17:30:25 | Left     | Recorder Reset | No pending email         |
| 17:31:55 | Gate     | Recorder Reset | Email sent (095599eb)    |
| 17:32:10 | Gate     | New Session    | 92fedae7                 |
| 17:33:11 | Garage   | Recorder Reset | No pending email         |
| 17:34:58 | Entrance | New Session    | a37ea336                 |
+----------+----------+----------------+--------------------------+

finish.
```

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

Ultralytics is required only to export YOLOv8 models to ONNX.
The detector itself runs entirely on ONNX Runtime.
```

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



---
### Footage

<img width="652" height="514" alt="image" src="https://github.com/user-attachments/assets/40604a3e-3ece-4447-b4f3-6e8b7bb77040" />





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
- Top-edge filter is now per-camera: [TOP_EDGE_CONFIG] (margin,high_conf) is
- loaded via load_camera_config() with day/night overrides and passed to the
- YOLO worker, instead of the single global pair. Cameras not listed fall
- back to the global [FILTERS] TOP_EDGE_MARGIN / TOP_EDGE_HIGH_CONF.

**Detector 3.25.4**
- One session per recording segment: a stationary person no longer spawns a new
  session every few seconds (camera parks in `WAITING_RESET` until `/session-reset`).
- Fixes pending-email loss — only one session can complete per reset cycle.
- YOLO worker reloads its own day/night config on transition.
- Removed a hardcoded thread-pool block that overrode `config.ini`.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
**current version:** 3.25.5, 1.8.7` 

See [LICENSE](LICENSE).
