### CCTV for RPI with person detection

<img width="638" height="497" alt="image" src="https://github.com/user-attachments/assets/33b99e49-9491-4b3e-bcb6-feb295ab96bf" />

### What it does
This is a multi-camera person-detection system. Six IP cameras feed RTSP streams into a Raspberry Pi 5 running YOLOv8. When a person is detected, it uploads up to 6 annotated JPEG frames to one of 6 dedicated Raspberry Pi 4B recorders. Each recorder saves the images and sends an HTTP reset signal back when done, allowing the detector to start a new session.

### Architecture & threading model
The system has several concurrent layers:

CameraStream — one thread per camera, continuously reads frames from RTSP via OpenCV/FFmpeg, auto-reconnects on failure. Thread-safe via a lock; callers get a .copy() of the latest frame.
Main loop — polls each camera every ANALYSIS_INTERVAL (4s), feeds a bounded multiprocessing queue (maxsize=10).
YOLO worker — a separate process (not thread) to bypass Python's GIL for CPU-intensive inference. Runs YOLOv8n ONNX and puts results onto a result queue.
handle_results — a thread consuming detections, managing session state, and dispatching uploads.
ThreadPoolExecutor (4 workers, queue cap 20) — handles async HTTP uploads with retry/backoff.
Flask webhook server — a daemon thread receiving reset signals and health check pings.
Session watchdog — a thread detecting stuck sessions and force-resetting them.


### Session state machine
Each camera has its own independent CameraState with a proper enum: IDLE → ACTIVE → WAITING_RESET → COMPLETED → IDLE. This is a solid design choice — it replaces what was previously a tangle of boolean flags (as noted in the v3.22.1 changelog). The lock on each state object ensures thread safety.
A few key protections:

POST_RESET_COOLDOWN (6s after reset) — prevents stale frames from the old session triggering a new one immediately.
active_session_id (UUID) — each upload checks its session ID still matches; old in-flight uploads from a previous session are silently dropped.
RESET_DEDUP_WINDOW (2s) — ignores duplicate reset POSTs from the recorder.
WATCHDOG_TIMEOUT (150s) — force-resets any session stuck in WAITING_RESET, protecting against recorder crashes.


### Upload resilience
The upload_task function implements retry logic with exponential backoff: 3 attempts, base 2s delay. It handles HTTP 429 (recorder busy), 503 (unavailable), and 401 (auth failure) distinctly. The UPLOAD_QUEUE_SIZE cap prevents RAM growth under sustained load — excess frames are dropped and counted via dropped_uploads.

### Overall assessment

This is well-structured production code for an embedded system. The progression through versions shows clear, disciplined iteration — each version changelog reflects real lessons learned (race conditions, retry logic, watchdog tuning). The use of a SessionState enum, per-camera locking, session ID validation, and backpressure on the upload queue are all solid engineering choices for a concurrent, real-world deployment.

### Recorder
### What it does

Each RPi 4B runs one instance of this script, pointed at one camera. It does two things simultaneously:

Continuously records the RTSP stream into 5-minute MP4 segments via ffmpeg, moved to a network share when complete
Receives JPEG detections from the detector via HTTP, saves them alongside the video, then signals back to the detector when the session is done

### Architecture

main thread        → recording_loop() — ffmpeg segments, forever
daemon thread      → Flask :5000 — receives uploads + health/reset endpoints  
daemon thread      → cleanup_worker() — deletes old recordings every 6h
ThreadPoolExecutor → move_to_share_background() — moves MP4s + sends reset

The reset signal to the detector is sent from move_to_share_background() — meaning the detector only gets unblocked after the MP4 segment finishes and is moved. This is an important design coupling to understand.

### The upload flow

When a detection image arrives at /upload:

Authenticates via X-API-KEY header
Checks if recording has started (current_video_prefix != "")
Detects a new session when detection_id changes
Enforces MAX_IMAGES_PER_SESSION (returns 429 if exceeded — which is what the detector's retry logic handles)
Saves the JPEG directly to the final camera directory with a filename embedding the video prefix, detection ID, and timestamp
Returns frame count to the detector

### Installation
-eof-
