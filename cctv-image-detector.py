#!/usr/bin/env python3
# ==============================================================================
# CCTV IMAGE DETECTOR - VERSION 3.26.1  cloude+deepseek
# ==============================================================================
#
# IMPROVEMENTS in v3.26.1:
#   1. Added DRAW_ZONES ([IMAGE], default false): when on, the per-camera
#      exclusion zones are drawn (orange, labeled "ZONE") on rejected images and
#      rejection emails — for ALL rejection reasons, not only exclude_zone — so
#      the regions can be visually tuned against the boxes being caught.
#
# IMPROVEMENTS in v3.26.0:
#   1. Added per-camera exclusion-zone filter [CAMERA_EXCLUDE_ZONE]: detections
#      whose bounding box is mostly inside a fixed ignore region are rejected.
#      Targets static false positives at a permanent location (clutter, fixtures
#      such as a hanging clothes rack) that score as "person". Runs FIRST, at ALL
#      confidence levels (before the FILTER_CONFIDENCE high-conf bypass). Rejection
#      is coverage-based (EXCLUDE_ZONE_COVERAGE, default 0.6 of box area), so a real
#      person clipping the edge of the region is still detected. Zones honor
#      day/night overrides and are reloaded by the worker on transition.
#
# IMPROVEMENTS in v3.25.5:
#   1. Top-edge filter is now per-camera: [TOP_EDGE_CONFIG] (margin,high_conf) is
#      loaded via load_camera_config() with day/night overrides and passed to the
#      YOLO worker, instead of the single global pair. Cameras not listed fall
#      back to the global [FILTERS] TOP_EDGE_MARGIN / TOP_EDGE_HIGH_CONF.
#
# IMPROVEMENTS in v3.25.4:
#   1. Fixed continuous-presence session spam: a person standing still no longer
#      spawns a new detection session every few seconds. After a session captures
#      MAX_IMAGES frames the camera parks in WAITING_RESET and resumes only when
#      the recorder's /session-reset arrives (one session per recording segment).
#   2. This also fixes pending-email loss: only one session can complete per reset
#      cycle, so pending_emails[camera] can no longer be overwritten before sending.
#   3. YOLO worker reloads its own day/night camera config on transition (the
#      worker is a separate process and never saw the main process's updates).
#   4. Removed hardcoded THREAD POOL SETTINGS that overwrote config.ini values;
#      removed unused merge_camera_config() helper.
#
# IMPROVEMENTS in v3.25.3:
#   1. Fixed session watchdog to use separate timeouts for different purposes
#      - ACTIVE_SESSION_IDLE_TIMEOUT (60s): How long an incomplete session can go
#        without a new detection before it's considered abandoned
#      - SESSION_TIMEOUT (300s): Legacy timeout for WAITING_RESET state (safety)
#      - Partial sessions now saved as pending emails when they timeouta
#   2. Added ACTIVE_SESSION_IDLE_TIMEOUT config option with 60s default
#
# IMPROVEMENTS in v3.25.2:
#   1. Decoupled image capture from email sending
#      - Camera returns to IDLE immediately after 3/3 frames are captured
#      - New detections can start right away on the same camera
#      - Email is held in a PendingEmail object and fired when the recorder
#        sends its /session-reset signal (end of 5-minute segment)
#      - PendingEmail watchdog expires and sends email after 360s if the
#        recorder reset never arrives (e.g. crash / missed signal)
#   2. Cleaned up CameraStream class (removed duplicate methods)
#   3. Fixed email sending from main process only
#
# IMPROVEMENTS in v3.25.1:
#   1. Fixed rate limiting logic - now correctly processes frames at ANALYSIS_INTERVAL
#   2. Improved debug logging for frame timing
#
# AUTHOR: yoosamui
# DATE: 2026-06-05
# ==============================================================================
import glob
import cv2
import multiprocessing
import threading
import time
import requests
import os
import sys
import uuid
import onnxruntime as ort
import numpy as np
import configparser
import datetime
from queue import Empty, Full
from urllib.parse import quote
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import logging
from collections import defaultdict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

VERSION = "3.26.1"

# ==========================================
# SILENCE LOGS
# ==========================================
werkzeug_log = logging.getLogger('werkzeug')
werkzeug_log.disabled = True
logging.getLogger('werkzeug').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


# ==========================================
# LOAD CONFIGURATION
# ==========================================

config = configparser.ConfigParser()

if not config.read("/etc/cctv/config.ini"):
    print("ERROR: Failed to load /etc/cctv/config.ini")
    sys.exit(1)

# ==========================================
# CONFIGURATION HELPERS
# ==========================================
def cfg_int(section, key):
    return config.getint(section, key)

def cfg_float(section, key):
    return config.getfloat(section, key)

def cfg_bool(section, key):
    return config.getboolean(section, key)

def cfg_str(section, key):
    return config.get(section, key)


# ==========================================
# TIME-BASED CONFIGURATION HELPERS
# ==========================================
DAY_START_HOUR = 7   # 7 AM
DAY_END_HOUR = 19    # 7 PM

def is_daytime():
    """Return True if current time is daytime."""
    now = datetime.datetime.now()
    current_hour = now.hour
    return DAY_START_HOUR <= current_hour < DAY_END_HOUR

def parse_zones(value):
    """Parse 'x1,y1,x2,y2; x1,y1,x2,y2; ...' into a list of (x1,y1,x2,y2) int
    tuples. Whitespace and trailing semicolons are tolerated; malformed groups
    (not exactly 4 ints) are skipped."""
    zones = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            coords = [int(c.strip()) for c in part.split(",")]
        except ValueError:
            continue
        if len(coords) == 4:
            zones.append(tuple(coords))
    return zones


def load_camera_config(section_prefix):
    """
    Load camera configuration with support for day/night overrides.

    Args:
        section_prefix: 'CAMERA_MIN_AREA', 'CAMERA_MAX_AREA', 'CAMERA_ASPECT_RATIO',
            'TOP_EDGE_CONFIG', or 'CAMERA_EXCLUDE_ZONE'

    Returns:
        Dictionary with camera configurations merged from base and night overrides
    """
    # Load base configuration (daytime defaults)
    base_config = {}
    for camera, value in config[section_prefix].items():
        camera_name = camera.capitalize()
        if section_prefix == "CAMERA_ASPECT_RATIO":
            min_ratio, max_ratio = map(float, value.split(","))
            base_config[camera_name] = (min_ratio, max_ratio)
        elif section_prefix == "TOP_EDGE_CONFIG":
            margin_str, conf_str = value.split(",")
            base_config[camera_name] = (int(margin_str), float(conf_str))
        elif section_prefix == "CAMERA_EXCLUDE_ZONE":
            base_config[camera_name] = parse_zones(value)
        else:
            base_config[camera_name] = int(value)

    # If it's nighttime, try to load night overrides
    if not is_daytime():
        night_section = f"NIGHT_{section_prefix}"
        if config.has_section(night_section):
            for camera, value in config[night_section].items():
                camera_name = camera.capitalize()
                if section_prefix == "CAMERA_ASPECT_RATIO":
                    min_ratio, max_ratio = map(float, value.split(","))
                    base_config[camera_name] = (min_ratio, max_ratio)
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🌙 NIGHT override: {camera_name} aspect ratio = ({min_ratio}, {max_ratio})")
                elif section_prefix == "TOP_EDGE_CONFIG":
                    margin_str, conf_str = value.split(",")
                    base_config[camera_name] = (int(margin_str), float(conf_str))
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🌙 NIGHT override: {camera_name} top-edge = (margin={int(margin_str)}, high_conf={float(conf_str)})")
                elif section_prefix == "CAMERA_EXCLUDE_ZONE":
                    base_config[camera_name] = parse_zones(value)
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🌙 NIGHT override: {camera_name} exclude zones = {base_config[camera_name]}")
                else:
                    base_config[camera_name] = int(value)
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🌙 NIGHT override: {camera_name} {section_prefix.split('_')[1]} = {value}")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ☀️  No night overrides for {section_prefix}, using daytime defaults")

    return base_config


# ==========================================
# CONFIGURATION
# ==========================================
ANALYSIS_INTERVAL = cfg_float("GENERAL", "ANALYSIS_INTERVAL")
MAX_IMAGES = cfg_int("GENERAL", "MAX_IMAGES")
COOLDOWN = cfg_float("GENERAL", "COOLDOWN")
CAM_THREAD_SLEEP = cfg_float("GENERAL", "CAM_THREAD_SLEEP")
WEBHOOK_PORT = cfg_int("GENERAL", "WEBHOOK_PORT")

YOLO_CONFIDENCE = cfg_float("YOLO", "CONFIDENCE")
YOLO_INPUT_SIZE = cfg_int("YOLO", "INPUT_SIZE")
YOLO_IOU = cfg_float("YOLO", "IOU")
YOLO_RESTART_DELAY = cfg_int("YOLO", "RESTART_DELAY")

JPEG_QUALITY = cfg_int("IMAGE", "JPEG_QUALITY")
DRAW_BOUNDING_BOXES = cfg_bool("IMAGE", "DRAW_BOUNDING_BOXES")
# Draw the per-camera exclusion zones on rejected images (for tuning).
DRAW_ZONES = config.getboolean("IMAGE", "DRAW_ZONES", fallback=False)

# How long an incomplete session can go without a new detection before it's
# considered abandoned and flushed to a pending email.
SESSION_TIMEOUT = cfg_int("SESSION", "SESSION_TIMEOUT")

WATCHDOG_TIMEOUT = cfg_int("SESSION", "WATCHDOG_TIMEOUT")
WATCHDOG_CHECK = cfg_int("SESSION", "WATCHDOG_CHECK")
RESET_DEDUP_WINDOW = cfg_int("SESSION", "RESET_DEDUP_WINDOW")

DEBUG_LOG_INTERVAL = cfg_float("DEBUG", "DEBUG_LOG_INTERVAL")
ENABLE_DEBUG_PRINTS = cfg_bool("DEBUG", "ENABLE_DEBUG_PRINTS")
SAVE_IMAGE_INTERVAL = cfg_float("DEBUG", "SAVE_IMAGE_INTERVAL")

SAVE_REJECTED_IMAGES = cfg_bool("DEBUG", "SAVE_REJECTED_IMAGES")
REJECTED_IMAGES_DIR = cfg_str("DEBUG", "REJECTED_IMAGES_DIR")
MAX_REJECTED_IMAGES = cfg_int("DEBUG", "MAX_REJECTED_IMAGES")

ENABLE_AREA_FILTER = cfg_bool("FILTERS", "ENABLE_AREA_FILTER")
ENABLE_ASPECT_FILTER = cfg_bool("FILTERS", "ENABLE_ASPECT_FILTER")
ENABLE_TOP_EDGE_FILTER = cfg_bool("FILTERS", "ENABLE_TOP_EDGE_FILTER")
ENABLE_DARK_FILTER = cfg_bool("FILTERS", "ENABLE_DARK_FILTER")

# Exclusion-zone filter — rejects detections that sit inside a fixed ignore
# region (static clutter, fixtures). Defensive fallbacks so a config that
# predates this feature still loads.
ENABLE_EXCLUDE_ZONE_FILTER = config.getboolean("FILTERS", "ENABLE_EXCLUDE_ZONE_FILTER", fallback=True)
# Fraction of the detection box that must fall inside a zone to reject it.
EXCLUDE_ZONE_COVERAGE = config.getfloat("FILTERS", "EXCLUDE_ZONE_COVERAGE", fallback=0.6)

MIN_PERSON_AREA = cfg_int("FILTERS", "MIN_PERSON_AREA")
MIN_ASPECT_RATIO = cfg_float("FILTERS", "MIN_ASPECT_RATIO")
MAX_ASPECT_RATIO = cfg_float("FILTERS", "MAX_ASPECT_RATIO")

TOP_EDGE_MARGIN = cfg_int("FILTERS", "TOP_EDGE_MARGIN")
TOP_EDGE_HIGH_CONF = cfg_float("FILTERS", "TOP_EDGE_HIGH_CONF")

DARKNESS_THRESHOLD = cfg_int("FILTERS", "DARKNESS_THRESHOLD")
DARK_PIXEL_RATIO = cfg_float("FILTERS", "DARK_PIXEL_RATIO")

UPLOAD_WORKERS = cfg_int("UPLOAD", "UPLOAD_WORKERS")
UPLOAD_QUEUE_SIZE = cfg_int("UPLOAD", "UPLOAD_QUEUE_SIZE")
UPLOAD_MAX_RETRIES = cfg_int("UPLOAD", "UPLOAD_MAX_RETRIES")
UPLOAD_RETRY_DELAY_BASE = cfg_int("UPLOAD", "UPLOAD_RETRY_DELAY_BASE")

TASK_QUEUE_SIZE = cfg_int("UPLOAD", "TASK_QUEUE_SIZE")
RESULT_QUEUE_SIZE = cfg_int("UPLOAD", "RESULT_QUEUE_SIZE")

FILTER_CONFIDENCE = cfg_float("FILTERS", "FILTER_CONFIDENCE")

# ==========================================
# CAMERA-SPECIFIC CONFIGURATION LOADING (with day/night support)
# ==========================================
CAMERA_MIN_AREA = load_camera_config("CAMERA_MIN_AREA")
CAMERA_MAX_AREA = load_camera_config("CAMERA_MAX_AREA")
CAMERA_ASPECT_RATIOS = load_camera_config("CAMERA_ASPECT_RATIO")
# Per-camera top-edge (margin, high_conf); falls back to global [FILTERS] values
# for any camera not listed in [TOP_EDGE_CONFIG].
CAMERA_TOP_EDGE = load_camera_config("TOP_EDGE_CONFIG")
# Per-camera exclusion zones (list of (x1,y1,x2,y2)); empty if section absent.
CAMERA_EXCLUDE_ZONES = (load_camera_config("CAMERA_EXCLUDE_ZONE")
                        if config.has_section("CAMERA_EXCLUDE_ZONE") else {})

# Print loaded configuration
print("\n" + "=" * 60)
print(f"TIME: {'☀️ DAY' if is_daytime() else '🌙 NIGHT'}")
print("CAMERA_MIN_AREA (final):")
for camera, value in CAMERA_MIN_AREA.items():
    print(f"  {camera} = {value}")
print("CAMERA_MAX_AREA (final):")
for camera, value in CAMERA_MAX_AREA.items():
    print(f"  {camera} = {value}")
print("CAMERA_ASPECT_RATIOS (final):")
for camera, value in CAMERA_ASPECT_RATIOS.items():
    print(f"  {camera} = {value[0]},{value[1]}")
print("CAMERA_TOP_EDGE (final, margin,high_conf):")
for camera, value in CAMERA_TOP_EDGE.items():
    print(f"  {camera} = {value[0]},{value[1]}")
print(f"EXCLUDE ZONES (enabled={ENABLE_EXCLUDE_ZONE_FILTER}, coverage={EXCLUDE_ZONE_COVERAGE}):")
for camera, value in CAMERA_EXCLUDE_ZONES.items():
    print(f"  {camera} = {value}")
print("=" * 60)


# ==========================================
# REJECTED IMAGES FUNCTIONS
# ==========================================
def ensure_rejected_dir():
    """Create rejected images directory if it doesn't exist"""
    if SAVE_REJECTED_IMAGES:
        try:
            os.makedirs(REJECTED_IMAGES_DIR, exist_ok=True)
            existing = glob.glob(os.path.join(REJECTED_IMAGES_DIR, "*.jpg"))
            if len(existing) > MAX_REJECTED_IMAGES:
                existing.sort()
                for f in existing[:-MAX_REJECTED_IMAGES]:
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"[WARN] Failed to remove old rejected image {f}: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to create rejected images directory {REJECTED_IMAGES_DIR}: {e}")


# ==========================================
# DRAW TEXT WITH BACKGROUND
# ==========================================
def draw_text_with_background(img, text, position, font_scale, bg_color, text_color, thickness=1):
    """Draw text with a colored background rectangle."""
    x, y = position
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    padding = 4
    cv2.rectangle(img,
                  (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + padding),
                  bg_color,
                  -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

def draw_text_safe(img, text, x, y, font_scale, color, thickness=1):
    """Draw text safely within image boundaries."""
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if x + text_w > img.shape[1]:
        x = img.shape[1] - text_w - 5
    if x < 0:
        x = 5
    if y - text_h < 0:
        y = text_h + 5
    if y > img.shape[0]:
        y = img.shape[0] - 5
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def compute_label_position(box, block_w, block_h, first_line_h, img_w, img_h, margin=10):
    """Pick a top-left anchor for a text block that stays inside the image and
    clear of the detection box.

    Returns (text_x, text_y) where text_y is the BASELINE of the first line.
    Tries, in order: right of the box, left of the box, below it, above it —
    taking the first region that fully fits. Only if none fit does it fall back
    to clamping the block into the frame (which may overlap the box).
    """
    x1, y1, x2, y2 = box
    # Right / left: horizontally clear of the box; clamp top into the frame.
    top = max(0, min(y1, img_h - block_h))
    if x2 + margin + block_w <= img_w:
        return x2 + margin, top + first_line_h
    if x1 - margin - block_w >= 0:
        return x1 - margin - block_w, top + first_line_h
    # Below / above: vertically clear of the box; clamp left into the frame.
    left = max(0, min(x1, img_w - block_w))
    if y2 + margin + block_h <= img_h:
        return left, y2 + margin + first_line_h
    if y1 - margin - block_h >= 0:
        return left, y1 - margin - block_h + first_line_h
    # Nothing fits cleanly: clamp the block's top-left corner into the frame.
    left = max(0, min(x1, img_w - block_w))
    top = max(0, min(y1, img_h - block_h))
    return left, top + first_line_h


def draw_exclude_zones(img, zones):
    """Draw exclusion-zone rectangles (orange, labeled) on an image in place."""
    if not zones:
        return
    orange = (0, 128, 255)
    for zx1, zy1, zx2, zy2 in zones:
        cv2.rectangle(img, (zx1, zy1), (zx2, zy2), orange, 2)
        label_y = zy1 + 14 if zy1 < 14 else zy1 - 5
        cv2.putText(img, "ZONE", (zx1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, orange, 1)


def save_rejected_image(camera_name, frame, box, confidence, reason,
                        aspect_ratio=None, area=None, min_value=None,
                        top_y=None, min_conf=None, zones=None):
    """Save rejected detection image for debugging."""
    if not SAVE_REJECTED_IMAGES:
        return
    try:
        ensure_rejected_dir()
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        img_height, img_width = frame.shape[:2]
        debug_frame = frame.copy()
        yellow = (0, 255, 255)
        if DRAW_ZONES:
            draw_exclude_zones(debug_frame, zones)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), yellow, 1)
        timestamp = time.strftime('%Y-%m-%d_%H%M%S')

        box_area = width * height
        # Common suffix with box geometry (x,y,w,h) and area for all rejections.
        geom = f"_box{x1}-{y1}-{width}x{height}_area{box_area}px"

        if reason == "area":
            filename = f"[{timestamp}]_{camera_name}_area_small_{area}px_conf{confidence:.2f}{geom}.jpg"
        elif reason == "area_too_large":
            filename = f"[{timestamp}]_{camera_name}_area_large_{area}px_conf{confidence:.2f}{geom}.jpg"
        elif reason == "aspect":
            ar = aspect_ratio if aspect_ratio is not None else 0
            filename = f"[{timestamp}]_{camera_name}_aspect_{ar:.2f}_conf{confidence:.2f}{geom}.jpg"
        elif reason == "top_edge":
            filename = f"[{timestamp}]_{camera_name}_topedge_y{top_y}_conf{confidence:.2f}{geom}.jpg"
        elif reason == "dark_pixels":
            filename = f"[{timestamp}]_{camera_name}_dark_conf{confidence:.2f}{geom}.jpg"
        elif reason == "exclude_zone":
            filename = f"[{timestamp}]_{camera_name}_excludezone_conf{confidence:.2f}{geom}.jpg"
        else:
            filename = f"[{timestamp}]_{camera_name}_{reason}_conf{confidence:.2f}{geom}.jpg"

        filepath = os.path.join(REJECTED_IMAGES_DIR, filename)

        if reason == "area":
            line1 = f"REJECTED: Area too small ({area}px < {min_value}px)"
        elif reason == "area_too_large":
            line1 = f"REJECTED: Area too large ({area}px > {min_value}px)"
        elif reason == "aspect":
            ar = aspect_ratio if aspect_ratio is not None else 0
            line1 = f"REJECTED: Bad aspect ratio ({ar:.2f})"
        elif reason == "top_edge":
            line1 = f"REJECTED: Top-edge (y1={top_y})"
        elif reason == "dark_pixels":
            dark_percent = int(DARK_PIXEL_RATIO * 100)
            line1 = f"REJECTED: Too many dark pixels (>{dark_percent}%)"
        elif reason == "exclude_zone":
            line1 = "REJECTED: In exclusion zone"
        else:
            line1 = f"REJECTED: {reason}"

        line2 = f"Detection Conf: {confidence:.2f}"
        line3 = f"YOLO Threshold: {YOLO_CONFIDENCE:.2f}"
        line4 = f"Filter Threshold: {FILTER_CONFIDENCE:.2f}"
        lines = [line1, line2, line3, line4]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        max_width = 0
        line_heights = []
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, w)
            line_heights.append(h)

        line_spacing = 22
        total_text_height = sum(line_heights) + (len(lines) - 1) * line_spacing
        text_x, text_y = compute_label_position(
            box, max_width, total_text_height, line_heights[0], img_width, img_height
        )

        for i, line in enumerate(lines):
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            y_pos = text_y + i * line_spacing
            cv2.rectangle(debug_frame,
                         (text_x - 3, y_pos - h - 2),
                         (text_x + w + 3, y_pos + 2),
                         (0, 255, 255), -1)
            cv2.putText(debug_frame, line, (text_x, y_pos), font, font_scale, (0, 0, 0), thickness)

        cv2.imwrite(filepath, debug_frame)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] Saved rejected image: {filepath}")

    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] save_rejected_image failed for {camera_name}: {e}")
        import traceback
        traceback.print_exc()


# ==========================================
# DARK PIXEL FILTER FUNCTION
# ==========================================
def is_dark_detection(frame, box, darkness_threshold=50, dark_pixel_ratio=0.3):
    """
    Check if a detection contains too many dark pixels.
    """
    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return False

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, darkness_threshold, 255, cv2.THRESH_BINARY_INV)
    dark_pixels = cv2.countNonZero(dark_mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0
    return dark_ratio > dark_pixel_ratio


# ==========================================
# SESSION STATE ENUM
# ==========================================
class SessionState(Enum):
    IDLE = 0
    ACTIVE = 1
    WAITING_RESET = 2   # camera parks here after a full session until the recorder's /session-reset arrives
    COMPLETED = 3

# ==========================================
# PENDING EMAIL
# Holds captured frames after a session completes, waiting for the recorder's
# /session-reset signal before sending the email.  The camera parks in
# WAITING_RESET (one session per recording segment) and resumes detection
# only when that signal arrives.
# ==========================================
class PendingEmail:
    def __init__(self, camera_name, detection_id, frames, frame_count):
        self.camera_name = camera_name
        self.detection_id = detection_id
        self.frames = frames          # list of {'confidence', 'frame', 'timestamp', 'frame_num'}
        self.frame_count = frame_count
        self.created_at = time.time()

# Global dict: camera_name -> PendingEmail
# Protected by pending_emails_lock
pending_emails = {}
pending_emails_lock = threading.Lock()

# ==========================================
# THREAD-SAFE METRICS COLLECTOR
# ==========================================
class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.pending_uploads = 0
        self.dropped_frames = 0
        self.dropped_uploads = 0

    def inc_dropped_frames(self):
        with self._lock:
            self.dropped_frames += 1

    def inc_dropped_uploads(self):
        with self._lock:
            self.dropped_uploads += 1

    def inc_pending_uploads(self):
        with self._lock:
            self.pending_uploads += 1

    def dec_pending_uploads(self):
        with self._lock:
            self.pending_uploads = max(0, self.pending_uploads - 1)

    def can_add_upload(self):
        with self._lock:
            return self.pending_uploads < UPLOAD_QUEUE_SIZE

    def get_stats(self):
        with self._lock:
            return {
                'pending_uploads': self.pending_uploads,
                'dropped_frames': self.dropped_frames,
                'dropped_uploads': self.dropped_uploads
            }

metrics = Metrics()
yolo_process = None

# ==========================================
# AUTHENTICATION
# ==========================================
load_dotenv("/etc/cctv/credentials.env")
CAM_PASS = os.getenv("CAM_PASS")
if not CAM_PASS:
    print("ERROR: CAM_PASS not found!")
    sys.exit(1)
password = quote(CAM_PASS)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-this-default-secret")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|timeout;5000000|reorder_queue_size;1024|buffer_size;1024000"
)

# ==========================================
# EMAIL ALERT CONFIGURATION
# ==========================================
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
ALERT_TO   = os.getenv("ALERT_TO")
ENABLE_EMAIL_ALERTS = bool(SMTP_USER and SMTP_PASS and ALERT_TO)

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

# ==========================================
# PER-CAMERA SESSION STATE
# ==========================================
class CameraState:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = SessionState.IDLE
        self.count = 0
        self.detection_id = None
        self.last_upload = 0
        self.last_queued_time = 0
        self.last_result_time = 0
        self.last_activity = 0
        self.last_waiting_start = 0
        self.last_reset_time = 0
        self.last_reset_processed = 0
        self.active_session_id = None
        self.last_processed_count = 0

        # Store frames for email (top MAX_IMAGES by confidence)
        self.session_frames = []
        self.session_frames_lock = threading.Lock()

camera_states = {n: CameraState() for n in NODES}
upload_executor = ThreadPoolExecutor(max_workers=UPLOAD_WORKERS, thread_name_prefix="upload")

# Separate executor for email alerts (non-critical, main process only)
email_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="email")

# ==========================================
# FLASK WEBHOOK SERVER
# ==========================================
app = Flask(__name__)

def verify_auth():
    api_key = request.headers.get('X-API-KEY')
    return api_key and api_key == WEBHOOK_SECRET

# ==========================================
# SESSION RESET WEBHOOK
# The camera is parked in WAITING_RESET when this arrives.  This handler
# moves it to COMPLETED (resuming detection), saves any mid-capture frames,
# and fires the pending email for the segment.
# ==========================================
@app.route('/session-reset', methods=['POST'])
def session_reset():
    if not verify_auth():
        return jsonify({"status": "unauthorized"}), 401
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error"}), 400
        camera_name = data.get('camera')
        if not camera_name or camera_name not in camera_states:
            return jsonify({"status": "ok"}), 200

        now = time.time()
        state = camera_states[camera_name]

        # Dedup guard — ignore duplicate signals within RESET_DEDUP_WINDOW
        with state.lock:
            if now - state.last_reset_processed < RESET_DEDUP_WINDOW:
                return '', 200
            state.last_reset_processed = now
            state.last_reset_time = now

            # If a session was still mid-capture (frames collected but never
            # reached MAX_IMAGES), the recorder segment is ending now — save
            # those frames as the pending email for this segment instead of
            # discarding them. A completed session already moved its frames to
            # pending_emails and left count==0, so this is skipped for those.
            if state.count > 0 and state.detection_id:
                with state.session_frames_lock:
                    frames_copy = state.session_frames.copy()
                    state.session_frames = []
                if frames_copy:
                    with pending_emails_lock:
                        pending_emails[camera_name] = PendingEmail(
                            camera_name, state.detection_id, frames_copy, len(frames_copy)
                        )

            # Resume detection: a completed session parks the camera in
            # WAITING_RESET until this signal arrives. Clear it so the camera
            # can detect again. COOLDOWN still applies via last_reset_time.
            state.state = SessionState.COMPLETED
            state.count = 0
            state.detection_id = None
            state.active_session_id = None
            state.last_processed_count = 0
            state.last_waiting_start = 0

        # Claim pending email (if any) and fire it
        with pending_emails_lock:
            pending = pending_emails.pop(camera_name, None)

        if pending:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📡 Recorder signaled: {camera_name} reset "
                  f"- sending email [SID:{pending.detection_id}] ({pending.frame_count} frames)")
            email_executor.submit(send_session_email_from_pending, pending)
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📡 Recorder signaled: {camera_name} reset "
                  f"- no pending email (camera already idle)")

        return '', 200
    except Exception as e:
        print(f"Error in session_reset: {e}")
        return jsonify({"status": "error"}), 500

@app.route('/reset', methods=['POST'])
def reset_legacy():
    response = make_response(session_reset())
    response.headers['Warning'] = '299 - "Deprecated endpoint: Use /session-reset instead"'
    return response

@app.route('/health', methods=['GET'])
def health_check():
    camera_stats = {}
    for name, state in camera_states.items():
        with state.lock:
            camera_stats[name] = {"state": state.state.name, "count": state.count}
    with pending_emails_lock:
        pending_count = len(pending_emails)
    return jsonify({
        "status": "running",
        "version": VERSION,
        "cameras": camera_stats,
        "pending_emails": pending_count
    }), 200

def start_webhook_server():
    threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=WEBHOOK_PORT,
                               debug=False, use_reloader=False, threaded=True),
        daemon=True
    ).start()
    print(f"🌐 Webhook server on port {WEBHOOK_PORT}")

# ==========================================
# UPLOAD FUNCTIONS
# ==========================================
def upload_task(camera_name, url, image_buffer, ts, current_count, max_images, detection_id):
    max_retries = UPLOAD_MAX_RETRIES
    state = camera_states[camera_name]

    with state.lock:
        if state.active_session_id and detection_id != state.active_session_id:
            return
        original_session_id = state.active_session_id

    for attempt in range(max_retries):
        try:
            files = {'image': (f"{camera_name}_{ts}.jpg", image_buffer, 'image/jpeg')}
            data = {
                'frame_num': current_count,
                'total_frames': max_images,
                'camera': camera_name,
                'detection_id': detection_id
            }
            headers = {'X-API-KEY': WEBHOOK_SECRET}
            response = requests.post(url, files=files, data=data, headers=headers, timeout=5)
            response.raise_for_status()

            # On last frame, park the camera in WAITING_RESET and register a PendingEmail.
            if current_count == max_images:
                with state.lock:
                    if (state.state == SessionState.ACTIVE and
                            state.active_session_id == original_session_id and
                            state.detection_id == detection_id):

                        # Collect frames before clearing state
                        with state.session_frames_lock:
                            frames_copy = state.session_frames.copy()
                            state.session_frames = []

                        frame_count = len(frames_copy)

                        # Register pending email
                        with pending_emails_lock:
                            pending_emails[camera_name] = PendingEmail(
                                camera_name, detection_id, frames_copy, frame_count
                            )

                        # Park in WAITING_RESET so a continuously-present person does
                        # not spawn a new session every few seconds. The camera resumes
                        # detection only when the recorder's /session-reset arrives (one
                        # session per recording segment). Frame bookkeeping is cleared
                        # because the frames now live in pending_emails.
                        old_sid = state.detection_id
                        state.state = SessionState.WAITING_RESET
                        state.last_waiting_start = time.time()
                        state.count = 0
                        state.detection_id = None
                        state.active_session_id = None
                        state.last_processed_count = 0

                        print(f"[{ts}] 🛑 {camera_name}: Last frame sent ({current_count}/{max_images}) "
                              f"[SID:{old_sid}] — waiting for recorder reset (no new sessions until then)")
                    else:
                        print(f"[{ts}] ⚠️ {camera_name}: Session changed during upload, "
                              f"skipping state change [SID:{detection_id}]")
            return

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                wait_time = UPLOAD_RETRY_DELAY_BASE * (attempt + 1)
                time.sleep(wait_time)
            elif e.response.status_code == 401:
                break
            else:
                break
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(UPLOAD_RETRY_DELAY_BASE)
            else:
                break

def draw_and_upload(camera_name, url, frame, detections, ts, current_count, max_images, detection_id):
    working_frame = frame.copy()
    yellow = (0, 255, 255)

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"PERSON {d['conf']:.2f}"
        if DRAW_BOUNDING_BOXES:
            cv2.rectangle(working_frame, (x1, y1), (x2, y2), yellow, 1)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(working_frame, (x1, y1 - 20), (x1 + w, y1), yellow, -1)
        cv2.putText(working_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    success, buffer = cv2.imencode('.jpg', working_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if success:
        if not metrics.can_add_upload():
            metrics.inc_dropped_uploads()
            return
        metrics.inc_pending_uploads()
        def upload_wrapper(*args, **kwargs):
            try:
                return upload_task(*args, **kwargs)
            finally:
                metrics.dec_pending_uploads()
        upload_executor.submit(upload_wrapper, camera_name, url, buffer.tobytes(),
                               ts, current_count, max_images, detection_id)

# ==========================================
# EMAIL — SEND SESSION EMAIL FROM PENDING
# ==========================================
def send_session_email_from_pending(pending: PendingEmail):
    """Unwrap PendingEmail and delegate to send_session_email."""
    send_session_email(pending.camera_name, pending.detection_id, pending.frames)

def send_session_email(camera_name, detection_id, session_frames):
    """Send one email per session with all captured frames."""
    if not ENABLE_EMAIL_ALERTS:
        return
    if not SMTP_USER or not SMTP_PASS or not ALERT_TO:
        return
    if not session_frames:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No frames to send for session {detection_id}")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = ALERT_TO
        msg['Subject'] = (f"🚨 CCTV Alert: Person detected on {camera_name} "
                          f"(Session {detection_id})")

        avg_conf = sum(f['confidence'] for f in session_frames) / len(session_frames)

        body = (
            f"<b>Camera:</b> {camera_name}<br>"
            f"<b>Session ID:</b> {detection_id}<br>"
            f"<b>Frames captured:</b> {len(session_frames)}/{MAX_IMAGES}<br>"
            f"<b>Average Confidence:</b> {avg_conf:.0%}<br>"
            f"<b>Time:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}<br><br>"
            f"<i>Attached: All frames from this detection session "
            f"(sorted by confidence).</i>"
        )
        msg.attach(MIMEText(body, 'html'))

        for idx, frame_data in enumerate(session_frames, 1):
            conf = frame_data['confidence']
            frame = frame_data['frame']
            frame_num = frame_data.get('frame_num', idx)
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if success:
                img = MIMEImage(
                    buffer.tobytes(),
                    name=f"{camera_name}_{detection_id}_frame{frame_num}_conf{conf:.0%}.jpg"
                )
                msg.attach(img)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Email sent for {camera_name} "
              f"- Session {detection_id} ({len(session_frames)} frames)")

    except Exception as e:
        print(f"[ERROR] Email alert failed for {camera_name}: {e}")


# ==========================================
# EMAIL — SEND REJECTION EMAIL
# ==========================================
def send_rejection_email(camera_name, frame, box, confidence, reason,
                         aspect_ratio=None, area=None, min_value=None):
    """Send email with rejected image for debugging."""
    if not ENABLE_EMAIL_ALERTS:
        return
    if not SMTP_USER or not SMTP_PASS or not ALERT_TO:
        return

    try:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        img_height, img_width = frame.shape[:2]

        annotated_frame = frame.copy()
        yellow = (0, 255, 255)
        if DRAW_ZONES:
            draw_exclude_zones(annotated_frame, CAMERA_EXCLUDE_ZONES.get(camera_name))
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), yellow, 1)

        if reason == "area":
            subject = f"⚠️ CCTV Rejection: Area too small on {camera_name}"
            line1 = f"REJECTED: Area too small ({area}px < {min_value}px)"
        elif reason == "area_too_large":
            subject = f"⚠️ CCTV Rejection: Area too large on {camera_name}"
            line1 = f"REJECTED: Area too large ({area}px > {min_value}px)"
        elif reason == "aspect":
            subject = f"⚠️ CCTV Rejection: Bad aspect ratio on {camera_name}"
            ar = aspect_ratio if aspect_ratio is not None else 0
            line1 = f"REJECTED: Bad aspect ratio ({ar:.2f})"
        elif reason == "top_edge":
            subject = f"⚠️ CCTV Rejection: Top-edge on {camera_name}"
            line1 = f"REJECTED: Top-edge (y1={y1})"
        elif reason == "dark_pixels":
            subject = f"⚠️ CCTV Rejection: Dark pixels on {camera_name}"
            dark_percent = int(DARK_PIXEL_RATIO * 100)
            line1 = f"REJECTED: Too many dark pixels (>{dark_percent}%)"
        elif reason == "exclude_zone":
            subject = f"⚠️ CCTV Rejection: Exclusion zone on {camera_name}"
            line1 = "REJECTED: In exclusion zone"
        else:
            subject = f"⚠️ CCTV Rejection: {reason} on {camera_name}"
            line1 = f"REJECTED: {reason}"

        line2 = f"Detection Conf: {confidence:.2f}"
        line3 = f"YOLO Conf/Filter Conf: {YOLO_CONFIDENCE:.2f}/{FILTER_CONFIDENCE:.2f}"
        line4 = f"Size: {width}x{height}px  Area: {width * height}px"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        lines = [line1, line2, line3, line4]

        max_width = 0
        line_heights = []
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, w)
            line_heights.append(h)

        line_spacing = 22
        total_text_height = sum(line_heights) + (len(lines) - 1) * line_spacing
        text_x, text_y = compute_label_position(
            box, max_width, total_text_height, line_heights[0], img_width, img_height
        )

        draw_text_with_background(annotated_frame, line1, (text_x, text_y),            0.5, yellow, (0, 0, 0))
        draw_text_with_background(annotated_frame, line2, (text_x, text_y + 24),       0.5, yellow, (0, 0, 0))
        draw_text_with_background(annotated_frame, line3, (text_x, text_y + 48),       0.5, yellow, (0, 0, 0))
        draw_text_with_background(annotated_frame, line4, (text_x, text_y + 72),       0.5, yellow, (0, 0, 0))

        success, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not success:
            print(f"[ERROR] Rejection email failed for {camera_name}: cannot encode frame")
            return

        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = ALERT_TO
        msg['Subject'] = subject

        body = (
            f"<b>Camera:</b> {camera_name}<br>"
            f"<b>Reason:</b> {line1}<br>"
            f"<b>Confidence:</b> {confidence:.0%}<br>"
            f"<b>Box:</b> {width}x{height}px<br>"
            f"<b>Time:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        msg.attach(MIMEText(body, 'html'))
        img = MIMEImage(buffer.tobytes(), name=f"rejected_{camera_name}_{reason}.jpg")
        msg.attach(img)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📧 Rejection email sent for {camera_name}: {reason}")

    except Exception as e:
        print(f"[ERROR] Rejection email failed for {camera_name}: {e}")


# ==========================================
# PENDING EMAIL WATCHDOG
# If the recorder reset signal never arrives (crash, missed POST, etc.),
# this fires the email after PENDING_EMAIL_TIMEOUT seconds so alerts
# are never silently lost.  Runs in the main process.
# ==========================================
PENDING_EMAIL_TIMEOUT = 360   # 5-minute segment + 60s grace

def pending_email_watchdog():
    """Expire and send pending emails that never received a recorder reset."""
    while True:
        time.sleep(30)
        now = time.time()
        expired = []
        with pending_emails_lock:
            for name, p in list(pending_emails.items()):
                if now - p.created_at > PENDING_EMAIL_TIMEOUT:
                    expired.append(pending_emails.pop(name))

        for pending in expired:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⏰ Pending email timeout: "
                  f"{pending.camera_name} [SID:{pending.detection_id}] — "
                  f"sending without recorder reset confirmation")
            email_executor.submit(send_session_email_from_pending, pending)


# ==========================================
# CONFIGURATION RELOAD THREAD (for day/night switching)
# ==========================================
def config_reload_thread():
    """Reload camera configurations every hour to handle day/night transitions."""
    global CAMERA_MIN_AREA, CAMERA_MAX_AREA, CAMERA_ASPECT_RATIOS, CAMERA_TOP_EDGE

    while True:
        time.sleep(3600)
        new_min_area = load_camera_config("CAMERA_MIN_AREA")
        if CAMERA_MIN_AREA != new_min_area:
            CAMERA_MIN_AREA = new_min_area
            CAMERA_MAX_AREA = load_camera_config("CAMERA_MAX_AREA")
            CAMERA_ASPECT_RATIOS = load_camera_config("CAMERA_ASPECT_RATIO")
            CAMERA_TOP_EDGE = load_camera_config("TOP_EDGE_CONFIG")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Camera configs updated "
                  f"for {'DAY ☀️' if is_daytime() else 'NIGHT 🌙'}")

 
# ==========================================
# YOLO WORKER PROCESS
# ==========================================
def yolo_worker_process(input_q, output_q, min_person_area, min_aspect_ratio, max_aspect_ratio,
                        enable_area_filter, enable_aspect_filter, enable_dark_filter,
                        camera_min_area_dict, camera_max_area_dict, camera_aspect_ratios_dict,
                        camera_top_edge_dict, camera_exclude_zones_dict,
                        enable_exclude_zone_filter, exclude_zone_coverage,
                        yolo_iou, debug_log_interval):

    import threading

    _last_area_log_time = {}
    _last_maxarea_log_time = {}
    _last_aspect_log_time = {}
    _last_topedge_log_time = {}
    _last_topedge_accept_log_time = {}
    _last_dark_log_time = {}
    _last_exclude_log_time = {}

    _last_area_save_time = {}
    _last_maxarea_save_time = {}
    _last_aspect_save_time = {}
    _last_topedge_save_time = {}
    _last_dark_save_time = {}
    _last_exclude_save_time = {}

    _log_lock = threading.Lock()

    _last_rejection_queue_time = {}
    _rejection_pending = []
    _rejection_lock = threading.Lock()

    def queue_rejection_email(camera_name, frame, box, confidence, reason, **kwargs):
        now = time.time()
        last = _last_rejection_queue_time.get(camera_name, 0)
        if now - last >= 300:
            _last_rejection_queue_time[camera_name] = now
            with _rejection_lock:
                _rejection_pending.append(
                    (camera_name, frame.copy() if frame is not None else None,
                     box, confidence, reason, kwargs)
                )

    def is_valid_person_detection_worker(camera_name, box, confidence,
                                         image_height, frame_id=None, original_frame=None):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        min_area = camera_min_area_dict.get(camera_name, min_person_area)
        max_area = camera_max_area_dict.get(camera_name) if camera_max_area_dict else None
        min_ratio, max_ratio = camera_aspect_ratios_dict.get(
            camera_name, (min_aspect_ratio, max_aspect_ratio)
        )

        # Camera's exclude zones — used by the filter below and, when DRAW_ZONES
        # is on, drawn on every rejected image (any reason) so they can be tuned.
        zones = camera_exclude_zones_dict.get(camera_name) if camera_exclude_zones_dict else None

        # Exclusion-zone filter — reject detections that sit inside a fixed
        # ignore region (static clutter, fixtures like a hanging clothes rack).
        # Runs FIRST, at ALL confidence levels (before the high-conf bypass
        # below) because a static object can occasionally score above
        # FILTER_CONFIDENCE. A detection is rejected only when at least
        # `exclude_zone_coverage` of its box area falls inside a zone, so a real
        # person who merely clips the edge of the region is still detected.
        if enable_exclude_zone_filter and zones:
            box_area = max(1, width * height)
            for zx1, zy1, zx2, zy2 in zones:
                ix1 = max(x1, zx1)
                iy1 = max(y1, zy1)
                ix2 = min(x2, zx2)
                iy2 = min(y2, zy2)
                if ix2 > ix1 and iy2 > iy1:
                    covered = (ix2 - ix1) * (iy2 - iy1) / box_area
                    if covered >= exclude_zone_coverage:
                        with _log_lock:
                            now = time.time()
                            if now - _last_exclude_log_time.get(camera_name, 0) >= debug_log_interval:
                                _last_exclude_log_time[camera_name] = now
                                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: "
                                      f"In exclude zone ({covered:.0%} of box in "
                                      f"{(zx1, zy1, zx2, zy2)}) conf={confidence:.2f} REJECTED!")

                        if original_frame is not None and SAVE_REJECTED_IMAGES:
                            now = time.time()
                            if now - _last_exclude_save_time.get(camera_name, 0) >= SAVE_IMAGE_INTERVAL:
                                _last_exclude_save_time[camera_name] = now
                                save_rejected_image(camera_name, original_frame, box, confidence,
                                                    "exclude_zone", zones=zones)

                        queue_rejection_email(camera_name, original_frame, box, confidence,
                                              "exclude_zone")
                        return False

        # Aspect-ratio (shape) filter — applied only to LOW-confidence detections
        # (conf < FILTER_CONFIDENCE). A standing person is taller than wide, but a
        # genuine person with arms outstretched / carrying something / bending can
        # produce a near-square box; rejecting those on shape drops real people.
        # In practice false-positive wide boxes (e.g. vehicles) score low, so the
        # shape check still catches them here, while high-confidence detections are
        # trusted and fall through to the bypass below.
        if enable_aspect_filter and width > 0 and confidence < FILTER_CONFIDENCE:
            aspect_ratio_val = height / width
            if aspect_ratio_val < min_ratio or aspect_ratio_val > max_ratio:
                with _log_lock:
                    now = time.time()
                    if now - _last_aspect_log_time.get(camera_name, 0) >= debug_log_interval:
                        _last_aspect_log_time[camera_name] = now
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: "
                              f"Bad aspect ratio ({aspect_ratio_val:.2f}) - "
                              f"box: {width}x{height}px conf={confidence:.2f} REJECTED!")

                if original_frame is not None and SAVE_REJECTED_IMAGES:
                    now = time.time()
                    if now - _last_aspect_save_time.get(camera_name, 0) >= SAVE_IMAGE_INTERVAL:
                        _last_aspect_save_time[camera_name] = now
                        save_rejected_image(camera_name, original_frame, box, confidence,
                                            "aspect", aspect_ratio=aspect_ratio_val, zones=zones)

                queue_rejection_email(camera_name, original_frame, box, confidence,
                                      "aspect", aspect_ratio=aspect_ratio_val)
                return False

        # High confidence bypass — trusted detection, shape check skipped above
        if confidence >= FILTER_CONFIDENCE:
            return True

        # Dark pixel filter
        if enable_dark_filter and confidence < FILTER_CONFIDENCE:
            if is_dark_detection(original_frame, box, DARKNESS_THRESHOLD, DARK_PIXEL_RATIO):
                with _log_lock:
                    now = time.time()
                    if now - _last_dark_log_time.get(camera_name, 0) >= debug_log_interval and ENABLE_DEBUG_PRINTS:
                        _last_dark_log_time[camera_name] = now
                        dark_percent = int(DARK_PIXEL_RATIO * 100)
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: "
                              f"Too many dark pixels (> {dark_percent}%) - "
                              f"box: {width}x{height}px conf={confidence:.2f} REJECTED!")

                if original_frame is not None and SAVE_REJECTED_IMAGES:
                    now = time.time()
                    if now - _last_dark_save_time.get(camera_name, 0) >= SAVE_IMAGE_INTERVAL:
                        _last_dark_save_time[camera_name] = now
                        save_rejected_image(camera_name, original_frame, box, confidence,
                                            "dark_pixels", min_value=int(DARK_PIXEL_RATIO * 100), zones=zones)

                queue_rejection_email(camera_name, original_frame, box, confidence, "dark_pixels")
                return False

        # Min area filter
        if enable_area_filter and area < min_area:
            with _log_lock:
                now = time.time()
                if now - _last_area_log_time.get(camera_name, 0) >= debug_log_interval and ENABLE_DEBUG_PRINTS:
                    _last_area_log_time[camera_name] = now
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: "
                          f"Area too small ({area}px < {min_area}px) - "
                          f"box: {width}x{height}px REJECTED!")

            if original_frame is not None and SAVE_REJECTED_IMAGES:
                now = time.time()
                if now - _last_area_save_time.get(camera_name, 0) >= SAVE_IMAGE_INTERVAL:
                    _last_area_save_time[camera_name] = now
                    save_rejected_image(camera_name, original_frame, box, confidence,
                                        "area", area=area, min_value=min_area, zones=zones)

            queue_rejection_email(camera_name, original_frame, box, confidence,
                                  "area", area=area, min_value=min_area)
            return False

        # Max area filter
        if max_area is not None and area > max_area:
            with _log_lock:
                now = time.time()
                if now - _last_maxarea_log_time.get(camera_name, 0) >= debug_log_interval:
                    _last_maxarea_log_time[camera_name] = now
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: "
                          f"Area too large ({area}px > {max_area}px) - "
                          f"box: {width}x{height}px conf={confidence:.2f} REJECTED!")

            if original_frame is not None and SAVE_REJECTED_IMAGES:
                now = time.time()
                if now - _last_maxarea_save_time.get(camera_name, 0) >= SAVE_IMAGE_INTERVAL:
                    _last_maxarea_save_time[camera_name] = now
                    save_rejected_image(camera_name, original_frame, box, confidence,
                                        "area_too_large", area=area, min_value=max_area, zones=zones)

            queue_rejection_email(camera_name, original_frame, box, confidence,
                                  "area_too_large", area=area, min_value=max_area)
            return False

        # Top-edge filter — per-camera (margin, high_conf) from [TOP_EDGE_CONFIG],
        # falling back to the global [FILTERS] values for any camera not listed.
        if ENABLE_TOP_EDGE_FILTER and image_height is not None:
            top_margin, top_high_conf = camera_top_edge_dict.get(
                camera_name, (TOP_EDGE_MARGIN, TOP_EDGE_HIGH_CONF)
            )
            LABEL_MARGIN = 12
            top_y = y1
            if top_y <= top_margin + LABEL_MARGIN:
                if confidence < top_high_conf:
                    with _log_lock:
                        now = time.time()
                        if now - _last_topedge_log_time.get(camera_name, 0) >= debug_log_interval:
                            _last_topedge_log_time[camera_name] = now
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ [DEBUG] {camera_name}: "
                                  f"Top-edge rejection - y1={top_y}, "
                                  f"conf={confidence:.2f}<{top_high_conf} REJECTED!")

                    if original_frame is not None and SAVE_REJECTED_IMAGES:
                        now = time.time()
                        if now - _last_topedge_save_time.get(camera_name, 0) >= SAVE_IMAGE_INTERVAL:
                            _last_topedge_save_time[camera_name] = now
                            save_rejected_image(camera_name, original_frame, box, confidence,
                                                "top_edge", top_y=top_y, min_conf=top_high_conf, zones=zones)

                    queue_rejection_email(camera_name, original_frame, box, confidence, "top_edge")
                    return False
                else:
                    with _log_lock:
                        now = time.time()
                        if now - _last_topedge_accept_log_time.get(camera_name, 0) >= debug_log_interval:
                            _last_topedge_accept_log_time[camera_name] = now
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ [DEBUG] {camera_name}: "
                                  f"Top-edge HIGH CONF ACCEPT - y1={top_y}, "
                                  f"conf={confidence:.2f}>={top_high_conf}")

        return True

    # --- YOLO model load ---
    try:
        print("Loading yolov8n.onnx with pure ONNX Runtime...")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        session = ort.InferenceSession(
            "yolov8n.onnx",
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        print("Pure ONNX Runtime initialized")
        print("Ultra-light + smart NMS (minimal CPU increase)")
    except Exception as e:
        print(f"❌ ONNX load failed: {e}")
        return

    # Day/night config reload (worker-local): this worker is a separate process and
    # will NOT see config dicts updated in the main process, so we re-check the
    # day/night transition here and reload directly from the config file.
    current_daytime = is_daytime()
    last_config_check = time.time()
    CONFIG_CHECK_INTERVAL = 60  # seconds

    while True:
        try:
            now_check = time.time()
            if now_check - last_config_check >= CONFIG_CHECK_INTERVAL:
                last_config_check = now_check
                if is_daytime() != current_daytime:
                    current_daytime = is_daytime()
                    camera_min_area_dict = load_camera_config("CAMERA_MIN_AREA")
                    camera_max_area_dict = load_camera_config("CAMERA_MAX_AREA")
                    camera_aspect_ratios_dict = load_camera_config("CAMERA_ASPECT_RATIO")
                    camera_top_edge_dict = load_camera_config("TOP_EDGE_CONFIG")
                    if config.has_section("CAMERA_EXCLUDE_ZONE"):
                        camera_exclude_zones_dict = load_camera_config("CAMERA_EXCLUDE_ZONE")
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔄 [WORKER] Camera configs "
                          f"reloaded for {'DAY ☀️' if current_daytime else 'NIGHT 🌙'}")
 
            name, frame, ts, capture_time = input_q.get(timeout=0.5)

            h, w = frame.shape[:2]
            scale_x = w / YOLO_INPUT_SIZE
            scale_y = h / YOLO_INPUT_SIZE

            img = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1).reshape(
                1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE
            )

            outputs = session.run(None, {input_name: img})
            predictions = outputs[0][0]

            detections = []
            for pred in predictions.T:
                confidence = np.max(pred[4:])
                if confidence < YOLO_CONFIDENCE:
                    continue
                if np.argmax(pred[4:]) != 0:
                    continue

                xc, yc, pw, ph = pred[:4]
                x1 = int((xc - pw / 2) * scale_x)
                y1 = int((yc - ph / 2) * scale_y)
                x2 = int((xc + pw / 2) * scale_x)
                y2 = int((yc + ph / 2) * scale_y)
                box = [x1, y1, x2, y2]

                if is_valid_person_detection_worker(name, box, confidence, h, None, frame):
                    detections.append({"box": box, "conf": float(confidence)})

            # NMS
            if len(detections) > 0:
                detections.sort(key=lambda x: x['conf'], reverse=True)
                filtered = []
                for d1 in detections:
                    keep = True
                    x1, y1, x2, y2 = d1['box']
                    area1 = (x2 - x1) * (y2 - y1)
                    for d2 in filtered:
                        xx1 = max(x1, d2['box'][0])
                        yy1 = max(y1, d2['box'][1])
                        xx2 = min(x2, d2['box'][2])
                        yy2 = min(y2, d2['box'][3])
                        if xx2 > xx1 and yy2 > yy1:
                            overlap = (xx2 - xx1) * (yy2 - yy1)
                            area2 = ((d2['box'][2] - d2['box'][0]) *
                                     (d2['box'][3] - d2['box'][1]))
                            union = area1 + area2 - overlap
                            iou = overlap / union if union > 0 else 0
                            if iou > yolo_iou:
                                keep = False
                                break
                    if keep:
                        filtered.append(d1)
                detections = filtered

            # Drain pending rejection emails and include in result
            with _rejection_lock:
                rejections = _rejection_pending.copy()
                _rejection_pending.clear()

            try:
                output_q.put_nowait((name, frame, detections, ts, capture_time, rejections))
            except Full:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Result queue full, "
                      f"dropping {name} detection")

        except Empty:
            continue
        except Exception as e:
            print(f"YOLO error: {e}")
            time.sleep(1)


def start_yolo_worker(task_q, result_q):
    global yolo_process
    if yolo_process and yolo_process.is_alive():
        return yolo_process
    yolo_process = multiprocessing.Process(
        target=yolo_worker_process,
        args=(task_q, result_q, MIN_PERSON_AREA, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO,
              ENABLE_AREA_FILTER, ENABLE_ASPECT_FILTER, ENABLE_DARK_FILTER,
              CAMERA_MIN_AREA, CAMERA_MAX_AREA, CAMERA_ASPECT_RATIOS,
              CAMERA_TOP_EDGE, CAMERA_EXCLUDE_ZONES,
              ENABLE_EXCLUDE_ZONE_FILTER, EXCLUDE_ZONE_COVERAGE,
              YOLO_IOU, DEBUG_LOG_INTERVAL),
        daemon=True
    )
    yolo_process.start()
    return yolo_process

def check_yolo_health(task_q, result_q):
    global yolo_process
    while True:
        time.sleep(YOLO_RESTART_DELAY)
        if yolo_process and not yolo_process.is_alive():
            print("⚠️ YOLO worker died! Restarting...")
            yolo_process.join(timeout=1)
            yolo_process = multiprocessing.Process(
                target=yolo_worker_process,
                args=(task_q, result_q, MIN_PERSON_AREA, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO,
                      ENABLE_AREA_FILTER, ENABLE_ASPECT_FILTER, ENABLE_DARK_FILTER,
                      CAMERA_MIN_AREA, CAMERA_MAX_AREA, CAMERA_ASPECT_RATIOS,
                      CAMERA_TOP_EDGE, CAMERA_EXCLUDE_ZONES,
                      ENABLE_EXCLUDE_ZONE_FILTER, EXCLUDE_ZONE_COVERAGE,
                      YOLO_IOU, DEBUG_LOG_INTERVAL),
                daemon=True
            )
            yolo_process.start()
            print("✅ YOLO worker restarted")


# ==========================================
# CAMERA STREAM
# ==========================================
class CameraStream:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.lock = threading.Lock()
        self.frame = None
        self.frame_time = 0
        self.running = True
        self.cap = None
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        consecutive_failures = 0
        max_failures = 5

        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if self.cap:
                        self.cap.release()
                    self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, 15)
                    if not self.cap.isOpened():
                        raise Exception("Failed to open camera")
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ {self.name}: Camera connected")
                    consecutive_failures = 0

                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.frame_time = time.time()
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ {self.name}: "
                              f"No frames after {max_failures} attempts, reconnecting...")
                        self.cap.release()
                        self.cap = None
                    time.sleep(0.05)

            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ {self.name}: Error: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                time.sleep(2)

    def get_frame(self, max_age=1.5):
        """Return the most recent frame only if it is fresher than max_age seconds."""
        with self.lock:
            if self.frame is not None:
                if time.time() - self.frame_time < max_age:
                    return self.frame.copy(), self.frame_time
        return None, 0

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ==========================================
# SESSION WATCHDOG - FIXED v3.25.3
# 
# Separate timeouts for different purposes:
#   SESSION_TIMEOUT: How long an incomplete session can go without a new
#     detection before it's considered abandoned. When this expires, the
#     session is cleaned up AND any captured frames are saved as a pending
#     email so they aren't lost.
#
#   WATCHDOG_TIMEOUT: Safety net for a camera stuck in WAITING_RESET. A full
#     session normally leaves WAITING_RESET when the recorder's /session-reset
#     arrives; if that signal is missed, this force-resets the camera and saves
#     any captured frames as a pending email.
#
#   PENDING_EMAIL_TIMEOUT (360s): Handled by separate pending_email_watchdog.
# ==========================================
def session_watchdog():
    while True:
        time.sleep(WATCHDOG_CHECK)
        now = time.time()
        for name, state in camera_states.items():
            with state.lock:
                # Handle COMPLETED state cleanup
                if state.state == SessionState.COMPLETED:
                    if now - state.last_reset_time > 5:
                        state.state = SessionState.IDLE
                    continue
                
                # Handle WAITING_RESET (safety net for a missed recorder reset)
                if (state.state == SessionState.WAITING_RESET and
                        state.last_waiting_start > 0):
                    if now - state.last_waiting_start > WATCHDOG_TIMEOUT:
                        print(f"⚠️ Watchdog: {name} stuck in WAITING_RESET. Force resetting.")
                        # Save any captured frames as pending email before cleanup
                        if state.count > 0 and state.detection_id:
                            with state.session_frames_lock:
                                frames_copy = state.session_frames.copy()
                                state.session_frames = []
                            with pending_emails_lock:
                                pending_emails[name] = PendingEmail(
                                    name, state.detection_id, frames_copy, state.count
                                )
                            print(f"📧 {name}: Saved {state.count} frames as pending email before force reset")
                        state.state = SessionState.IDLE
                        state.count = 0
                        state.detection_id = None
                        state.active_session_id = None
                        state.last_processed_count = 0
                
                # Handle ACTIVE session idle timeout (NO new detections for a while)
                elif state.state == SessionState.ACTIVE and state.count > 0:
                    if now - state.last_activity > SESSION_TIMEOUT:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⏰ Session idle timeout: {name} - no detections for {SESSION_TIMEOUT}s")
                        
                        # Save partial session as pending email so frames aren't lost
                        if state.count > 0 and state.detection_id:
                            with state.session_frames_lock:
                                frames_copy = state.session_frames.copy()
                                state.session_frames = []
                            with pending_emails_lock:
                                pending_emails[name] = PendingEmail(
                                    name, state.detection_id, frames_copy, state.count
                                )
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📧 {name}: Saved {state.count} frames as pending email before cleanup")
                        
                        state.state = SessionState.IDLE
                        state.count = 0
                        state.detection_id = None
                        state.active_session_id = None
                        state.last_processed_count = 0


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    start_webhook_server()
    task_q = multiprocessing.Queue(maxsize=TASK_QUEUE_SIZE)
    result_q = multiprocessing.Queue(maxsize=RESULT_QUEUE_SIZE)
    start_yolo_worker(task_q, result_q)
    threading.Thread(target=check_yolo_health, args=(task_q, result_q), daemon=True).start()
    streams = {n: CameraStream(n, cfg["cam_rtsp"]) for n, cfg in NODES.items()}
    time.sleep(2)

    print("=" * 60)
    print(f"CCTV DETECTOR v{VERSION}")
    print(f"ANALYSIS_INTERVAL: {ANALYSIS_INTERVAL}s")
    print(f"MAX_IMAGES: {MAX_IMAGES} per camera")
    print(f"UPLOAD_WORKERS: {UPLOAD_WORKERS}")
    print(f"UPLOAD_QUEUE_SIZE: {UPLOAD_QUEUE_SIZE}")
    print(f"UPLOAD_MAX_RETRIES: {UPLOAD_MAX_RETRIES}")
    print(f"UPLOAD_RETRY_DELAY_BASE: {UPLOAD_RETRY_DELAY_BASE}s")
    print(f"YOLO_AUTO_RESTART: Enabled")
    print(f"YOLO_CONFIDENCE: {YOLO_CONFIDENCE}")
    print(f"YOLO_INPUT_SIZE: {YOLO_INPUT_SIZE}")
    print(f"YOLO_IOU: {YOLO_IOU}")
    print(f"WEBHOOK_PORT: {WEBHOOK_PORT}")
    print(f"SESSION_TIMEOUT: {SESSION_TIMEOUT}s (idle detection timeout for active sessions)")
    print(f"WATCHDOG: {WATCHDOG_CHECK}s check interval, {WATCHDOG_TIMEOUT}s timeout for stuck sessions")
    print(f"COOLDOWN: {COOLDOWN}s (unified cooldown)")
    print(f"RESET_DEDUP_WINDOW: {RESET_DEDUP_WINDOW}s (ignore duplicate resets)")
    print(f"PENDING_EMAIL_TIMEOUT: {PENDING_EMAIL_TIMEOUT}s (safety net if recorder reset missed)")
    print(f"MIN_PERSON_AREA: {MIN_PERSON_AREA}")
    print(f"MIN_ASPECT_RATIO: {MIN_ASPECT_RATIO}")
    print(f"DRAW_BOUNDING_BOXES: {DRAW_BOUNDING_BOXES}")
    print(f"TOP_EDGE_FILTER: {ENABLE_TOP_EDGE_FILTER} (per-camera [TOP_EDGE_CONFIG]; "
          f"global fallback margin={TOP_EDGE_MARGIN}px, high_conf={TOP_EDGE_HIGH_CONF})")
    print(f"AREA_FILTER: {ENABLE_AREA_FILTER}")
    print(f"ASPECT_FILTER: {ENABLE_ASPECT_FILTER}")
    print(f"DARK_FILTER: {ENABLE_DARK_FILTER} (threshold={DARKNESS_THRESHOLD}, "
          f"ratio={DARK_PIXEL_RATIO:.0%}, min_conf={FILTER_CONFIDENCE})")
    print(f"MAX_AREA_FILTER: Enabled (per-camera thresholds)")
    print(f"DEBUG_LOG_INTERVAL: {DEBUG_LOG_INTERVAL}s")
    print(f"ENABLE_DEBUG_PRINTS: {ENABLE_DEBUG_PRINTS}")
    print(f"SAVE_IMAGE_INTERVAL: {SAVE_IMAGE_INTERVAL}s (rate-limited image saving)")
    print("EMAIL CONFIGURATION DEBUG:")
    print(f"  SMTP_HOST: {SMTP_HOST}")
    print(f"  SMTP_PORT: {SMTP_PORT}")
    print(f"  SMTP_USER: {'***SET***' if SMTP_USER else 'NOT SET'}")
    print(f"  SMTP_PASS: {'***SET***' if SMTP_PASS else 'NOT SET'}")
    print(f"  ALERT_TO: {ALERT_TO if ALERT_TO else 'NOT SET'}")
    print(f"  ENABLE_EMAIL_ALERTS: {ENABLE_EMAIL_ALERTS}")
    print("=" * 60)

    threading.Thread(target=session_watchdog, daemon=True).start()
    threading.Thread(target=pending_email_watchdog, daemon=True).start()
    threading.Thread(target=config_reload_thread, daemon=True).start()

    # ==========================================
    # RESULT HANDLER
    # ==========================================
    _main_rejection_email_time = {}

    def handle_results():
        while True:
            try:
                name, frame, detections, ts, capture_time, rejections = result_q.get(timeout=0.01)

                # --- Process rejection emails from YOLO worker ---
                for (cam, rej_frame, box, conf, reason, kwargs) in rejections:
                    now = time.time()
                    last = _main_rejection_email_time.get(cam, 0)
                    if now - last >= 300:
                        _main_rejection_email_time[cam] = now
                        if rej_frame is not None:
                            email_executor.submit(
                                send_rejection_email, cam, rej_frame, box, conf, reason, **kwargs
                            )

                now = time.time()
                state = camera_states[name]

                with state.lock:
                    state.last_result_time = now

                if not detections:
                    continue

                with state.lock:
                    cooldown_remaining = (COOLDOWN - (now - state.last_reset_time)
                                         if state.last_reset_time > 0 else 0)

                if cooldown_remaining > 0:
                    continue

                current_count = None
                detection_id = None

                with state.lock:
                    if state.state == SessionState.IDLE:
                        state.state = SessionState.ACTIVE
                        state.count = 0
                        state.active_session_id = None
                        state.last_processed_count = 0

                    if not (state.state == SessionState.ACTIVE and state.count < MAX_IMAGES):
                        continue

                    if state.count == 0 and (now - state.last_upload < COOLDOWN):
                        continue

                    next_count = state.count + 1
                    if next_count <= state.last_processed_count:
                        continue

                    if state.count == 0:
                        state.detection_id = str(uuid.uuid4())[:8]
                        state.active_session_id = state.detection_id
                        with state.session_frames_lock:
                            state.session_frames = []
                        print(f"[{ts}] 🆔 {name}: New detection session {state.detection_id}")

                    state.count += 1
                    state.last_processed_count = state.count
                    state.last_activity = capture_time if capture_time > 0 else now
                    current_count = state.count
                    detection_id = state.detection_id
                    conf = detections[0]['conf'] if detections else 0

                    box = detections[0]['box']
                    x1, y1, x2, y2 = box
                    width = int(box[2] - box[0])
                    height = int(box[3] - box[1])

                    print(f"[{ts}] ⚡ >>> {name}: {state.count}/{MAX_IMAGES} "
                          f"[SID:{detection_id}] conf={conf:.2f} "
                          f"box={width}x{height}px area={width*height}px {y1}px")

                    # Store frame for pending email
                    with state.session_frames_lock:
                        state.session_frames.append({
                            'confidence': conf,
                            'frame': frame.copy(),
                            'timestamp': ts,
                            'frame_num': state.count
                        })
                        state.session_frames.sort(key=lambda x: x['confidence'], reverse=True)
                        state.session_frames = state.session_frames[:MAX_IMAGES]

                if current_count is None or detection_id is None:
                    continue

                if frame is not None:
                    draw_and_upload(name, NODES[name]["rpi_url"], frame, detections,
                                    ts, current_count, MAX_IMAGES, detection_id)

                with state.lock:
                    state.last_upload = now

            except Empty:
                continue
            except Exception as e:
                print(f"Handler error: {e}")

    threading.Thread(target=handle_results, daemon=True).start()

    # ==========================================
    # MAIN CAPTURE LOOP
    # ==========================================
    try:
        while True:
            now = time.time()
            for name in NODES:
                state = camera_states[name]

                with state.lock:
                    if state.state == SessionState.WAITING_RESET:
                        continue
                    time_since_last = now - state.last_queued_time

                if time_since_last < ANALYSIS_INTERVAL:
                    continue

                frame, frame_time = streams[name].get_frame()
                if frame is None:
                    continue

                try:
                    task_q.put_nowait((name, frame, time.strftime('%Y-%m-%d %H:%M:%S'), frame_time))
                    with state.lock:
                        state.last_queued_time = now
                except Full:
                    metrics.inc_dropped_frames()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nShutting down...")
        upload_executor.shutdown(wait=True)
        email_executor.shutdown(wait=True)
        if yolo_process:
            yolo_process.terminate()
            yolo_process.join(timeout=2)
        for s in streams.values():
            s.stop()
        sys.exit(0)
