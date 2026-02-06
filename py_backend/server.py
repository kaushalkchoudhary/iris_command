"""FastAPI control server: all API endpoints, overlay/frame/metrics management, FFmpeg publishing."""

import os
import yaml
import uuid
import shutil
import threading
import time
import subprocess
import json
import logging
import asyncio
import math
import re
from typing import Dict, List, Optional
from pathlib import Path
from collections import deque
import multiprocessing as mp
from datetime import datetime

import requests as http_requests
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from login import (
    login_user, add_user, delete_user, change_password, list_users,
    create_session, get_session, touch_session, delete_session, purge_expired_sessions,
)
from sam import (
    load_sam_model, sam_annotate_frame, sam_worker,
    sam_model_loaded, sam_results_lock, sam_results, sam_threads,
)
from log_utils import env_log_path, new_log_path
from report import generate_report, REPORTS_DIR
from helpers import get_video_duration, JPEG_QUALITY

IRIS_LOCAL = os.environ.get("IRIS_LOCAL", "0") == "1"
DEFAULT_MEDIAMTX_HOST = "127.0.0.1" if IRIS_LOCAL else "mediamtx1.stagingbot.xyz"
MEDIAMTX_HOST = os.environ.get("IRIS_MEDIAMTX_HOST", DEFAULT_MEDIAMTX_HOST)
DEFAULT_MEDIAMTX_API = f"http://{MEDIAMTX_HOST}:9997"
MEDIAMTX_API = os.environ.get("IRIS_MEDIAMTX_API", DEFAULT_MEDIAMTX_API)


# ── Pydantic models ──

class LoginRequest(BaseModel):
    username: str
    password: str

class UserCreateRequest(BaseModel):
    username: str
    password: str

class UserDeleteRequest(BaseModel):
    username: str

class PasswordChangeRequest(BaseModel):
    username: str
    new_password: str

class OverlayUpdate(BaseModel):
    heatmap: Optional[bool] = None
    heatmap_full: Optional[bool] = None
    heatmap_trails: Optional[bool] = None
    trails: Optional[bool] = None
    bboxes: Optional[bool] = None
    confidence: Optional[float] = None

class SourceIndexRequest(BaseModel):
    index: int

class SourceStartRequest(BaseModel):
    index: int
    mode: Optional[str] = None

class ActiveSourcesRequest(BaseModel):
    indexes: List[int]

class ConfidenceUpdate(BaseModel):
    confidence: float

class SamStartRequest(BaseModel):
    source: str
    prompt: str
    confidence: float = 0.7
    show_boxes: bool = True
    show_masks: bool = True

class SamStopRequest(BaseModel):
    source: str

class SamUpdateRequest(BaseModel):
    source: str
    confidence: Optional[float] = None
    show_boxes: Optional[bool] = None
    show_masks: Optional[bool] = None

class FrontendLogEntry(BaseModel):
    level: str
    message: str
    context: Optional[dict] = None
    ts: Optional[str] = None


# ── Config paths ──

RTSP_CONFIG_PATH = Path("config/rtsp_links.yml")
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads" / "recordings"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALERTS_DIR = DATA_DIR / "alerts"
ALERTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Thread-safe state ──

alerts_lock = threading.Lock()
alerts: deque = deque(maxlen=50)
alert_cooldowns: Dict[str, float] = {}
ALERT_COOLDOWN_SECONDS = 15

overlay_lock = threading.Lock()
overlays: Dict[str, Dict[str, bool]] = {}

metrics_lock = threading.Lock()
metrics: Dict[str, dict] = {}

frame_lock = threading.Condition()
frame_buffer: Dict[str, bytes] = {}
frame_sequences: Dict[str, int] = {}
frame_bgr_buffer: Dict[str, np.ndarray] = {}

raw_frame_lock = threading.Condition()
raw_frame_buffer: Dict[str, bytes] = {}
raw_frame_sequences: Dict[str, int] = {}
raw_frame_bgr_buffer: Dict[str, np.ndarray] = {}

jobs_lock = threading.Lock()
jobs: Dict[str, dict] = {}

source_lock = threading.Lock()
running_sources: Dict[int, dict] = {}

# FFmpeg RTSP publishers (processed stream -> MediaMTX)
ffmpeg_publishers: Dict[str, subprocess.Popen] = {}
ffmpeg_log_files: Dict[str, object] = {}
ffmpeg_writer_threads: Dict[str, threading.Thread] = {}
ffmpeg_stop_events: Dict[str, threading.Event] = {}
ffmpeg_next_start: Dict[str, float] = {}
ffmpeg_frame_sequences: Dict[str, int] = {}
ffmpeg_state_lock = threading.Lock()
ffmpeg_starting: set = set()
raw_ffmpeg_publishers: Dict[str, subprocess.Popen] = {}
raw_ffmpeg_log_files: Dict[str, object] = {}
raw_ffmpeg_writer_threads: Dict[str, threading.Thread] = {}
raw_ffmpeg_stop_events: Dict[str, threading.Event] = {}
MEDIAMTX_RTSP_PORT = int(os.environ.get("IRIS_MEDIAMTX_RTSP_PORT", "8554"))
# Always publish to the local MediaMTX instance (same machine).
# The remote hostname is only for frontend WebRTC consumption via Cloudflare.
MEDIAMTX_PUBLISH_BASE = os.environ.get(
    "IRIS_MEDIAMTX_PUBLISH_BASE",
    f"rtsp://127.0.0.1:{MEDIAMTX_RTSP_PORT}",
)
PROCESSED_FPS = int(os.environ.get("IRIS_PROCESSED_FPS", "30"))
KEYFRAME_INTERVAL_SEC = int(os.environ.get("IRIS_KEYFRAME_INTERVAL_SEC", "1"))
PROCESSED_BITRATE = os.environ.get("IRIS_PROCESSED_BITRATE", "12M")
PROCESSED_MAXRATE = os.environ.get("IRIS_PROCESSED_MAXRATE", "16M")
PROCESSED_BUFSIZE = os.environ.get("IRIS_PROCESSED_BUFSIZE", "10M")
PROCESSED_NVENC_PRESET = os.environ.get("IRIS_PROCESSED_NVENC_PRESET", "p5")
RAW_BITRATE = os.environ.get("IRIS_RAW_BITRATE", "3M")
RAW_MAXRATE = os.environ.get("IRIS_RAW_MAXRATE", "4M")
RAW_BUFSIZE = os.environ.get("IRIS_RAW_BUFSIZE", "2M")
PERSIST_ACTIVE_SOURCES = os.environ.get("IRIS_PERSIST_ACTIVE_SOURCES", "0") == "1"
MAX_RTSP_STREAMS = int(os.environ.get("IRIS_MAX_RTSP_STREAMS", "4"))
MAX_UPLOAD_STREAMS = int(os.environ.get("IRIS_MAX_UPLOAD_STREAMS", "2"))
USE_NVENC = os.environ.get("IRIS_USE_NVENC", "1") == "1"
RAW_USE_NVENC = os.environ.get("IRIS_RAW_USE_NVENC", "0") == "1"
# For uploaded files, publish raw via direct ffmpeg->MediaMTX low-latency transcode path.
UPLOAD_RAW_PASSTHROUGH = os.environ.get("IRIS_UPLOAD_RAW_PASSTHROUGH", "0") == "1"
PROCESSED_JPEG_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
RAW_JPEG_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

# Active mode tracking
active_mode: Optional[str] = None

# Fixed overlay config per mode
MODE_OVERLAYS = {
    "congestion": {"heatmap": True, "heatmap_full": False, "heatmap_trails": True, "trails": False, "bboxes": False},
    "vehicle":    {"heatmap": False, "heatmap_full": False, "heatmap_trails": False, "trails": False, "bboxes": True, "bbox_label": "class"},
    "flow":       {"heatmap": False, "heatmap_full": False, "heatmap_trails": False, "trails": True,  "bboxes": True, "bbox_label": "speed"},
    "forensics":  {"heatmap": False, "heatmap_full": False, "heatmap_trails": False, "trails": False, "bboxes": False, "bbox_label": "class"},
    "crowd":      {"heatmap": True, "heatmap_full": False, "heatmap_trails": False, "trails": False, "bboxes": False, "bbox_label": "class"},
}

MODE_CONFIDENCE = {
    "congestion": 0.20,
    "vehicle":    0.25,
    "flow":       0.20,
    "forensics":  0.20,
    "crowd":      0.25,
}

confidence_lock = threading.Lock()
confidence_settings: Dict[str, float] = {}

start_source_callback = None
start_upload_callback = None

frontend_log_lock = threading.Lock()
frontend_log_file = None

# ── Auth/session state ──

auth_lock = threading.Lock()
auth_sessions: Dict[str, dict] = {}
auth_user_token: Dict[str, str] = {}
allowed_ips_last_seen: Dict[str, float] = {}
AUTH_IDLE_TTL_SECONDS = int(os.environ.get("IRIS_AUTH_IDLE_TTL_SECONDS", "1800"))
AUTH_MAX_IPS = int(os.environ.get("IRIS_AUTH_MAX_IPS", "2"))
AUTH_PUBLIC_PATHS = {
    "/api/login",
    "/api/health",
    "/api/logs/frontend",
}

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://iriscmd.stagingbot.xyz",
    "https://drone.magicboxhub.net",
]
_cors_origins_raw = os.environ.get("IRIS_CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS))
CORS_ORIGINS = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
MAGICBOXHUB_ORIGIN_RE = re.compile(r"^https://([a-zA-Z0-9-]+\.)*magicboxhub\.net$")


def _client_ip_from_request(request: Request) -> str:
    xfwd = request.headers.get("x-forwarded-for", "").strip()
    if xfwd:
        ip = xfwd.split(",")[0].strip()
        if ip:
            return ip
    xreal = request.headers.get("x-real-ip", "").strip()
    if xreal:
        return xreal
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _extract_bearer_token(request: Request) -> str:
    authz = request.headers.get("authorization", "").strip()
    if authz.lower().startswith("bearer "):
        return authz[7:].strip()
    fallback = request.headers.get("x-auth-token", "").strip()
    if fallback:
        return fallback
    q = request.query_params.get("auth_token", "").strip()
    return q


def _purge_stale_auth(now: Optional[float] = None):
    if now is None:
        now = time.time()
    purge_expired_sessions(AUTH_IDLE_TTL_SECONDS, int(now))
    stale_tokens = []
    for token, sess in list(auth_sessions.items()):
        last_seen = float(sess.get("last_seen", 0.0))
        if now - last_seen > AUTH_IDLE_TTL_SECONDS:
            stale_tokens.append((token, sess.get("username")))
    for token, username in stale_tokens:
        auth_sessions.pop(token, None)
        if username and auth_user_token.get(username) == token:
            auth_user_token.pop(username, None)

    for ip, last_seen in list(allowed_ips_last_seen.items()):
        if now - float(last_seen) > AUTH_IDLE_TTL_SECONDS:
            allowed_ips_last_seen.pop(ip, None)


def _register_ip_or_raise(ip: str):
    now = time.time()
    _purge_stale_auth(now)
    if ip in allowed_ips_last_seen:
        allowed_ips_last_seen[ip] = now
        return
    if len(allowed_ips_last_seen) >= AUTH_MAX_IPS:
        raise HTTPException(403, f"Maximum distinct client IPs reached ({AUTH_MAX_IPS}).")
    allowed_ips_last_seen[ip] = now


def _is_origin_allowed(origin: str) -> bool:
    if not origin:
        return False
    if origin in CORS_ORIGINS:
        return True
    return bool(MAGICBOXHUB_ORIGIN_RE.match(origin))


def _cors_headers_for_request(request: Request) -> Dict[str, str]:
    origin = request.headers.get("origin", "").strip()
    if _is_origin_allowed(origin):
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Vary": "Origin",
        }
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    }


# ── RTSP config helpers ──

def _extract_rtsp_links_from_mediamtx_cfg() -> List[str]:
    """Best-effort fallback: derive RTSP source URLs from MediaMTX path config."""
    mediamtx_cfg_path = Path("config/mediamtx.yml")
    if not mediamtx_cfg_path.exists():
        return []
    try:
        with open(mediamtx_cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return []

    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    if not isinstance(paths, dict):
        return []

    out = []
    for _, p in paths.items():
        if not isinstance(p, dict):
            continue
        src = str(p.get("source", "")).strip()
        if src.startswith("rtsp://"):
            out.append(src)
    return out


def _normalize_rtsp_config(cfg: dict | None) -> dict:
    if not isinstance(cfg, dict):
        cfg = {}
    links = cfg.get("rtsp_links")
    active = cfg.get("active_sources")
    overlays_cfg = cfg.get("overlays")

    if not isinstance(links, list):
        links = []
    if not isinstance(active, list):
        active = []
    if not isinstance(overlays_cfg, dict):
        overlays_cfg = {}

    # Auto-heal: if rtsp_links disappeared, recover from mediamtx path sources.
    if not links:
        recovered = _extract_rtsp_links_from_mediamtx_cfg()
        if recovered:
            print(f"[CONFIG] Recovered {len(recovered)} rtsp_links from config/mediamtx.yml")
            links = recovered

    return {"rtsp_links": links, "active_sources": active, "overlays": overlays_cfg}


def load_rtsp_config():
    if not RTSP_CONFIG_PATH.exists():
        return {"rtsp_links": [], "active_sources": [], "overlays": {}}
    with open(RTSP_CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f) or {}
    return _normalize_rtsp_config(raw)

def save_rtsp_config(cfg):
    cfg = _normalize_rtsp_config(cfg)
    with open(RTSP_CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

def source_display_name(url: str) -> str:
    return url.rstrip("/").split("/")[-1]

def mask_rtsp(url: str) -> str:
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":")[0]
        return f"{scheme}://{user}:****@{host}"
    return url


def _write_frontend_log(entry: FrontendLogEntry):
    global frontend_log_file
    with frontend_log_lock:
        if frontend_log_file is None:
            log_path = env_log_path("frontend") or new_log_path("frontend")
            frontend_log_file = open(log_path, "a", buffering=1)
        ts = entry.ts or time.strftime("%Y-%m-%dT%H:%M:%S")
        level = (entry.level or "info").upper()
        msg = entry.message or ""
        context = ""
        if entry.context:
            try:
                context = " " + json.dumps(entry.context, separators=(",", ":"), ensure_ascii=True)
            except Exception:
                context = ""
        frontend_log_file.write(f"{ts} {level} {msg}{context}\n")


# ── Mediamtx helpers ──

def _mediamtx_add_path_sync(name: str, rtsp_url: str):
    try:
        http_requests.delete(f"{MEDIAMTX_API}/v3/config/paths/delete/{name}", timeout=2)
    except Exception:
        pass
    try:
        r = http_requests.post(
            f"{MEDIAMTX_API}/v3/config/paths/add/{name}",
            json={"source": rtsp_url, "sourceOnDemand": False},
            timeout=3,
        )
        print(f"[MEDIAMTX] Added path {name} (status={r.status_code})")
    except Exception as e:
        print(f"[MEDIAMTX] Failed to add path {name}: {e}")

def _mediamtx_remove_path_sync(name: str):
    try:
        r = http_requests.delete(f"{MEDIAMTX_API}/v3/config/paths/delete/{name}", timeout=2)
        print(f"[MEDIAMTX] Deleted path {name} (status={r.status_code})")
    except Exception as e:
        print(f"[MEDIAMTX] Failed to delete path {name}: {e}")

def mediamtx_add_path(name: str, rtsp_url: str):
    threading.Thread(target=_mediamtx_add_path_sync, args=(name, rtsp_url), daemon=True).start()

def mediamtx_remove_path(name: str):
    threading.Thread(target=_mediamtx_remove_path_sync, args=(name,), daemon=True).start()


# ── FFmpeg publisher management ──

def _wait_for_frame(name: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    with frame_lock:
        last_seq = frame_sequences.get(name, 0)
    while time.time() < deadline:
        with frame_lock:
            if frame_sequences.get(name, 0) > last_seq:
                data = frame_buffer.get(name)
                if data:
                    return data
            frame_lock.wait(timeout=0.3)
    return None

def _wait_for_frame_bgr(name: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    with frame_lock:
        last_seq = frame_sequences.get(name, 0)
    while time.time() < deadline:
        with frame_lock:
            if frame_sequences.get(name, 0) > last_seq:
                frame = frame_bgr_buffer.get(name)
                if frame is not None:
                    return frame
            frame_lock.wait(timeout=0.3)
    return None

def _wait_for_raw_bgr_frame(name: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    with raw_frame_lock:
        last_seq = raw_frame_sequences.get(name, 0)
    while time.time() < deadline:
        with raw_frame_lock:
            if raw_frame_sequences.get(name, 0) > last_seq:
                frame = raw_frame_bgr_buffer.get(name)
                if frame is not None:
                    return frame
            raw_frame_lock.wait(timeout=0.3)
    return None


def _ffmpeg_writer_worker(name: str, proc: subprocess.Popen, stop_event: threading.Event, width: int, height: int):
    last_seq = -1
    last_frame = None
    frame_interval = 1.0 / PROCESSED_FPS
    next_send_time = time.monotonic()
    while not stop_event.is_set():
        # Rate-limit to match PROCESSED_FPS so FFmpeg timestamps stay in sync.
        now = time.monotonic()
        sleep_dur = next_send_time - now
        if sleep_dur > 0:
            time.sleep(sleep_dur)
        next_send_time = max(time.monotonic(), next_send_time) + frame_interval

        with frame_lock:
            seq = frame_sequences.get(name, 0)
            frame = frame_bgr_buffer.get(name)

        if stop_event.is_set():
            break
        if frame is not None and seq > last_seq:
            if frame.shape[1] != width or frame.shape[0] != height:
                print(f"[FFMPEG] Frame size changed for {name} ({width}x{height} -> {frame.shape[1]}x{frame.shape[0]}), restarting.")
                break
            last_frame = frame
            last_seq = seq

        if last_frame is None:
            continue

        try:
            if proc.stdin is None:
                print(f"[FFMPEG] Writer stdin unavailable for {name}, stopping writer loop.")
                break
            proc.stdin.write(last_frame.tobytes())
            ffmpeg_frame_sequences[name] = ffmpeg_frame_sequences.get(name, 0) + 1
        except BrokenPipeError:
            break
        except Exception as e:
            print(f"[FFMPEG] Writer error for {name}: {e}")
            break

    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass

def _raw_ffmpeg_writer_worker(name: str, proc: subprocess.Popen, stop_event: threading.Event, width: int, height: int):
    last_seq = -1
    while not stop_event.is_set():
        with raw_frame_lock:
            while raw_frame_sequences.get(name, 0) <= last_seq and not stop_event.is_set():
                raw_frame_lock.wait(timeout=0.3)
            seq = raw_frame_sequences.get(name, 0)
            frame = raw_frame_bgr_buffer.get(name)

        if stop_event.is_set():
            break
        if frame is None or seq <= last_seq:
            time.sleep(0.01)
            continue

        last_seq = seq
        if frame.shape[1] != width or frame.shape[0] != height:
            print(f"[FFMPEG RAW] Frame size changed for {name} ({width}x{height} -> {frame.shape[1]}x{frame.shape[0]}), restarting.")
            break

        try:
            proc.stdin.write(frame.tobytes())
            proc.stdin.flush()
        except BrokenPipeError:
            break
        except Exception as e:
            print(f"[FFMPEG RAW] Writer error for {name}: {e}")
            break

    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass


def _start_ffmpeg_publisher(name: str, wait_timeout: float = 8.0) -> bool:
    with ffmpeg_state_lock:
        if name in ffmpeg_starting:
            return False
        existing = ffmpeg_publishers.get(name)
        if existing is not None and existing.poll() is None:
            return True
        ffmpeg_starting.add(name)

    # Keep the in-flight "starting" marker while replacing any stale publisher
    # so concurrent start attempts can't race and spawn duplicates.
    _stop_ffmpeg_publisher(name, clear_starting=False)
    rtsp_url = f"{MEDIAMTX_PUBLISH_BASE}/processed_{name}"

    frame = _wait_for_frame_bgr(name, timeout=wait_timeout)
    if frame is None:
        print(f"[FFMPEG] No frames available for {name}, skipping publisher start.")
        with ffmpeg_state_lock:
            ffmpeg_starting.discard(name)
        return False

    height, width = frame.shape[:2]
    gop = max(1, PROCESSED_FPS * KEYFRAME_INTERVAL_SEC)
    if USE_NVENC:
        encoder_args = [
            "-c:v", "h264_nvenc",
            "-preset", PROCESSED_NVENC_PRESET,
            "-tune", "ll",
            "-rc", "cbr",
            "-b:v", PROCESSED_BITRATE,
            "-maxrate", PROCESSED_MAXRATE,
            "-bufsize", PROCESSED_BUFSIZE,
            "-spatial-aq", "1",
            "-aq-strength", "8",
            "-forced-idr", "1",
        ]
    else:
        encoder_args = [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", PROCESSED_BITRATE,
            "-maxrate", PROCESSED_MAXRATE,
            "-bufsize", PROCESSED_BUFSIZE,
            "-x264-params", f"keyint={gop}:min-keyint={gop}:scenecut=0",
        ]

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "warning",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(PROCESSED_FPS),
        "-i", "pipe:0",
        "-vf", "format=yuv420p",
        *encoder_args,
        "-g", str(gop),
        "-bf", "0",
        "-threads", "1",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]

    try:
        ffmpeg_log_path = env_log_path("ffmpeg") or (new_log_path("ffmpeg").parent / "ffmpeg.log")
        ffmpeg_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(ffmpeg_log_path, "a", buffering=1)
        ffmpeg_log_files[name] = log_file
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
        ffmpeg_publishers[name] = proc
        stop_event = threading.Event()
        ffmpeg_stop_events[name] = stop_event
        thread = threading.Thread(
            target=_ffmpeg_writer_worker,
            args=(name, proc, stop_event, width, height),
            daemon=True,
        )
        ffmpeg_writer_threads[name] = thread
        thread.start()
        print(f"[FFMPEG] Started publisher for {name} -> processed_{name} (pid={proc.pid})")
        with ffmpeg_state_lock:
            ffmpeg_starting.discard(name)
        return True
    except Exception as e:
        print(f"[FFMPEG] Failed to start publisher for {name}: {e}")
        with ffmpeg_state_lock:
            ffmpeg_starting.discard(name)
        return False

def _start_raw_ffmpeg_publisher(name: str, wait_timeout: float = 10.0) -> bool:
    _stop_raw_ffmpeg_publisher(name)
    rtsp_url = f"{MEDIAMTX_PUBLISH_BASE}/{name}"

    frame = _wait_for_raw_bgr_frame(name, timeout=wait_timeout)
    if frame is None:
        print(f"[FFMPEG RAW] No frames available for {name}, skipping raw publisher start.")
        return False

    height, width = frame.shape[:2]
    gop = max(1, PROCESSED_FPS * KEYFRAME_INTERVAL_SEC)
    if RAW_USE_NVENC:
        encoder_args = [
            "-c:v", "h264_nvenc",
            "-preset", "p1",
            "-tune", "ll",
            "-rc", "cbr",
            "-b:v", RAW_BITRATE,
            "-maxrate", RAW_MAXRATE,
            "-bufsize", RAW_BUFSIZE,
            "-forced-idr", "1",
        ]
    else:
        encoder_args = [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", RAW_BITRATE,
            "-maxrate", RAW_MAXRATE,
            "-bufsize", RAW_BUFSIZE,
            "-x264-params", f"keyint={gop}:min-keyint={gop}:scenecut=0",
        ]

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "warning",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(PROCESSED_FPS),
        "-i", "pipe:0",
        "-vf", "format=yuv420p",
        *encoder_args,
        "-g", str(gop),
        "-bf", "0",
        "-threads", "1",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]

    try:
        ffmpeg_log_path = env_log_path("ffmpeg") or (new_log_path("ffmpeg").parent / "ffmpeg.log")
        ffmpeg_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(ffmpeg_log_path, "a", buffering=1)
        raw_ffmpeg_log_files[name] = log_file
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
        raw_ffmpeg_publishers[name] = proc
        stop_event = threading.Event()
        raw_ffmpeg_stop_events[name] = stop_event
        thread = threading.Thread(
            target=_raw_ffmpeg_writer_worker,
            args=(name, proc, stop_event, width, height),
            daemon=True,
        )
        raw_ffmpeg_writer_threads[name] = thread
        thread.start()
        print(f"[FFMPEG RAW] Started publisher for {name} -> {name} (pid={proc.pid})")
        return True
    except Exception as e:
        print(f"[FFMPEG RAW] Failed to start publisher for {name}: {e}")
        return False

def _start_raw_ffmpeg_publisher_async(name: str):
    def _run():
        for _ in range(3):
            if _start_raw_ffmpeg_publisher(name):
                return
            time.sleep(1.5)
    threading.Thread(target=_run, daemon=True).start()

def _start_upload_raw_passthrough_publisher(name: str, file_path: str) -> bool:
    """Publish upload file directly to MediaMTX raw path with low-latency transcode."""
    _stop_raw_ffmpeg_publisher(name)
    rtsp_url = f"{MEDIAMTX_PUBLISH_BASE}/{name}"
    gop = max(1, PROCESSED_FPS * KEYFRAME_INTERVAL_SEC)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "warning",
        "-re",
        "-stream_loop", "-1",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-i", file_path,
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", str(gop),
        "-keyint_min", str(gop),
        "-sc_threshold", "0",
        "-b:v", RAW_BITRATE,
        "-maxrate", RAW_MAXRATE,
        "-bufsize", RAW_BUFSIZE,
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]

    try:
        ffmpeg_log_path = env_log_path("ffmpeg") or (new_log_path("ffmpeg").parent / "ffmpeg.log")
        ffmpeg_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(ffmpeg_log_path, "a", buffering=1)
        raw_ffmpeg_log_files[name] = log_file
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
        raw_ffmpeg_publishers[name] = proc
        stop_event = threading.Event()
        raw_ffmpeg_stop_events[name] = stop_event
        print(f"[FFMPEG RAW] Started upload low-latency transcode for {name} -> {name} (pid={proc.pid})")
        return True
    except Exception as e:
        print(f"[FFMPEG RAW] Failed upload low-latency transcode for {name}: {e}")
        return False

def _start_upload_raw_passthrough_async(name: str, file_path: str):
    def _run():
        for _ in range(3):
            if _start_upload_raw_passthrough_publisher(name, file_path):
                return
            time.sleep(1.0)
    threading.Thread(target=_run, daemon=True).start()

def _start_ffmpeg_publisher_async(name: str):
    def _run():
        for _ in range(5):
            if _start_ffmpeg_publisher(name):
                return
            time.sleep(1.0)
    threading.Thread(target=_run, daemon=True).start()

def _stop_ffmpeg_publisher(name: str, clear_starting: bool = True):
    if clear_starting:
        with ffmpeg_state_lock:
            ffmpeg_starting.discard(name)
    ffmpeg_next_start.pop(name, None)
    ffmpeg_frame_sequences.pop(name, None)
    stop_event = ffmpeg_stop_events.pop(name, None)
    if stop_event:
        stop_event.set()

    proc = ffmpeg_publishers.pop(name, None)
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    thread = ffmpeg_writer_threads.pop(name, None)
    if thread:
        try:
            thread.join(timeout=1)
        except Exception:
            pass

    log_file = ffmpeg_log_files.pop(name, None)
    if log_file:
        try:
            log_file.close()
        except Exception:
            pass
        print(f"[FFMPEG] Stopped publisher for {name}")

def _stop_raw_ffmpeg_publisher(name: str):
    stop_event = raw_ffmpeg_stop_events.pop(name, None)
    if stop_event:
        stop_event.set()

    proc = raw_ffmpeg_publishers.pop(name, None)
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    thread = raw_ffmpeg_writer_threads.pop(name, None)
    if thread:
        try:
            thread.join(timeout=1)
        except Exception:
            pass

    log_file = raw_ffmpeg_log_files.pop(name, None)
    if log_file:
        try:
            log_file.close()
        except Exception:
            pass
        print(f"[FFMPEG RAW] Stopped publisher for {name}")

_ffmpeg_monitor_sources: set = set()
_ffmpeg_monitor_thread: threading.Thread = None
_ffmpeg_monitor_stop = threading.Event()

def _is_source_name_active(name: str) -> bool:
    cfg = load_rtsp_config()
    links = cfg.get("rtsp_links", [])
    with source_lock:
        for idx in running_sources.keys():
            if 1 <= idx <= len(links) and source_display_name(links[idx - 1]) == name:
                return True
    with upload_sources_lock:
        if name in upload_sources:
            return True
    return False

def _ffmpeg_monitor_worker():
    while not _ffmpeg_monitor_stop.is_set():
        try:
            for name in list(_ffmpeg_monitor_sources):
                with ffmpeg_state_lock:
                    proc = ffmpeg_publishers.get(name)
                now = time.time()
                next_at = ffmpeg_next_start.get(name, 0)

                if proc is None:
                    if not _is_source_name_active(name):
                        _ffmpeg_monitor_sources.discard(name)
                        ffmpeg_next_start.pop(name, None)
                        continue
                    if now >= next_at:
                        ok = _start_ffmpeg_publisher(name)
                        if not ok:
                            ffmpeg_next_start[name] = now + 3.0
                        else:
                            ffmpeg_next_start.pop(name, None)
                    continue

                poll = proc.poll()
                if poll is not None:
                    source_active = _is_source_name_active(name)

                    if source_active:
                        print(f"[FFMPEG] Publisher for {name} exited (code={poll}), restarting...")
                        _stop_ffmpeg_publisher(name)
                        ffmpeg_next_start[name] = now + 1.0
                    else:
                        _ffmpeg_monitor_sources.discard(name)
        except Exception as e:
            print(f"[FFMPEG Monitor] Error: {e}")

        _ffmpeg_monitor_stop.wait(2)

def _ensure_ffmpeg_monitor():
    global _ffmpeg_monitor_thread
    if _ffmpeg_monitor_thread is None or not _ffmpeg_monitor_thread.is_alive():
        _ffmpeg_monitor_stop.clear()
        _ffmpeg_monitor_thread = threading.Thread(target=_ffmpeg_monitor_worker, daemon=True)
        _ffmpeg_monitor_thread.start()
        print("[FFMPEG] Monitor thread started")

def _stop_all_ffmpeg_publishers():
    for name in list(ffmpeg_publishers.keys()):
        _stop_ffmpeg_publisher(name)
    for name in list(raw_ffmpeg_publishers.keys()):
        _stop_raw_ffmpeg_publisher(name)


# ── Overlay / metrics / frame management ──

def ensure_overlay(name: str):
    with overlay_lock:
        if name in overlays:
            existing = overlays[name]
            # Ensure all required keys exist
            if isinstance(existing, dict) and "trails" in existing and "heatmap" in existing and "bboxes" in existing:
                if "heatmap_full" not in existing:
                    existing["heatmap_full"] = existing.get("heatmap", True)
                if "heatmap_trails" not in existing:
                    existing["heatmap_trails"] = existing.get("heatmap", True)
                if "confidence" not in existing:
                    existing["confidence"] = 0.15
                return
        
        # Initialize with defaults
        default_state = {
            "heatmap": True,
            "heatmap_full": True,
            "heatmap_trails": True,
            "trails": True,
            "bboxes": True,
            "confidence": 0.15,
        }
        overlays[name] = default_state
        print(f"[OVERLAY] Initialized {name}: {default_state}")

def get_overlay_state(stream: str) -> Dict[str, bool]:
    ensure_overlay(stream)
    with overlay_lock:
        return dict(overlays.get(stream, {
            "heatmap": True, 
            "heatmap_full": True, 
            "heatmap_trails": True, 
            "trails": True, 
            "bboxes": True
        }))

def _resolve_source_mode(name: str, source_metrics: Optional[dict] = None) -> Optional[str]:
    """Resolve mode with per-source precedence before falling back to global mode."""
    with upload_sources_lock:
        upload_info = upload_sources.get(name)
        if upload_info and upload_info.get("mode"):
            return upload_info.get("mode")
    with overlay_lock:
        overlay_mode = (overlays.get(name) or {}).get("active_mode")
        if overlay_mode:
            return overlay_mode
    if isinstance(source_metrics, dict):
        metric_mode = source_metrics.get("mode")
        if metric_mode:
            return metric_mode
    return active_mode


def _to_json_safe(value):
    """Convert nested metric values to JSON-safe builtins."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    # numpy / torch scalars
    try:
        if hasattr(value, "item"):
            return _to_json_safe(value.item())
    except Exception:
        pass
    return str(value)


def _default_metrics_for_mode(mode: Optional[str]) -> Dict[str, object]:
    base = {
        "fps": 0.0,
        "detection_count": 0,
        "current_detection_count": 0,
    }
    if mode == "vehicle":
        base.update({
            "class_counts": {},
            "state_counts": {"moving": 0, "stopped": 0, "abnormal": 0},
            "behavior_counts": {"stable": 0, "start_stop": 0, "erratic": 0},
            "type_influence": {},
            "attention_list": [],
            "high_impact_vehicles": [],
            "dwell_stats": {"avg_dwell": 0.0, "max_dwell": 0.0, "over_threshold": 0, "longest_vehicles": []},
        })
    elif mode in ("flow", "congestion"):
        base.update({
            "congestion_index": 0,
            "traffic_density": 0,
            "mobility_index": 0,
            "stalled_pct": 0,
            "slow_pct": 0,
            "medium_pct": 0,
            "fast_pct": 0,
            "class_counts": {},
            "hot_regions": {"active_count": 0, "severity_counts": {"HIGH": 0, "MODERATE": 0, "LOW": 0}, "regions": []},
        })
    elif mode == "crowd":
        base.update({
            "crowd_count": 0,
            "crowd_density": 0,
            "avg_density": 0,
            "peak_density": 0,
            "peak_count": 0,
            "risk_score": 0,
            "density_class": "sparse",
            "operational_status": "MONITOR",
            "zones": [],
            "zone_distribution": {"sparse": 0, "gathering": 0, "dense": 0, "critical": 0},
            "hotspots": [],
            "flow_summary": {"total_inflow": 0, "total_outflow": 0, "net_flow": 0},
            "anomalies": [],
        })
    return base

def update_metrics(stream: str, data: dict):
    # Handle critical workflow signals first
    if data.get("__started__"):
        with source_lock:
            for info in running_sources.values():
                if info.get("name") == stream:
                    info["started_processing"] = True
        with upload_sources_lock:
            info = upload_sources.get(stream)
            if info:
                info["started_processing"] = True
                if not info.get("started_at"):
                    info["started_at"] = time.time()

    if data.get("__finished__"):
        # Ignore finished flag for uploads to avoid premature completion UI.
        # Reports are triggered manually by the user.
        return

    stream_mode = _resolve_source_mode(stream, data)
    if stream_mode == "forensics":
        # Suppress standard metrics data store in Forensics mode
        return

    # Ensure UI can consume metrics as soon as any valid payload arrives.
    with source_lock:
        for info in running_sources.values():
            if info.get("name") == stream:
                info["started_processing"] = True
    with upload_sources_lock:
        info = upload_sources.get(stream)
        if info:
            info["started_processing"] = True
            if not info.get("started_at"):
                info["started_at"] = time.time()

    with metrics_lock:
        payload = _to_json_safe(data or {})
        if not isinstance(payload, dict):
            payload = {}
        if stream_mode and "mode" not in payload:
            payload["mode"] = stream_mode
        mode_for_defaults = payload.get("mode") or stream_mode
        defaults = _default_metrics_for_mode(mode_for_defaults)
        defaults.update(payload)
        metrics[stream] = defaults

def update_frame(stream: str, frame: np.ndarray):
    if frame is None or not isinstance(frame, np.ndarray):
        return

    try:
        ret_enc, buffer = cv2.imencode(".jpg", frame, PROCESSED_JPEG_ENCODE_PARAMS)
        if not ret_enc:
            return
        jpeg_data = buffer.tobytes()
    except Exception:
        return

    with frame_lock:
        frame_buffer[stream] = jpeg_data
        frame_bgr_buffer[stream] = frame
        frame_sequences[stream] = frame_sequences.get(stream, 0) + 1
        frame_lock.notify_all()

def update_raw_frame(stream: str, frame: np.ndarray):
    if frame is None or not isinstance(frame, np.ndarray):
        return

    try:
        ret_enc, buffer = cv2.imencode(".jpg", frame, RAW_JPEG_ENCODE_PARAMS)
        if not ret_enc:
            return
        jpeg_data = buffer.tobytes()
    except Exception:
        return

    # Mark source/upload active on first raw frame for snappier UI state.
    with source_lock:
        for info in running_sources.values():
            if info.get("name") == stream:
                info["started_processing"] = True
    with upload_sources_lock:
        info = upload_sources.get(stream)
        if info:
            info["started_processing"] = True
            if not info.get("started_at"):
                info["started_at"] = time.time()

    with raw_frame_lock:
        raw_frame_buffer[stream] = jpeg_data
        raw_frame_bgr_buffer[stream] = frame
        raw_frame_sequences[stream] = raw_frame_sequences.get(stream, 0) + 1
        raw_frame_lock.notify_all()

def get_all_metrics():
    with metrics_lock:
        raw_m = metrics.copy()
    m = {k: _to_json_safe(v) for k, v in dict(raw_m).items()}

    # Merge RTSP start times
    with source_lock:
        for info in running_sources.values():
            name = info.get("name")
            start_t = info.get("start_time")
            if name:
                if name not in m:
                    m[name] = {}
                if start_t:
                    m[name]["start_time"] = start_t

    # Merge Upload duration
    with upload_sources_lock:
        for name, info in upload_sources.items():
            if name not in m:
                m[name] = {}
            dur = info.get("duration", 0.0)
            if dur > 0:
                m[name]["total_duration"] = dur

    # Ensure each active source has a complete metrics schema for the UI.
    for name, data in list(m.items()):
        if not isinstance(data, dict):
            data = {}
        mode = _resolve_source_mode(name, data)
        defaults = _default_metrics_for_mode(mode)
        defaults.update(data)
        if mode and "mode" not in defaults:
            defaults["mode"] = mode
        m[name] = defaults

    return m


# ── Alert management ──

def add_alert(source: str, congestion: int, metrics_data: dict, screenshot_data: bytes):
    with alerts_lock:
        now = time.time()

        last_alert = alert_cooldowns.get(source, 0)
        if now - last_alert < ALERT_COOLDOWN_SECONDS:
            return None

        alert_cooldowns[source] = now

        alert_id = f"{source}_{int(now * 1000)}"
        screenshot_path = ALERTS_DIR / f"{alert_id}.jpg"

        try:
            with open(screenshot_path, "wb") as f:
                f.write(screenshot_data)
        except Exception as e:
            print(f"[ALERT] Failed to save screenshot: {e}")
            return None

        if congestion >= 60:
            severity = "critical"
        elif congestion >= 40:
            severity = "high"
        else:
            severity = "medium"

        alert = {
            "id": alert_id,
            "source": source,
            "severity": severity,
            "congestion": congestion,
            "timestamp": now,
            "time_str": time.strftime("%H:%M:%S"),
            "screenshot": f"/api/alerts/{alert_id}/screenshot",
            "metrics": {
                "congestion_index": metrics_data.get("congestion_index", congestion),
                "traffic_density": metrics_data.get("traffic_density", 0),
                "mobility_index": metrics_data.get("mobility_index", 0),
                "detection_count": metrics_data.get("detection_count", 0),
                "stalled_pct": metrics_data.get("stalled_pct", 0),
                "slow_pct": metrics_data.get("slow_pct", 0),
                "medium_pct": metrics_data.get("medium_pct", 0),
                "fast_pct": metrics_data.get("fast_pct", 0),
            }
        }

        alerts.appendleft(alert)
        print(f"[ALERT] New {severity} alert for {source}: congestion={congestion}%")
        return alert

def load_stored_alerts():
    loaded = 0
    for jpg in sorted(ALERTS_DIR.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True):
        stem = jpg.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        source = parts[0]
        try:
            ts = int(parts[1]) / 1000.0
        except ValueError:
            continue

        alert = {
            "id": stem,
            "source": source,
            "severity": "medium",
            "congestion": 0,
            "timestamp": ts,
            "time_str": time.strftime("%H:%M:%S", time.localtime(ts)),
            "screenshot": f"/api/alerts/{stem}/screenshot",
            "metrics": {},
        }
        with alerts_lock:
            alerts.append(alert)
        loaded += 1
        if loaded >= 50:
            break
    if loaded:
        print(f"[ALERT] Loaded {loaded} stored alerts from disk")

def get_alerts(limit: int = 20):
    with alerts_lock:
        return list(alerts)[:limit]

def clear_alerts():
    with alerts_lock:
        alerts.clear()
        alert_cooldowns.clear()
    for f in ALERTS_DIR.glob("*.jpg"):
        try:
            f.unlink()
        except:
            pass


# ── Source management ──

def _stop_all_sources():
    global active_mode
    stopped = 0
    import torch
    import gc
    import time

    print(f"[STOP_ALL] Initiating aggressive cleanup (Previous mode: {active_mode})")

    # 1. Stop all primary inference processes
    with source_lock:
        for idx, src in list(running_sources.items()):
            src["stop"].set()
            proc = src.get("process")
            if proc:
                try:
                    proc.terminate()
                    t_start = time.time()
                    while proc.is_alive() and time.time() - t_start < 0.5:
                        time.sleep(0.01)
                    if proc.is_alive():
                        proc.kill()
                except:
                    pass
            stopped += 1
        running_sources.clear()

    # 2. Stop all SAM workers and unload model
    for src_name, info in list(sam_threads.items()):
        print(f"[STOP_ALL] Stopping SAM worker for {src_name}")
        info["stop_event"].set()
    sam_threads.clear()

    try:
        from sam import unload_sam_model
        unload_sam_model()
    except:
        pass

    # 3. Clear SAM analytical results
    with sam_results_lock:
        sam_results.clear()

    # 4. Stop all FFmpeg publishers (streaming)
    _ffmpeg_monitor_sources.clear()
    _stop_all_ffmpeg_publishers()

    # 4b. Stop all upload processes and clear state
    _stop_all_uploads(delete_files=False)

    # 5. Clear ALL shared state/buffers
    with metrics_lock:
        metrics.clear()
    with overlay_lock:
        overlays.clear()
    with frame_lock:
        frame_buffer.clear()
        frame_bgr_buffer.clear()
        frame_sequences.clear()
    with raw_frame_lock:
        raw_frame_buffer.clear()
        raw_frame_bgr_buffer.clear()
        raw_frame_sequences.clear()

    # 6. Force Resource Reclamation
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[STOP_ALL] GPU memory cleared")
    except:
        pass

    # 7. Reset active mode and config
    active_mode = None 
    cfg = load_rtsp_config()
    cfg["active_sources"] = []
    save_rtsp_config(cfg)

    print(f"[STOP_ALL] Cleanup complete. Stopped {stopped} sources.")
    return stopped


# ── FastAPI app ──

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"^https://([a-zA-Z0-9-]+\.)*magicboxhub\.net$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def auth_guard(request: Request, call_next):
    path = request.url.path
    method = request.method.upper()

    # Skip non-API paths and CORS preflight.
    if not path.startswith("/api/") or method == "OPTIONS":
        return await call_next(request)

    # Public API paths.
    if path in AUTH_PUBLIC_PATHS:
        return await call_next(request)

    cors_headers = _cors_headers_for_request(request)

    token = _extract_bearer_token(request)
    if not token:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"}, headers=cors_headers)

    client_ip = _client_ip_from_request(request)
    client_tab = request.headers.get("x-client-tab", "").strip() or request.query_params.get("x_client_tab", "").strip()
    if not client_tab:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"}, headers=cors_headers)

    with auth_lock:
        _purge_stale_auth()
        sess = auth_sessions.get(token)
        if not sess:
            db_sess = get_session(token)
            if db_sess:
                sess = db_sess
                auth_sessions[token] = dict(sess)
                auth_user_token[sess.get("username")] = token
        if not sess:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"}, headers=cors_headers)
        if sess.get("tab_id") and sess.get("tab_id") != client_tab:
            return JSONResponse(status_code=401, content={"detail": "Session restricted to one browser tab"}, headers=cors_headers)
        if sess.get("ip") and sess.get("ip") != client_ip:
            return JSONResponse(status_code=401, content={"detail": "Session IP mismatch"}, headers=cors_headers)
        try:
            _register_ip_or_raise(client_ip)
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"detail": str(e.detail)}, headers=cors_headers)
        sess["last_seen"] = time.time()
        auth_sessions[token] = sess
        touch_session(token, int(sess["last_seen"]))

    request.state.auth_username = sess.get("username")
    request.state.auth_ip = client_ip
    return await call_next(request)

@app.exception_handler(413)
async def _payload_too_large_handler(request, exc):
    """Return 413 with CORS headers for large file uploads."""
    return JSONResponse(
        status_code=413,
        content={"detail": "File too large. Maximum upload size is 500MB."},
        headers=_cors_headers_for_request(request),
    )

@app.exception_handler(Exception)
async def _global_exception_handler(request, exc):
    """Return 500 with CORS headers so the browser doesn't mask the real error."""
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers=_cors_headers_for_request(request),
    )


# ── Auth endpoints ──

@app.post("/api/login")
def api_login(req: LoginRequest, request: Request):
    result = login_user(req.username, req.password)
    if not result.get("success"):
        raise HTTPException(status_code=401, detail=result.get("error", "Invalid credentials"))

    client_ip = _client_ip_from_request(request)
    client_tab = request.headers.get("x-client-tab", "").strip()
    if not client_tab:
        raise HTTPException(status_code=401, detail="Missing client tab identifier")

    token = uuid.uuid4().hex
    now = time.time()
    with auth_lock:
        _register_ip_or_raise(client_ip)
        old = auth_user_token.get(req.username)
        if old:
            auth_sessions.pop(old, None)
            delete_session(old)
        auth_user_token[req.username] = token
        auth_sessions[token] = {
            "username": req.username,
            "ip": client_ip,
            "tab_id": client_tab,
            "created_at": now,
            "last_seen": now,
        }
        create_session(token, req.username, client_ip, client_tab, int(now))

    return {"success": True, "username": req.username, "token": token}

@app.post("/api/logout")
def api_logout(request: Request):
    token = _extract_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    with auth_lock:
        sess = auth_sessions.pop(token, None)
        if not sess:
            db_sess = get_session(token)
            if not db_sess:
                raise HTTPException(status_code=401, detail="Unauthorized")
            sess = db_sess
        uname = sess.get("username")
        if uname and auth_user_token.get(uname) == token:
            auth_user_token.pop(uname, None)
        delete_session(token)
    return {"success": True}

@app.get("/api/users")
def api_list_users():
    return list_users()

@app.post("/api/users/add")
def api_add_user(req: UserCreateRequest):
    return add_user(req.username, req.password)

@app.post("/api/users/delete")
def api_delete_user(req: UserDeleteRequest):
    return delete_user(req.username)

@app.post("/api/users/change_password")
def api_change_password(req: PasswordChangeRequest):
    return change_password(req.username, req.new_password)


# ── Frontend log endpoint ──

@app.post("/api/logs/frontend")
def api_frontend_log(entry: FrontendLogEntry):
    _write_frontend_log(entry)
    return {"status": "ok"}


# ── Health endpoint ──

@app.get("/api/health")
def api_health():
    with source_lock:
        rtsp_count = len(running_sources)
    with upload_sources_lock:
        upload_count = len(upload_sources)
    return {
        "status": "ok",
        "active_mode": active_mode,
        "rtsp_sources": rtsp_count,
        "uploads": upload_count,
    }


# ── Overlay endpoints ──

@app.get("/api/overlays")
def get_overlays():
    with overlay_lock:
        try:
            return {k: dict(v) for k, v in overlays.items()}
        except:
            return dict(overlays)

@app.get("/api/overlays/{name}")
def get_overlay(name: str):
    ensure_overlay(name)
    with overlay_lock:
        try:
            return dict(overlays[name])
        except:
            return {"heatmap": True, "heatmap_full": True, "heatmap_trails": True, "trails": True, "bboxes": True}

@app.post("/api/overlays/{name}")
def set_overlay(name: str, update: OverlayUpdate):
    ensure_overlay(name)
    with overlay_lock:
        try:
            cur = dict(overlays[name])
        except:
            cur = {
                "heatmap": True,
                "heatmap_full": True,
                "heatmap_trails": True,
                "trails": True,
                "bboxes": True,
                "confidence": 0.15,
            }

        new_confidence = cur.get("confidence", 0.15)
        if update.confidence is not None:
            new_confidence = max(0.05, min(0.95, update.confidence))

        heatmap_val = update.heatmap if update.heatmap is not None else cur.get("heatmap", True)
        new_state = {
            "heatmap": heatmap_val,
            "heatmap_full": update.heatmap_full if update.heatmap_full is not None else cur.get("heatmap_full", heatmap_val),
            "heatmap_trails": update.heatmap_trails if update.heatmap_trails is not None else cur.get("heatmap_trails", heatmap_val),
            "trails": update.trails if update.trails is not None else cur.get("trails", True),
            "bboxes": update.bboxes if update.bboxes is not None else cur.get("bboxes", True),
            "confidence": new_confidence,
        }
        # Preserve mode context and label behavior set by mode configs.
        if "active_mode" in cur:
            new_state["active_mode"] = cur.get("active_mode")
        if "bbox_label" in cur:
            new_state["bbox_label"] = cur.get("bbox_label")
        # Keep heatmap children coherent.
        if not new_state["heatmap"]:
            new_state["heatmap_full"] = False
            new_state["heatmap_trails"] = False
        else:
            # Turning heatmap back on should re-enable per-vehicle heat layer by default
            # unless caller explicitly controls heatmap_trails.
            if update.heatmap is True and update.heatmap_trails is None:
                new_state["heatmap_trails"] = True
            if update.heatmap_full:
                new_state["heatmap"] = True
                if update.heatmap_trails is None:
                    new_state["heatmap_trails"] = False
            if update.heatmap_full is False and update.heatmap_trails is None:
                new_state["heatmap_trails"] = True
            if update.heatmap_trails:
                new_state["heatmap"] = True
        if cur.get("active_mode") == "congestion":
            # Congestion mode: only heatmaps, no trails/boxes.
            new_state["trails"] = False
            new_state["bboxes"] = False
        overlays[name] = dict(new_state)
        print(
            f"[OVERLAY] {name} updated: heatmap={new_state['heatmap']}, "
            f"heatmap_full={new_state['heatmap_full']}, heatmap_trails={new_state['heatmap_trails']}, "
            f"trails={new_state['trails']}, bboxes={new_state['bboxes']}, confidence={new_state['confidence']}"
        )

    cfg = load_rtsp_config()
    cfg.setdefault("overlays", {})[name] = new_state
    save_rtsp_config(cfg)
    return new_state


# ── Confidence endpoints ──

@app.get("/api/confidence/{name}")
def get_confidence(name: str):
    ensure_overlay(name)
    with overlay_lock:
        try:
            conf = overlays[name].get("confidence", 0.15)
        except:
            conf = 0.15
    return {"source": name, "confidence": conf}

@app.post("/api/confidence/{name}")
def set_confidence(name: str, update: ConfidenceUpdate):
    ensure_overlay(name)
    new_conf = max(0.05, min(0.95, update.confidence))

    with overlay_lock:
        try:
            cur = dict(overlays[name])
        except:
            cur = {
                "heatmap": True,
                "heatmap_full": True,
                "heatmap_trails": True,
                "trails": True,
                "bboxes": True,
                "confidence": 0.15,
            }
        cur["confidence"] = new_conf
        overlays[name] = cur

    print(f"[CONFIDENCE] {name} set to {new_conf}")

    cfg = load_rtsp_config()
    cfg.setdefault("overlays", {})[name] = cur
    save_rtsp_config(cfg)
    return {"source": name, "confidence": new_conf}

@app.get("/api/mode/confidence")
def get_mode_confidence():
    return {"mode_confidence": MODE_CONFIDENCE, "active_mode": active_mode}


# ── Source endpoints ──

@app.get("/api/sources")
def list_sources():
    cfg = load_rtsp_config()
    active = set(running_sources.keys())
    out = []

    for i, url in enumerate(cfg.get("rtsp_links", [])):
        idx = i + 1
        name = source_display_name(url)
        ensure_overlay(name)
        out.append({
            "index": idx,
            "name": name,
            "url": mask_rtsp(url),
            "active": idx in active,
            "start_time": running_sources[idx].get("start_time") if idx in active else None,
        })

    return {"sources": out, "active_sources": sorted(active)}

@app.post("/api/sources/start")
def start_source(req: SourceStartRequest):
    global active_mode

    cfg = load_rtsp_config()
    rtsp_links = cfg.get("rtsp_links", [])
    if req.index < 1 or req.index > len(rtsp_links):
        raise HTTPException(400, "Invalid source index or empty rtsp_links config")

    source_mode = req.mode or active_mode
    if req.mode and active_mode and req.mode != active_mode:
        print(f"[MODE] Running mixed modes in parallel: global={active_mode}, source={req.mode}")
    if req.mode and not active_mode:
        active_mode = req.mode

    url = rtsp_links[req.index - 1]
    name = source_display_name(url)

    base_overlay = MODE_OVERLAYS.get(
        source_mode,
        {"heatmap": False, "heatmap_full": False, "heatmap_trails": False, "trails": False, "bboxes": False},
    )
    mode_confidence = MODE_CONFIDENCE.get(source_mode, 0.15)
    overlay_config = {**base_overlay, "confidence": mode_confidence, "active_mode": source_mode}

    with source_lock:
        if req.index in running_sources:
            print(f"[START] Source {req.index} ({name}) already active, skipping restart but updating config.")
            # Update the shared overlay dict so the running process picks up changes
            with overlay_lock:
                overlays[name] = dict(overlay_config)
            
            if PERSIST_ACTIVE_SOURCES:
                # Ensure it is in active_sources config
                active = cfg.get("active_sources") or []
                if req.index not in active:
                    active.append(req.index)
                cfg["active_sources"] = active
                save_rtsp_config(cfg)
                
            return {"status": "started", "index": req.index, "mode": source_mode, "info": "already_running"}

        # Calculate total active streams (including this new one) for dynamic GPU allocation
        with upload_sources_lock:
            upload_count = len(upload_sources)
        active_streams = len(running_sources) + upload_count + 1

        if len(running_sources) >= MAX_RTSP_STREAMS:
            raise HTTPException(429, f"Maximum RTSP streams reached ({MAX_RTSP_STREAMS}).")

        process, stop = start_source_callback(req.index, url, name, overlay_config, active_streams)
        if process is None:
            raise HTTPException(500, "Failed to start source")
        running_sources[req.index] = {"process": process, "stop": stop, "start_time": time.time(), "name": name}

    with overlay_lock:
        overlays[name] = dict(overlay_config)
    print(f"[OVERLAY] {name} set from mode config: {overlay_config}")

    _ffmpeg_monitor_sources.add(name)
    ffmpeg_next_start[name] = time.time() + 1.0
    _start_ffmpeg_publisher_async(name)
    _ensure_ffmpeg_monitor()

    if PERSIST_ACTIVE_SOURCES:
        active = cfg.get("active_sources") or []
        if req.index not in active:
            active.append(req.index)
        cfg["active_sources"] = active
        save_rtsp_config(cfg)
    return {"status": "started", "index": req.index, "mode": source_mode}

@app.post("/api/sources/stop")
def stop_source(req: SourceIndexRequest):
    cfg = load_rtsp_config()
    name = None
    if 1 <= req.index <= len(cfg.get("rtsp_links", [])):
        name = source_display_name(cfg["rtsp_links"][req.index - 1])

    with source_lock:
        src = running_sources.pop(req.index, None)
        if src:
            src["stop"].set()
            # Aggressively terminate the process
            proc = src.get("process")
            if proc:
                try:
                    proc.terminate()
                    # Wait briefly for graceful exit, then force kill if needed
                    import time
                    t_start = time.time()
                    while proc.is_alive() and time.time() - t_start < 1.0:
                        time.sleep(0.05)
                    if proc.is_alive():
                        proc.kill()
                except:
                    pass

    if name:
        _ffmpeg_monitor_sources.discard(name)
        _stop_ffmpeg_publisher(name)
        # Clear stale UI state
        with metrics_lock:
            metrics.pop(name, None)
        with overlay_lock:
            overlays.pop(name, None)
        with frame_lock:
            frame_buffer.pop(name, None)
            frame_bgr_buffer.pop(name, None)
            frame_sequences.pop(name, 0)
        with raw_frame_lock:
            raw_frame_buffer.pop(name, None)
            raw_frame_bgr_buffer.pop(name, None)
            raw_frame_sequences.pop(name, None)

    if PERSIST_ACTIVE_SOURCES:
        cfg["active_sources"] = [i for i in cfg.get("active_sources", []) if i != req.index]
        save_rtsp_config(cfg)
    
    # Flush GPU memory if all sources stopped
    if not running_sources:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    return {"status": "stopped", "index": req.index}

@app.post("/api/sources/stop_all")
def stop_all_sources():
    global active_mode
    count = _stop_all_sources()
    active_mode = None
    return {"status": "stopped", "count": count}


# ── Metrics / Alerts endpoints ──

@app.get("/api/metrics")
def get_metrics():
    return get_all_metrics()

@app.get("/api/processed/{name}/ready")
def processed_ready(name: str):
    proc = ffmpeg_publishers.get(name)
    running = proc is not None and proc.poll() is None
    has_frames = ffmpeg_frame_sequences.get(name, 0) > 0
    return {"name": name, "ready": bool(running and has_frames)}

@app.get("/api/alerts")
def api_get_alerts(limit: int = 20):
    return {"alerts": get_alerts(limit)}

@app.get("/api/alerts/{alert_id}/screenshot")
def api_get_alert_screenshot(alert_id: str):
    screenshot_path = ALERTS_DIR / f"{alert_id}.jpg"
    if not screenshot_path.exists():
        raise HTTPException(404, "Screenshot not found")
    return FileResponse(screenshot_path, media_type="image/jpeg")

@app.delete("/api/alerts")
def api_clear_alerts():
    clear_alerts()
    return {"status": "cleared"}


# ── Report generation endpoint ──

@app.post("/api/report/{name}")
def api_generate_report(name: str):
    """Generate a Markdown incident report for a given source.

    Captures current frame screenshot, metrics, and alerts.
    Returns a ZIP file containing the report and screenshots.
    """
    import zipfile
    import io

    print(f"[REPORT] Generating report for {name}")
    
    # Grab current frame screenshot
    with frame_lock:
        screenshot = frame_buffer.get(name)
        if screenshot and not isinstance(screenshot, (bytes, bytearray)):
            print(f"[REPORT] WARNING: Screenshot for {name} is {type(screenshot)}, ignoring")
            screenshot = None
        
        if screenshot:
            print(f"[REPORT] Found screenshot for {name} ({len(screenshot)} bytes)")
        else:
            print(f"[REPORT] WARNING: No screenshot found for {name}")

    # Grab current metrics for this source
    with metrics_lock:
        source_metrics = metrics.get(name) or {}
        # Inject start_time from running_sources or upload_sources
        source_start = None
        
        # Check RTSP sources
        with source_lock:
            for info in running_sources.values():
                if info.get("name") == name:
                    source_start = info.get("start_time")
                    break
        
        # Check Uploads if not found
        if not source_start:
             with upload_sources_lock:
                info = upload_sources.get(name)
                if info:
                    source_start = info.get("start_time")
        
        if source_start:
            source_metrics["start_time"] = source_start

        if source_metrics:
            print(f"[REPORT] Found metrics for {name}")
        else:
            print(f"[REPORT] WARNING: No metrics found for {name}")

    # Resolve per-source mode (overrides global active_mode)
    report_mode = _resolve_source_mode(name, source_metrics)

    # Forensics special: Inject data from sam_results
    if report_mode == "forensics":
        try:
            from sam import sam_results, sam_results_lock
            with sam_results_lock:
                sam_info = sam_results.get(name)
                if sam_info:
                    source_metrics["prompt"] = sam_info.get("prompt")
                    source_metrics["detection_count"] = sam_info.get("count", 0)
                    source_metrics["session_history"] = sam_info.get("session_history", [])
        except Exception as e:
            print(f"[REPORT] Failed to inject SAM metrics: {e}")

    # Grab alerts filtered for this source
    with alerts_lock:
        source_alerts = [a for a in alerts if a.get("source") == name]
        print(f"[REPORT] Found {len(source_alerts)} alerts for {name}")

    # Forensics special: Trigger VLM Analysis
    vlm_narrative = None
    if report_mode == "forensics":
        # Check if SAM has actually analyzed anything
        with sam_results_lock:
            sam_info = sam_results.get(name)
            if not sam_info:
                raise HTTPException(400, "Forensic data matching this feed not found. Did you start a forensic search?")
            
            history = sam_info.get("session_history", [])
            if not history:
                raise HTTPException(400, "Forensic analysis cycle in progress. Please wait for at least one detection or analysis frame to be recorded.")

        try:
            from sam import generate_vlm_analysis
            vlm_narrative = generate_vlm_analysis(name)
        except Exception as e:
            print(f"[REPORT] VLM Analysis failed: {e}")

    try:
        pdf_bytes = generate_report(
            source_name=name,
            screenshot_bytes=screenshot,
            metrics_data=source_metrics,
            alerts_list=source_alerts,
            active_mode=report_mode,
            vlm_narrative=vlm_narrative,
        )
        # print(f"[REPORT] PDF generated successfully ({len(pdf_bytes)} bytes)")
    except Exception as e:
        print(f"[REPORT] ERROR generating PDF: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Report generation failed: {e}")

    # Return PDF directly
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        print(f"[REPORT] CRITICAL: generate_report returned {type(pdf_bytes)}, forcing cleanup")
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode('latin-1')
            
    pdf_buffer = io.BytesIO(pdf_bytes)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_filename = f"report_{name}_{timestamp}.pdf"

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{pdf_filename}"'},
    )


# ── Stream endpoint ──

@app.get("/api/stream/{name}")
def stream_video(name: str):
    def generate():
        last_processed_seq = -1
        last_raw_seq = -1
        while True:
            frame = None

            with frame_lock:
                processed_seq = frame_sequences.get(name, 0)
                if processed_seq > last_processed_seq:
                    frame = frame_buffer.get(name)
                    last_processed_seq = processed_seq

            if frame is None:
                with raw_frame_lock:
                    raw_seq = raw_frame_sequences.get(name, 0)
                    if raw_seq > last_raw_seq:
                        frame = raw_frame_buffer.get(name)
                        last_raw_seq = raw_seq

            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # No fresh frame on either path yet.
                time.sleep(0.01)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/api/raw/{name}")
def stream_raw_video(name: str):
    def generate():
        last_seq = -1
        while True:
            with raw_frame_lock:
                while raw_frame_sequences.get(name, 0) <= last_seq:
                    if not raw_frame_lock.wait(timeout=0.2):
                        break

                frame = raw_frame_buffer.get(name)
                last_seq = raw_frame_sequences.get(name, 0)

            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.001)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


# ── Jobs endpoints ──

@app.get("/api/jobs")
def list_jobs():
    with jobs_lock:
        return {"jobs": list(jobs.values())}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        if job_id in jobs:
            return jobs[job_id]
    raise HTTPException(404, "Job not found")


# ── Upload endpoints ──

upload_sources_lock = threading.Lock()
upload_sources: Dict[str, dict] = {}
upload_name_counter = 0

def _next_upload_name() -> str:
    global upload_name_counter
    with upload_sources_lock:
        upload_name_counter += 1
        return f"upload{upload_name_counter}"

def _stop_all_uploads(delete_files: bool = False):
    """Stop all upload processes. Optionally delete files and entries."""
    with upload_sources_lock:
        names = list(upload_sources.keys())

    for name in names:
        _ffmpeg_monitor_sources.discard(name)
        _stop_ffmpeg_publisher(name)
        _stop_raw_ffmpeg_publisher(name)

    with upload_sources_lock:
        for name, info in list(upload_sources.items()):
            info["stop"].set()
            proc = info.get("process")
            if proc:
                try:
                    proc.terminate()
                    proc.join(timeout=2)
                except:
                    pass
            info["process"] = None
            info["stopped"] = True
            if delete_files:
                try:
                    Path(info["file_path"]).unlink()
                except:
                    pass
                upload_sources.pop(name, None)

    # Clear per-upload buffers/metrics/overlays (runtime state only)
    with metrics_lock:
        for name in names:
            metrics.pop(name, None)
    with overlay_lock:
        for name in names:
            overlays.pop(name, None)
    with frame_lock:
        for name in names:
            frame_buffer.pop(name, None)
            frame_bgr_buffer.pop(name, None)
            frame_sequences.pop(name, 0)
    with raw_frame_lock:
        for name in names:
            raw_frame_buffer.pop(name, None)
            raw_frame_bgr_buffer.pop(name, None)
            raw_frame_sequences.pop(name, None)

def _stop_upload_by_name(name: str, delete_file: bool = False):
    _ffmpeg_monitor_sources.discard(name)
    _stop_ffmpeg_publisher(name)
    _stop_raw_ffmpeg_publisher(name)
    with upload_sources_lock:
        info = upload_sources.get(name)
        if not info:
            return
        info["stop"].set()
        proc = info.get("process")
        if proc:
            try:
                proc.terminate()
                proc.join(timeout=2)
            except Exception:
                pass
        if delete_file:
            try:
                Path(info["file_path"]).unlink()
            except Exception:
                pass
        upload_sources.pop(name, None)

    with metrics_lock:
        metrics.pop(name, None)
    with overlay_lock:
        overlays.pop(name, None)
    with frame_lock:
        frame_buffer.pop(name, None)
        frame_bgr_buffer.pop(name, None)
        frame_sequences.pop(name, 0)
    with raw_frame_lock:
        raw_frame_buffer.pop(name, None)
        raw_frame_bgr_buffer.pop(name, None)
        raw_frame_sequences.pop(name, None)

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...), mode: Optional[str] = Form(None)):
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    allowed_exts = {".mp4", ".mkv", ".avi", ".mov"}
    if ext not in allowed_exts:
        # Fallback on MIME for cases with odd/missing extension.
        content_type = (file.content_type or "").lower()
        allowed_mimes = {"video/mp4", "video/x-matroska", "video/quicktime", "video/x-msvideo"}
        if content_type not in allowed_mimes and not content_type.startswith("video/"):
            raise HTTPException(400, "Unsupported file format. Use MP4, MKV, AVI, or MOV.")

    global active_mode
    upload_mode = mode or active_mode
    if mode and active_mode and mode != active_mode:
        print(f"[MODE] Running mixed modes in parallel: global={active_mode}, upload={mode}")
    elif mode and not active_mode:
        active_mode = mode

    original_name = Path(file.filename).stem
    job_id = uuid.uuid4().hex
    name = _next_upload_name()
    ensure_overlay(name)

    # Ensure directory exists in case it was deleted at runtime
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # If a previous upload with the same original name exists, stop and remove it
    with upload_sources_lock:
        existing = [n for n, info in upload_sources.items() if info.get("original_name") == original_name]
    for n in existing:
        _stop_upload_by_name(n, delete_file=False)

    target = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Probe duration
    duration = get_video_duration(str(target))

    base_overlay = MODE_OVERLAYS.get(
        upload_mode,
        {"heatmap": False, "heatmap_full": False, "heatmap_trails": False, "trails": False, "bboxes": False},
    )
    mode_confidence = MODE_CONFIDENCE.get(upload_mode, 0.15)
    overlay_config = {**base_overlay, "confidence": mode_confidence, "active_mode": upload_mode}

    with overlay_lock:
        overlays[name] = dict(overlay_config)

    is_crowd_mode = upload_mode == "crowd"
    print(f"[UPLOAD] {name} overlays from mode={upload_mode}, is_crowd={is_crowd_mode}: {overlay_config}")

    # Calculate total active streams for dynamic GPU allocation
    with upload_sources_lock:
        upload_count = len(upload_sources)
    active_streams = len(running_sources) + upload_count + 1

    if upload_count >= MAX_UPLOAD_STREAMS:
        raise HTTPException(429, f"Maximum upload streams reached ({MAX_UPLOAD_STREAMS}).")

    # realtime=True so uploads play in real time with live overlays
    process, stop = start_upload_callback(str(target), name, overlay_config, is_crowd_mode, active_streams, realtime=True)
    if process is None:
        raise HTTPException(500, "Failed to start upload inference")
    if UPLOAD_RAW_PASSTHROUGH:
        _start_upload_raw_passthrough_async(name, str(target))
    else:
        _start_raw_ffmpeg_publisher_async(name)

    with upload_sources_lock:
        upload_sources[name] = {
            "process": process,
            "stop": stop,
            "file_path": str(target),
            "job_id": job_id,
            "original_name": original_name,
            "duration": duration,
            "started_processing": False,
            "started_at": None,
            "mode": upload_mode,
        }

    with jobs_lock:
        jobs[job_id] = {"id": job_id, "name": name, "status": "processing"}

    _ffmpeg_monitor_sources.add(name)
    ffmpeg_next_start[name] = time.time() + 1.0
    _start_ffmpeg_publisher_async(name)
    _ensure_ffmpeg_monitor()

    return {
        "job_id": job_id,
        "name": name,
        "status": "started",
    }

@app.post("/api/uploads/{name}/restart")
def restart_upload(name: str):
    """Stop and restart an existing upload."""
    print(f"[UPLOAD] Restarting {name}")
    
    # 1. Stop if running
    with upload_sources_lock:
        if name in upload_sources:
            info = upload_sources[name]
            info["stop"].set()
            if "process" in info:
                try:
                    info["process"].join(timeout=2)
                except:
                    pass
            # Don't pop yet, we need the file path
            file_path = info["file_path"]
            original_name = info.get("original_name")
            job_id = info.get("job_id")
            info["stopped"] = False
        else:
            raise HTTPException(404, "Upload not active or found")

    # 2. Start again
    # Preserve the original mode of this upload if available
    upload_mode = None
    with upload_sources_lock:
        upload_mode = upload_sources.get(name, {}).get("mode")

    base_overlay = MODE_OVERLAYS.get(
        upload_mode or active_mode,
        {"heatmap": False, "heatmap_full": False, "heatmap_trails": False, "trails": False, "bboxes": False},
    )
    mode_confidence = MODE_CONFIDENCE.get(upload_mode or active_mode, 0.15)
    overlay_config = {**base_overlay, "confidence": mode_confidence, "active_mode": (upload_mode or active_mode)}
    
    is_crowd_mode = (upload_mode or active_mode) == "crowd"
    
    # Update overlays dict
    with overlay_lock:
        overlays[name] = dict(overlay_config)

    # Recalculate streams count
    with source_lock:
        rtsp_count = len(running_sources)
    with upload_sources_lock:
        # It's still in the dict, so count is same
        upload_count = len(upload_sources)
    
    active_streams = rtsp_count + upload_count

    # Cleanup old process info in ffmpeg monitor just in case
    _ffmpeg_monitor_sources.discard(name)
    _stop_ffmpeg_publisher(name)
    _stop_raw_ffmpeg_publisher(name)

    # Start new process
    process, stop = start_upload_callback(file_path, name, overlay_config, is_crowd_mode, active_streams, realtime=True)
    if process is None:
        raise HTTPException(500, "Failed to restart upload inference")
    if UPLOAD_RAW_PASSTHROUGH:
        _start_upload_raw_passthrough_async(name, file_path)
    else:
        _start_raw_ffmpeg_publisher_async(name)

    with upload_sources_lock:
        upload_sources[name] = {
            "process": process,
            "stop": stop,
            "file_path": file_path,
            "job_id": job_id,
            "original_name": original_name,
            "started_at": None,
            "mode": (upload_mode or active_mode),
        }
    
    # Reset job status
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = "processing"

    # Restart monitor
    _ffmpeg_monitor_sources.add(name)
    ffmpeg_next_start[name] = time.time() + 1.0
    _start_ffmpeg_publisher_async(name)
    _ensure_ffmpeg_monitor()

    return {"status": "restarted", "name": name}

@app.get("/api/uploads")
def list_uploads():
    with upload_sources_lock:
        return {
            "uploads": [
                {
                    "name": name,
                    "original_name": info.get("original_name", name),
                    "job_id": info.get("job_id"),
                    "stream_url": f"/api/stream/{name}",
                    "stopped": info.get("stopped", False),
                }
                for name, info in upload_sources.items()
            ]
        }

@app.post("/api/uploads/{name}/stop")
def stop_upload(name: str):
    _ffmpeg_monitor_sources.discard(name)
    _stop_ffmpeg_publisher(name)
    _stop_raw_ffmpeg_publisher(name)
    with upload_sources_lock:
        info = upload_sources.get(name)
        if info:
            info["stop"].set()
            proc = info.get("process")
            if proc:
                try:
                    proc.terminate()
                    proc.join(timeout=2)
                except:
                    pass
            info["process"] = None
            info["stopped"] = True
            return {"status": "stopped", "name": name}
    raise HTTPException(404, "Upload not found")

@app.delete("/api/uploads/{name}")
def delete_upload(name: str):
    _ffmpeg_monitor_sources.discard(name)
    _stop_ffmpeg_publisher(name)
    _stop_raw_ffmpeg_publisher(name)
    with upload_sources_lock:
        info = upload_sources.pop(name, None)
        if info:
            info["stop"].set()
            proc = info.get("process")
            if proc:
                try:
                    proc.terminate()
                    proc.join(timeout=2)
                except:
                    pass
            try:
                Path(info["file_path"]).unlink()
            except:
                pass
            return {"status": "deleted", "name": name}
    raise HTTPException(404, "Upload not found")


# ── SAM3 endpoints ──

@app.post("/api/sam/start")
def sam_start(req: SamStartRequest):
    if not load_sam_model():
        raise HTTPException(503, "SAM3 model failed to load")

    existing = sam_threads.pop(req.source, None)
    if existing:
        existing["stop_event"].set()
        # Non-blocking or short-timeout join to avoid hanging API
        # existing["thread"].join(timeout=1) 

    with sam_results_lock:
        sam_results[req.source] = {"session_history": [], "vlm_analysis": None}

    settings_ref = {
        "confidence": req.confidence,
        "show_boxes": req.show_boxes,
        "show_masks": req.show_masks,
    }

    stop_event = threading.Event()
    t = threading.Thread(
        target=sam_worker,
        args=(req.source, req.prompt, req.confidence, stop_event,
              req.show_boxes, req.show_masks, settings_ref),
        kwargs={"raw_frame_lock": raw_frame_lock, "raw_frame_buffer": raw_frame_buffer},
        daemon=True,
    )
    t.start()
    sam_threads[req.source] = {
        "thread": t,
        "stop_event": stop_event,
        "prompt": req.prompt,
        "confidence": req.confidence,
        "show_boxes": req.show_boxes,
        "show_masks": req.show_masks,
        "settings_ref": settings_ref,
    }
    return {"status": "started", "source": req.source, "prompt": req.prompt}

@app.post("/api/sam/update")
def sam_update(req: SamUpdateRequest):
    info = sam_threads.get(req.source)
    if not info:
        raise HTTPException(404, "SAM not running for this source")

    settings_ref = info.get("settings_ref")
    if not settings_ref:
        raise HTTPException(500, "Settings ref not available")

    if req.confidence is not None:
        settings_ref["confidence"] = req.confidence
        info["confidence"] = req.confidence
    if req.show_boxes is not None:
        settings_ref["show_boxes"] = req.show_boxes
        info["show_boxes"] = req.show_boxes
    if req.show_masks is not None:
        settings_ref["show_masks"] = req.show_masks
        info["show_masks"] = req.show_masks

    return {
        "status": "updated",
        "source": req.source,
        "confidence": settings_ref["confidence"],
        "show_boxes": settings_ref["show_boxes"],
        "show_masks": settings_ref["show_masks"],
    }

@app.get("/api/sam/result/{source}")
def sam_result(source: str):
    with sam_results_lock:
        result = sam_results.get(source)
    if result is None:
        return {"status": "no_result", "source": source}
    return result

@app.post("/api/sam/stop")
def sam_stop(req: SamStopRequest):
    info = sam_threads.pop(req.source, None)
    if info:
        info["stop_event"].set()
        return {"status": "stopped", "source": req.source}
    return {"status": "not_running", "source": req.source}

@app.get("/api/sam/status")
def sam_status():
    from sam import sam_model_loaded as _loaded
    active = list(sam_threads.keys())
    return {
        "model_loaded": _loaded,
        "active_sources": active,
        "details": {
            src: {
                "prompt": info["prompt"],
                "confidence": info["confidence"],
                "show_boxes": info.get("show_boxes", True),
                "show_masks": info.get("show_masks", True),
            }
            for src, info in sam_threads.items()
        },
    }


# ── Server entry point ──

def run_control_server(start_source_fn, start_upload_fn, initial_overlays):
    global start_source_callback, start_upload_callback, overlays

    start_source_callback = start_source_fn
    start_upload_callback = start_upload_fn

    if initial_overlays is not None:
        overlays = initial_overlays
        print(f"[OVERLAY] Using shared Manager dict (type: {type(initial_overlays).__name__})")
    else:
        overlays = {}
        print("[OVERLAY] Using local dict (no shared dict provided)")

    cfg = load_rtsp_config()
    rtsp_links = cfg.get("rtsp_links", [])

    # Clear persisted runtime state so each run starts clean.
    if cfg.get("active_sources") or cfg.get("overlays"):
        cfg["active_sources"] = []
        cfg["overlays"] = {}
        save_rtsp_config(cfg)

    saved_overlays = {}
    for name, state in saved_overlays.items():
        with overlay_lock:
            heatmap_val = state.get("heatmap", True)
            overlays[name] = {
                "heatmap": heatmap_val,
                "heatmap_full": state.get("heatmap_full", heatmap_val),
                "heatmap_trails": state.get("heatmap_trails", heatmap_val),
                "trails": state.get("trails", True),
                "bboxes": state.get("bboxes", True),
                "confidence": state.get("confidence", 0.15),
            }
            print(f"[OVERLAY] Loaded saved state for {name}: {overlays[name]}")

    for url in rtsp_links:
        ensure_overlay(source_display_name(url))

    class _ShutdownNoiseFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.exc_info:
                exc_type = record.exc_info[0]
                if exc_type in (asyncio.CancelledError, KeyboardInterrupt):
                    return False
            return True

    logging.getLogger("uvicorn.error").addFilter(_ShutdownNoiseFilter())

    import uvicorn
    import copy
    import threading
    import time

    # --- Access Log Filtering ---
    class AccessLogFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.args and len(record.args) >= 5:
                status_code = record.args[4]
                path = record.args[2]
                
                if status_code == 200:
                    if "/api/health" in path or "/api/metrics" in path:
                        return False
                    if "/api/sam/result/" in path or "/api/sam/status" in path:
                         return False
                    if "/api/uploads" in path:
                         return False
            return True


    # --- Periodic Status Logger ---
    def _status_logger_worker():
        while True:
            time.sleep(15)
            try:
                # Count resources
                with source_lock:
                    n_rtsp = len(running_sources)
                with upload_sources_lock:
                    n_upload = sum(1 for u in upload_sources.values() if u.get("started_processing"))
                n_sam = len(sam_threads)
                
                print(f"[SYSTEM] Status: OK | RTSP Streams: {n_rtsp} | Uploads: {n_upload} | SAM Workers: {n_sam}")
            except Exception:
                pass

    threading.Thread(target=_status_logger_worker, daemon=True).start()

    # Define standard Uvicorn log config
    log_config = uvicorn.config.LOGGING_CONFIG.copy()
    
    # Add our filter to the configuration
    log_config["filters"] = log_config.get("filters", {})
    log_config["filters"]["access_filter"] = {
        "()": lambda: AccessLogFilter()
    }
    
    # Apply filter to access logger
    log_config["loggers"]["uvicorn.access"]["filters"] = ["access_filter"]

    # Run with explicit log_config
    uvicorn.run(app, host="0.0.0.0", port=9010, log_config=log_config)
