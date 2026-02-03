"""FastAPI control server: all API endpoints, overlay/frame/metrics management, FFmpeg publishing."""

import yaml
import uuid
import shutil
import threading
import time
import subprocess
import json
import logging
import asyncio
from typing import Dict, List, Optional
from pathlib import Path
from collections import deque
import multiprocessing as mp

import requests as http_requests
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from login import login_user, add_user, delete_user, change_password, list_users
from sam import (
    load_sam_model, sam_annotate_frame, sam_worker,
    sam_model_loaded, sam_results_lock, sam_results, sam_threads,
)
from log_utils import new_log_path

MEDIAMTX_API = "http://127.0.0.1:9997"


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

raw_frame_lock = threading.Lock()
raw_frame_buffer: Dict[str, bytes] = {}

jobs_lock = threading.Lock()
jobs: Dict[str, dict] = {}

source_lock = threading.Lock()
running_sources: Dict[int, dict] = {}

# FFmpeg RTSP publishers
ffmpeg_publishers: Dict[str, subprocess.Popen] = {}
ffmpeg_log_files: Dict[str, object] = {}
FFMPEG_PUBLISH_PORT = 9010
MEDIAMTX_RTSP_PORT = 8554

# Active mode tracking
active_mode: Optional[str] = None

# Fixed overlay config per mode
MODE_OVERLAYS = {
    "congestion": {"heatmap": True, "heatmap_full": True, "heatmap_trails": False, "trails": False, "bboxes": False},
    "vehicle":    {"heatmap": False, "heatmap_full": False, "heatmap_trails": False, "trails": False, "bboxes": True},
    "flow":       {"heatmap": False, "heatmap_full": False, "heatmap_trails": False, "trails": True,  "bboxes": True},
    "forensics":  {"heatmap": False, "heatmap_full": False, "heatmap_trails": False, "trails": False, "bboxes": False},
    "crowd":      {"heatmap": True, "heatmap_full": False, "heatmap_trails": False, "trails": False, "bboxes": False},
}

MODE_CONFIDENCE = {
    "congestion": 0.15,
    "vehicle":    0.20,
    "flow":       0.15,
    "forensics":  0.15,
    "crowd":      0.25,
}

confidence_lock = threading.Lock()
confidence_settings: Dict[str, float] = {}

start_source_callback = None
start_upload_callback = None

frontend_log_lock = threading.Lock()
frontend_log_file = None


# ── RTSP config helpers ──

def load_rtsp_config():
    if not RTSP_CONFIG_PATH.exists():
        return {"rtsp_links": [], "active_sources": [], "overlays": {}}
    with open(RTSP_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}

def save_rtsp_config(cfg):
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
            frontend_log_file = open(new_log_path("frontend"), "a", buffering=1)
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

def _start_ffmpeg_publisher(name: str, retry_count: int = 3):
    _stop_ffmpeg_publisher(name)
    mjpeg_url = f"http://127.0.0.1:{FFMPEG_PUBLISH_PORT}/api/stream/{name}"
    rtsp_url = f"rtsp://127.0.0.1:{MEDIAMTX_RTSP_PORT}/processed_{name}"

    for attempt in range(retry_count):
        try:
            resp = http_requests.get(mjpeg_url, stream=True, timeout=2)
            if resp.status_code == 200:
                resp.close()
                break
        except Exception:
            pass
        print(f"[FFMPEG] Waiting for {name} stream to be available... (attempt {attempt + 1}/{retry_count})")
        time.sleep(2)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "warning",
        "-f", "mjpeg",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "10",
        "-reconnect_at_eof", "1",
        "-i", mjpeg_url,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-g", "30",
        "-bf", "0",
        "-pix_fmt", "yuv420p",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]
    try:
        log_file = open(new_log_path(f"ffmpeg-{name}"), "a", buffering=1)
        ffmpeg_log_files[name] = log_file
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
        ffmpeg_publishers[name] = proc
        print(f"[FFMPEG] Started publisher for {name} -> processed_{name} (pid={proc.pid})")
    except Exception as e:
        print(f"[FFMPEG] Failed to start publisher for {name}: {e}")

def _stop_ffmpeg_publisher(name: str):
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
    log_file = ffmpeg_log_files.pop(name, None)
    if log_file:
        try:
            log_file.close()
        except Exception:
            pass
        print(f"[FFMPEG] Stopped publisher for {name}")

_ffmpeg_monitor_sources: set = set()
_ffmpeg_monitor_thread: threading.Thread = None
_ffmpeg_monitor_stop = threading.Event()

def _ffmpeg_monitor_worker():
    while not _ffmpeg_monitor_stop.is_set():
        try:
            for name in list(_ffmpeg_monitor_sources):
                proc = ffmpeg_publishers.get(name)
                if proc is None:
                    continue

                poll = proc.poll()
                if poll is not None:
                    with source_lock:
                        source_active = any(
                            source_display_name(load_rtsp_config().get("rtsp_links", [])[idx - 1]) == name
                            for idx in running_sources.keys()
                            if idx <= len(load_rtsp_config().get("rtsp_links", []))
                        )

                    with upload_sources_lock:
                        if name in upload_sources:
                            source_active = True

                    if source_active:
                        print(f"[FFMPEG] Publisher for {name} exited (code={poll}), restarting...")
                        _start_ffmpeg_publisher(name)
                    else:
                        _ffmpeg_monitor_sources.discard(name)
        except Exception as e:
            print(f"[FFMPEG Monitor] Error: {e}")

        _ffmpeg_monitor_stop.wait(5)

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


# ── Overlay / metrics / frame management ──

def ensure_overlay(name: str):
    with overlay_lock:
        try:
            if name in overlays:
                existing = overlays[name]
                has_keys = False
                try:
                    has_keys = "trails" in existing and "heatmap" in existing and "bboxes" in existing
                except:
                    pass
                if has_keys:
                    if "heatmap_full" not in existing:
                        existing["heatmap_full"] = existing.get("heatmap", True)
                    if "heatmap_trails" not in existing:
                        existing["heatmap_trails"] = existing.get("heatmap", True)
                    if "confidence" not in existing:
                        existing["confidence"] = 0.15
                        overlays[name] = existing
                    return
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
        except Exception as e:
            overlays[name] = {
                "heatmap": True,
                "heatmap_full": True,
                "heatmap_trails": True,
                "trails": True,
                "bboxes": True,
                "confidence": 0.15,
            }
            print(f"[OVERLAY] Initialized {name} (after error: {e})")

def get_overlay_state(stream: str) -> Dict[str, bool]:
    ensure_overlay(stream)
    with overlay_lock:
        try:
            return dict(overlays[stream])
        except:
            return {"heatmap": True, "heatmap_full": True, "heatmap_trails": True, "trails": True, "bboxes": True}

def update_metrics(stream: str, data: dict):
    with metrics_lock:
        metrics[stream] = data

def update_frame(stream: str, jpeg_data: bytes):
    with frame_lock:
        frame_buffer[stream] = jpeg_data
        frame_sequences[stream] = frame_sequences.get(stream, 0) + 1
        frame_lock.notify_all()

def update_raw_frame(stream: str, jpeg_data: bytes):
    with raw_frame_lock:
        raw_frame_buffer[stream] = jpeg_data

def get_all_metrics():
    with metrics_lock:
        return metrics.copy()


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

    with source_lock:
        for idx, src in list(running_sources.items()):
            src["stop"].set()
            stopped += 1
        running_sources.clear()

    for src_name, info in list(sam_threads.items()):
        info["stop_event"].set()
    sam_threads.clear()

    with sam_results_lock:
        sam_results.clear()

    cfg = load_rtsp_config()
    cfg["active_sources"] = []
    save_rtsp_config(cfg)

    _ffmpeg_monitor_sources.clear()
    _stop_all_ffmpeg_publishers()

    print(f"[STOP_ALL] Stopped {stopped} sources, cleared SAM workers, stopped ffmpeg publishers")
    return stopped


# ── FastAPI app ──

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "https://iriscmdapi.stagingbot.xyz", "https://mediamtx1.stagingbot.xyz", "https://iriscommand.stagingbot.xyz"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth endpoints ──

@app.post("/api/login")
def api_login(req: LoginRequest):
    return login_user(req.username, req.password)

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
        })

    return {"sources": out, "active_sources": sorted(active)}

@app.post("/api/sources/start")
def start_source(req: SourceStartRequest):
    global active_mode

    cfg = load_rtsp_config()
    if req.index < 1 or req.index > len(cfg["rtsp_links"]):
        raise HTTPException(400)

    if req.mode and req.mode != active_mode:
        print(f"[MODE] Switching from {active_mode} -> {req.mode}, stopping all sources")
        _stop_all_sources()
        active_mode = req.mode

    with source_lock:
        if req.index in running_sources:
            print(f"[START] Source {req.index} already active, restarting...")
            src = running_sources.pop(req.index)
            src["stop"].set()
            time.sleep(1.0)

            if 1 <= req.index <= len(cfg["rtsp_links"]):
                prev_name = source_display_name(cfg["rtsp_links"][req.index - 1])
                _stop_ffmpeg_publisher(prev_name)

        url = cfg["rtsp_links"][req.index - 1]
        name = source_display_name(url)

        base_overlay = MODE_OVERLAYS.get(
            active_mode,
            {"heatmap": False, "heatmap_full": False, "heatmap_trails": False, "trails": False, "bboxes": False},
        )
        mode_confidence = MODE_CONFIDENCE.get(active_mode, 0.15)
        overlay_config = {**base_overlay, "confidence": mode_confidence}

        process, stop = start_source_callback(req.index, url, name, overlay_config)
        running_sources[req.index] = {"process": process, "stop": stop}

    with overlay_lock:
        overlays[name] = dict(overlay_config)
    print(f"[OVERLAY] {name} set from mode config: {overlay_config}")

    def _delayed_ffmpeg_start(stream_name):
        time.sleep(3)
        _start_ffmpeg_publisher(stream_name)
        _ffmpeg_monitor_sources.add(stream_name)
        _ensure_ffmpeg_monitor()
    threading.Thread(target=_delayed_ffmpeg_start, args=(name,), daemon=True).start()

    active = cfg.get("active_sources", [])
    if req.index not in active:
        active.append(req.index)
    cfg["active_sources"] = active
    save_rtsp_config(cfg)
    return {"status": "started", "index": req.index, "mode": active_mode}

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

    if name:
        _ffmpeg_monitor_sources.discard(name)
        _stop_ffmpeg_publisher(name)

    cfg["active_sources"] = [i for i in cfg.get("active_sources", []) if i != req.index]
    save_rtsp_config(cfg)
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


# ── Stream endpoint ──

@app.get("/api/stream/{name}")
def stream_video(name: str):
    def generate():
        last_seq = -1
        while True:
            with frame_lock:
                while frame_sequences.get(name, 0) <= last_seq:
                    if not frame_lock.wait(timeout=0.3):
                        break

                frame = frame_buffer.get(name)
                last_seq = frame_sequences.get(name, 0)

            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.033)

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

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".mkv", ".avi", ".mov")):
        raise HTTPException(400, "Unsupported file format. Use MP4, MKV, AVI, or MOV.")

    job_id = uuid.uuid4().hex
    original_name = Path(file.filename).stem
    name = f"upload_{original_name}_{job_id[:8]}"
    ensure_overlay(name)

    target = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)

    base_overlay = MODE_OVERLAYS.get(
        active_mode,
        {"heatmap": False, "heatmap_full": False, "heatmap_trails": False, "trails": False, "bboxes": False},
    )
    mode_confidence = MODE_CONFIDENCE.get(active_mode, 0.15)
    overlay_config = {**base_overlay, "confidence": mode_confidence}

    with overlay_lock:
        overlays[name] = dict(overlay_config)

    is_crowd_mode = active_mode == "crowd"
    print(f"[UPLOAD] {name} overlays from active_mode={active_mode}, is_crowd={is_crowd_mode}: {overlay_config}")

    process, stop = start_upload_callback(str(target), name, overlay_config, is_crowd_mode)

    with upload_sources_lock:
        upload_sources[name] = {
            "process": process,
            "stop": stop,
            "file_path": str(target),
            "job_id": job_id,
            "original_name": original_name,
        }

    with jobs_lock:
        jobs[job_id] = {"id": job_id, "name": name, "status": "processing"}

    def _delayed_ffmpeg_start(stream_name):
        time.sleep(3)
        _start_ffmpeg_publisher(stream_name)
        _ffmpeg_monitor_sources.add(stream_name)
        _ensure_ffmpeg_monitor()
    threading.Thread(target=_delayed_ffmpeg_start, args=(name,), daemon=True).start()

    return {
        "job_id": job_id,
        "name": name,
        "stream_url": f"/api/stream/{name}",
        "status": "started",
    }

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
                }
                for name, info in upload_sources.items()
            ]
        }

@app.delete("/api/uploads/{name}")
def stop_upload(name: str):
    _ffmpeg_monitor_sources.discard(name)
    _stop_ffmpeg_publisher(name)
    with upload_sources_lock:
        info = upload_sources.pop(name, None)
        if info:
            info["stop"].set()
            try:
                Path(info["file_path"]).unlink()
            except:
                pass
            return {"status": "stopped", "name": name}
    raise HTTPException(404, "Upload not found")


# ── SAM3 endpoints ──

@app.post("/api/sam/start")
def sam_start(req: SamStartRequest):
    if not load_sam_model():
        raise HTTPException(503, "SAM3 model failed to load")

    existing = sam_threads.get(req.source)
    if existing:
        existing["stop_event"].set()
        existing["thread"].join(timeout=5)

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

    saved_overlays = cfg.get("overlays", {})
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
    uvicorn.run(app, host="0.0.0.0", port=9010)
