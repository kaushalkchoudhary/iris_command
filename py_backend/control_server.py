import yaml
import uuid
import shutil
import threading
import time
from typing import Dict, List, Optional
from pathlib import Path
from collections import deque
import multiprocessing as mp

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Valid credentials (username -> password)
VALID_CREDENTIALS = {
    "admin": "admin123",
    "commandcentre": "command@2024",
    "command_admin": "iris_admin#2024",
}

class LoginRequest(BaseModel):
    username: str
    password: str

RTSP_CONFIG_PATH = Path("data/rtsp_links.yml")
UPLOAD_DIR = RTSP_CONFIG_PATH.parent / "uploads" / "recordings"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Alerts storage
ALERTS_DIR = RTSP_CONFIG_PATH.parent / "alerts"
ALERTS_DIR.mkdir(parents=True, exist_ok=True)

alerts_lock = threading.Lock()
alerts: deque = deque(maxlen=50)  # Keep last 50 alerts
alert_cooldowns: Dict[str, float] = {}  # Prevent spam: source -> last alert time
ALERT_COOLDOWN_SECONDS = 30  # Minimum time between alerts from same source

overlay_lock = threading.Lock()
overlays: Dict[str, Dict[str, bool]] = {}

metrics_lock = threading.Lock()
metrics: Dict[str, dict] = {}

frame_lock = threading.Condition()
frame_buffer: Dict[str, bytes] = {}
frame_sequences: Dict[str, int] = {}

jobs_lock = threading.Lock()
jobs: Dict[str, dict] = {}

source_lock = threading.Lock()
running_sources: Dict[int, dict] = {}

start_source_callback = None
start_upload_callback = None

def load_rtsp_config():
    if not RTSP_CONFIG_PATH.exists():
        return {"rtsp_links": [], "active_sources": [], "overlays": {}}
    with open(RTSP_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}

def save_rtsp_config(cfg):
    with open(RTSP_CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f)

def source_display_name(url: str) -> str:
    return url.rstrip("/").split("/")[-1]

def mask_rtsp(url: str) -> str:
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":")[0]
        return f"{scheme}://{user}:****@{host}"
    return url

def ensure_overlay(name: str):
    """Ensure overlay state exists for a stream. Default: all overlays ON."""
    with overlay_lock:
        try:
            # Check if key exists - use direct access for Manager dict compatibility
            if name in overlays:
                existing = overlays[name]
                # Handle DictProxy - check if it has the expected keys
                has_keys = False
                try:
                    has_keys = "trails" in existing and "heatmap" in existing and "bboxes" in existing
                except:
                    pass
                if has_keys:
                    return  # Already valid
            # Initialize with defaults - use a plain dict for Manager compatibility
            default_state = {"heatmap": True, "trails": True, "bboxes": True}
            overlays[name] = default_state
            print(f"[OVERLAY] Initialized {name}: {default_state}")
        except Exception as e:
            overlays[name] = {"heatmap": True, "trails": True, "bboxes": True}
            print(f"[OVERLAY] Initialized {name} (after error: {e})")

def get_overlay_state(stream: str) -> Dict[str, bool]:
    ensure_overlay(stream)
    with overlay_lock:
        try:
            return dict(overlays[stream])  # Convert to regular dict
        except:
            return {"heatmap": True, "trails": True, "bboxes": True}

def update_metrics(stream: str, data: dict):
    with metrics_lock:
        metrics[stream] = data

def update_frame(stream: str, jpeg_data: bytes):
    with frame_lock:
        frame_buffer[stream] = jpeg_data
        frame_sequences[stream] = frame_sequences.get(stream, 0) + 1
        frame_lock.notify_all()

def get_all_metrics():
    with metrics_lock:
        return metrics.copy()


def add_alert(source: str, congestion: int, metrics_data: dict, screenshot_data: bytes):
    """Add a new congestion alert with screenshot."""
    with alerts_lock:
        now = time.time()

        # Check cooldown to prevent alert spam
        last_alert = alert_cooldowns.get(source, 0)
        if now - last_alert < ALERT_COOLDOWN_SECONDS:
            return None

        alert_cooldowns[source] = now

        # Generate alert ID and save screenshot
        alert_id = f"{source}_{int(now * 1000)}"
        screenshot_path = ALERTS_DIR / f"{alert_id}.jpg"

        try:
            with open(screenshot_path, "wb") as f:
                f.write(screenshot_data)
        except Exception as e:
            print(f"[ALERT] Failed to save screenshot: {e}")
            return None

        # Determine severity based on congestion level
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


def get_alerts(limit: int = 20):
    """Get recent alerts."""
    with alerts_lock:
        return list(alerts)[:limit]


def clear_alerts():
    """Clear all alerts."""
    with alerts_lock:
        alerts.clear()
        alert_cooldowns.clear()
    # Optionally clean up old screenshots
    for f in ALERTS_DIR.glob("*.jpg"):
        try:
            f.unlink()
        except:
            pass


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class OverlayUpdate(BaseModel):
    heatmap: Optional[bool] = None
    trails: Optional[bool] = None
    bboxes: Optional[bool] = None

class SourceIndexRequest(BaseModel):
    index: int

class ActiveSourcesRequest(BaseModel):
    indexes: List[int]

@app.post("/api/login")
def api_login(req: LoginRequest):
    valid_password = VALID_CREDENTIALS.get(req.username)
    if valid_password and valid_password == req.password:
        return {"success": True, "username": req.username}
    return {"success": False, "error": "Invalid credentials"}

@app.get("/api/overlays")
def get_overlays():
    with overlay_lock:
        # Convert Manager dict to regular dict for JSON serialization
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
            return {"heatmap": True, "trails": True, "bboxes": True}

@app.post("/api/overlays/{name}")
def set_overlay(name: str, update: OverlayUpdate):
    ensure_overlay(name)
    with overlay_lock:
        try:
            cur = dict(overlays[name])
        except:
            cur = {"heatmap": True, "trails": True, "bboxes": True}

        new_state = {
            "heatmap": update.heatmap if update.heatmap is not None else cur.get("heatmap", True),
            "trails": update.trails if update.trails is not None else cur.get("trails", True),
            "bboxes": update.bboxes if update.bboxes is not None else cur.get("bboxes", True),
        }
        # Assign as a new dict to ensure Manager dict synchronization
        overlays[name] = dict(new_state)
        print(f"[OVERLAY] {name} updated: heatmap={new_state['heatmap']}, trails={new_state['trails']}, bboxes={new_state['bboxes']}")

    # Save to config for persistence
    cfg = load_rtsp_config()
    cfg.setdefault("overlays", {})[name] = new_state
    save_rtsp_config(cfg)

    # Return the actual state from the shared dict to confirm
    return new_state


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
            "active": idx in active
        })

    return {"sources": out, "active_sources": sorted(active)}

@app.post("/api/sources/start")
def start_source(req: SourceIndexRequest):
    cfg = load_rtsp_config()
    if req.index < 1 or req.index > len(cfg["rtsp_links"]):
        raise HTTPException(400)

    with source_lock:
        if req.index in running_sources:
            return {"status": "already_running"}

        url = cfg["rtsp_links"][req.index - 1]
        name = source_display_name(url)
        ensure_overlay(name)

        stop = mp.Event()
        process = start_source_callback(req.index, url, stop, name)
        running_sources[req.index] = {"process": process, "stop": stop}

    # Mark only this source as active in config
    cfg["active_sources"] = [req.index]
    save_rtsp_config(cfg)
    return {"status": "started", "index": req.index}

@app.post("/api/sources/stop")
def stop_source(req: SourceIndexRequest):
    with source_lock:
        src = running_sources.pop(req.index, None)
        if src:
            src["stop"].set()

    cfg = load_rtsp_config()
    cfg["active_sources"] = [i for i in cfg.get("active_sources", []) if i != req.index]
    save_rtsp_config(cfg)
    return {"status": "stopped", "index": req.index}

@app.get("/api/metrics")
def get_metrics():
    return get_all_metrics()


@app.get("/api/alerts")
def api_get_alerts(limit: int = 20):
    """Get recent congestion alerts."""
    return {"alerts": get_alerts(limit)}


@app.get("/api/alerts/{alert_id}/screenshot")
def api_get_alert_screenshot(alert_id: str):
    """Get screenshot for a specific alert."""
    screenshot_path = ALERTS_DIR / f"{alert_id}.jpg"
    if not screenshot_path.exists():
        raise HTTPException(404, "Screenshot not found")
    return FileResponse(screenshot_path, media_type="image/jpeg")


@app.delete("/api/alerts")
def api_clear_alerts():
    """Clear all alerts."""
    clear_alerts()
    return {"status": "cleared"}


@app.get("/api/stream/{name}")
def stream_video(name: str):
    def generate():
        last_seq = -1
        while True:
            with frame_lock:
                # Wait for a new frame if the current sequence matches what we last sent
                while frame_sequences.get(name, 0) <= last_seq:
                    # Timeout every 1s to allow checking for connection closure
                    if not frame_lock.wait(timeout=1.0):
                        break
                
                frame = frame_buffer.get(name)
                last_seq = frame_sequences.get(name, 0)
            
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # Fallback to prevent infinite loop on empty buffer
                import time
                time.sleep(0.1)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

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

# Track uploaded video sources
upload_sources_lock = threading.Lock()
upload_sources: Dict[str, dict] = {}  # name -> {process, stop, file_path, job_id}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".mkv", ".avi", ".mov")):
        raise HTTPException(400, "Unsupported file format. Use MP4, MKV, AVI, or MOV.")

    job_id = uuid.uuid4().hex
    original_name = Path(file.filename).stem
    # Create unique name to avoid conflicts
    name = f"upload_{original_name}_{job_id[:8]}"
    ensure_overlay(name)

    target = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)

    stop = mp.Event()
    process = start_upload_callback(str(target), stop, name)

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

    return {
        "job_id": job_id,
        "name": name,
        "stream_url": f"/api/stream/{name}",
        "status": "started"
    }


@app.get("/api/uploads")
def list_uploads():
    """List all active uploaded video streams."""
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
    """Stop an uploaded video stream."""
    with upload_sources_lock:
        info = upload_sources.pop(name, None)
        if info:
            info["stop"].set()
            # Optionally delete the file
            try:
                Path(info["file_path"]).unlink()
            except:
                pass
            return {"status": "stopped", "name": name}
    raise HTTPException(404, "Upload not found")

def run_control_server(start_source_fn, start_upload_fn, initial_overlays):
    global start_source_callback, start_upload_callback, overlays

    start_source_callback = start_source_fn
    start_upload_callback = start_upload_fn

    # Use the Manager dict if provided, otherwise create a local dict
    if initial_overlays is not None:
        overlays = initial_overlays
        print(f"[OVERLAY] Using shared Manager dict (type: {type(initial_overlays).__name__})")
    else:
        overlays = {}
        print("[OVERLAY] Using local dict (no shared dict provided)")

    cfg = load_rtsp_config()
    rtsp_links = cfg.get("rtsp_links", [])

    # Load saved overlay states from config
    saved_overlays = cfg.get("overlays", {})
    for name, state in saved_overlays.items():
        with overlay_lock:
            overlays[name] = {
                "heatmap": state.get("heatmap", True),
                "trails": state.get("trails", True),
                "bboxes": state.get("bboxes", True),
            }
            print(f"[OVERLAY] Loaded saved state for {name}: {overlays[name]}")

    # Initialize overlays for all configured sources
    for url in rtsp_links:
        ensure_overlay(source_display_name(url))

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9010)
