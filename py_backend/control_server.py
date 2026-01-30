import yaml
import uuid
import shutil
import threading
from typing import Dict, List, Optional
from pathlib import Path
import multiprocessing as mp

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

class LoginRequest(BaseModel):
    username: str
    password: str

RTSP_CONFIG_PATH = Path("data/rtsp_links.yml")
UPLOAD_DIR = RTSP_CONFIG_PATH.parent / "uploads" / "recordings"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
        if name not in overlays:
            overlays[name] = {"heatmap": True, "trails": True, "bboxes": True}
            print(f"[+] Initialized overlay state for {name}: all ON")

def get_overlay_state(stream: str) -> Dict[str, bool]:
    ensure_overlay(stream)
    with overlay_lock:
        return overlays[stream].copy()  # Return copy to avoid thread issues

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
    return {"success": req.username == ADMIN_USERNAME and req.password == ADMIN_PASSWORD}

@app.get("/api/overlays")
def get_overlays():
    with overlay_lock:
        return overlays

@app.get("/api/overlays/{name}")
def get_overlay(name: str):
    ensure_overlay(name)
    with overlay_lock:
        return overlays[name]

@app.post("/api/overlays/{name}")
def set_overlay(name: str, update: OverlayUpdate):
    ensure_overlay(name)
    with overlay_lock:
        cur = overlays[name]
        overlays[name] = {
            "heatmap": update.heatmap if update.heatmap is not None else cur["heatmap"],
            "trails": update.trails if update.trails is not None else cur["trails"],
            "bboxes": update.bboxes if update.bboxes is not None else cur["bboxes"],
        }

    cfg = load_rtsp_config()
    cfg.setdefault("overlays", {})[name] = overlays[name]
    save_rtsp_config(cfg)
    return overlays[name]


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

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".mkv")):
        raise HTTPException(400)

    job_id = uuid.uuid4().hex
    name = Path(file.filename).stem
    ensure_overlay(name)

    target = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)

    stop = mp.Event()
    start_upload_callback(str(target), stop, name)

    with jobs_lock:
        jobs[job_id] = {"id": job_id, "name": name, "status": "processing"}

    return {"job_id": job_id}

def run_control_server(start_source_fn, start_upload_fn, initial_overlays):
    global start_source_callback, start_upload_callback, overlays

    start_source_callback = start_source_fn
    start_upload_callback = start_upload_fn
    overlays = initial_overlays or {}

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

    # Initialize overlays for all configured sources
    for url in rtsp_links:
        ensure_overlay(source_display_name(url))

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9010)
