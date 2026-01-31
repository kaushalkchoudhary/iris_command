import yaml
import uuid
import shutil
import threading
import time
import sys
import base64
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

MEDIAMTX_API = "http://127.0.0.1:9997"

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
ALERT_COOLDOWN_SECONDS = 15  # Minimum time between alerts from same source

overlay_lock = threading.Lock()
overlays: Dict[str, Dict[str, bool]] = {}

metrics_lock = threading.Lock()
metrics: Dict[str, dict] = {}

frame_lock = threading.Condition()
frame_buffer: Dict[str, bytes] = {}
frame_sequences: Dict[str, int] = {}

raw_frame_lock = threading.Lock()
raw_frame_buffer: Dict[str, bytes] = {}  # Raw frames before YOLO overlays (for SAM)

jobs_lock = threading.Lock()
jobs: Dict[str, dict] = {}

source_lock = threading.Lock()
running_sources: Dict[int, dict] = {}

# Active mode tracking — when mode changes, all sources are stopped first
active_mode: Optional[str] = None

# Fixed overlay config per mode — hardcoded, not user-changeable
MODE_OVERLAYS = {
    "congestion": {"heatmap": True, "trails": False, "bboxes": False},
    "vehicle":    {"heatmap": False, "trails": False, "bboxes": True},
    "flow":       {"heatmap": False, "trails": True,  "bboxes": True},
    "forensics":  {"heatmap": False, "trails": False, "bboxes": False},
}

start_source_callback = None
start_upload_callback = None

# ── SAM3 globals ──
SAM_DIR = Path(__file__).resolve().parent.parent / "sam"
sam_lock = threading.Lock()
sam_model = None
sam_processor = None
sam_model_loaded = False

sam_results_lock = threading.Lock()
sam_results: Dict[str, dict] = {}  # source -> {annotated_frame, detections, count, timestamp}
sam_threads: Dict[str, dict] = {}  # source -> {thread, stop_event, prompt, confidence}


def _load_sam_model():
    """Lazy-load SAM3 model on CUDA."""
    global sam_model, sam_processor, sam_model_loaded
    if sam_model_loaded:
        return True

    try:
        import torch
        sam_str = str(SAM_DIR)
        if sam_str not in sys.path:
            sys.path.insert(0, sam_str)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint = SAM_DIR / "sam3.pt"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[SAM3] Loading model on CUDA from {checkpoint} ...")
        sam_model = build_sam3_image_model(checkpoint_path=str(checkpoint), device="cuda")
        sam_processor = Sam3Processor(sam_model)
        sam_model_loaded = True
        print("[SAM3] Model loaded successfully on CUDA")
        return True
    except Exception as e:
        print(f"[SAM3] Failed to load model: {e}")
        sam_model = None
        sam_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def _sam_annotate_frame(jpeg_data: bytes, prompt: str, confidence: float = 0.7,
                        show_boxes: bool = True, show_masks: bool = True):
    """Run SAM3 inference on a JPEG frame and return annotated image + detections."""
    import torch

    img_array = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        return None, []

    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to("cuda")

    with sam_lock:
        sam_processor.confidence_threshold = confidence
        with torch.amp.autocast("cuda", dtype=torch.float16):
            state = sam_processor.set_image(tensor)
            output = sam_processor.set_text_prompt(state=state, prompt=prompt)

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    detections = []
    if masks is not None and len(masks) > 0:
        for mask, box, score in zip(masks, boxes, scores):
            s = score.cpu().item() if hasattr(score, "cpu") else float(score)
            b = box.cpu().numpy() if hasattr(box, "cpu") else np.array(box)
            m = mask.cpu().numpy() if hasattr(mask, "cpu") else np.array(mask)
            if m.ndim == 3:
                m = m.squeeze()
            bi = b.astype(int)

            if show_masks:
                yellow_overlay = img_array.copy()
                yellow_overlay[m > 0.5] = [0, 255, 255]
                img_array = cv2.addWeighted(img_array, 0.65, yellow_overlay, 0.35, 0)

            if show_boxes:
                cv2.rectangle(img_array, (bi[0], bi[1]), (bi[2], bi[3]), (0, 255, 255), 2)
                label = f"{s:.0%}"
                cv2.putText(img_array, label, (bi[0], bi[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            detections.append({
                "score": round(s, 3),
                "box": [int(bi[0]), int(bi[1]), int(bi[2]), int(bi[3])],
            })

    _, jpeg_out = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg_out.tobytes(), detections


def _sam_worker(source: str, prompt: str, confidence: float, stop_event: threading.Event,
                show_boxes: bool = True, show_masks: bool = True, settings_ref: dict = None):
    """Background thread: grab frame every 2s, run SAM3, store result."""
    print(f"[SAM3] Worker started for {source} with prompt='{prompt}' conf={confidence}")
    while not stop_event.is_set():
        try:
            cur_confidence = confidence
            cur_show_boxes = show_boxes
            cur_show_masks = show_masks
            if settings_ref:
                cur_confidence = settings_ref.get("confidence", confidence)
                cur_show_boxes = settings_ref.get("show_boxes", show_boxes)
                cur_show_masks = settings_ref.get("show_masks", show_masks)

            with raw_frame_lock:
                jpeg_data = raw_frame_buffer.get(source)

            if jpeg_data is None:
                stop_event.wait(1.0)
                continue

            annotated_jpeg, detections = _sam_annotate_frame(
                jpeg_data, prompt, cur_confidence, cur_show_boxes, cur_show_masks
            )
            if annotated_jpeg is not None:
                b64 = base64.b64encode(annotated_jpeg).decode("ascii")
                with sam_results_lock:
                    sam_results[source] = {
                        "annotated_frame": b64,
                        "detections": detections,
                        "count": len(detections),
                        "prompt": prompt,
                        "confidence": cur_confidence,
                        "show_boxes": cur_show_boxes,
                        "show_masks": cur_show_masks,
                        "timestamp": time.time(),
                    }
        except Exception as e:
            print(f"[SAM3] Worker error for {source}: {e}")

        stop_event.wait(2.0)

    with sam_results_lock:
        sam_results.pop(source, None)
    print(f"[SAM3] Worker stopped for {source}")

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

def _mediamtx_add_path_sync(name: str, rtsp_url: str):
    try:
        http_requests.delete(
            f"{MEDIAMTX_API}/v3/config/paths/delete/{name}", timeout=2,
        )
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
        r = http_requests.delete(
            f"{MEDIAMTX_API}/v3/config/paths/delete/{name}", timeout=2,
        )
        print(f"[MEDIAMTX] Deleted path {name} (status={r.status_code})")
    except Exception as e:
        print(f"[MEDIAMTX] Failed to delete path {name}: {e}")


def mediamtx_add_path(name: str, rtsp_url: str):
    threading.Thread(
        target=_mediamtx_add_path_sync, args=(name, rtsp_url), daemon=True
    ).start()


def mediamtx_remove_path(name: str):
    threading.Thread(
        target=_mediamtx_remove_path_sync, args=(name,), daemon=True
    ).start()


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
                    return
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
            return dict(overlays[stream])
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

def update_raw_frame(stream: str, jpeg_data: bytes):
    with raw_frame_lock:
        raw_frame_buffer[stream] = jpeg_data

def get_all_metrics():
    with metrics_lock:
        return metrics.copy()


def add_alert(source: str, congestion: int, metrics_data: dict, screenshot_data: bytes):
    """Add a new congestion alert with screenshot."""
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
    """Load previously saved alerts from disk."""
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


def _stop_all_sources():
    """Stop every running source and SAM worker."""
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

    print(f"[STOP_ALL] Stopped {stopped} sources, cleared SAM workers")
    return stopped


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

class SourceStartRequest(BaseModel):
    index: int
    mode: Optional[str] = None

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
        overlays[name] = dict(new_state)
        print(f"[OVERLAY] {name} updated: heatmap={new_state['heatmap']}, trails={new_state['trails']}, bboxes={new_state['bboxes']}")

    cfg = load_rtsp_config()
    cfg.setdefault("overlays", {})[name] = new_state
    save_rtsp_config(cfg)

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
            return {"status": "already_running"}

        url = cfg["rtsp_links"][req.index - 1]
        name = source_display_name(url)

        overlay_config = MODE_OVERLAYS.get(active_mode, {"heatmap": False, "trails": False, "bboxes": False})

        process, stop = start_source_callback(req.index, url, name, overlay_config)
        running_sources[req.index] = {"process": process, "stop": stop}

    with overlay_lock:
        overlays[name] = dict(overlay_config)
    print(f"[OVERLAY] {name} set from mode config: {overlay_config}")

    active = cfg.get("active_sources", [])
    if req.index not in active:
        active.append(req.index)
    cfg["active_sources"] = active
    save_rtsp_config(cfg)
    return {"status": "started", "index": req.index, "mode": active_mode}

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

@app.post("/api/sources/stop_all")
def stop_all_sources():
    """Stop every running source and SAM worker."""
    global active_mode
    count = _stop_all_sources()
    active_mode = None
    return {"status": "stopped", "count": count}

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
                import time
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

    overlay_config = MODE_OVERLAYS.get(active_mode, {"heatmap": False, "trails": False, "bboxes": False})

    with overlay_lock:
        overlays[name] = dict(overlay_config)
    print(f"[UPLOAD] {name} overlays from active_mode={active_mode}: {overlay_config}")

    process, stop = start_upload_callback(str(target), name, overlay_config)

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

# ── SAM3 Pydantic models ──

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


@app.post("/api/sam/start")
def sam_start(req: SamStartRequest):
    if not _load_sam_model():
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
        target=_sam_worker,
        args=(req.source, req.prompt, req.confidence, stop_event,
              req.show_boxes, req.show_masks, settings_ref),
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
    active = list(sam_threads.keys())
    return {
        "model_loaded": sam_model_loaded,
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
            overlays[name] = {
                "heatmap": state.get("heatmap", True),
                "trails": state.get("trails", True),
                "bboxes": state.get("bboxes", True),
            }
            print(f"[OVERLAY] Loaded saved state for {name}: {overlays[name]}")

    for url in rtsp_links:
        ensure_overlay(source_display_name(url))

    # Skip loading stored alerts — only show new alerts from this session

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9010)
