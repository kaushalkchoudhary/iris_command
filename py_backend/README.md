# IRIS Command — Python Backend

Real-time drone/camera video analytics using YOLO + ByteTrack (vehicles) and CCN density counting (crowds), served via FastAPI with RTSP re-publishing through MediaMTX.

## Module Structure

```
py_backend/
  app.py          Entry point — multiprocessing setup, relay worker, main()
  server.py       FastAPI endpoints, overlay/frame/metrics state, FFmpeg, alerts
  yolobyte.py     YOLO + ByteTrack inference pipeline (process_stream, process_upload_stream)
  crowd.py        CCN crowd counting model + CrowdAnalyticsState
  overlays.py     TrailRenderer, HeatmapRenderer, CrowdHeatmapRenderer
  helpers.py      FrameCapture, BboxSmoother, shared constants
  sam.py          SAM3 (Segment Anything 3) integration for forensics
  login.py        User authentication (SQLite)
  config/
    rtsp_links.yml      RTSP source configuration
    mediamtx.yml        MediaMTX RTSP server config
  models/
    yolov11n-visdrone.pt   YOLO vehicle detection weights
    best_head.pt           YOLO head detection weights
    crowd-model.pth        CCN crowd counting weights
  data/
    uploads/              Uploaded recordings (auto-created)
    alerts/               Alerts cache (auto-created)
    auth.db               User credentials (auto-created)
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended; CPU fallback supported)
- FFmpeg (for RTSP re-publishing)
- MediaMTX (for RTSP relay server)

## Setup

```bash
cd py_backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Make sure model weight files are present in `models/`:
- `yolov11n-visdrone.pt` (vehicle detection)
- `best_head.pt` (head/crowd detection)
- `crowd-model.pth` (CCN density counting)

## Running

### Development (foreground)

```bash
source .venv/bin/activate
python -u app.py
```

The FastAPI server starts on **port 9010**.

### Production (background script)

```bash
./start_backend.sh start      # Start MediaMTX + backend
./start_backend.sh stop       # Stop all services
./start_backend.sh restart    # Restart everything
./start_backend.sh status     # Show service status
./start_backend.sh logs       # Show recent logs
```

Logs are written to `../logs/inference.log` and `../logs/mediamtx.log`.

### Systemd (persistent service)

```bash
# Install service files
sudo cp service/iris-backend.service /etc/systemd/system/
sudo cp service/iris-mediamtx.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable and start
sudo systemctl enable iris-mediamtx iris-backend
sudo systemctl start iris-mediamtx
sudo systemctl start iris-backend

# Management
sudo systemctl status iris-backend
sudo systemctl restart iris-backend
journalctl -u iris-backend -f     # follow logs
```

**Note:** If you previously had the old `inference.py`-based service installed, re-copy `iris-backend.service` to `/etc/systemd/system/` and run `sudo systemctl daemon-reload` to pick up the new `app.py` entry point.

## API Endpoints

All endpoints are served at `http://<host>:9010`.

### Sources

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/sources` | List all configured RTSP sources |
| POST | `/api/sources/start` | Start inference on a source (`{index, mode}`) |
| POST | `/api/sources/stop` | Stop a source (`{index}`) |
| POST | `/api/sources/stop_all` | Stop all running sources |

### Streaming & Metrics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/stream/{name}` | MJPEG stream for a source |
| GET | `/api/metrics` | Current analytics metrics for all sources |

### Overlays & Confidence

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/overlays` | All overlay states |
| GET/POST | `/api/overlays/{name}` | Get/set overlay toggles (heatmap, trails, bboxes) |
| GET/POST | `/api/confidence/{name}` | Get/set detection confidence threshold |
| GET | `/api/mode/confidence` | Default confidence per mode |

### File Upload

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload a video file for analysis |
| GET | `/api/uploads` | List active upload streams |
| DELETE | `/api/uploads/{name}` | Stop an upload stream |

### SAM3 Forensics

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/sam/start` | Start SAM3 on a source (`{source, prompt, confidence}`) |
| POST | `/api/sam/stop` | Stop SAM3 for a source |
| POST | `/api/sam/update` | Update SAM3 settings live |
| GET | `/api/sam/result/{source}` | Get latest SAM3 annotated frame + detections |
| GET | `/api/sam/status` | SAM3 model and worker status |

### Alerts

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/alerts` | List recent congestion alerts |
| GET | `/api/alerts/{id}/screenshot` | Alert screenshot image |
| DELETE | `/api/alerts` | Clear all alerts |

### Auth

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/login` | Login (`{username, password}`) |
| GET | `/api/users` | List users |
| POST | `/api/users/add` | Add user |
| POST | `/api/users/delete` | Delete user |
| POST | `/api/users/change_password` | Change password |

## Analysis Modes

Set via `mode` field in `/api/sources/start`:

| Mode | Model | Overlays | Description |
|------|-------|----------|-------------|
| `congestion` | YOLO + ByteTrack | Heatmap | Traffic congestion analysis |
| `vehicle` | YOLO + ByteTrack | Bounding boxes | Vehicle detection and tracking |
| `flow` | YOLO + ByteTrack | Trails + boxes | Traffic flow visualization |
| `forensics` | YOLO + SAM3 | None (SAM overlay separate) | Object segmentation |
| `crowd` | CCN + YOLO heads | Heatmap | Crowd density and counting |

## Architecture

```
app.py  (main process)
  |
  +-- server.py  (FastAPI on port 9010, runs in main process)
  |     +-- sam.py  (SAM3 background threads)
  |     +-- login.py  (auth)
  |
  +-- relay_worker  (thread: bridges multiprocess queues -> server state)
  |
  +-- process_stream  (spawned process per RTSP source)
  |     +-- yolobyte.py  (YOLO/ByteTrack inference loop)
  |     +-- helpers.py  (FrameCapture, BboxSmoother)
  |     +-- overlays.py  (trail/heatmap rendering)
  |     +-- crowd.py  (CCN model, CrowdAnalyticsState)
  |
  +-- process_upload_stream  (spawned process per uploaded file)
```

Each inference process runs in its own subprocess (via `multiprocessing.spawn`) and communicates with the main server through queues for frames, metrics, and alerts.
