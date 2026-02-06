# IRIS Command

Integrated real-time video analytics for multi-drone camera feeds. The stack runs a Python FastAPI backend with YOLO + ByteTrack and CCN crowd counting, and a React dashboard with WebRTC/HLS streaming via MediaMTX.

![Congestion Panel](congestion_panel.png)
![Traffic Flow](traffic_flow.png)

**Highlights**
- Multi-source RTSP ingestion with on-demand processing.
- Real-time analytics overlays (heatmap, trails, bboxes) by mode.
- WebRTC + HLS streaming through MediaMTX.
- FastAPI control plane with per-source metrics and alerts.
- Structured logging to `logs/YYYY-MM-DD/` with timestamped files.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                               IRIS Command                                 │
├──────────────────────────────┬─────────────────────────────────────────────┤
│      Python Backend          │               Frontend (React)              │
├──────────────────────────────┼─────────────────────────────────────────────┤
│ • FastAPI control server     │ • WebRTC + HLS playback                     │
│ • YOLO + ByteTrack           │ • Multi-view dashboard                      │
│ • CCN crowd counting         │ • Mode-based analytics UI                   │
│ • FFmpeg RTSP publishing     │ • Metrics and alert overlays                │
│ • MediaMTX API integration   │ • Framer Motion UI                          │
└──────────────────────────────┴─────────────────────────────────────────────┘
```

## Repo Structure

```
iris_command/
├── py_backend/                # Python backend (FastAPI + inference)
│   ├── app.py
│   ├── server.py
│   ├── yolobyte.py
│   ├── crowd.py
│   ├── config/                # YAML config
│   │   ├── rtsp_links.yml
│   │   └── mediamtx.yml
│   ├── models/                # Model weights
│   │   ├── yolov11n-visdrone.pt
│   │   ├── best_head.pt
│   │   └── crowd-model.pth
│   ├── service/               # systemd units + wrappers
│   └── start_backend.sh
├── frontend/                  # React dashboard
├── logs/                      # Daily logs (runtime)
├── congestion_panel.png
├── traffic_flow.png
└── README.md
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU + CUDA (recommended; CPU fallback supported)
- FFmpeg
- MediaMTX
- Node.js 18+ (frontend)

## Configuration

**RTSP Sources**
Edit `py_backend/config/rtsp_links.yml`:

```yaml
rtsp_links:
  - rtsp://user:pass@host:554/stream1
  - rtsp://user:pass@host:554/stream2
active_sources: []
overlays:
  stream1:
    heatmap: true
    trails: true
    bboxes: true
```

**MediaMTX**
Edit `py_backend/config/mediamtx.yml`. WebRTC, RTSP, HLS ports are set here.

Default ports:
- Backend API: `9010`
- MediaMTX RTSP: `8554`
- MediaMTX WebRTC: `8889`
- MediaMTX HLS: `8888`
- MediaMTX API: `9997`

## Quick Start (Development)

**Backend**

```bash
cd py_backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -u app.py
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Open the dashboard at `http://localhost:5173`.

## Production Run (Script)

```bash
cd py_backend
./start_backend.sh start
```

Common commands:
- `./start_backend.sh stop`
- `./start_backend.sh restart`
- `./start_backend.sh status`
- `./start_backend.sh logs`

## Production Run (systemd)

```bash
# Optional: tune runtime/GPU/session limits first.
sudo cp py_backend/service/iris-command.env /etc/default/iris-command
sudo nano /etc/default/iris-command

# Install + enable + restart both units.
sudo py_backend/service/install_systemd.sh

# Verify.
systemctl status iris-mediamtx.service iris-backend.service
journalctl -u iris-mediamtx.service -u iris-backend.service -f
```

## API Examples

**List sources**
```bash
curl -s http://localhost:9010/api/sources
```

**Start processing a source**
```bash
curl -s -X POST http://localhost:9010/api/sources/start \
  -H 'Content-Type: application/json' \
  -d '{"index":1,"mode":"congestion"}'
```

**Stop a source**
```bash
curl -s -X POST http://localhost:9010/api/sources/stop \
  -H 'Content-Type: application/json' \
  -d '{"index":1}'
```

**Get metrics**
```bash
curl -s http://localhost:9010/api/metrics
```

## Logging

All logs are written to the repo root `logs/` folder and grouped by day:

```
logs/
  2026-02-03/
    backend-10-15-02-12345.log
    mediamtx-10-15-01.log
    ffmpeg-bcpdrone2-10-15-04-12345.log
    frontend-10-15-30-12345.log
```

No logs are written to stdout, journald, or any other location.

## Troubleshooting

- **WHEP 404 on processed streams**
  Ensure `py_backend/config/mediamtx.yml` includes:
  ```yaml
  paths:
    all:
      source: publisher
  ```

- **Backend 400 on `/api/sources/start`**
  Verify `py_backend/config/rtsp_links.yml` exists and the `index` is in range.

- **Processed stream takes time to appear**
  The backend starts FFmpeg after a short delay and retries if the MJPEG stream is not ready. The UI also waits a few seconds before attempting WHEP.

## License

Proprietary. All rights reserved.
