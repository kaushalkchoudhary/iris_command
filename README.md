# IRIS Command

Real-time drone video analytics platform with YOLO + ByteTrack object detection and tracking.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         IRIS Command                            │
├──────────────────────────┬──────────────────────────────────────┤
│       Backend (Rust)     │          Frontend (React)            │
├──────────────────────────┼──────────────────────────────────────┤
│ • TensorRT inference     │ • WebRTC video streaming             │
│ • ByteTrack tracking     │ • Real-time analytics dashboard      │
│ • Dynamic multi-drone    │ • Overlay controls (H/T/B)           │
│ • RTSP re-publishing     │ • Multi-view grid layout             │
│ • REST API (port 9010)   │ • Tailwind + Framer Motion           │
└──────────────────────────┴──────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA 12+
- TensorRT 8+
- Rust 1.70+
- Node.js 18+
- MediaMTX (for RTSP relay)

### Backend

```bash
cd backend

# Build with TensorRT (default)
cargo build --release

# Run
cargo run --release

# API available at http://localhost:9010
```

### Frontend

```bash
cd frontend
yarn install
yarn dev

# Open http://localhost:5173
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/sources` | List all drones with status |
| POST | `/api/sources/start` | Start processing `{"index": N}` |
| POST | `/api/sources/stop` | Stop processing `{"index": N}` |
| POST | `/api/sources/active` | Set active drones `{"indexes": [1,2,3]}` |
| GET | `/api/overlays` | Get all overlay states |
| POST | `/api/overlays/{name}` | Update overlays `{"heatmap":true,...}` |

## Configuration

### RTSP Sources (`backend/data/rtsp_links.yml`)

```yaml
rtsp_links:
  - rtsp://user:pass@host:554/stream1
  - rtsp://user:pass@host:554/stream2
active_sources:
  - 1
  - 2
overlays:
  stream1:
    heatmap: true
    trails: true
    bboxes: true
```

## Project Structure

```
iris_command/
├── backend/               # Rust backend
│   ├── src/
│   │   ├── main.rs       # Entry point, video pipeline
│   │   ├── analytics.rs  # Tracking, heatmap, metrics
│   │   ├── control.rs    # REST API server
│   │   ├── helpers.rs    # Video I/O, config
│   │   ├── state.rs      # Core types, constants
│   │   ├── trt.rs        # TensorRT bindings
│   │   └── trt_yolo.rs   # YOLO inference
│   ├── data/
│   │   ├── rtsp_links.yml
│   │   └── *.engine      # TensorRT models
│   └── Cargo.toml
├── frontend/              # React frontend
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   ├── package.json
│   └── vite.config.js
├── CLAUDE.md              # AI assistant instructions
└── README.md
```

## License

Proprietary - All rights reserved

