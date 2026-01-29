# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IRIS Command is a drone/camera video analytics platform with two main components:
- **Backend**: Rust-based real-time video analytics using YOLO + ByteTrack for object detection and tracking
- **Frontend**: React dashboard for visualization with 3D globe and analytics displays

## Build & Run Commands

### Backend (Rust)

```bash
# Build
cd backend && cargo build --release

# Run with default source (data/drone8.mp4 or first RTSP link)
cargo run --release

# Run specific video file
cargo run --release -- /path/to/video.mp4

# Run with RTSP source (use --index N or --N shorthand)
cargo run --release -- --rtsp --index 2
cargo run --release -- --rtsp --3

# GPU builds
cargo build --release --features cuda      # CUDA (Linux/Windows)
cargo build --release --features coreml    # CoreML (macOS)
cargo build --release --features nvidia    # CUDA + TensorRT

# Install binary
cargo install --path .
```

### Frontend (React/Vite)

```bash
cd frontend
yarn install
yarn dev      # Development server
yarn build    # Production build
yarn lint     # ESLint
yarn preview  # Preview production build
```

## Architecture

### Backend Structure (`backend/src/`)

- **main.rs**: Entry point, video pipeline orchestration, CLI argument parsing, inference loop
- **analytics.rs**: Object tracking logic using ByteTrack, trail management, heatmap generation, system stats
- **state.rs**: Core types (BBox, Detection, Track, SpeedClass) and tuning constants (thresholds, tracker params)
- **helpers.rs**: Video I/O, frame preprocessing, overlay rendering, RTSP link loading
- **db.rs**: SQLite persistence for analytics data
- **mediamtx.rs**: RTSP re-publishing via FFmpeg
- **trt.rs**: TensorRT engine utilities

**Key dependencies**: `ultralytics-inference` (YOLO), `jamtrack-rs` (ByteTrack), `opencv` (video capture/display)

### Frontend Structure (`frontend/src/`)

- **App.jsx**: Main routing and camera analytics component with use-case configs (traffic, crowd, safety, perimeter)
- **components/Dashboard/**: Header, Footer, LeftPanel, RightPanel, WelcomeScreen, IRISLoader
- **components/Globe/**: 3D Earth visualization using react-three-fiber (Scene, Earth, Markers)
- **components/UI/**: Shared UI primitives (Card)

**Tech stack**: React 19, Vite, Tailwind CSS v4, Framer Motion, react-three-fiber/drei

### Data Flow

1. Video source (local file or RTSP) → OpenCV capture
2. Frames → YOLO inference (TensorRT/CUDA/MPS/CPU) → detections
3. Detections → ByteTrack → stable track IDs with trails
4. Tracks → speed classification, heatmap accumulation
5. Annotated frames → display window + optional RTSP publish + file output
6. Analytics → SQLite aggregation (60s intervals)

## Runtime Configuration

- **Model file**: `backend/data/yolov11n-visdrone.onnx`
- **RTSP sources**: `backend/data/rtsp_links.yml`
- **Output directory**: `backend/runs/drone_analysis/`
- **RTSP publish URL**: Set `MEDIAMTX_PUBLISH_URL` env var (default: `rtsp://127.0.0.1:8554/analytics`)

## Key Constants (state.rs)

- `CONF_THRESH`: Detection confidence threshold (0.10)
- `TRACK_THRESH`, `HIGH_THRESH`, `MATCH_THRESH`: ByteTrack thresholds
- `THRESH_STALLED/SLOW/MEDIUM`: Speed classification thresholds (px/sec)
- `TARGET_CLASSES`: VisDrone class IDs to track [3,4,5,7,8,9]
