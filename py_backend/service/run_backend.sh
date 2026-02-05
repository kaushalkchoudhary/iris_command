#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$PROJECT_DIR/../logs"
DATE_DIR="$(date '+%Y-%m-%d')"
LOG_DIR="$LOG_ROOT/$DATE_DIR"
TS="$(date '+%H-%M-%S')"
BACKEND_LOG="$LOG_DIR/backend-$TS.log"

mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"
exec env PYTHONUNBUFFERED=1 \
  IRIS_LOG_PATH="$BACKEND_LOG" \
  OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|buffer_size;262144|analyzeduration;50000|probesize;50000|fflags;nobuffer|flags;low_delay" \
  "$PROJECT_DIR/.venv/bin/python" -u app.py >>"$BACKEND_LOG" 2>&1
