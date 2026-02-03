#!/bin/bash
# IRIS Command Backend Startup Script
# Properly runs mediamtx and app.py in background with logging
#
# Usage:
#   ./start_backend.sh start     # Start all services
#   ./start_backend.sh stop      # Stop all services
#   ./start_backend.sh restart   # Restart all services
#   ./start_backend.sh status    # Show service status
#   ./start_backend.sh logs      # Show recent logs
#
# This script handles:
# - Running Python with unbuffered output (critical for nohup/background)
# - Proper process isolation (start_new_session)
# - Automatic mediamtx startup
# - FFmpeg publisher monitoring (handled by server.py)

# Don't use set -e as pkill returns non-zero when no process found

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
LOG_ROOT="$PROJECT_DIR/logs"
DATE_DIR="$(date '+%Y-%m-%d')"
LOG_DIR="$LOG_ROOT/$DATE_DIR"
DATA_DIR="$SCRIPT_DIR/data"
CONFIG_DIR="$SCRIPT_DIR/config"

mkdir -p "$LOG_DIR"
TS="$(date '+%H-%M-%S')"
MEDIAMTX_LOG="$LOG_DIR/mediamtx-$TS.log"
BACKEND_LOG="$LOG_DIR/backend-$TS.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

stop_services() {
    log "Stopping existing services..."

    # Stop app.py (and legacy inference.py)
    pkill -f "python.*app\.py" 2>/dev/null || true
    pkill -f "python.*inference\.py" 2>/dev/null || true

    # Stop mediamtx
    pkill -f "mediamtx" 2>/dev/null || true

    # Stop any ffmpeg publishers
    pkill -f "ffmpeg.*processed_" 2>/dev/null || true

    sleep 2
    log "Services stopped"
}

start_mediamtx() {
    log "Starting MediaMTX..."

    # Check if mediamtx binary exists
    if ! command -v mediamtx &> /dev/null; then
        if [ -f "$DATA_DIR/mediamtx" ]; then
            MEDIAMTX_BIN="$DATA_DIR/mediamtx"
        else
            error "mediamtx not found!"
            return 1
        fi
    else
        MEDIAMTX_BIN="mediamtx"
    fi

    # Start mediamtx with config
    cd "$SCRIPT_DIR"
    $MEDIAMTX_BIN "$CONFIG_DIR/mediamtx.yml" >> "$MEDIAMTX_LOG" 2>&1 &
    MEDIAMTX_PID=$!
    echo $MEDIAMTX_PID > "$LOG_DIR/mediamtx.pid"

    # Wait for mediamtx to start
    sleep 2
    if kill -0 $MEDIAMTX_PID 2>/dev/null; then
        log "MediaMTX started (PID: $MEDIAMTX_PID)"
        return 0
    else
        error "MediaMTX failed to start"
        return 1
    fi
}

start_inference() {
    log "Starting inference backend..."

    cd "$SCRIPT_DIR"

    # Use virtual environment python if available
    if [ -x ".venv/bin/python" ]; then
        PYTHON_BIN=".venv/bin/python"
    else
        PYTHON_BIN="python"
    fi

    # Key fix: Use unbuffered Python output and proper environment
    # PYTHONUNBUFFERED=1 ensures frames are sent immediately without buffering
    # This is critical for nohup/background operation
    PYTHONUNBUFFERED=1 \
    IRIS_LOG_PATH="$BACKEND_LOG" \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|buffer_size;262144|analyzeduration;50000|probesize;50000|fflags;nobuffer|flags;low_delay" \
    "$PYTHON_BIN" -u app.py >/dev/null 2>&1 &

    INFERENCE_PID=$!
    echo $INFERENCE_PID > "$LOG_DIR/inference.pid"

    # Wait briefly and check if it started
    sleep 3
    if kill -0 $INFERENCE_PID 2>/dev/null; then
        log "Inference backend started (PID: $INFERENCE_PID)"
        return 0
    else
        error "Inference backend failed to start. Check $BACKEND_LOG"
        return 1
    fi
}

status() {
    echo ""
    log "Service Status:"
    echo "──────────────────────────────────────"

    # Check mediamtx
    if pgrep -f "mediamtx" > /dev/null; then
        MPID=$(pgrep -f "mediamtx" | head -1)
        echo -e "  MediaMTX:   ${GREEN}Running${NC} (PID: $MPID)"
    else
        echo -e "  MediaMTX:   ${RED}Stopped${NC}"
    fi

    # Check app.py
    if pgrep -f "python.*app\.py" > /dev/null; then
        IPID=$(pgrep -f "python.*app\.py" | head -1)
        echo -e "  Backend:    ${GREEN}Running${NC} (PID: $IPID)"
    else
        echo -e "  Backend:    ${RED}Stopped${NC}"
    fi

    # Check FFmpeg publishers
    FFMPEG_COUNT=$(pgrep -f "ffmpeg.*processed_" | wc -l)
    if [ "$FFMPEG_COUNT" -gt 0 ]; then
        echo -e "  FFmpeg:     ${GREEN}$FFMPEG_COUNT publisher(s) active${NC}"
    else
        echo -e "  FFmpeg:     ${YELLOW}No active publishers${NC}"
    fi

    echo "──────────────────────────────────────"
    echo ""
}

case "$1" in
    start)
        stop_services
        start_mediamtx
        start_inference
        status
        ;;
    stop)
        stop_services
        status
        ;;
    restart)
        stop_services
        start_mediamtx
        start_inference
        status
        ;;
    status)
        status
        ;;
    logs)
        echo "=== Backend Log (latest) ==="
        LATEST_BACKEND_LOG="$(ls -1t "$LOG_DIR"/backend-*.log 2>/dev/null | head -1)"
        if [ -n "$LATEST_BACKEND_LOG" ]; then
            tail -50 "$LATEST_BACKEND_LOG"
        else
            echo "No backend log found"
        fi
        echo ""
        echo "=== MediaMTX Log (latest) ==="
        LATEST_MEDIAMTX_LOG="$(ls -1t "$LOG_DIR"/mediamtx-*.log 2>/dev/null | head -1)"
        if [ -n "$LATEST_MEDIAMTX_LOG" ]; then
            tail -30 "$LATEST_MEDIAMTX_LOG"
        else
            echo "No mediamtx log found"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
