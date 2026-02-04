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
PROJECT_DIR="$SCRIPT_DIR"
LOG_ROOT="$PROJECT_DIR/logs"
LOG_DIR="$LOG_ROOT"
DATA_DIR="$SCRIPT_DIR/data"
CONFIG_DIR="$SCRIPT_DIR/config"

mkdir -p "$LOG_DIR"
MEDIAMTX_LOG="$LOG_DIR/mediamtx.log"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
MEDIAMTX_PID_FILE="$LOG_DIR/mediamtx.pid"
BACKEND_PID_FILE="$LOG_DIR/inference.pid"

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

is_pid_running() {
    local pid="$1"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

backend_pids() {
    local pid cwd
    while read -r pid; do
        [ -z "$pid" ] && continue
        cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
        if [ "$cwd" = "$SCRIPT_DIR" ]; then
            echo "$pid"
        fi
    done < <(pgrep -f "python.*(app|inference)\.py" 2>/dev/null || true)
}

mediamtx_pids() {
    pgrep -f "mediamtx.*$CONFIG_DIR/mediamtx.yml" 2>/dev/null || true
}

ffmpeg_pids() {
    pgrep -f "ffmpeg.*processed_" 2>/dev/null || true
}

cleanup_leftovers() {
    for pid_file in "$MEDIAMTX_PID_FILE" "$BACKEND_PID_FILE"; do
        if [ -f "$pid_file" ]; then
            local pid
            pid="$(cat "$pid_file" 2>/dev/null || true)"
            if ! is_pid_running "$pid"; then
                rm -f "$pid_file"
            fi
        fi
    done
}

kill_pids_gracefully() {
    local label="$1"
    shift
    local pids=("$@")
    local pid alive attempts

    if [ "${#pids[@]}" -eq 0 ]; then
        return 0
    fi

    log "Stopping $label PID(s): ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true

    attempts=0
    while [ "$attempts" -lt 20 ]; do
        alive=0
        for pid in "${pids[@]}"; do
            if is_pid_running "$pid"; then
                alive=1
                break
            fi
        done
        [ "$alive" -eq 0 ] && return 0
        sleep 0.5
        attempts=$((attempts + 1))
    done

    warn "$label did not stop with SIGTERM, forcing SIGKILL"
    kill -9 "${pids[@]}" 2>/dev/null || true
}

is_fully_started() {
    local backend_pid mediamtx_pid
    backend_pid="$(backend_pids | head -1)"
    mediamtx_pid="$(mediamtx_pids | head -1)"
    [ -n "$backend_pid" ] && [ -n "$mediamtx_pid" ]
}

stop_services() {
    cleanup_leftovers
    log "Stopping existing services..."

    mapfile -t ffmpeg_list < <(ffmpeg_pids)
    mapfile -t backend_list < <(backend_pids)
    mapfile -t mediamtx_list < <(mediamtx_pids)

    kill_pids_gracefully "FFmpeg publishers" "${ffmpeg_list[@]}"
    kill_pids_gracefully "backend" "${backend_list[@]}"
    kill_pids_gracefully "MediaMTX" "${mediamtx_list[@]}"

    rm -f "$MEDIAMTX_PID_FILE" "$BACKEND_PID_FILE"
    cleanup_leftovers
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
    echo "$MEDIAMTX_PID" > "$MEDIAMTX_PID_FILE"

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
    IRIS_FRONTEND_LOG_PATH="$FRONTEND_LOG" \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|buffer_size;262144|analyzeduration;50000|probesize;50000|fflags;nobuffer|flags;low_delay" \
    "$PYTHON_BIN" -u app.py >> "$BACKEND_LOG" 2>&1 &

    INFERENCE_PID=$!
    echo "$INFERENCE_PID" > "$BACKEND_PID_FILE"

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
    MPID="$(mediamtx_pids | head -1)"
    if [ -n "$MPID" ]; then
        echo -e "  MediaMTX:   ${GREEN}Running${NC} (PID: $MPID)"
    else
        echo -e "  MediaMTX:   ${RED}Stopped${NC}"
    fi

    # Check app.py
    IPID="$(backend_pids | head -1)"
    if [ -n "$IPID" ]; then
        echo -e "  Backend:    ${GREEN}Running${NC} (PID: $IPID)"
    else
        echo -e "  Backend:    ${RED}Stopped${NC}"
    fi

    # Check FFmpeg publishers
    FFMPEG_COUNT="$(ffmpeg_pids | wc -l)"
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
        cleanup_leftovers
        if is_fully_started; then
            MPID="$(mediamtx_pids | head -1)"
            IPID="$(backend_pids | head -1)"
            warn "Already started (MediaMTX PID: $MPID, Backend PID: $IPID)"
            status
            exit 0
        fi
        if [ -n "$(backend_pids | head -1)" ] || [ -n "$(mediamtx_pids | head -1)" ]; then
            warn "Detected partially running services; cleaning up before start"
            stop_services
        fi
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
        if [ -f "$BACKEND_LOG" ]; then
            tail -50 "$BACKEND_LOG"
        else
            echo "No backend log found"
        fi
        echo ""
        echo "=== MediaMTX Log (latest) ==="
        if [ -f "$MEDIAMTX_LOG" ]; then
            tail -30 "$MEDIAMTX_LOG"
        else
            echo "No mediamtx log found"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
