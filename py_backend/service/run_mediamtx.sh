#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$PROJECT_DIR/../logs"
DATE_DIR="$(date '+%Y-%m-%d')"
LOG_DIR="$LOG_ROOT/$DATE_DIR"
TS="$(date '+%H-%M-%S')"
MEDIAMTX_LOG="$LOG_DIR/mediamtx-$TS.log"

mkdir -p "$LOG_DIR"

if command -v mediamtx >/dev/null 2>&1; then
  MEDIAMTX_BIN="$(command -v mediamtx)"
else
  MEDIAMTX_BIN="/usr/local/bin/mediamtx"
fi

exec "$MEDIAMTX_BIN" "$PROJECT_DIR/config/mediamtx.yml" >> "$MEDIAMTX_LOG" 2>&1
