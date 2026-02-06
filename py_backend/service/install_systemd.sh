#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BACKEND_SERVICE_SRC="$SCRIPT_DIR/iris-backend.service"
MEDIAMTX_SERVICE_SRC="$SCRIPT_DIR/iris-mediamtx.service"
ENV_SRC="$SCRIPT_DIR/iris-command.env"

BACKEND_SERVICE_DST="/etc/systemd/system/iris-backend.service"
MEDIAMTX_SERVICE_DST="/etc/systemd/system/iris-mediamtx.service"
ENV_DST="/etc/default/iris-command"

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo $0"
  exit 1
fi

install -m 0644 "$BACKEND_SERVICE_SRC" "$BACKEND_SERVICE_DST"
install -m 0644 "$MEDIAMTX_SERVICE_SRC" "$MEDIAMTX_SERVICE_DST"

if [[ ! -f "$ENV_DST" ]]; then
  install -m 0644 "$ENV_SRC" "$ENV_DST"
  echo "Created $ENV_DST"
else
  echo "Kept existing $ENV_DST"
fi

systemctl daemon-reload
systemctl enable iris-mediamtx.service iris-backend.service
systemctl restart iris-mediamtx.service
systemctl restart iris-backend.service

echo "Installed and restarted services."
echo "Check status with:"
echo "  systemctl status iris-mediamtx.service iris-backend.service"
echo "Follow logs with:"
echo "  journalctl -u iris-mediamtx.service -u iris-backend.service -f"
echo "Daily file logs are also under:"
echo "  $ROOT_DIR/../logs"

