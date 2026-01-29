#!/usr/bin/env bash
  set -euo pipefail

  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /data/yolov11n-visdrone.onnx [out.engine] [fp16|fp32|int8]"
    exit 1
  fi

  ONNX_PATH="$1"
  ENGINE_PATH="${2:-${ONNX_PATH%.onnx}.engine}"
  PRECISION="${3:-fp16}"

  TRTEXEC="${TRTEXEC:-trtexec}"

  if ! command -v "$TRTEXEC" >/dev/null 2>&1; then
    echo "trtexec not found. Set TRTEXEC=/path/to/trtexec or add it to PATH."
    exit 1
  fi

  FLAGS=(
    --onnx="$ONNX_PATH"
    --saveEngine="$ENGINE_PATH"
    --explicitBatch
    --workspace=4096
  )

  case "$PRECISION" in
    fp16) FLAGS+=(--fp16) ;;
    fp32) ;;
    int8) FLAGS+=(--int8) ;;
    *) echo "Unknown precision: $PRECISION"; exit 1 ;;
  esac

  "$TRTEXEC" "${FLAGS[@]}"

  echo "Wrote engine: $ENGINE_PATH"