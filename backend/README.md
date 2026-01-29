# Drone Analytics

Real-time drone video analytics and tracking using YOLO + ByteTrack. Supports local video files and RTSP streams, with optional RTSP re-publish of annotated frames.

## Quick start

```bash
# build
cargo build --release

# install binary locally
cargo install --path .
```

## Usage

```bash
drone_analytics [SOURCE] [FLAGS]
```

- `SOURCE` can be a local video path or an RTSP URL.
- If no `SOURCE` is provided, the app selects a source in this order:
  1) The first entry in `data/rtsp_links.yml`, if present and non-empty
  2) `data/drone8.mp4`

## Flags

- `--rtsp` Use RTSP source list from `data/rtsp_links.yml`.
- `--index <N>` Choose the Nth RTSP source (1-based) from `data/rtsp_links.yml`.
- `--indexes <N1,N2,...>` Choose multiple RTSP sources (1-based) from `data/rtsp_links.yml`.
- `--<N>` Shorthand for `--index <N>` (e.g. `--3`).
- `--mp4` Write output as MP4 (default is MKV).
- `--mkv` Write output as MKV.
- `--threads <N>` Set inference thread count.
- `--stride <N>` Process every Nth frame (reduces CPU load; default 2).
- `--headless` Disable GUI window (useful on servers without a display).
- `--device <DEVICE>` Set device via `ultralytics_inference::Device::from_str`.
- `--cpu` Force CPU device.
- `--gpu` Use GPU device (CUDA on Linux/Windows, MPS on macOS).

## Examples

```bash
# run default source
drone_analytics

# run a local file and save MP4
drone_analytics /path/to/video.mp4 --mp4

# run RTSP source list, choose camera #2
drone_analytics --rtsp --index 2

# run RTSP source list, choose camera #5 (shorthand)
drone_analytics --rtsp --5

# run multiple RTSP sources in parallel (one thread per source)
drone_analytics --rtsp --indexes 2,5,7

# run a specific RTSP URL
drone_analytics rtsp://user:pass@host:8554/stream

# run on GPU
drone_analytics --gpu
```

## Output

- Annotated video is written to `runs/drone_analysis/` as `drone_analysis_YYYYMMDD_HHMMSS.mkv` (or `.mp4`).
- For RTSP sources, output writing is capped to `60s` (see `RTSP_SAVE_SECONDS` in `src/state.rs`).

## RTSP publish (optional)

When the input is RTSP, the processed frames are also published via FFmpeg to an RTSP server:

- Default publish URL: `rtsp://127.0.0.1:8554/analytics`
- Override with `MEDIAMTX_PUBLISH_URL` environment variable.
- RTSP publishing requires `ffmpeg` in your PATH.
- A minimal MediaMTX config is provided in `data/mediamtx.yml`.

Example:

```bash
MEDIAMTX_PUBLISH_URL=rtsp://127.0.0.1:8554/analytics \
  drone_analytics --rtsp --index 1
```

## Controls

- `q` Quit
- `h` Toggle heatmap overlay (when UI is enabled)
- `t` Toggle trails overlay (when UI is enabled)
- `b` Toggle bounding boxes (when UI is enabled)

Note: when multiple sources are selected, the app runs headless and UI controls are disabled.

## Models & data

- Expected model file: `data/yolov11n-visdrone.onnx`
- TensorRT engine path (when enabled): `data/yolov11n-visdrone.engine`
- RTSP list: `data/rtsp_links.yml`

## Build notes

On Linux/Windows the default inference device is `TensorRT:0` (falls back to `CUDA:0`, then CPU on load failure). On macOS the default is `MPS`.
To run with a TensorRT engine file, build with `--features "nvidia trt_engine"` and ensure `data/yolov11n-visdrone.engine` exists.
OpenCV capture uses the FFmpeg options in `OPENCV_FFMPEG_CAPTURE_OPTIONS` when set, otherwise it uses a software decode-friendly default.

The project uses `opencv` and `ultralytics-inference`. If you want GPU or CoreML support, enable features at build time:

```bash
# CUDA (Linux/Windows)
cargo install --path . --features cuda
drone_analytics --gpu

# CoreML (macOS)
cargo install --path . --features coreml
drone_analytics --gpu
```
