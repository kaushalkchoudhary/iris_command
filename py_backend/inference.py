import os
import cv2
import time
import threading
import subprocess
import shutil
import numpy as np
import gc
import multiprocessing as mp

from ultralytics import YOLO
import supervision as sv
import torch

from control_server import (
    run_control_server,
    update_metrics,
    update_frame,
    update_raw_frame,
    get_overlay_state,
    ensure_overlay,
    add_alert,
)

# Shared communication objects
spawn_ctx = None
frame_queue = None
raw_frame_queue = None
metrics_queue = None
alert_queue = None
overlay_shared_dict = None

# Alert settings
CONGESTION_ALERT_THRESHOLD = 40  # Trigger alert when congestion >= this value


def relay_worker(stop_event, f_q, m_q, a_q, rf_q):
    """Relay metrics, frames, raw frames, and alerts from multiprocess queues to control server."""
    while not stop_event.is_set():
        did_work = False

        try:
            for _ in range(60):
                if f_q.empty():
                    break
                name, data = f_q.get_nowait()
                update_frame(name, data)
                did_work = True
        except:
            pass

        try:
            for _ in range(30):
                if rf_q.empty():
                    break
                name, data = rf_q.get_nowait()
                update_raw_frame(name, data)
                did_work = True
        except:
            pass

        try:
            for _ in range(20):
                if m_q.empty():
                    break
                name, data = m_q.get_nowait()
                update_metrics(name, data)
                did_work = True
        except:
            pass

        try:
            for _ in range(5):
                if a_q.empty():
                    break
                source, congestion, metrics_data, screenshot = a_q.get_nowait()
                add_alert(source, congestion, metrics_data, screenshot)
                did_work = True
        except:
            pass

        if not did_work:
            time.sleep(0.001)
        else:
            time.sleep(0.0005)


# Optimized RTSP settings - lower buffer for real-time
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;262144|analyzeduration;50000|probesize;50000|fflags;nobuffer|flags;low_delay"
)

MODEL_PATH_VEHICLE = "data/yolov11n-visdrone.pt"
MODEL_PATH_CROWD = "data/best_head.pt"
TARGET_FPS = 30

# Inference settings
INFERENCE_SIZE = 320
SKIP_FRAMES = 1
JPEG_QUALITY = 70
MAX_DET = 100
BBOX_SMOOTH_ALPHA = 0.45

# GPU memory budget
YOLO_GPU_MEMORY_FRACTION = 0.35

# Drones that use crowd counting model
CROWD_DRONES = {"bcpdrone10", "bcpdrone12"}

# Speed thresholds in pixels/second
THRESH_STALLED = 8.0
THRESH_SLOW = 50.0
THRESH_MEDIUM = 160.0

EMA_ALPHA = 0.2


def classify_speed(speed_px_s: float) -> int:
    if speed_px_s < THRESH_STALLED:
        return 0
    elif speed_px_s < THRESH_SLOW:
        return 1
    elif speed_px_s < THRESH_MEDIUM:
        return 2
    return 3


class AnalyticsState:
    """Track analytics state for a single video source."""

    __slots__ = (
        "width", "height", "fps", "frame_area", "track_positions",
        "traffic_density_ema", "mobility_index_ema",
        "fps_frame_count", "fps_last_time", "fps_value",
        "class_names"
    )

    def __init__(self, width: int, height: int, fps: float, class_names: dict = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_area = max(1.0, float(width * height))
        self.track_positions = {}
        self.traffic_density_ema = 0.0
        self.mobility_index_ema = 0.0
        self.fps_frame_count = 0
        self.fps_last_time = time.time()
        self.fps_value = 0.0
        self.class_names = class_names or {}

    def update(self, tracked_detections) -> dict:
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_last_time
        if elapsed >= 1.0:
            self.fps_value = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_last_time = time.time()

        if tracked_detections is None or tracked_detections.tracker_id is None:
            return self._build_metrics(0, [0, 0, 0, 0], 0.0, {})

        speed_counts = [0, 0, 0, 0]
        total_area = 0.0
        new_positions = {}
        class_counts = {}

        xyxys = tracked_detections.xyxy
        tids = tracked_detections.tracker_id
        class_ids = tracked_detections.class_id

        for i in range(len(tids)):
            x1, y1, x2, y2 = xyxys[i]
            tid = tids[i]
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            total_area += (x2 - x1) * (y2 - y1)

            if class_ids is not None:
                cls_id = int(class_ids[i])
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            speed_px_s = 0.0
            if tid in self.track_positions:
                old_cx, old_cy, old_speed = self.track_positions[tid]
                dx, dy = cx - old_cx, cy - old_cy
                raw_speed = np.sqrt(dx * dx + dy * dy) * self.fps
                speed_px_s = old_speed * 0.4 + raw_speed * 0.6 if old_speed > 0 else raw_speed

            new_positions[tid] = (cx, cy, speed_px_s)
            speed_counts[classify_speed(speed_px_s)] += 1

        self.track_positions = new_positions
        return self._build_metrics(len(tracked_detections), speed_counts, total_area, class_counts)

    def _build_metrics(self, detection_count: int, speed_counts: list, total_area: float, class_counts: dict = None) -> dict:
        total = max(1, sum(speed_counts))
        stalled_pct = round(speed_counts[0] * 100 // total)
        slow_pct = round(speed_counts[1] * 100 // total)
        medium_pct = round(speed_counts[2] * 100 // total)
        fast_pct = round(speed_counts[3] * 100 // total)

        density = min(100.0, (total_area / self.frame_area) * 650.0)
        if self.traffic_density_ema > 0:
            self.traffic_density_ema = self.traffic_density_ema * 0.8 + density * 0.2
        else:
            self.traffic_density_ema = density

        mobility = 0.0
        if total > 0:
            mobility = (speed_counts[0] * 10 + speed_counts[1] * 40 +
                        speed_counts[2] * 70 + speed_counts[3] * 95) / total
        if self.mobility_index_ema > 0:
            self.mobility_index_ema = self.mobility_index_ema * 0.8 + mobility * 0.2
        else:
            self.mobility_index_ema = mobility

        t_density = int(self.traffic_density_ema + 0.5)
        m_index = int(self.mobility_index_ema + 0.5)
        congestion = min(100, max(0, int(t_density * 0.6 + (100 - m_index) * 0.4 + 0.5)))

        return {
            "fps": round(self.fps_value, 1),
            "congestion_index": congestion,
            "traffic_density": t_density,
            "mobility_index": m_index,
            "stalled_pct": stalled_pct,
            "slow_pct": slow_pct,
            "medium_pct": medium_pct,
            "fast_pct": fast_pct,
            "detection_count": detection_count,
            "class_counts": class_counts or {},
        }


def _setup_child_logging(name):
    """Redirect child process stdout/stderr to a log file so we can debug crashes."""
    import sys
    log_path = f"/tmp/iris_child_{name}.log"
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    print(f"[CHILD] {name}: logging to {log_path}")


class RTSPPublisher:
    """Publish processed frames to MediaMTX via FFmpeg RTSP push.
    Uses NVENC if available, falls back to libx264 ultrafast.
    Encodes with baseline profile for browser/HLS/WebRTC compatibility."""

    def __init__(self, name, width, height, fps=25):
        self.name = name
        self.alive = False
        self.proc = None
        url = f"rtsp://127.0.0.1:8554/processed_{name}"

        base_input = [
            "ffmpeg",
            "-y",
            "-loglevel", "warning",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(int(fps)),
            "-i", "-",
        ]
        base_output = ["-f", "rtsp", "-rtsp_transport", "tcp", url]

        # Try NVENC first with a quick probe
        nvenc_ok = False
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                 "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, text=True, timeout=5
            )
            nvenc_ok = r.returncode == 0
        except:
            pass

        codecs_to_try = []
        if nvenc_ok:
            codecs_to_try.append(("h264_nvenc", [
                "-pix_fmt", "yuv420p",
                "-c:v", "h264_nvenc",
                "-preset", "p1",
                "-tune", "ll",
                "-profile:v", "baseline",
                "-level", "3.1",
                "-rc", "cbr",
                "-b:v", "2M",
                "-maxrate", "2M",
                "-bufsize", "1M",
                "-g", str(fps),
                "-bf", "0",
            ]))

        # Always have libx264 as reliable fallback
        # -pix_fmt yuv420p required: bgr24 input is 4:4:4, baseline needs 4:2:0
        codecs_to_try.append(("libx264", [
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-b:v", "2M",
            "-g", str(fps),
            "-bf", "0",
        ]))

        for codec_name, codec_args in codecs_to_try:
            cmd = base_input + codec_args + base_output
            print(f"[RTSP-PUB] {name}: trying {codec_name} baseline @ {width}x{height}")
            try:
                self.proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
                # Give FFmpeg a moment to fail on init
                import time as _time
                _time.sleep(0.5)
                if self.proc.poll() is not None:
                    # Process already exited — codec failed
                    stderr_out = self.proc.stderr.read().decode("utf-8", errors="replace")
                    print(f"[RTSP-PUB] {name}: {codec_name} failed: {stderr_out.strip()[:200]}")
                    self.proc = None
                    continue
                # Success
                self.alive = True
                self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
                self._stderr_thread.start()
                print(f"[RTSP-PUB] {name}: publishing to {url} via {codec_name}")
                break
            except Exception as e:
                print(f"[RTSP-PUB] {name}: {codec_name} spawn error: {e}")
                self.proc = None
                continue

        if not self.alive:
            print(f"[RTSP-PUB] {name}: ALL codecs failed, no publisher")

    def _read_stderr(self):
        """Read FFmpeg stderr in background for debugging."""
        try:
            for line in self.proc.stderr:
                msg = line.decode("utf-8", errors="replace").strip()
                if msg:
                    print(f"[RTSP-PUB] {self.name} ffmpeg: {msg}")
        except:
            pass

    def write(self, frame):
        """Write a BGR numpy frame to the FFmpeg stdin."""
        if not self.alive or self.proc is None:
            return False
        try:
            self.proc.stdin.write(frame.tobytes())
            return True
        except (BrokenPipeError, OSError):
            self.alive = False
            return False

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except:
                pass
            try:
                self.proc.wait(timeout=3)
            except:
                self.proc.kill()
            self.alive = False
            print(f"[RTSP-PUB] {self.name}: closed")


class FrameCapture:
    """Threaded frame capture with corrupt frame detection."""

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        self.frame_ready = threading.Event()
        self.consecutive_failures = 0
        if self.cap.isOpened():
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _is_corrupt(self, frame):
        if frame is None or frame.size == 0:
            return True
        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            return True
        # Fast check: sample 4 pixels at corners — if all identical, likely corrupt
        tl = frame[2, 2]
        tr = frame[2, w - 3]
        bl = frame[h - 3, 2]
        br = frame[h - 3, w - 3]
        if np.array_equal(tl, tr) and np.array_equal(bl, br) and np.array_equal(tl, bl):
            return True
        return False

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and not self._is_corrupt(frame):
                with self.lock:
                    self.ret, self.frame = True, frame
                    self.consecutive_failures = 0
                self.frame_ready.set()
                time.sleep(0.001)
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures > 30:
                    time.sleep(0.5)
                else:
                    time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        self.cap.release()

    def get(self, prop):
        return self.cap.get(prop)

    def isOpened(self):
        return self.cap.isOpened()


class TrailRenderer:
    """Efficient trail rendering with grounded animal-tail style."""

    def __init__(self, max_len=20):
        self.trails = {}
        self.max_len = max_len

    def update(self, tracked_detections):
        current_ids = set()

        if tracked_detections is not None and tracked_detections.tracker_id is not None:
            xyxys = tracked_detections.xyxy
            tids = tracked_detections.tracker_id

            for i in range(len(tids)):
                x1, y1, x2, y2 = xyxys[i]
                tid = tids[i]
                gx = int((x1 + x2) * 0.5)
                gy = int(y2)
                current_ids.add(tid)

                if tid not in self.trails:
                    self.trails[tid] = []
                trail = self.trails[tid]
                trail.append((gx, gy))
                if len(trail) > self.max_len:
                    trail.pop(0)

        stale = []
        for tid in self.trails:
            if tid not in current_ids:
                trail = self.trails[tid]
                if trail:
                    trail.pop(0)
                if not trail:
                    stale.append(tid)
        for tid in stale:
            del self.trails[tid]

        return current_ids

    def render(self, frame, current_ids):
        for tid, trail in self.trails.items():
            if len(trail) < 2:
                continue
            n = len(trail)
            for i in range(n - 1):
                progress = (i + 1) / n
                thickness = max(1, int(1 + 2 * progress))
                # Purple-to-cyan gradient trail
                r = int(168 * (1 - progress) + 0 * progress)
                g = int(85 * (1 - progress) + 255 * progress)
                b = int(247 * (1 - progress) + 255 * progress)
                alpha = 0.3 + 0.7 * progress
                color = (int(b * alpha), int(g * alpha), int(r * alpha))
                cv2.line(frame, trail[i], trail[i + 1], color, thickness, cv2.LINE_AA)


class HeatmapRenderer:
    """Per-vehicle JET heatmap tail — fat rocket-exhaust style.
    Blue at the far tail → cyan → green → yellow → red near vehicle.
    Drawn as continuous fat filled polylines, lightly blurred for glow."""

    def __init__(self, max_len=40):
        self.trails = {}   # tid -> list of (cx, bot_y, half_w)
        self.max_len = max_len
        self._jet_lut = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_JET
        ).reshape(256, 3)

    def update(self, tracked_detections):
        current_ids = set()

        if tracked_detections is not None and tracked_detections.tracker_id is not None:
            xyxys = tracked_detections.xyxy
            tids = tracked_detections.tracker_id

            for i in range(len(tids)):
                x1, y1, x2, y2 = xyxys[i]
                tid = tids[i]
                current_ids.add(tid)
                cx = int((x1 + x2) * 0.5)
                bot_y = int(y2)
                half_w = max(int((x2 - x1) * 0.6), 6)

                if tid not in self.trails:
                    self.trails[tid] = []
                trail = self.trails[tid]
                trail.append((cx, bot_y, half_w))
                if len(trail) > self.max_len:
                    trail.pop(0)

        stale = []
        for tid in self.trails:
            if tid not in current_ids:
                trail = self.trails[tid]
                if trail:
                    trail.pop(0)
                if not trail:
                    stale.append(tid)
        for tid in stale:
            del self.trails[tid]

        return current_ids

    def render(self, frame, current_ids):
        h, w = frame.shape[:2]
        jet = self._jet_lut

        # Draw heat trails on a SEPARATE black canvas (don't touch the frame yet)
        heat = np.zeros_like(frame)
        has_heat = False

        for tid, trail in self.trails.items():
            n = len(trail)
            if n < 3:
                continue
            has_heat = True

            # Draw segments as fat filled quads from tail to vehicle
            for i in range(n - 1):
                p0 = i / max(n - 1, 1)
                p1 = (i + 1) / max(n - 1, 1)

                cx0, by0, hw0 = trail[i]
                cx1, by1, hw1 = trail[i + 1]

                # Width: VERY fat — 1.5x half-width, min 8px
                w0 = max(int(hw0 * (0.4 + 0.6 * p0) * 1.5), 8)
                w1 = max(int(hw1 * (0.4 + 0.6 * p1) * 1.5), 8)

                # JET color
                p_mid = (p0 + p1) * 0.5
                jet_idx = min(255, max(0, int(p_mid * 240 + 10)))
                color = (int(jet[jet_idx][0]), int(jet[jet_idx][1]), int(jet[jet_idx][2]))

                # Filled quad
                pts = np.array([
                    [cx0 - w0, by0],
                    [cx0 + w0, by0],
                    [cx1 + w1, by1],
                    [cx1 - w1, by1],
                ], dtype=np.int32)
                cv2.fillConvexPoly(heat, pts, color, cv2.LINE_AA)

            # Bright red head at vehicle position
            cx_head, by_head, hw_head = trail[-1]
            head_r = max(int(hw_head * 1.2), 8)
            cv2.circle(heat, (cx_head, by_head), head_r,
                       (int(jet[250][0]), int(jet[250][1]), int(jet[250][2])), -1, cv2.LINE_AA)

        if not has_heat:
            return

        # Blur ONLY the heat canvas — keeps the video sharp
        heat = cv2.GaussianBlur(heat, (21, 21), 0)

        # Composite: where heat is non-zero, blend it onto the frame
        # Create mask from heat brightness
        gray = cv2.cvtColor(heat, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

        # Dilate mask slightly for softer edges
        mask = cv2.GaussianBlur(mask, (11, 11), 0)

        # Alpha blend: heat over frame using mask
        mask_f = mask.astype(np.float32) / 255.0 * 0.75  # max 75% opacity
        mask_3 = mask_f[:, :, np.newaxis]
        blended = frame.astype(np.float32) * (1.0 - mask_3) + heat.astype(np.float32) * mask_3
        np.copyto(frame, blended.astype(np.uint8))


class BboxSmoother:
    """EMA-smooth bounding box positions per track ID to reduce jitter."""

    def __init__(self, alpha=BBOX_SMOOTH_ALPHA):
        self.alpha = alpha
        self.smoothed = {}

    def smooth(self, tracked_detections):
        if tracked_detections is None or tracked_detections.tracker_id is None:
            self.smoothed.clear()
            return None

        xyxys = tracked_detections.xyxy
        tids = tracked_detections.tracker_id
        smoothed_xyxy = xyxys.copy().astype(np.float64)

        active_ids = set()
        for i in range(len(tids)):
            tid = tids[i]
            active_ids.add(tid)
            raw = xyxys[i].astype(np.float64)

            if tid in self.smoothed:
                prev = self.smoothed[tid]
                s = prev * (1.0 - self.alpha) + raw * self.alpha
                self.smoothed[tid] = s
                smoothed_xyxy[i] = s
            else:
                self.smoothed[tid] = raw
                smoothed_xyxy[i] = raw

        stale = [t for t in self.smoothed if t not in active_ids]
        for t in stale:
            del self.smoothed[t]

        return smoothed_xyxy


def _render_overlays(frame, overlay, tracked, trail_renderer, heatmap_renderer, bbox_smoother, current_ids, CLASS_NAMES):
    """Render enabled overlays onto the frame. Modifies frame in-place."""
    if overlay.get("heatmap", False):
        heatmap_renderer.render(frame, current_ids)

    if overlay.get("trails", False):
        trail_renderer.render(frame, current_ids)

    if overlay.get("bboxes", False) and tracked is not None and tracked.tracker_id is not None:
        smoothed_xyxy = bbox_smoother.smooth(tracked)
        tids = tracked.tracker_id
        class_ids = tracked.class_id

        for i in range(len(tids)):
            x1, y1, x2, y2 = map(int, smoothed_xyxy[i])
            tid = tids[i]

            # Color by class for visual distinction
            cls_id = int(class_ids[i]) if class_ids is not None else 0
            colors = {
                3: (0, 255, 200),   # car - cyan-green
                4: (0, 200, 255),   # van - sky blue
                5: (80, 127, 255),  # truck - orange-red
                7: (255, 180, 0),   # bus - blue
                8: (200, 0, 255),   # motor - magenta
                9: (0, 255, 100),   # bicycle - green
                0: (0, 255, 255),   # head (crowd) - yellow
            }
            color = colors.get(cls_id, (0, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Compact label — class name only, small tag
            cls_name = CLASS_NAMES.get(cls_id, "")
            if cls_name:
                label = cls_name.upper()
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
                ly = y1 - 2 if y1 > 16 else y2 + th + 4
                lx = x1
                cv2.rectangle(frame, (lx, ly - th - 2), (lx + tw + 4, ly + 2), color, -1)
                cv2.putText(frame, label, (lx + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        bbox_smoother.smooth(tracked)


def process_stream(index, name, url, stop_event, f_q, m_q, a_q, rf_q, overlay_dict):
    """Process RTSP stream: YOLO for metrics + overlays published via RTSP to MediaMTX."""
    _setup_child_logging(name)
    print(f"[+] Starting optimized inference: {name}")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    cap = FrameCapture(url)
    if not cap.isOpened():
        print(f"[!] Failed to open source: {name}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Load model
    device = "cpu"
    half_precision = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(YOLO_GPU_MEMORY_FRACTION)
        device = "cuda"
        half_precision = True
        torch.backends.cudnn.benchmark = True

    is_crowd = name in CROWD_DRONES
    model_path = MODEL_PATH_CROWD if is_crowd else MODEL_PATH_VEHICLE

    model = YOLO(model_path, task="detect")
    if device == "cuda":
        model.to(device)

    target_classes = None if is_crowd else [3, 4, 5, 7, 8, 9]
    CLASS_NAMES = {0: "head"} if is_crowd else {
        3: "car", 4: "van", 5: "truck", 7: "bus", 8: "motor", 9: "bicycle"
    }
    conf_thresh = 0.25 if is_crowd else 0.15

    tracker = sv.ByteTrack(
        frame_rate=int(src_fps),
        track_activation_threshold=0.2,
        lost_track_buffer=45,
        minimum_matching_threshold=0.75,
        minimum_consecutive_frames=3,
    )
    trail_renderer = TrailRenderer(max_len=20)
    heatmap_renderer = HeatmapRenderer(max_len=18)
    bbox_smoother = BboxSmoother(alpha=BBOX_SMOOTH_ALPHA)
    analytics = AnalyticsState(w, h, src_fps, CLASS_NAMES)

    # Read overlay config once — fixed per mode, not changeable
    overlay = {"heatmap": False, "trails": False, "bboxes": False}
    try:
        if name in overlay_dict:
            raw = overlay_dict[name]
            if isinstance(raw, dict):
                overlay = {
                    "heatmap": raw.get("heatmap", False),
                    "trails": raw.get("trails", False),
                    "bboxes": raw.get("bboxes", False),
                }
    except:
        pass

    has_any_overlay = overlay.get("heatmap") or overlay.get("trails") or overlay.get("bboxes")
    is_forensics = not has_any_overlay
    print(f"[+] {name}: model loaded, overlay={overlay}, forensics={is_forensics}")

    # Start RTSP publisher for processed frames (non-forensics modes)
    publisher = None
    if not is_forensics:
        publisher = RTSPPublisher(name, w, h, fps=min(int(src_fps), TARGET_FPS))

    last_metrics_time = time.time()
    frame_count = 0
    fps_times = []
    actual_fps = 0.0

    print(f"[+] {name}: entering main loop, cap.isOpened={cap.isOpened()}, stop={stop_event.is_set()}")

    try:
        while cap.isOpened() and not stop_event.is_set():
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1

            if SKIP_FRAMES > 1 and frame_count % SKIP_FRAMES != 0:
                continue

            fps_times.append(frame_start)
            if len(fps_times) > 30:
                fps_times.pop(0)
            if len(fps_times) >= 2:
                elapsed = fps_times[-1] - fps_times[0]
                if elapsed > 0:
                    actual_fps = (len(fps_times) - 1) / elapsed

            # YOLO inference
            results = model.predict(
                frame,
                imgsz=INFERENCE_SIZE,
                conf=conf_thresh,
                max_det=MAX_DET,
                verbose=False,
                classes=target_classes,
                half=half_precision and device == "cuda",
                device=device,
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            tracked = tracker.update_with_detections(detections)

            # Update trail/heatmap state
            current_ids = trail_renderer.update(tracked)
            heatmap_renderer.update(tracked)

            # Metrics (throttled)
            now = time.time()
            if now - last_metrics_time >= 0.25:
                metrics = analytics.update(tracked)
                metrics["fps"] = round(actual_fps, 1)
                try:
                    if not m_q.full():
                        m_q.put_nowait((name, metrics))
                except:
                    pass
                last_metrics_time = now

                congestion = metrics.get("congestion_index", 0)
                if congestion >= CONGESTION_ALERT_THRESHOLD:
                    try:
                        _, screenshot_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not a_q.full():
                            a_q.put_nowait((name, congestion, metrics, screenshot_buf.tobytes()))
                    except:
                        pass

            if is_forensics:
                # Forensics: raw frames for SAM via queue (no publisher)
                ret_enc, buffer = cv2.imencode(".jpg", frame, encode_params)
                if ret_enc:
                    data = buffer.tobytes()
                    try:
                        if rf_q.full():
                            try: rf_q.get_nowait()
                            except: pass
                        rf_q.put_nowait((name, data))
                    except: pass
            else:
                # Render overlays and publish via RTSP (H.264 → WebRTC)
                _render_overlays(frame, overlay, tracked, trail_renderer, heatmap_renderer, bbox_smoother, current_ids, CLASS_NAMES)
                if publisher and publisher.alive:
                    publisher.write(frame)
    finally:
        if publisher:
            publisher.close()
        cap.release()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def start_backend(idx, url, name, overlay_config=None):
    """Start inference process for a source with overlay config from mode."""
    if overlay_config:
        overlay_shared_dict[name] = overlay_config
    else:
        overlay_shared_dict[name] = get_overlay_state(name)
    stop = spawn_ctx.Event()
    p = spawn_ctx.Process(
        target=process_stream,
        args=(idx, name, url, stop, frame_queue, metrics_queue, alert_queue, raw_frame_queue, overlay_shared_dict),
        daemon=True,
    )
    p.start()
    return p, stop


def start_upload_backend(file_path, name, overlay_config=None):
    """Start inference on an uploaded video file with mode-specific overlays."""
    if overlay_config:
        overlay_shared_dict[name] = overlay_config
    else:
        overlay_shared_dict[name] = get_overlay_state(name)
    stop = spawn_ctx.Event()
    p = spawn_ctx.Process(
        target=process_upload_stream,
        args=(name, file_path, stop, frame_queue, metrics_queue, alert_queue, raw_frame_queue, overlay_shared_dict),
        daemon=True,
    )
    p.start()
    return p, stop


def process_upload_stream(name, file_path, stop_event, f_q, m_q, a_q, rf_q, overlay_dict):
    """Process uploaded video: YOLO + overlays published via RTSP to MediaMTX."""
    _setup_child_logging(f"upload_{name}")
    print(f"[+] Starting upload inference: {name} from {file_path}")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"[!] Failed to open uploaded file: {file_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Load model
    device = "cpu"
    half_precision = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(YOLO_GPU_MEMORY_FRACTION)
        device = "cuda"
        half_precision = True
        torch.backends.cudnn.benchmark = True

    model = YOLO(MODEL_PATH_VEHICLE, task="detect")
    if device == "cuda":
        model.to(device)

    target_classes = [3, 4, 5, 7, 8, 9]
    CLASS_NAMES = {3: "car", 4: "van", 5: "truck", 7: "bus", 8: "motor", 9: "bicycle"}
    conf_thresh = 0.15

    tracker = sv.ByteTrack(
        frame_rate=int(src_fps),
        track_activation_threshold=0.2,
        lost_track_buffer=45,
        minimum_matching_threshold=0.75,
        minimum_consecutive_frames=3,
    )
    trail_renderer = TrailRenderer(max_len=20)
    heatmap_renderer = HeatmapRenderer(max_len=18)
    bbox_smoother = BboxSmoother(alpha=BBOX_SMOOTH_ALPHA)
    analytics = AnalyticsState(w, h, src_fps, CLASS_NAMES)

    # Read overlay config once — fixed per mode
    overlay = {"heatmap": False, "trails": False, "bboxes": False}
    try:
        if name in overlay_dict:
            raw = overlay_dict[name]
            if isinstance(raw, dict):
                overlay = {
                    "heatmap": raw.get("heatmap", False),
                    "trails": raw.get("trails", False),
                    "bboxes": raw.get("bboxes", False),
                }
    except:
        pass

    has_any_overlay = overlay.get("heatmap") or overlay.get("trails") or overlay.get("bboxes")
    is_forensics = not has_any_overlay
    print(f"[+] {name}: model loaded, overlay={overlay}, forensics={is_forensics}")

    # Start RTSP publisher for processed frames
    publisher = RTSPPublisher(name, w, h, fps=min(int(src_fps), TARGET_FPS))

    last_metrics_time = time.time()
    frame_interval = 1.0 / src_fps
    fps_times = []
    actual_fps = 0.0

    try:
        while not stop_event.is_set():
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker = sv.ByteTrack(frame_rate=int(src_fps))
                trail_renderer = TrailRenderer(max_len=20)
                heatmap_renderer = HeatmapRenderer(max_len=24)
                continue

            fps_times.append(frame_start)
            if len(fps_times) > 30:
                fps_times.pop(0)
            if len(fps_times) >= 2:
                elapsed = fps_times[-1] - fps_times[0]
                if elapsed > 0:
                    actual_fps = (len(fps_times) - 1) / elapsed

            # YOLO inference
            results = model.predict(
                frame,
                imgsz=INFERENCE_SIZE,
                conf=conf_thresh,
                max_det=MAX_DET,
                verbose=False,
                classes=target_classes,
                half=half_precision and device == "cuda",
                device=device,
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            tracked = tracker.update_with_detections(detections)

            current_ids = trail_renderer.update(tracked)
            heatmap_renderer.update(tracked)

            # Metrics (throttled)
            now = time.time()
            if now - last_metrics_time >= 0.25:
                metrics = analytics.update(tracked)
                metrics["fps"] = round(actual_fps, 1)
                try:
                    if not m_q.full():
                        m_q.put_nowait((name, metrics))
                except:
                    pass
                last_metrics_time = now

                congestion = metrics.get("congestion_index", 0)
                if congestion >= CONGESTION_ALERT_THRESHOLD:
                    try:
                        _, screenshot_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not a_q.full():
                            a_q.put_nowait((name, congestion, metrics, screenshot_buf.tobytes()))
                    except:
                        pass

            if is_forensics:
                # Raw frames for SAM via queue
                ret_enc, buffer = cv2.imencode(".jpg", frame, encode_params)
                if ret_enc:
                    data = buffer.tobytes()
                    try:
                        if rf_q.full():
                            try: rf_q.get_nowait()
                            except: pass
                        rf_q.put_nowait((name, data))
                    except: pass
                # Also publish raw via RTSP for display
                if publisher.alive:
                    publisher.write(frame)
            else:
                # Render overlays, publish via RTSP
                _render_overlays(frame, overlay, tracked, trail_renderer, heatmap_renderer, bbox_smoother, current_ids, CLASS_NAMES)
                if publisher.alive:
                    publisher.write(frame)

            elapsed = time.time() - frame_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        publisher.close()
        cap.release()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[+] Upload inference stopped: {name}")


def main():
    global spawn_ctx, overlay_shared_dict, frame_queue, raw_frame_queue, metrics_queue, alert_queue
    ctx = mp.get_context("spawn")
    spawn_ctx = ctx

    manager = ctx.Manager()
    overlay_shared_dict = manager.dict()
    frame_queue = ctx.Queue(maxsize=60)
    raw_frame_queue = ctx.Queue(maxsize=30)
    metrics_queue = ctx.Queue(maxsize=20)
    alert_queue = ctx.Queue(maxsize=10)

    stop_relay = threading.Event()
    relay_t = threading.Thread(target=relay_worker, args=(stop_relay, frame_queue, metrics_queue, alert_queue, raw_frame_queue), daemon=True)
    relay_t.start()

    try:
        run_control_server(start_backend, start_upload_backend, overlay_shared_dict)
    finally:
        stop_relay.set()
        relay_t.join(timeout=1.0)


if __name__ == "__main__":
    main()
