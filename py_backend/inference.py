import os
import cv2
import time
import threading
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
        try:
            for _ in range(30):
                if f_q.empty():
                    break
                name, data = f_q.get_nowait()
                update_frame(name, data)
        except:
            pass

        # Relay raw frames for SAM
        try:
            for _ in range(30):
                if rf_q.empty():
                    break
                name, data = rf_q.get_nowait()
                update_raw_frame(name, data)
        except:
            pass

        try:
            for _ in range(10):
                if m_q.empty():
                    break
                name, data = m_q.get_nowait()
                update_metrics(name, data)
        except:
            pass

        # Process alert queue
        try:
            for _ in range(5):
                if a_q.empty():
                    break
                source, congestion, metrics_data, screenshot = a_q.get_nowait()
                add_alert(source, congestion, metrics_data, screenshot)
        except:
            pass

        time.sleep(0.002)


# Optimized RTSP settings - lower buffer for real-time
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;262144|analyzeduration;50000|probesize;50000|fflags;nobuffer|flags;low_delay"
)

MODEL_PATH_VEHICLE = "data/yolov11n-visdrone.pt"
MODEL_PATH_CROWD = "data/best_head.pt"
TARGET_FPS = 30

# Inference settings
INFERENCE_SIZE = 320  # Smaller inference = less GPU memory + faster
SKIP_FRAMES = 1  # Process every frame for smooth tracking
JPEG_QUALITY = 70  # Lower = faster encoding
MAX_DET = 100  # Max detections per frame
BBOX_SMOOTH_ALPHA = 0.45  # EMA smoothing for bbox positions (lower = smoother, higher = snappier)

# GPU memory budget: cap each YOLO process so SAM3 has room
YOLO_GPU_MEMORY_FRACTION = 0.35  # Each YOLO process gets at most 35% of GPU

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
        "fps_frame_count", "fps_last_time", "fps_value"
    )

    def __init__(self, width: int, height: int, fps: float):
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

    def update(self, tracked_detections) -> dict:
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_last_time
        if elapsed >= 1.0:
            self.fps_value = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_last_time = time.time()

        if tracked_detections is None or tracked_detections.tracker_id is None:
            return self._build_metrics(0, [0, 0, 0, 0], 0.0)

        speed_counts = [0, 0, 0, 0]
        total_area = 0.0
        new_positions = {}

        xyxys = tracked_detections.xyxy
        tids = tracked_detections.tracker_id

        for i in range(len(tids)):
            x1, y1, x2, y2 = xyxys[i]
            tid = tids[i]
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            total_area += (x2 - x1) * (y2 - y1)

            speed_px_s = 0.0
            if tid in self.track_positions:
                old_cx, old_cy, old_speed = self.track_positions[tid]
                dx, dy = cx - old_cx, cy - old_cy
                raw_speed = np.sqrt(dx * dx + dy * dy) * self.fps
                speed_px_s = old_speed * 0.4 + raw_speed * 0.6 if old_speed > 0 else raw_speed

            new_positions[tid] = (cx, cy, speed_px_s)
            speed_counts[classify_speed(speed_px_s)] += 1

        self.track_positions = new_positions
        return self._build_metrics(len(tracked_detections), speed_counts, total_area)

    def _build_metrics(self, detection_count: int, speed_counts: list, total_area: float) -> dict:
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
        }


class FrameCapture:
    """Threaded frame capture with corrupt frame detection."""

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        # Reduce buffer size for lower latency
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
        """Fast corrupt frame detection."""
        if frame is None:
            return True
        # Check if frame is mostly black or has invalid values
        if frame.size == 0:
            return True
        # Quick check: sample a few pixels
        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            return True
        # Check for all-zero or all-same frames (corrupt)
        sample = frame[::h // 4, ::w // 4]
        if sample.std() < 1.0:  # Nearly uniform = likely corrupt
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
                    # Try to reconnect
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
        self.trails = {}  # tid -> list of (x, y)
        self.max_len = max_len

    def update(self, tracked_detections):
        """Update trails with new detections - attached to bottom of bbox (grounded)."""
        current_ids = set()

        if tracked_detections is not None and tracked_detections.tracker_id is not None:
            xyxys = tracked_detections.xyxy
            tids = tracked_detections.tracker_id

            for i in range(len(tids)):
                x1, y1, x2, y2 = xyxys[i]
                tid = tids[i]
                # Attach to BOTTOM-CENTER of bbox (grounded, like animal feet/wheels)
                gx = int((x1 + x2) * 0.5)
                gy = int(y2)  # Bottom edge - grounded
                current_ids.add(tid)

                if tid not in self.trails:
                    self.trails[tid] = []
                trail = self.trails[tid]
                trail.append((gx, gy))
                if len(trail) > self.max_len:
                    trail.pop(0)

        # Fade out old trails gradually
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
        """Render thin grounded trails like animal tails dragging behind."""
        for tid, trail in self.trails.items():
            if len(trail) < 2:
                continue

            n = len(trail)

            # Draw thin gradient trail - fades from tail to head
            for i in range(n - 1):
                # Progress: 0 at oldest (tail tip), 1 at newest (attached to object)
                progress = (i + 1) / n

                # Color: faint cyan at tail -> bright cyan at head
                intensity = int(80 + 175 * progress)
                color = (intensity, int(200 * progress + 55), int(180 * progress + 75))

                # 1px thin like an animal tail
                cv2.line(frame, trail[i], trail[i + 1], color, 1, cv2.LINE_AA)


class HeatmapRenderer:
    """Renders thin grounded heat trails on the road behind vehicles - like tire marks."""

    def __init__(self, max_len=20):
        self.heat_trails = {}  # tid -> list of (x, y, radius)
        self.max_len = max_len

    def update(self, tracked_detections):
        """Update heat trails - grounded at bottom of bbox (road level)."""
        current_ids = set()

        if tracked_detections is not None and tracked_detections.tracker_id is not None:
            xyxys = tracked_detections.xyxy
            tids = tracked_detections.tracker_id

            for i in range(len(tids)):
                x1, y1, x2, y2 = xyxys[i]
                tid = tids[i]
                current_ids.add(tid)

                # GROUNDED: bottom-center of bbox (where wheels touch road)
                gx = int((x1 + x2) * 0.5)
                gy = int(y2)  # Bottom edge - on the road

                # Thin radius for the heat trail
                r = int(min(x2 - x1, y2 - y1) * 0.25)

                if tid not in self.heat_trails:
                    self.heat_trails[tid] = []
                trail = self.heat_trails[tid]
                trail.append((gx, gy, r))
                if len(trail) > self.max_len:
                    trail.pop(0)

        # Fade out old trails gradually
        stale = []
        for tid in self.heat_trails:
            if tid not in current_ids:
                trail = self.heat_trails[tid]
                if trail:
                    trail.pop(0)
                if not trail:
                    stale.append(tid)
        for tid in stale:
            del self.heat_trails[tid]

        return current_ids

    def render(self, frame, current_ids):
        """Render thin grounded heat trails using COLORMAP_JET."""
        h, w = frame.shape[:2]

        # Use scaled-down mask for performance
        scale = 0.5
        sh, sw = int(h * scale), int(w * scale)
        heat_mask = np.zeros((sh, sw), dtype=np.uint8)

        for tid, pts in self.heat_trails.items():
            if tid not in current_ids:
                continue
            if len(pts) < 2:
                continue

            # Use trail points (excluding newest to keep heat behind vehicle)
            sub_pts = pts[:-1] if len(pts) > 2 else pts
            n = len(sub_pts)
            if n < 2:
                continue

            for i in range(n):
                # Scale coordinates
                tx = int(sub_pts[i][0] * scale)
                ty = int(sub_pts[i][1] * scale)
                tr = int(sub_pts[i][2] * scale)

                # Phase: 0 at oldest (tail), 1 at newest (near vehicle)
                phase = (i + 1) / n

                # Intensity increases towards vehicle (40 to 230)
                intensity = int(40 + 190 * phase)

                # Thin radius - smaller near tail, slightly larger near vehicle
                curr_r = max(2, int(tr * (0.5 + 0.5 * phase)))

                # Draw small circle
                cv2.circle(heat_mask, (tx, ty), curr_r, intensity, -1)

                # Connect with thin lines for smooth trail
                if i > 0:
                    px = int(sub_pts[i - 1][0] * scale)
                    py = int(sub_pts[i - 1][1] * scale)
                    pr = int(sub_pts[i - 1][2] * scale)

                    prev_phase = i / n
                    prev_intensity = int(40 + 190 * prev_phase)
                    line_intensity = (intensity + prev_intensity) // 2

                    prev_r = max(2, int(pr * (0.5 + 0.5 * prev_phase)))
                    line_thick = max(1, (curr_r + prev_r) // 2)

                    cv2.line(heat_mask, (px, py), (tx, ty), line_intensity, line_thick)

        # Apply colormap and blend if there's any heat
        if np.any(heat_mask > 0):
            # Gaussian blur for smooth edges (smaller kernel for thinner look)
            blur = cv2.GaussianBlur(heat_mask, (7, 7), 0)

            # Apply JET colormap (blue -> green -> yellow -> red)
            colored = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

            # Scale back up to full resolution
            colored_full = cv2.resize(colored, (w, h), interpolation=cv2.INTER_LINEAR)
            blur_full = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)

            # Create mask where heat exists
            mask = blur_full > 10

            # Blend with good intensity
            frame[mask] = cv2.addWeighted(frame[mask], 0.45, colored_full[mask], 0.55, 0)


class BboxSmoother:
    """EMA-smooth bounding box positions per track ID to reduce jitter."""

    def __init__(self, alpha=BBOX_SMOOTH_ALPHA):
        self.alpha = alpha
        self.smoothed = {}  # tid -> [x1, y1, x2, y2] as floats

    def smooth(self, tracked_detections):
        """Smooth bbox positions in-place on the detections object. Returns smoothed xyxy copy."""
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

        # Remove stale tracks
        stale = [t for t in self.smoothed if t not in active_ids]
        for t in stale:
            del self.smoothed[t]

        return smoothed_xyxy


def process_stream(index, name, url, stop_event, f_q, m_q, a_q, rf_q, overlay_dict):
    print(f"[+] Starting optimized inference: {name}")

    # Pre-allocate JPEG encoding params
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    # ── 1. Connect to RTSP immediately and start pushing raw frames ──
    cap = FrameCapture(url)
    if not cap.isOpened():
        print(f"[!] Failed to open source: {name}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Push raw frames while model loads so the feed shows instantly
    print(f"[+] {name}: streaming raw frames while model loads...")
    raw_frame_count = 0
    while not stop_event.is_set() and raw_frame_count < 200:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue
        raw_frame_count += 1
        # Push every 3rd raw frame to keep the feed alive without flooding
        if raw_frame_count % 3 == 0:
            ret_enc, buffer = cv2.imencode(".jpg", frame, encode_params)
            if ret_enc:
                try:
                    if f_q.full():
                        try: f_q.get_nowait()
                        except: pass
                    f_q.put_nowait((name, buffer.tobytes()))
                except: pass
        # Break out early once we've pushed a few frames and can start loading model
        if raw_frame_count >= 6:
            break

    # ── 2. Load model (frames already visible in frontend) ──
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
    heatmap_renderer = HeatmapRenderer(max_len=12)
    bbox_smoother = BboxSmoother(alpha=BBOX_SMOOTH_ALPHA)
    analytics = AnalyticsState(w, h, src_fps)

    print(f"[+] {name}: model loaded, switching to processed frames")

    last_metrics_time = time.time()
    frame_count = 0

    # For accurate FPS calculation with smoothing
    fps_times = []
    actual_fps = 0.0

    # Cache for overlay state - refresh periodically
    cached_overlay = {"heatmap": True, "trails": True, "bboxes": True}
    last_overlay_check = 0

    # ── 3. Main inference loop ──
    while cap.isOpened() and not stop_event.is_set():
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1

        # Skip frames for performance (process every Nth)
        if SKIP_FRAMES > 1 and frame_count % SKIP_FRAMES != 0:
            continue

        # Accurate FPS calculation using rolling window
        fps_times.append(frame_start)
        # Keep last 30 frame times for smoothing
        if len(fps_times) > 30:
            fps_times.pop(0)
        if len(fps_times) >= 2:
            elapsed = fps_times[-1] - fps_times[0]
            if elapsed > 0:
                actual_fps = (len(fps_times) - 1) / elapsed

        # Get overlay state - refresh every 50ms for real-time toggle response
        if frame_start - last_overlay_check > 0.05:
            last_overlay_check = frame_start
            try:
                # Read from Manager dict - use direct key access for latest value
                if name in overlay_dict:
                    raw = overlay_dict[name]
                    # Convert to regular dict (handles DictProxy and dict)
                    if isinstance(raw, dict):
                        cached_overlay = {
                            "heatmap": raw.get("heatmap", True),
                            "trails": raw.get("trails", True),
                            "bboxes": raw.get("bboxes", True),
                        }
                    else:
                        # Try to extract values from proxy
                        cached_overlay = {
                            "heatmap": bool(raw.get("heatmap", True)),
                            "trails": bool(raw.get("trails", True)),
                            "bboxes": bool(raw.get("bboxes", True)),
                        }
            except Exception as e:
                pass  # Keep cached value on error

        overlay = cached_overlay

        # Run inference at lower resolution for speed
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

        # Track detections
        detections = sv.Detections.from_ultralytics(results)
        tracked = tracker.update_with_detections(detections)

        # Update trails and heatmap
        current_ids = trail_renderer.update(tracked)
        heatmap_renderer.update(tracked)

        # Update metrics more frequently for responsive UI
        now = time.time()
        if now - last_metrics_time >= 0.2:
            metrics = analytics.update(tracked)
            metrics["fps"] = round(actual_fps, 1)
            try:
                if not m_q.full():
                    m_q.put_nowait((name, metrics))
            except:
                pass
            last_metrics_time = now

            # Check for high congestion and trigger alert
            congestion = metrics.get("congestion_index", 0)
            if congestion >= CONGESTION_ALERT_THRESHOLD:
                try:
                    # Encode current frame as screenshot for alert
                    _, screenshot_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not a_q.full():
                        a_q.put_nowait((name, congestion, metrics, screenshot_buf.tobytes()))
                except:
                    pass

        # Send raw frame (before overlays) for SAM forensics
        ret_raw, raw_buf = cv2.imencode(".jpg", frame, encode_params)
        if ret_raw:
            try:
                if rf_q.full():
                    try: rf_q.get_nowait()
                    except: pass
                rf_q.put_nowait((name, raw_buf.tobytes()))
            except: pass

        # Render overlays
        out = frame

        # Heatmap - localized thermal trails behind vehicles
        if overlay.get("heatmap", True):
            heatmap_renderer.render(out, current_ids)

        # Trails - thin animal-tail style
        if overlay.get("trails", True):
            trail_renderer.render(out, current_ids)

        # Bounding boxes - smoothed
        if overlay.get("bboxes", True) and tracked is not None and tracked.tracker_id is not None:
            smoothed_xyxy = bbox_smoother.smooth(tracked)
            tids = tracked.tracker_id
            class_ids = tracked.class_id

            for i in range(len(tids)):
                x1, y1, x2, y2 = map(int, smoothed_xyxy[i])
                tid = tids[i]

                # Thin yellow box
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)

                # Compact label
                cls_name = CLASS_NAMES.get(int(class_ids[i]), "") if class_ids is not None else ""
                label = f"#{tid}"
                if cls_name:
                    label += f" {cls_name}"

                # Label position
                ly = y1 - 4 if y1 > 15 else y2 + 12
                cv2.putText(out, label, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            bbox_smoother.smooth(tracked)  # Keep smoother state updated even if not rendering

        # Encode and send frame
        ret_enc, buffer = cv2.imencode(".jpg", out, encode_params)
        if ret_enc:
            try:
                # Non-blocking put, drop if full
                if f_q.full():
                    try:
                        f_q.get_nowait()
                    except:
                        pass
                f_q.put_nowait((name, buffer.tobytes()))
            except:
                pass

    cap.release()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def start_backend(idx, url, name):
    overlay_shared_dict[name] = get_overlay_state(name)
    stop = spawn_ctx.Event()
    p = spawn_ctx.Process(
        target=process_stream,
        args=(idx, name, url, stop, frame_queue, metrics_queue, alert_queue, raw_frame_queue, overlay_shared_dict),
        daemon=True,
    )
    p.start()
    return p, stop


def start_upload_backend(file_path, name):
    """Start inference on an uploaded video file."""
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
    """Process an uploaded video file with inference - loops the video."""
    print(f"[+] Starting upload inference: {name} from {file_path}")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    # ── 1. Open file and push raw frames instantly ──
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"[!] Failed to open uploaded file: {file_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Push raw frames while model loads
    print(f"[+] {name}: streaming raw frames while model loads...")
    for _ in range(6):
        if stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        ret_enc, buffer = cv2.imencode(".jpg", frame, encode_params)
        if ret_enc:
            try:
                if f_q.full():
                    try: f_q.get_nowait()
                    except: pass
                f_q.put_nowait((name, buffer.tobytes()))
            except: pass

    # ── 2. Load model (feed already visible) ──
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

    print(f"[+] {name}: model loaded, switching to processed frames")

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
    heatmap_renderer = HeatmapRenderer(max_len=12)
    bbox_smoother = BboxSmoother(alpha=BBOX_SMOOTH_ALPHA)
    analytics = AnalyticsState(w, h, src_fps)

    last_metrics_time = time.time()

    # FPS control - match source FPS
    frame_interval = 1.0 / src_fps
    fps_times = []
    actual_fps = 0.0

    # Cache for overlay state - refresh periodically
    cached_overlay = {"heatmap": True, "trails": True, "bboxes": True}
    last_overlay_check = 0

    while not stop_event.is_set():
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            # Loop the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracker = sv.ByteTrack(frame_rate=int(src_fps))  # Reset tracker
            trail_renderer = TrailRenderer(max_len=20)  # Reset trails
            heatmap_renderer = HeatmapRenderer(max_len=12)  # Reset heatmap
            continue

        # FPS calculation
        fps_times.append(frame_start)
        if len(fps_times) > 30:
            fps_times.pop(0)
        if len(fps_times) >= 2:
            elapsed = fps_times[-1] - fps_times[0]
            if elapsed > 0:
                actual_fps = (len(fps_times) - 1) / elapsed

        # Get overlay state - refresh every 50ms for real-time toggle response
        if frame_start - last_overlay_check > 0.05:
            last_overlay_check = frame_start
            try:
                # Read from Manager dict - use direct key access for latest value
                if name in overlay_dict:
                    raw = overlay_dict[name]
                    # Convert to regular dict (handles DictProxy and dict)
                    if isinstance(raw, dict):
                        cached_overlay = {
                            "heatmap": raw.get("heatmap", True),
                            "trails": raw.get("trails", True),
                            "bboxes": raw.get("bboxes", True),
                        }
                    else:
                        # Try to extract values from proxy
                        cached_overlay = {
                            "heatmap": bool(raw.get("heatmap", True)),
                            "trails": bool(raw.get("trails", True)),
                            "bboxes": bool(raw.get("bboxes", True)),
                        }
            except Exception:
                pass  # Keep cached value on error

        overlay = cached_overlay

        # Run inference
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

        # Track detections
        detections = sv.Detections.from_ultralytics(results)
        tracked = tracker.update_with_detections(detections)

        # Update trails and heatmap
        current_ids = trail_renderer.update(tracked)
        heatmap_renderer.update(tracked)

        # Update metrics
        now = time.time()
        if now - last_metrics_time >= 0.2:
            metrics = analytics.update(tracked)
            metrics["fps"] = round(actual_fps, 1)
            try:
                if not m_q.full():
                    m_q.put_nowait((name, metrics))
            except:
                pass
            last_metrics_time = now

            # Check for high congestion alert
            congestion = metrics.get("congestion_index", 0)
            if congestion >= CONGESTION_ALERT_THRESHOLD:
                try:
                    _, screenshot_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not a_q.full():
                        a_q.put_nowait((name, congestion, metrics, screenshot_buf.tobytes()))
                except:
                    pass

        # Send raw frame (before overlays) for SAM forensics
        ret_raw, raw_buf = cv2.imencode(".jpg", frame, encode_params)
        if ret_raw:
            try:
                if rf_q.full():
                    try: rf_q.get_nowait()
                    except: pass
                rf_q.put_nowait((name, raw_buf.tobytes()))
            except: pass

        # Render overlays
        out = frame

        # Heatmap - localized thermal trails behind vehicles
        if overlay.get("heatmap", True):
            heatmap_renderer.render(out, current_ids)

        # Trails - thin animal-tail style
        if overlay.get("trails", True):
            trail_renderer.render(out, current_ids)

        if overlay.get("bboxes", True) and tracked is not None and tracked.tracker_id is not None:
            smoothed_xyxy = bbox_smoother.smooth(tracked)
            tids = tracked.tracker_id
            class_ids = tracked.class_id

            for i in range(len(tids)):
                x1, y1, x2, y2 = map(int, smoothed_xyxy[i])
                tid = tids[i]
                # Thin yellow box
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cls_name = CLASS_NAMES.get(int(class_ids[i]), "") if class_ids is not None else ""
                label = f"#{tid}"
                if cls_name:
                    label += f" {cls_name}"
                ly = y1 - 4 if y1 > 15 else y2 + 12
                cv2.putText(out, label, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            bbox_smoother.smooth(tracked)

        # Encode and send frame
        ret_enc, buffer = cv2.imencode(".jpg", out, encode_params)
        if ret_enc:
            try:
                if f_q.full():
                    try:
                        f_q.get_nowait()
                    except:
                        pass
                f_q.put_nowait((name, buffer.tobytes()))
            except:
                pass

        # Control frame rate to match source
        elapsed = time.time() - frame_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

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
    frame_queue = ctx.Queue(maxsize=15)
    raw_frame_queue = ctx.Queue(maxsize=15)
    metrics_queue = ctx.Queue(maxsize=10)
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
