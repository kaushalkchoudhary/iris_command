import os
import cv2
import time
import threading
import subprocess
import numpy as np

from ultralytics import YOLO
import supervision as sv

from control_server import (
    run_control_server,
    update_metrics,
    get_overlay_state,
)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

ENGINE_PATH_VEHICLE = "data/yolov11n-visdrone.engine"
ENGINE_PATH_CROWD = "data/best_head.onnx"
RTSP_OUT_BASE = "rtsp://127.0.0.1:8554/processed_"
TARGET_FPS = 10

# Drones that use crowd counting model
CROWD_DRONES = {"bcpdrone10", "bcpdrone12"}

# Speed thresholds in pixels/second (matching Rust backend)
THRESH_STALLED = 8.0
THRESH_SLOW = 50.0
THRESH_MEDIUM = 160.0

# EMA smoothing factor for metrics
EMA_ALPHA = 0.2


def classify_speed(speed_px_s: float) -> int:
    """Classify speed into bucket: 0=stalled, 1=slow, 2=medium, 3=fast"""
    if speed_px_s < THRESH_STALLED:
        return 0
    elif speed_px_s < THRESH_SLOW:
        return 1
    elif speed_px_s < THRESH_MEDIUM:
        return 2
    else:
        return 3


class AnalyticsState:
    """Track analytics state for a single video source."""

    def __init__(self, width: int, height: int, fps: float):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_area = max(1.0, float(width * height))

        # Track positions: track_id -> (last_cx, last_cy, speed_px_s)
        self.track_positions = {}

        # EMA smoothed values
        self.traffic_density_ema = 0.0
        self.mobility_index_ema = 0.0

        # FPS tracking
        self.fps_frame_count = 0
        self.fps_last_time = time.time()
        self.fps_value = 0.0

    def update(self, tracked_detections) -> dict:
        """
        Update analytics with new detections and return metrics dict.
        tracked_detections: supervision Detections with tracker_id set.
        """
        # Update FPS
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_last_time
        if elapsed >= 1.0:
            self.fps_value = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_last_time = time.time()

        if tracked_detections is None or tracked_detections.tracker_id is None:
            return self._build_metrics(0, [0, 0, 0, 0], 0.0)

        # Calculate speeds and classify
        speed_counts = [0, 0, 0, 0]  # stalled, slow, medium, fast
        total_area = 0.0
        new_positions = {}

        for i, (xyxy, tid) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            total_area += w * h

            # Calculate speed
            speed_px_s = 0.0
            if tid in self.track_positions:
                old_cx, old_cy, old_speed = self.track_positions[tid]
                dx = cx - old_cx
                dy = cy - old_cy
                raw_speed = np.sqrt(dx * dx + dy * dy) * self.fps
                # Smooth speed
                speed_px_s = old_speed * 0.4 + raw_speed * 0.6 if old_speed > 0 else raw_speed

            new_positions[tid] = (cx, cy, speed_px_s)

            # Classify and count
            bucket = classify_speed(speed_px_s)
            speed_counts[bucket] += 1

        self.track_positions = new_positions

        return self._build_metrics(len(tracked_detections), speed_counts, total_area)

    def _build_metrics(self, detection_count: int, speed_counts: list, total_area: float) -> dict:
        """Build metrics dictionary from raw values."""
        total = max(1, sum(speed_counts))

        # Speed percentages
        stalled_pct = round(speed_counts[0] / total * 100)
        slow_pct = round(speed_counts[1] / total * 100)
        medium_pct = round(speed_counts[2] / total * 100)
        fast_pct = round(speed_counts[3] / total * 100)

        # Traffic density: bbox area relative to frame area
        density = min(100.0, (total_area / self.frame_area) * 100.0)
        if self.traffic_density_ema <= 0:
            self.traffic_density_ema = density
        else:
            self.traffic_density_ema = self.traffic_density_ema * (1 - EMA_ALPHA) + density * EMA_ALPHA
        traffic_density_pct = int(round(min(100, max(0, self.traffic_density_ema))))

        # Mobility index: weighted average of speed buckets
        if total > 0:
            w_stalled, w_slow, w_medium, w_fast = 10.0, 40.0, 70.0, 95.0
            mobility = (
                speed_counts[0] * w_stalled +
                speed_counts[1] * w_slow +
                speed_counts[2] * w_medium +
                speed_counts[3] * w_fast
            ) / total
        else:
            mobility = 0.0

        if self.mobility_index_ema <= 0:
            self.mobility_index_ema = mobility
        else:
            self.mobility_index_ema = self.mobility_index_ema * (1 - EMA_ALPHA) + mobility * EMA_ALPHA
        mobility_index_pct = int(round(min(100, max(0, self.mobility_index_ema))))

        # Congestion index: combination of density and inverted mobility
        congestion = int(round(
            traffic_density_pct * 0.6 + (100 - mobility_index_pct) * 0.4
        ))
        congestion = min(100, max(0, congestion))

        return {
            "fps": round(self.fps_value, 1),
            "congestion_index": congestion,
            "traffic_density": traffic_density_pct,
            "mobility_index": mobility_index_pct,
            "stalled_pct": stalled_pct,
            "slow_pct": slow_pct,
            "medium_pct": medium_pct,
            "fast_pct": fast_pct,
            "detection_count": detection_count,
        }


def start_rtsp_publisher(w, h, fps, name):
    rtsp_url = RTSP_OUT_BASE + name
    print(f"[→] Publishing to {rtsp_url}")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "warning",
        # Input options
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(int(fps)),
        "-i", "pipe:0",
        # Output options
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-g", str(int(fps * 2)),  # keyframe interval
        "-pix_fmt", "yuv420p",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]

    try:
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**6,
        )
    except Exception as e:
        print(f"[!] Failed to start ffmpeg for {name}: {e}")
        return None

    # Give ffmpeg a moment to start
    time.sleep(0.1)

    # Check if process is still running
    if p.poll() is not None:
        _, stderr = p.communicate()
        print(f"[!] ffmpeg exited immediately for {name}: {stderr.decode()}")
        return None

    # Send a few blank frames to initialize the stream
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(5):
        try:
            p.stdin.write(blank.tobytes())
            p.stdin.flush()
        except BrokenPipeError:
            _, stderr = p.communicate()
            print(f"[!] Failed to initialize RTSP publisher for {name}: {stderr.decode()}")
            return None
        time.sleep(0.05)

    print(f"[✓] RTSP publisher started for {name}")
    return p


def process_stream(index, name, url, stop_event):
    print(f"[+] Starting inference: {name}")

    if url.startswith("rtsp"):
        cap = cv2.VideoCapture(f"{url}?rtsp_transport=tcp", cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"[!] Failed to open source: {name}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Frame skipping: crowd model needs more skip due to ONNX being slower
    is_crowd_stream = name in CROWD_DRONES
    if is_crowd_stream:
        stride = 5  # Process every 5th frame for crowd model
    else:
        stride = 2  # Process every 2nd frame for vehicle model
    proc_fps = src_fps / stride

    publisher = start_rtsp_publisher(w, h, proc_fps, name)
    if publisher is None:
        print(f"[!] Failed to start RTSP publisher for {name}, aborting")
        cap.release()
        return

    # Select model based on drone type
    is_crowd_model = name in CROWD_DRONES
    if is_crowd_model:
        model_path = ENGINE_PATH_CROWD
        target_classes = None  # Crowd model - detect all classes (heads)
        CLASS_NAMES = {0: "head"}
        print(f"[*] Using CROWD model for {name}")
    else:
        model_path = ENGINE_PATH_VEHICLE
        target_classes = [3, 4, 5, 7, 8, 9]  # VisDrone vehicle classes
        CLASS_NAMES = {
            3: "car",
            4: "van",
            5: "truck",
            7: "bus",
            8: "motor",
            9: "bicycle"
        }
        print(f"[*] Using VEHICLE model for {name}")

    model = YOLO(model_path, task="detect")

    tracker = sv.ByteTrack(frame_rate=proc_fps)

    # Heatmap for activity visualization
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Trail history: track_id -> list of (x, y) points
    trail_history = {}
    MAX_TRAIL_LEN = 50  # Longer trails for better visualization

    # Speed colors for indicator dots (stalled=red, slow=orange, medium=yellow, fast=green)
    SPEED_COLORS = [
        (0, 0, 255),    # Red - stalled
        (0, 165, 255),  # Orange - slow
        (0, 255, 255),  # Yellow - medium
        (0, 255, 0),    # Green - fast
    ]

    # Initialize analytics state for this stream
    analytics = AnalyticsState(w, h, proc_fps)
    last_metrics_update = time.time()
    METRICS_UPDATE_INTERVAL = 0.5  # Update metrics every 500ms

    frame_idx = 0

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        overlay = get_overlay_state(name)
        show_heatmap = overlay["heatmap"]
        show_trails = overlay["trails"]
        show_bboxes = overlay["bboxes"]

        # Decay heatmap
        heatmap *= 0.97

        # Run inference with appropriate class filter
        predict_args = {
            "conf": 0.25 if is_crowd_model else 0.10,
            "max_det": 500,
            "verbose": False,
        }
        if target_classes is not None:
            predict_args["classes"] = target_classes

        results = model.predict(frame, **predict_args)[0]

        detections = sv.Detections.from_ultralytics(results)
        tracked = tracker.update_with_detections(detections)

        # Track current frame's track IDs for cleanup
        current_track_ids = set()

        if tracked is not None and tracked.tracker_id is not None:
            for i, (xyxy, tid) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
                x1, y1, x2, y2 = xyxy
                cx = int((x1 + x2) / 2)
                cy = int(y2)  # Bottom center for ground contact point

                current_track_ids.add(tid)

                # Update trail history
                if tid not in trail_history:
                    trail_history[tid] = []
                trail_history[tid].append((cx, cy))
                if len(trail_history[tid]) > MAX_TRAIL_LEN:
                    trail_history[tid].pop(0)

                # Add to heatmap (smaller, more localized circles)
                cv2.circle(heatmap, (cx, cy), 8, 1.5, -1)

        # Cleanup old tracks from trail history
        stale_ids = [tid for tid in trail_history if tid not in current_track_ids]
        for tid in stale_ids:
            # Keep trails for a bit after track disappears, then remove
            if len(trail_history[tid]) > 0:
                trail_history[tid] = trail_history[tid][1:]  # Fade out
            if len(trail_history[tid]) == 0:
                del trail_history[tid]

        # Clamp heatmap to prevent oversaturation
        np.clip(heatmap, 0, 50, out=heatmap)

        # Update analytics and send metrics to frontend
        metrics = analytics.update(tracked)
        now = time.time()
        if now - last_metrics_update >= METRICS_UPDATE_INTERVAL:
            update_metrics(name, metrics)
            last_metrics_update = now

        out = frame.copy()

        # Draw heatmap overlay (subtle, localized)
        if show_heatmap:
            sm = cv2.GaussianBlur(heatmap, (31, 31), 0)
            mv = sm.max()
            if mv > 1.0:
                norm = np.clip(sm / mv * 255, 0, 255).astype(np.uint8)
                # Use a warmer colormap for better visibility
                colored = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
                # Only show where there's significant heat
                mask = norm > 25
                if np.any(mask):
                    out[mask] = cv2.addWeighted(out[mask], 0.7, colored[mask], 0.3, 0)

        # Draw trails as fading polylines (blue gradient like reference)
        if show_trails:
            for tid, points in trail_history.items():
                if len(points) < 2:
                    continue
                # Draw trail with fading effect (older = dimmer, blue gradient)
                for j in range(1, len(points)):
                    alpha = j / len(points)  # 0 to 1, newer = brighter
                    # Blue-cyan gradient trail (BGR format)
                    color = (
                        int(255 * alpha),      # B - bright blue
                        int(200 * alpha),      # G - some green for cyan tint
                        int(50 * alpha)        # R - minimal red
                    )
                    thickness = max(2, int(2 + alpha * 3))
                    pt1 = points[j - 1]
                    pt2 = points[j]
                    cv2.line(out, pt1, pt2, color, thickness, cv2.LINE_AA)

        # Draw bounding boxes and small labels
        if show_bboxes and tracked is not None and tracked.tracker_id is not None:
            # Get class IDs if available
            class_ids = tracked.class_id if hasattr(tracked, 'class_id') and tracked.class_id is not None else [None] * len(tracked.xyxy)

            for i, (xyxy, tid) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
                x1, y1, x2, y2 = map(int, xyxy)

                # Get speed bucket for this track
                speed_bucket = 0
                if tid in analytics.track_positions:
                    _, _, speed = analytics.track_positions[tid]
                    speed_bucket = classify_speed(speed)

                # Draw bbox with orange/yellow color (thin line)
                bbox_color = (0, 180, 255)  # Orange in BGR
                cv2.rectangle(out, (x1, y1), (x2, y2), bbox_color, 1, cv2.LINE_AA)

                # Get class name
                cls_id = class_ids[i] if i < len(class_ids) else None
                cls_name = CLASS_NAMES.get(int(cls_id), "") if cls_id is not None else ""

                # Build compact label (just ID if no class, or "ID cls" if class known)
                label = f"#{tid}" if not cls_name else f"#{tid} {cls_name}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.28  # Smaller font
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Label position (above bbox)
                lx = x1
                ly = max(y1 - 2, th + 4)

                # Small orange label background
                padding = 2
                cv2.rectangle(out, (lx, ly - th - padding),
                             (lx + tw + padding * 2 + 6, ly + padding), (0, 100, 200), -1)

                # White text
                cv2.putText(out, label, (lx + padding, ly - 1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                # Tiny speed indicator dot
                dot_x = lx + tw + padding * 2 + 1
                dot_y = ly - th // 2
                dot_color = SPEED_COLORS[speed_bucket]
                cv2.circle(out, (dot_x + 2, dot_y), 3, dot_color, -1)

        # Write frame to RTSP publisher
        if publisher is not None:
            try:
                publisher.stdin.write(out.tobytes())
                publisher.stdin.flush()
            except (BrokenPipeError, OSError):
                print(f"[!] RTSP publisher pipe broken for {name}")
                break

        frame_idx += 1

    # Cleanup
    cap.release()
    if publisher is not None:
        try:
            publisher.stdin.close()
        except:
            pass
        publisher.wait(timeout=2)
    print(f"[-] Stopped inference: {name}")


def start_source_backend(index, url, stop_event, name):
    t = threading.Thread(target=process_stream, args=(index, name, url, stop_event), daemon=True)
    t.start()
    return t


def start_upload_backend(file_path, stop_event, name):
    t = threading.Thread(target=process_stream, args=(0, name, file_path, stop_event), daemon=True)
    t.start()
    return t


def main():
    run_control_server(start_source_backend, start_upload_backend, {})


if __name__ == "__main__":
    main()
