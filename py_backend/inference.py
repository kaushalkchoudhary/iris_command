import os
import cv2
import time
import threading
import subprocess
import numpy as np
import gc
import multiprocessing as mp
import signal

from ultralytics import YOLO
import supervision as sv
import torch

from control_server import (
    run_control_server,
    update_metrics,
    update_frame,
    get_overlay_state,
    ensure_overlay,
)

# Shared communication objects
frame_queue = None
metrics_queue = None
overlay_shared_dict = None

def relay_worker(stop_event, f_q, m_q):
    """Relay metrics and frames from multiprocess queues to control server."""
    while not stop_event.is_set():
        # Batch process frames to reduce lock contention
        try:
            for _ in range(20): # Process up to 20 frames per cycle
                if f_q.empty(): break
                name, data = f_q.get_nowait()
                update_frame(name, data)
        except: pass
        
        # Batch process metrics
        try:
            for _ in range(10):
                if m_q.empty(): break
                name, data = m_q.get_nowait()
                update_metrics(name, data)
        except: pass
        
        time.sleep(0.005) # Extreme low latency relay

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000|analyzeduration;100000|probesize;100000"

MODEL_PATH_VEHICLE = "data/yolov11n-visdrone.pt"
MODEL_PATH_CROWD = "data/best_head.pt"
RTSP_OUT_BASE = "rtsp://127.0.0.1:8554/processed_"
TARGET_FPS = 25

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
        """Update analytics with new detections."""
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

        for i, (xyxy, tid) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
            x1, y1, x2, y2 = xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            total_area += (x2 - x1) * (y2 - y1)

            speed_px_s = 0.0
            if tid in self.track_positions:
                old_cx, old_cy, old_speed = self.track_positions[tid]
                raw_speed = np.sqrt((cx - old_cx)**2 + (cy - old_cy)**2) * self.fps
                speed_px_s = old_speed * 0.4 + raw_speed * 0.6 if old_speed > 0 else raw_speed

            new_positions[tid] = (cx, cy, speed_px_s)
            speed_counts[classify_speed(speed_px_s)] += 1

        self.track_positions = new_positions
        return self._build_metrics(len(tracked_detections), speed_counts, total_area)

    def _build_metrics(self, detection_count: int, speed_counts: list, total_area: float) -> dict:
        total = max(1, sum(speed_counts))
        stalled_pct = round(speed_counts[0] / total * 100)
        slow_pct = round(speed_counts[1] / total * 100)
        medium_pct = round(speed_counts[2] / total * 100)
        fast_pct = round(speed_counts[3] / total * 100)

        density = min(100.0, (total_area / self.frame_area) * 100.0 * 6.5)
        self.traffic_density_ema = self.traffic_density_ema * (1-EMA_ALPHA) + density * EMA_ALPHA if self.traffic_density_ema > 0 else density
        
        mobility = 0.0
        if total > 0:
            weights = [10.0, 40.0, 70.0, 95.0]
            mobility = sum(c * w for c, w in zip(speed_counts, weights)) / total
        self.mobility_index_ema = self.mobility_index_ema * (1-EMA_ALPHA) + mobility * EMA_ALPHA if self.mobility_index_ema > 0 else mobility

        t_density = int(round(self.traffic_density_ema))
        m_index = int(round(self.mobility_index_ema))
        congestion = min(100, max(0, int(round(t_density * 0.6 + (100 - m_index) * 0.4))))

        return {
            "fps": round(self.fps_value, 1),
            "congestion_index": congestion,
            "traffic_density": t_density,
            "mobility_index": m_index,
            "stalled_pct": stalled_pct, "slow_pct": slow_pct, "medium_pct": medium_pct, "fast_pct": fast_pct,
            "detection_count": detection_count,
        }


class FrameCapture:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.frame, self.ret, self.running = None, False, True
        self.lock = threading.Lock()
        if self.cap.isOpened():
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame
            if not ret: time.sleep(1.0)
            else: time.sleep(0.005)

    def read(self):
        with self.lock: return self.ret, self.frame

    def release(self):
        self.running = False
        if hasattr(self, 'thread'): self.thread.join(timeout=1.0)
        self.cap.release()

    def get(self, prop): return self.cap.get(prop)
    def isOpened(self): return self.cap.isOpened()


def process_stream(index, name, url, stop_event, f_q, m_q, overlay_dict):
    print(f"[+] Starting inference: {name}")
    # Fix for CUDA initialization in new process
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    cap = FrameCapture(url)
    if not cap.isOpened():
        print(f"[!] Failed to open source: {name}")
        return

    w, h, src_fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 25
    
    is_crowd = name in CROWD_DRONES
    model = YOLO(MODEL_PATH_CROWD if is_crowd else MODEL_PATH_VEHICLE, task="detect")
    if torch.cuda.is_available(): model.to("cuda")

    target_classes = None if is_crowd else [3, 4, 5, 7, 8, 9]
    CLASS_NAMES = {0: "head"} if is_crowd else {3: "car", 4: "van", 5: "truck", 7: "bus", 8: "motor", 9: "bicycle"}

    tracker = sv.ByteTrack(frame_rate=src_fps)
    trail_history = {} # tid -> list of (cx, cy, r)
    MAX_TRAIL_LEN = 30
    
    analytics = AnalyticsState(w, h, src_fps)
    last_metrics_update = time.time()

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Get overlay state from shared dict
        overlay = overlay_dict.get(name, {"heatmap": True, "trails": True, "bboxes": True})
        
        results = model.predict(frame, conf=0.25 if is_crowd else 0.1, max_det=500, verbose=False, classes=target_classes)[0]
        tracked = tracker.update_with_detections(sv.Detections.from_ultralytics(results))
        
        current_track_ids = set()
        if tracked is not None and tracked.tracker_id is not None:
            for i, (xyxy, tid) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
                x1, y1, x2, y2 = xyxy
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                r = int(min(x2-x1, y2-y1) * 0.4)
                current_track_ids.add(tid)
                if tid not in trail_history: trail_history[tid] = []
                trail_history[tid].append((cx, cy, r))
                if len(trail_history[tid]) > MAX_TRAIL_LEN: trail_history[tid].pop(0)

        # Cleanup stale tracks
        for tid in [t for t in trail_history if t not in current_track_ids]:
            if trail_history[tid]: trail_history[tid].pop(0)
            else: del trail_history[tid]

        if time.time() - last_metrics_update >= 0.5:
            m_q.put((name, analytics.update(tracked)))
            last_metrics_update = time.time()

        out = frame.copy()

        # Heatmap Rendering (Optimized with scaling)
        if overlay["heatmap"]:
            scale = 0.5
            sh, sw = int(h * scale), int(w * scale)
            tail_mask = np.zeros((sh, sw), dtype=np.uint8)
            for tid, pts in trail_history.items():
                if tid not in current_track_ids: continue
                sub_pts = pts[-15:]
                for i in range(len(sub_pts)):
                    tx, ty, tr = int(sub_pts[i][0] * scale), int(sub_pts[i][1] * scale), int(sub_pts[i][2] * scale)
                    phase = (i + 1) / len(sub_pts)
                    intensity = int(35 + 220 * phase)
                    curr_r = int(tr * (0.7 + 0.3 * phase))
                    cv2.circle(tail_mask, (tx, ty), curr_r, intensity, -1)
                    if i > 0:
                        px, py = int(sub_pts[i-1][0] * scale), int(sub_pts[i-1][1] * scale)
                        pr = int(sub_pts[i-1][2] * scale)
                        prev_phase = i / len(sub_pts)
                        prev_intensity = int(35 + 220 * prev_phase)
                        line_intensity = (intensity + prev_intensity) // 2
                        line_thick = (curr_r + int(pr * (0.7 + 0.3 * prev_phase))) // 2
                        cv2.line(tail_mask, (px, py), (tx, ty), line_intensity, max(1, line_thick))
            
            if np.any(tail_mask > 0):
                blur = cv2.GaussianBlur(tail_mask, (11, 11), 0)
                colored = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
                colored_full = cv2.resize(colored, (w, h), interpolation=cv2.INTER_LINEAR)
                blur_full = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)
                m = blur_full > 5
                out[m] = cv2.addWeighted(out[m], 0.45, colored_full[m], 0.55, 0)

        # Trails Rendering (1px Blue)
        if overlay["trails"]:
            for tid, pts in trail_history.items():
                if len(pts) < 2: continue
                for i in range(1, len(pts)):
                    fade = (i + 1) / len(pts)
                    color = (int(180 * fade), int(220 * fade), int(255 * fade))
                    cv2.line(out, pts[i-1][:2], pts[i][:2], color, 1, cv2.LINE_AA)

        # Bounding Boxes Rendering (1px Orange)
        if overlay["bboxes"] and tracked is not None and tracked.tracker_id is not None:
            for i, (xyxy, tid) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 255), 1, cv2.LINE_4)
                label = f"#{tid}"
                if not is_crowd:
                    _, _, speed = analytics.track_positions.get(tid, (0,0,0))
                    label += f" {CLASS_NAMES.get(int(tracked.class_id[i]), '')}"
                cv2.putText(out, label, (x1, x1 if y1 < 10 else y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)

        # Output MJPEG via queue
        ret_enc, buffer = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret_enc: 
            try:
                if f_q.full(): f_q.get_nowait()
                f_q.put_nowait((name, buffer.tobytes()))
            except: pass

    cap.release()
    gc.collect()

def start_backend(idx, url, stop, name):
    # Overlay is ensured in control_server, we sync it to our shared dict
    overlay_shared_dict[name] = get_overlay_state(name)
    
    # Use multiprocessing Process instead of Thread
    p = mp.Process(target=process_stream, args=(idx, name, url, stop, frame_queue, metrics_queue, overlay_shared_dict), daemon=True)
    p.start()
    return p

def main():
    global overlay_shared_dict, frame_queue, metrics_queue
    ctx = mp.get_context('spawn')
    
    # Initialize shared memory
    manager = ctx.Manager()
    overlay_shared_dict = manager.dict()
    frame_queue = ctx.Queue(maxsize=10)
    metrics_queue = ctx.Queue(maxsize=10)
    
    # Start the relay thread to sync data across processes
    stop_relay = threading.Event()
    relay_t = threading.Thread(target=relay_worker, args=(stop_relay, frame_queue, metrics_queue), daemon=True)
    relay_t.start()
    
    try:
        # Start control server (FastAPI) which will spawn worker processes
        run_control_server(start_backend, start_backend, overlay_shared_dict)
    finally:
        stop_relay.set()
        relay_t.join(timeout=1.0)

if __name__ == "__main__": main()
