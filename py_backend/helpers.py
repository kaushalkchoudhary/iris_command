"""Shared utilities for video capture, bbox smoothing, and constants."""

import os
import cv2
import time
import threading
import numpy as np

# Optimized RTSP settings - lower buffer for real-time
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;262144|analyzeduration;50000|probesize;50000|fflags;nobuffer|flags;low_delay"
)

# Inference settings
INFERENCE_SIZE = 1280
TARGET_FPS = 30
SKIP_FRAMES = 1
JPEG_QUALITY = 75
MAX_DET = 50
BBOX_SMOOTH_ALPHA = 0.45

# GPU memory budget: cap each YOLO process so SAM3 has room
YOLO_GPU_MEMORY_FRACTION = 0.35


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
        if frame.size == 0:
            return True
        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            return True
        sample = frame[::h // 4, ::w // 4]
        if sample.std() < 1.0:
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


def raw_loader_worker(cap, stop_event, model_loading, name, f_q, encode_params):
    """Keep stream alive at low FPS while model loads."""
    while model_loading.is_set() and not stop_event.is_set():
        t_start = time.time()
        ret, frame = cap.read()
        if ret and frame is not None:
            ret_enc, buffer = cv2.imencode(".jpg", frame, encode_params)
            if ret_enc:
                try:
                    if f_q.full():
                        try: f_q.get_nowait()
                        except: pass
                    f_q.put_nowait((name, buffer.tobytes()))
                except: pass

        # Cap at ~5 FPS (0.2s)
        elapsed = time.time() - t_start
        sleep_time = max(0.01, 0.2 - elapsed)
        time.sleep(sleep_time)
