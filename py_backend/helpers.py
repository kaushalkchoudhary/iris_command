"""Shared utilities for video capture, bbox smoothing, and constants."""

import os
import cv2
import time
import threading
import subprocess
import numpy as np

# Optimized RTSP settings - lower buffer for real-time
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;262144|analyzeduration;50000|probesize;50000|fflags;nobuffer|flags;low_delay"
)

# Inference settings (env-overridable)
INFERENCE_SIZE = int(os.environ.get("IRIS_INFERENCE_SIZE", "960"))
TARGET_FPS = int(os.environ.get("IRIS_TARGET_FPS", "30"))
SKIP_FRAMES = int(os.environ.get("IRIS_SKIP_FRAMES", "1"))
JPEG_QUALITY = int(os.environ.get("IRIS_JPEG_QUALITY", "80"))
MAX_DET = int(os.environ.get("IRIS_MAX_DET", "50"))
BBOX_SMOOTH_ALPHA = float(os.environ.get("IRIS_BBOX_SMOOTH_ALPHA", "0.12"))

# Tracker tuning (env-overridable)
TRACK_ACTIVATION_THRESHOLD = float(os.environ.get("IRIS_TRACK_ACTIVATION", "0.25"))
TRACK_LOST_BUFFER = int(os.environ.get("IRIS_TRACK_BUFFER", "120"))
TRACK_MATCH_THRESHOLD = float(os.environ.get("IRIS_TRACK_MATCH_THRESHOLD", "0.35"))
TRACK_MIN_CONSEC = int(os.environ.get("IRIS_TRACK_MIN_CONSEC", "2"))

# Confidence defaults
VEHICLE_CONF_THRESH = float(os.environ.get("IRIS_VEHICLE_CONF", "0.20"))
CROWD_CONF_THRESH = float(os.environ.get("IRIS_CROWD_CONF", "0.25"))

# GPU memory budget: base value, dynamically adjusted based on active streams
YOLO_GPU_MEMORY_FRACTION = 0.20


def get_dynamic_gpu_fraction(active_streams: int) -> float:
    """Calculate GPU memory fraction based on number of active streams.

    More streams = less GPU per stream to avoid OOM.
    Fewer streams = more GPU for better performance.
    """
    if active_streams <= 1:
        return 0.70  # Single stream gets most of GPU
    elif active_streams == 2:
        return 0.40  # 2 streams: 40% each
    elif active_streams == 3:
        return 0.30  # 3 streams: 30% each
    elif active_streams <= 5:
        return 0.20  # 4-5 streams: 20% each
    else:
        return 0.15  # 6+ streams: 15% each (may need to queue)

GPU_DECODE_DEFAULT = os.environ.get("IRIS_GPU_DECODE", "1") == "1"
GPU_DECODE_UPLOADS = os.environ.get("IRIS_GPU_DECODE_UPLOADS", "1") == "1"


def _has_cuda_device() -> bool:
    return os.path.exists("/dev/nvidia0")


def _probe_stream_info(url: str):
    """Best-effort probe for width/height/fps/codec via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,codec_name",
        "-of", "csv=p=0",
        "-analyzeduration", "500000",
        "-probesize", "32k",
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except Exception:
        return None

    if proc.returncode != 0:
        return None

    parts = proc.stdout.strip().split(",")
    if len(parts) < 2:
        return None

    try:
        width = int(parts[0])
        height = int(parts[1])
    except Exception:
        return None

    fps = 0.0
    if len(parts) >= 3 and parts[2]:
        try:
            num, den = parts[2].split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except Exception:
            fps = 0.0

    codec = parts[3].strip() if len(parts) >= 4 else ""
    return width, height, fps, codec


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


class FFmpegFrameCapture:
    """FFmpeg-based capture with NVDEC for lower-latency GPU decode."""

    def __init__(self, url, width, height, fps=0.0, codec=""):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        self.frame_ready = threading.Event()
        self.consecutive_failures = 0
        self.frame_size = self.width * self.height * 3

        decoder = None
        codec = (codec or "").lower()
        if codec == "h264":
            decoder = "h264_cuvid"
        elif codec in ("hevc", "h265"):
            decoder = "hevc_cuvid"

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "500000",
            "-probesize", "32k",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
        ]

        if decoder:
            cmd += ["-c:v", decoder]

        cmd += [
            "-i", url,
            "-vf", "hwdownload,format=bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception:
            self.proc = None
            return

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _is_corrupt(self, frame):
        if frame is None:
            return True
        if frame.size == 0:
            return True
        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            return True
        sample = frame[:: max(1, h // 4), :: max(1, w // 4)]
        if sample.std() < 1.0:
            return True
        return False

    def _update(self):
        while self.running:
            if self.proc is None or self.proc.poll() is not None:
                time.sleep(0.2)
                continue

            data = self.proc.stdout.read(self.frame_size)
            if not data or len(data) != self.frame_size:
                self.consecutive_failures += 1
                if self.consecutive_failures > 30:
                    time.sleep(0.5)
                else:
                    time.sleep(0.01)
                continue

            frame = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
            if not self._is_corrupt(frame):
                with self.lock:
                    self.ret, self.frame = True, frame
                    self.consecutive_failures = 0
                self.frame_ready.set()
            time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def isOpened(self):
        return self.proc is not None and self.proc.poll() is None


class FFmpegFileCapture:
    """FFmpeg-based file capture with NVDEC for GPU decode."""

    def __init__(self, path, width, height, fps=0.0, codec=""):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        self.frame_ready = threading.Event()
        self.consecutive_failures = 0
        self.frame_size = self.width * self.height * 3

        decoder = None
        codec = (codec or "").lower()
        if codec == "h264":
            decoder = "h264_cuvid"
        elif codec in ("hevc", "h265"):
            decoder = "hevc_cuvid"

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel", "error",
            "-re",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
        ]

        if decoder:
            cmd += ["-c:v", decoder]

        cmd += [
            "-i", path,
            "-vf", "hwdownload,format=bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception:
            self.proc = None
            return

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _is_corrupt(self, frame):
        if frame is None:
            return True
        if frame.size == 0:
            return True
        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            return True
        sample = frame[:: max(1, h // 4), :: max(1, w // 4)]
        if sample.std() < 1.0:
            return True
        return False

    def _update(self):
        while self.running:
            if self.proc is None or self.proc.poll() is not None:
                time.sleep(0.2)
                continue

            data = self.proc.stdout.read(self.frame_size)
            if not data or len(data) != self.frame_size:
                self.consecutive_failures += 1
                if self.consecutive_failures > 30:
                    time.sleep(0.5)
                else:
                    time.sleep(0.01)
                continue

            frame = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
            if not self._is_corrupt(frame):
                with self.lock:
                    self.ret, self.frame = True, frame
                    self.consecutive_failures = 0
                self.frame_ready.set()
            time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def isOpened(self):
        return self.proc is not None and self.proc.poll() is None


def create_capture(url: str):
    """Create a capture object, preferring GPU decode when available."""
    if GPU_DECODE_DEFAULT and _has_cuda_device():
        info = _probe_stream_info(url)
        if info:
            width, height, fps, codec = info
            cap = FFmpegFrameCapture(url, width, height, fps, codec)
            # Wait briefly for first frame; fallback if none.
            if cap.isOpened():
                cap.frame_ready.wait(timeout=2.0)
                if cap.frame is not None:
                    return cap
            cap.release()
    return FrameCapture(url)


def create_file_capture(path: str):
    """Create a file capture object, preferring GPU decode when available."""
    if GPU_DECODE_UPLOADS and _has_cuda_device():
        info = _probe_stream_info(path)
        if info:
            width, height, fps, codec = info
            cap = FFmpegFileCapture(path, width, height, fps, codec)
            if cap.isOpened():
                cap.frame_ready.wait(timeout=2.0)
                if cap.frame is not None:
                    return cap
            cap.release()
    return cv2.VideoCapture(path)

def get_video_duration(path: str) -> float:
    """Get duration of a video file in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration", "-of", "csv=p=0", path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return float(out)
    except:
        return 0.0


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


def raw_loader_worker(cap, stop_event, model_loading, name, f_q, rf_q, encode_params):
    """Feeds processed queue (f_q) and raw queue (rf_q) while model is loading."""
    # Use PROCESSED_FPS if available, else default
    fps = float(os.environ.get("IRIS_PROCESSED_FPS", "30"))
    interval = 1.0 / fps

    while model_loading.is_set() and not stop_event.is_set():
        t_start = time.time()
        ret, frame = cap.read()
        if ret and frame is not None:
            ret_enc, buffer = cv2.imencode(".jpg", frame, encode_params)
            if ret_enc:
                data = buffer.tobytes()
                # 1. Push to processed queue so UI shows video immediately
                try:
                    if f_q.full(): f_q.get_nowait()
                    f_q.put_nowait((name, data))
                except: pass
                
                # 2. Push to raw queue for SAM to consume
                if rf_q:
                    try:
                        if rf_q.full(): rf_q.get_nowait()
                        rf_q.put_nowait((name, data))
                    except: pass

        elapsed = time.time() - t_start
        sleep_time = max(0.001, interval - elapsed)
        time.sleep(sleep_time)
