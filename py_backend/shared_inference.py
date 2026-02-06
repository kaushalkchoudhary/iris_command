"""Shared GPU inference service for RTSP and upload streams.

Runs a single (or limited) set of models on GPU and batches frames from multiple
streams to avoid per-stream model duplication.
"""

import os
import time
import threading
import base64
from collections import deque

import cv2
import numpy as np

from helpers import (
    create_capture,
    create_file_capture,
    BboxSmoother,
    raw_loader_worker,
    INFERENCE_SIZE,
    JPEG_QUALITY,
    MAX_DET,
    BBOX_SMOOTH_ALPHA,
    TARGET_FPS,
    TRACK_ACTIVATION_THRESHOLD,
    TRACK_LOST_BUFFER,
    TRACK_MATCH_THRESHOLD,
    TRACK_MIN_CONSEC,
    VEHICLE_CONF_THRESH,
    CROWD_CONF_THRESH,
)
from overlays import TrailRenderer, HeatmapRenderer, CrowdHeatmapRenderer, FullHeatmapRenderer
from crowd import CrowdAnalyticsState, CrowdCounter
from yolobyte import AnalyticsState, classify_speed
from sam import sam_results, sam_results_lock


MODEL_PATH_VEHICLE = "models/yolov11n-visdrone.pt"
MODEL_PATH_CROWD_YOLO = "models/best_head.pt"
MODEL_PATH_CROWD_CCN = "models/crowd-model.pth"
USE_CCN_FOR_CROWD = True


class ThreadHandle:
    """Process-like handle for a stream backed by threads."""

    def __init__(self, name, stop_event, thread, manager):
        self.name = name
        self._stop_event = stop_event
        self._thread = thread
        self._manager = manager

    def terminate(self):
        self._stop_event.set()
        self._manager.remove_stream(self.name)

    def kill(self):
        self.terminate()

    def is_alive(self):
        return self._thread.is_alive()

    def join(self, timeout=None):
        self._thread.join(timeout=timeout)


class StreamState:
    def __init__(self, name, stream_type, mode, overlay_dict, is_upload=False, realtime=False):
        self.name = name
        self.stream_type = stream_type  # "rtsp" | "upload"
        self.mode = mode  # "vehicle" | "crowd" | "forensics" | None
        self.overlay_dict = overlay_dict
        self.is_upload = is_upload
        self.realtime = realtime
        self.src_fps = 0.0

        self.cap = None
        self.stop_event = threading.Event()
        self.decode_thread = None

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_seq = 0
        self.last_processed_seq = 0
        self.last_frame_time = 0.0

        self.started_signaled = False
        self.last_metrics_time = time.time()
        self.fps_times = []
        self.actual_fps = 0.0

        self.cached_overlay = {
            "heatmap": True,
            "heatmap_full": True,
            "heatmap_trails": True,
            "trails": True,
            "bboxes": True,
            "confidence": VEHICLE_CONF_THRESH,
            "bbox_label": "speed",
        }
        self.last_overlay_check = 0.0

        # Per-stream trackers/renderers/analytics
        self.tracker = None
        self.trail_renderer = None
        self.heatmap_renderer = None
        self.full_heatmap_renderer = None
        self.bbox_smoother = None
        self.analytics = None
        self.crowd_analytics = None
        self.ccn_counter = None
        self.last_ccn_time = 0.0
        self.ccn_interval = 1.0 / 5.0
        self.cached_ccn_result = None
        self.class_names = None

        self.finished = False
        self.has_any_frame = False
        self.decode_failures = 0
        self.decode_start = time.time()
        self.first_frame_logged = False


class SharedInferenceManager:
    def __init__(self, frame_queue, metrics_queue, alert_queue, raw_frame_queue, overlay_dict):
        self.frame_queue = frame_queue
        self.metrics_queue = metrics_queue
        self.alert_queue = alert_queue
        self.raw_frame_queue = raw_frame_queue
        self.overlay_dict = overlay_dict

        self.streams = {}
        self.streams_lock = threading.Lock()

        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

        self._stop_event = threading.Event()
        self._inference_thread = None

        # Model cache
        self.device = None
        self.half_precision = False
        self.vehicle_model = None
        self.crowd_model = None

    def start(self):
        if self._inference_thread and self._inference_thread.is_alive():
            return
        self._stop_event.clear()
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)

    def _ensure_models(self):
        if self.device is None:
            import torch
            from ultralytics import YOLO
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.half_precision = self.device == "cuda"
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                try:
                    print(f"[CUDA] SharedInference: device={torch.cuda.get_device_name(0)} mem={torch.cuda.get_device_properties(0).total_memory // (1024**2)}MB")
                except Exception as e:
                    print(f"[CUDA] SharedInference: device info unavailable ({e})")
            else:
                print("[CUDA] SharedInference: CUDA not available, using CPU")

            self._YOLO = YOLO

        if self.vehicle_model is None:
            self.vehicle_model = self._YOLO(MODEL_PATH_VEHICLE, task="detect")
            if self.device == "cuda":
                self.vehicle_model.to(self.device)

        if self.crowd_model is None:
            self.crowd_model = self._YOLO(MODEL_PATH_CROWD_YOLO, task="detect")
            if self.device == "cuda":
                self.crowd_model.to(self.device)

    def add_rtsp_stream(self, idx, name, url, overlay_config=None):
        mode = None
        if overlay_config and isinstance(overlay_config, dict):
            mode = overlay_config.get("active_mode")

        state = StreamState(name, "rtsp", mode, self.overlay_dict, is_upload=False, realtime=True)
        state.cap = create_capture(url)
        if not state.cap or not state.cap.isOpened():
            print(f"[!] Failed to open RTSP source: {name}")
            return None

        self._init_stream_state(state)
        self._register_stream(state)

        t = threading.Thread(target=self._decode_loop, args=(state, url), daemon=True)
        state.decode_thread = t
        t.start()
        self.start()
        return ThreadHandle(name, state.stop_event, t, self)

    def add_upload_stream(self, file_path, name, overlay_config=None, is_crowd=False, realtime=False):
        mode = None
        if overlay_config and isinstance(overlay_config, dict):
            mode = overlay_config.get("active_mode")

        if not os.path.exists(file_path):
            print(f"[UPLOAD] {name}: file not found: {file_path}")
            return None

        state = StreamState(name, "upload", mode, self.overlay_dict, is_upload=True, realtime=realtime)
        state.cap = create_file_capture(file_path)
        if not state.cap or not state.cap.isOpened():
            print(f"[!] Failed to open upload file: {file_path}")
            return None

        print(f"[UPLOAD] {name}: started decode for {file_path} (mode={mode})")

        self._init_stream_state(state)
        self._register_stream(state)

        t = threading.Thread(target=self._decode_loop, args=(state, file_path), daemon=True)
        state.decode_thread = t
        t.start()
        self.start()
        return ThreadHandle(name, state.stop_event, t, self)

    def remove_stream(self, name):
        with self.streams_lock:
            state = self.streams.pop(name, None)
        if state:
            state.stop_event.set()
            try:
                if state.cap:
                    state.cap.release()
            except Exception:
                pass

    def _register_stream(self, state: StreamState):
        with self.streams_lock:
            self.streams[state.name] = state

    def _init_stream_state(self, state: StreamState):
        w = int(state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = state.cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not src_fps or src_fps < 5:
            src_fps = TARGET_FPS
        state.src_fps = src_fps

        if state.mode == "crowd":
            state.cached_overlay["confidence"] = CROWD_CONF_THRESH
            state.crowd_analytics = CrowdAnalyticsState(w, h, src_fps)
            state.heatmap_renderer = CrowdHeatmapRenderer(accumulate_frames=8)
            if USE_CCN_FOR_CROWD:
                state.ccn_counter = CrowdCounter(MODEL_PATH_CROWD_CCN, "cuda" if self.device == "cuda" else "cpu")
        else:
            state.cached_overlay["confidence"] = VEHICLE_CONF_THRESH
            import supervision as sv
            state.tracker = sv.ByteTrack(
                frame_rate=int(src_fps),
                track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
                lost_track_buffer=TRACK_LOST_BUFFER,
                minimum_matching_threshold=TRACK_MATCH_THRESHOLD,
                minimum_consecutive_frames=TRACK_MIN_CONSEC,
            )
            state.trail_renderer = TrailRenderer(max_len=15)
            state.heatmap_renderer = HeatmapRenderer(max_len=8)
            state.full_heatmap_renderer = FullHeatmapRenderer()
            state.bbox_smoother = BboxSmoother(alpha=BBOX_SMOOTH_ALPHA)
            state.class_names = {3: "car", 4: "van", 5: "truck", 7: "bus", 8: "motor", 9: "bicycle"}
            state.analytics = AnalyticsState(w, h, src_fps, state.class_names)

    def _decode_loop(self, state: StreamState, source_id):
        # Optional FPS caps to avoid runaway decode
        rtsp_cap = float(os.environ.get("IRIS_RTSP_FPS_CAP", "0"))
        upload_cap = float(os.environ.get("IRIS_UPLOAD_FPS_CAP", "0"))
        cap_fps = upload_cap if state.is_upload else rtsp_cap
        if state.is_upload:
            base_fps = state.src_fps if state.src_fps > 0 else TARGET_FPS
            # Always throttle uploads to real-time unless a lower cap is specified
            cap_fps = base_fps
            if upload_cap and upload_cap > 0:
                cap_fps = min(cap_fps, upload_cap)
        frame_interval = (1.0 / cap_fps) if cap_fps and cap_fps > 0 else 0.0

        while not state.stop_event.is_set():
            t0 = time.time()
            ret, frame = state.cap.read()
            if not ret or frame is None:
                state.decode_failures += 1
                if state.is_upload:
                    # Allow transient read failures (ffmpeg/NVDEC warmup)
                    if not state.has_any_frame and (time.time() - state.decode_start) < 5.0:
                        time.sleep(0.05)
                        continue
                    if not state.has_any_frame and (time.time() - state.decode_start) >= 10.0:
                        print(f"[UPLOAD] {state.name}: GPU decode stalled >10s, no frames received")
                    # Loop the video for uploads
                    if state.has_any_frame:
                        # Reset to beginning
                        state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        state.decode_failures = 0
                        # Small pause to prevent CPU spin if seek fails repeatedly
                        time.sleep(0.01)
                        continue

                    if not state.finished:
                        state.finished = True
                        try:
                            if not self.metrics_queue.full():
                                self.metrics_queue.put_nowait((state.name, {"__finished__": True}))
                        except Exception:
                            pass
                    break
                time.sleep(0.01)
                continue

            state.decode_failures = 0
            state.has_any_frame = True
            if state.is_upload and not state.first_frame_logged:
                state.first_frame_logged = True
                print(f"[UPLOAD] {state.name}: first frame received (fps={state.src_fps:.1f}, realtime={state.realtime})")

            # Push raw frame for SAM
            try:
                ret_enc, buffer = cv2.imencode(".jpg", frame, self.encode_params)
                if ret_enc:
                    if self.raw_frame_queue.full():
                        try:
                            self.raw_frame_queue.get_nowait()
                        except Exception:
                            pass
                    self.raw_frame_queue.put_nowait((state.name, buffer.tobytes()))
            except Exception:
                pass

            # Forensics mode: bypass inference and publish raw
            if state.mode == "forensics":
                try:
                    annotated = None
                    with sam_results_lock:
                        info = sam_results.get(state.name)
                        if info and info.get("annotated_frame"):
                            annotated = base64.b64decode(info["annotated_frame"])
                    if annotated:
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except Exception:
                                pass
                        self.frame_queue.put_nowait((state.name, annotated))
                    else:
                        ret_enc, buffer = cv2.imencode(".jpg", frame, self.encode_params)
                        if ret_enc:
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except Exception:
                                    pass
                            self.frame_queue.put_nowait((state.name, buffer.tobytes()))
                except Exception:
                    pass
            else:
                with state.frame_lock:
                    state.latest_frame = frame
                    state.latest_seq += 1
                    state.last_frame_time = time.time()

            # Throttle decode if requested
            if frame_interval > 0:
                elapsed = time.time() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _refresh_overlay(self, state: StreamState, now):
        if now - state.last_overlay_check < 0.2:
            return
        state.last_overlay_check = now
        try:
            if state.name in self.overlay_dict:
                raw = self.overlay_dict[state.name]
                if isinstance(raw, dict):
                    heatmap_val = raw.get("heatmap", True)
                    state.cached_overlay = {
                        "heatmap": heatmap_val,
                        "heatmap_full": raw.get("heatmap_full", heatmap_val),
                        "heatmap_trails": raw.get("heatmap_trails", heatmap_val),
                        "trails": raw.get("trails", True),
                        "bboxes": raw.get("bboxes", True),
                        "confidence": raw.get("confidence", state.cached_overlay.get("confidence", VEHICLE_CONF_THRESH)),
                        "bbox_label": str(raw.get("bbox_label", "speed")),
                    }
        except Exception:
            pass

    def _inference_loop(self):
        import supervision as sv
        self._ensure_models()

        while not self._stop_event.is_set():
            to_process = []
            with self.streams_lock:
                states = list(self.streams.values())

            # Collect latest frames
            for state in states:
                if state.mode == "forensics":
                    continue
                with state.frame_lock:
                    if state.latest_seq <= state.last_processed_seq or state.latest_frame is None:
                        continue
                    frame = state.latest_frame.copy()
                    seq = state.latest_seq
                to_process.append((state, frame, seq))

            if not to_process:
                time.sleep(0.002)
                continue

            # Group by mode
            crowd_batch = []
            vehicle_batch = []
            for state, frame, seq in to_process:
                if state.mode == "crowd":
                    crowd_batch.append((state, frame, seq))
                else:
                    vehicle_batch.append((state, frame, seq))

            if vehicle_batch:
                by_conf = {}
                for state, frame, seq in vehicle_batch:
                    conf = float(state.cached_overlay.get("confidence", VEHICLE_CONF_THRESH))
                    key = round(conf, 3)
                    by_conf.setdefault(key, []).append((state, frame, seq, conf))

                for _, items in by_conf.items():
                    frames = [f for _, f, _, _ in items]
                    conf = items[0][3]
                    results = self.vehicle_model.predict(
                        frames,
                        imgsz=INFERENCE_SIZE,
                        conf=conf,
                        max_det=MAX_DET,
                        verbose=False,
                        classes=[3, 4, 5, 7, 8, 9],
                        half=self.half_precision and self.device == "cuda",
                        device=self.device,
                    )
                    for (state, frame, seq, _), res in zip(items, results):
                        self._process_vehicle_frame(state, frame, res, sv)
                        state.last_processed_seq = seq

            if crowd_batch:
                by_conf = {}
                for state, frame, seq in crowd_batch:
                    conf = float(state.cached_overlay.get("confidence", CROWD_CONF_THRESH))
                    key = round(conf, 3)
                    by_conf.setdefault(key, []).append((state, frame, seq, conf))

                for _, items in by_conf.items():
                    frames = [f for _, f, _, _ in items]
                    conf = items[0][3]
                    results = self.crowd_model.predict(
                        frames,
                        imgsz=INFERENCE_SIZE,
                        conf=conf,
                        max_det=MAX_DET,
                        verbose=False,
                        classes=[0],
                        half=self.half_precision and self.device == "cuda",
                        device=self.device,
                    )
                    for (state, frame, seq, _), res in zip(items, results):
                        self._process_crowd_frame(state, frame, res, sv)
                        state.last_processed_seq = seq

    def _process_vehicle_frame(self, state: StreamState, frame, result, sv):
        now = time.time()
        self._refresh_overlay(state, now)
        overlay = state.cached_overlay
        current_conf = overlay.get("confidence", VEHICLE_CONF_THRESH)

        detections = sv.Detections.from_ultralytics(result)
        tracked = state.tracker.update_with_detections(detections) if state.tracker else detections
        smoothed_xyxy = None
        if tracked is not None and tracked.tracker_id is not None and state.bbox_smoother:
            smoothed_xyxy = state.bbox_smoother.smooth(tracked)
            if smoothed_xyxy is not None:
                tracked.xyxy = smoothed_xyxy

        out = frame.copy()
        current_ids = state.trail_renderer.update(tracked) if state.trail_renderer else set()

        if state.heatmap_renderer:
            state.heatmap_renderer.update(tracked)
        if state.full_heatmap_renderer:
            state.full_heatmap_renderer.update(tracked, frame.shape)

        if overlay.get("heatmap_full", overlay.get("heatmap", True)) and state.full_heatmap_renderer:
            state.full_heatmap_renderer.render(out)
        if overlay.get("heatmap_trails", overlay.get("heatmap", True)) and state.heatmap_renderer:
            state.heatmap_renderer.render(out, current_ids)
        if overlay.get("trails", True) and state.trail_renderer:
            state.trail_renderer.render(out, current_ids)

        if overlay.get("bboxes", True) and tracked is not None and tracked.tracker_id is not None:
            _SPEED_LABELS = {0: "STALLED", 1: "SLOW", 2: "MEDIUM", 3: "FAST"}
            _SPEED_COLORS = {
                0: (0, 0, 220),
                1: (0, 165, 255),
                2: (0, 210, 130),
                3: (210, 180, 0),
            }
            bbox_label_mode = overlay.get("bbox_label", "speed")
            if smoothed_xyxy is None:
                smoothed_xyxy = tracked.xyxy
            tids = tracked.tracker_id
            class_ids = tracked.class_id

            for i in range(len(tids)):
                sx1, sy1, sx2, sy2 = smoothed_xyxy[i]
                pw = (sx2 - sx1) * 0.05
                ph = (sy2 - sy1) * 0.05
                x1 = int(max(0, sx1 - pw))
                y1 = int(max(0, sy1 - ph))
                x2 = int(min(out.shape[1], sx2 + pw))
                y2 = int(min(out.shape[0], sy2 + ph))

                tid = tids[i]
                pos = state.analytics.track_positions.get(tid) if state.analytics else None
                spd_px = pos[2] if pos else 0.0
                spd_cls = classify_speed(spd_px)
                spd_color = _SPEED_COLORS[spd_cls]

                cv2.rectangle(out, (x1, y1), (x2, y2), spd_color, 2)

                cls_id = class_ids[i] if class_ids is not None and i < len(class_ids) else None
                cls_name = state.class_names.get(int(cls_id), "obj") if cls_id is not None else "obj"
                if bbox_label_mode == "speed":
                    label = f"{_SPEED_LABELS[spd_cls]}"
                elif bbox_label_mode == "class":
                    label = f"{cls_name}"
                else:
                    label = f"{tid}"

                cv2.putText(out, label, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, spd_color, 2)

        # Metrics
        if state.analytics:
            if now - state.last_metrics_time >= 0.2:
                metrics = state.analytics.update(tracked)
                state._update_fps(now)
                metrics["fps"] = round(state.actual_fps, 1)
                try:
                    if not self.metrics_queue.full():
                        if not state.started_signaled:
                            metrics["__started__"] = True
                            state.started_signaled = True
                        self.metrics_queue.put_nowait((state.name, metrics))
                except Exception:
                    pass
                state.last_metrics_time = now

        # Publish frame
        try:
            ret_enc, buffer = cv2.imencode(".jpg", out, self.encode_params)
            if ret_enc:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Exception:
                        pass
                self.frame_queue.put_nowait((state.name, buffer.tobytes()))
        except Exception:
            pass

    def _process_crowd_frame(self, state: StreamState, frame, result, sv):
        now = time.time()
        self._refresh_overlay(state, now)
        overlay = state.cached_overlay

        out = frame.copy()

        if USE_CCN_FOR_CROWD and state.ccn_counter:
            if now - state.last_ccn_time >= state.ccn_interval:
                state.last_ccn_time = now
                try:
                    state.cached_ccn_result = state.ccn_counter.count(frame)
                except Exception as e:
                    print(f"CCN Error: {e}")

        if state.cached_ccn_result:
            crowd_count = state.cached_ccn_result['count']
            ccn_density = state.cached_ccn_result['density_map']
            if overlay.get("heatmap", True):
                heatmap = state.cached_ccn_result['heatmap']
                out = cv2.addWeighted(out, 0.6, heatmap, 0.4, 0)
        else:
            crowd_count = 0
            h, w = out.shape[:2]
            ccn_density = np.zeros((h, w), dtype=np.float32)

        if overlay.get("bboxes", True):
            detections = sv.Detections.from_ultralytics(result)
            tracked = detections
            for i in range(len(tracked.xyxy)):
                x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
            if not state.cached_ccn_result:
                crowd_count = len(tracked.xyxy)
        else:
            tracked = None

        if state.crowd_analytics and now - state.last_metrics_time >= 0.2:
            metrics = state.crowd_analytics.update(crowd_count, ccn_density)
            state._update_fps(now)
            metrics["fps"] = round(state.actual_fps, 1)
            try:
                if not self.metrics_queue.full():
                    if not state.started_signaled:
                        metrics["__started__"] = True
                        state.started_signaled = True
                    self.metrics_queue.put_nowait((state.name, metrics))
            except Exception:
                pass
            state.last_metrics_time = now

        try:
            ret_enc, buffer = cv2.imencode(".jpg", out, self.encode_params)
            if ret_enc:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Exception:
                        pass
                self.frame_queue.put_nowait((state.name, buffer.tobytes()))
        except Exception:
            pass


def _state_update_fps(self, now):
    self.fps_times.append(now)
    if len(self.fps_times) > 15:
        self.fps_times.pop(0)
    if len(self.fps_times) >= 2:
        elapsed = self.fps_times[-1] - self.fps_times[0]
        if elapsed > 0:
            self.actual_fps = (len(self.fps_times) - 1) / elapsed


StreamState._update_fps = _state_update_fps
