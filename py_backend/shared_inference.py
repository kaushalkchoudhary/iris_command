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
from sam import sam_results, sam_results_lock, sam_threads


MODEL_PATH_VEHICLE = os.environ.get("IRIS_VEHICLE_MODEL_PATH", "models/yolov11n-visdrone.pt")
MODEL_PATH_CROWD_YOLO = "models/best_head.pt"
MODEL_PATH_CROWD_CCN = "models/crowd-model.pth"
USE_CCN_FOR_CROWD = True
UPLOAD_INFERENCE_SIZE = int(os.environ.get("IRIS_UPLOAD_INFERENCE_SIZE", "640"))
UPLOAD_MAX_DET = int(os.environ.get("IRIS_UPLOAD_MAX_DET", "35"))
CONGESTION_INFERENCE_SIZE = int(os.environ.get("IRIS_CONGESTION_INFERENCE_SIZE", "640"))
CONGESTION_MAX_DET = int(os.environ.get("IRIS_CONGESTION_MAX_DET", "30"))
# Vehicle-route tracker profile (ported from yolo-bytetrack-vehicle-tracking reference).
VEH_TRACK_ACTIVATION = float(os.environ.get("IRIS_VEH_TRACK_ACTIVATION", "0.25"))
VEH_TRACK_BUFFER = int(os.environ.get("IRIS_VEH_TRACK_BUFFER", "30"))
VEH_TRACK_MATCH = float(os.environ.get("IRIS_VEH_TRACK_MATCH", "0.8"))
VEH_TRACK_MIN_CONSEC = int(os.environ.get("IRIS_VEH_TRACK_MIN_CONSEC", "1"))
FLOW_TRACK_ACTIVATION = float(os.environ.get("IRIS_FLOW_TRACK_ACTIVATION", str(VEH_TRACK_ACTIVATION)))
FLOW_TRACK_BUFFER = int(os.environ.get("IRIS_FLOW_TRACK_BUFFER", str(VEH_TRACK_BUFFER)))
FLOW_TRACK_MATCH = float(os.environ.get("IRIS_FLOW_TRACK_MATCH", str(VEH_TRACK_MATCH)))
FLOW_TRACK_MIN_CONSEC = int(os.environ.get("IRIS_FLOW_TRACK_MIN_CONSEC", str(VEH_TRACK_MIN_CONSEC)))


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
        # Legacy crowdanalysis profile processed at ~1 FPS for stable crowd estimates.
        ccn_fps = max(0.1, float(os.environ.get("IRIS_CROWD_CCN_FPS", "1.0")))
        self.ccn_interval = 1.0 / ccn_fps
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
        # Upload metadata FPS can be noisy/wrong; keep a stable realtime baseline.
        if state.is_upload and (src_fps < 24 or src_fps > 120):
            src_fps = TARGET_FPS
        state.src_fps = src_fps

        if state.mode == "crowd":
            state.cached_overlay["confidence"] = CROWD_CONF_THRESH
            state.crowd_analytics = CrowdAnalyticsState(w, h, src_fps)
            state.heatmap_renderer = CrowdHeatmapRenderer(accumulate_frames=8)
            state.full_heatmap_renderer = FullHeatmapRenderer(
                red_hotspots_only=True,
                point_radius_scale=0.75,
                point_strength=1.8,
            )
            if USE_CCN_FOR_CROWD:
                state.ccn_counter = CrowdCounter(MODEL_PATH_CROWD_CCN, "cuda" if self.device == "cuda" else "cpu")
        else:
            state.cached_overlay["confidence"] = VEHICLE_CONF_THRESH
            if state.mode != "congestion":
                if state.mode == "vehicle":
                    track_activation = VEH_TRACK_ACTIVATION
                    lost_buffer = VEH_TRACK_BUFFER
                    match_threshold = VEH_TRACK_MATCH
                    min_consec = VEH_TRACK_MIN_CONSEC
                elif state.mode == "flow":
                    track_activation = FLOW_TRACK_ACTIVATION
                    lost_buffer = FLOW_TRACK_BUFFER
                    match_threshold = FLOW_TRACK_MATCH
                    min_consec = FLOW_TRACK_MIN_CONSEC
                else:
                    track_activation = TRACK_ACTIVATION_THRESHOLD
                    lost_buffer = TRACK_LOST_BUFFER
                    match_threshold = TRACK_MATCH_THRESHOLD
                    min_consec = TRACK_MIN_CONSEC
                import supervision as sv
                state.tracker = sv.ByteTrack(
                    frame_rate=int(src_fps),
                    track_activation_threshold=track_activation,
                    lost_track_buffer=lost_buffer,
                    minimum_matching_threshold=match_threshold,
                    minimum_consecutive_frames=min_consec,
                )
            if state.mode != "congestion":
                state.trail_renderer = TrailRenderer(max_len=15)
            state.heatmap_renderer = HeatmapRenderer(max_len=8)
            state.full_heatmap_renderer = FullHeatmapRenderer()
            if state.mode != "congestion":
                state.bbox_smoother = BboxSmoother(alpha=BBOX_SMOOTH_ALPHA)
            state.class_names = {3: "car", 4: "van", 5: "truck", 7: "bus", 8: "motor", 9: "bicycle"}
            state.analytics = AnalyticsState(w, h, src_fps, state.class_names)

    def _decode_loop(self, state: StreamState, source_id):
        # Optional FPS caps to avoid runaway decode
        rtsp_cap = float(os.environ.get("IRIS_RTSP_FPS_CAP", "0"))
        upload_cap = float(os.environ.get("IRIS_UPLOAD_FPS_CAP", "0"))
        upload_target = float(os.environ.get("IRIS_UPLOAD_TARGET_FPS", str(TARGET_FPS)))
        cap_fps = upload_cap if state.is_upload else rtsp_cap
        if state.is_upload:
            # Drive uploads by configured target FPS for smoother realtime cadence.
            base_fps = upload_target if upload_target > 0 else (state.src_fps if state.src_fps > 0 else TARGET_FPS)
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

            # Push raw frame for low-overhead raw publishing/SAM relay.
            try:
                if self.raw_frame_queue.full():
                    try:
                        self.raw_frame_queue.get_nowait()
                    except Exception:
                        pass
                self.raw_frame_queue.put_nowait((state.name, frame.copy()))
            except Exception:
                pass

            # Forensics mode: bypass inference and publish raw
            if state.mode == "forensics":
                try:
                    annotated = None
                    active_sam = False
                    sam_info = sam_threads.get(state.name)
                    if sam_info:
                        stop_evt = sam_info.get("stop_event")
                        active_sam = bool(stop_evt is not None and not stop_evt.is_set())
                    if active_sam:
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
                    if state.mode == "congestion":
                        imgsz = CONGESTION_INFERENCE_SIZE
                        max_det = CONGESTION_MAX_DET
                    else:
                        imgsz = UPLOAD_INFERENCE_SIZE if state.is_upload else INFERENCE_SIZE
                        max_det = UPLOAD_MAX_DET if state.is_upload else MAX_DET
                    key = (round(conf, 3), int(imgsz), int(max_det))
                    by_conf.setdefault(key, []).append((state, frame, seq, conf, imgsz, max_det))

                for _, items in by_conf.items():
                    frames = [f for _, f, _, _, _, _ in items]
                    conf = items[0][3]
                    imgsz = items[0][4]
                    max_det = items[0][5]
                    results = self.vehicle_model.predict(
                        frames,
                        imgsz=imgsz,
                        conf=conf,
                        max_det=max_det,
                        verbose=False,
                        classes=[3, 4, 5, 7, 8, 9],
                        half=self.half_precision and self.device == "cuda",
                        device=self.device,
                    )
                    for (state, frame, seq, _, _, _), res in zip(items, results):
                        self._process_vehicle_frame(state, frame, res, sv)
                        state.last_processed_seq = seq

            if crowd_batch:
                by_conf = {}
                for state, frame, seq in crowd_batch:
                    conf = float(state.cached_overlay.get("confidence", CROWD_CONF_THRESH))
                    imgsz = UPLOAD_INFERENCE_SIZE if state.is_upload else INFERENCE_SIZE
                    max_det = UPLOAD_MAX_DET if state.is_upload else MAX_DET
                    key = (round(conf, 3), int(imgsz), int(max_det))
                    by_conf.setdefault(key, []).append((state, frame, seq, conf, imgsz, max_det))

                for _, items in by_conf.items():
                    frames = [f for _, f, _, _, _, _ in items]
                    conf = items[0][3]
                    imgsz = items[0][4]
                    max_det = items[0][5]
                    results = self.crowd_model.predict(
                        frames,
                        imgsz=imgsz,
                        conf=conf,
                        max_det=max_det,
                        verbose=False,
                        classes=[0],
                        half=self.half_precision and self.device == "cuda",
                        device=self.device,
                    )
                    for (state, frame, seq, _, _, _), res in zip(items, results):
                        self._process_crowd_frame(state, frame, res, sv)
                        state.last_processed_seq = seq

    def _process_vehicle_frame(self, state: StreamState, frame, result, sv):
        now = time.time()
        self._refresh_overlay(state, now)
        overlay = state.cached_overlay
        mode = state.mode or overlay.get("active_mode")
        is_congestion_mode = mode == "congestion"
        heatmap_enabled = bool(overlay.get("heatmap", True))
        full_heatmap_enabled = heatmap_enabled and bool(overlay.get("heatmap_full", False))
        trails_enabled = bool(overlay.get("trails", True)) and not is_congestion_mode
        # In congestion mode we still want per-vehicle heatmap; only trails/boxes are disabled.
        heatmap_trails_enabled = heatmap_enabled and bool(overlay.get("heatmap_trails", True))
        bboxes_enabled = bool(overlay.get("bboxes", True)) and not is_congestion_mode

        detections = sv.Detections.from_ultralytics(result)
        tracked = detections if is_congestion_mode else (state.tracker.update_with_detections(detections) if state.tracker else detections)
        smoothed_xyxy = None
        if bboxes_enabled and tracked is not None and tracked.tracker_id is not None and state.bbox_smoother:
            smoothed_xyxy = state.bbox_smoother.smooth(tracked)
            if smoothed_xyxy is not None:
                tracked.xyxy = smoothed_xyxy

        out = frame.copy()
        current_ids = state.trail_renderer.update(tracked) if trails_enabled and state.trail_renderer else set()

        if heatmap_trails_enabled and state.heatmap_renderer:
            state.heatmap_renderer.update(tracked, frame.shape)
        if full_heatmap_enabled and state.full_heatmap_renderer:
            state.full_heatmap_renderer.update(tracked, frame.shape)

        if full_heatmap_enabled and state.full_heatmap_renderer:
            state.full_heatmap_renderer.render(out)
        if heatmap_trails_enabled and state.heatmap_renderer:
            state.heatmap_renderer.render(out, current_ids)
        if trails_enabled and state.trail_renderer:
            state.trail_renderer.render(out, current_ids)

        if bboxes_enabled and tracked is not None and tracked.tracker_id is not None:
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
            confs = getattr(tracked, "confidence", None)

            for i in range(len(tids)):
                sx1, sy1, sx2, sy2 = smoothed_xyxy[i]
                x1 = int(max(0, sx1))
                y1 = int(max(0, sy1))
                x2 = int(min(out.shape[1], sx2))
                y2 = int(min(out.shape[0], sy2))

                tid = tids[i]
                pos = state.analytics.track_positions.get(tid) if state.analytics else None
                spd_px = pos[2] if pos else 0.0
                spd_cls = classify_speed(spd_px)
                spd_color = _SPEED_COLORS[spd_cls]
                vel_samples = 0
                if state.analytics:
                    vel_samples = int(state.analytics.track_velocity_samples.get(tid, 0))

                cls_id = class_ids[i] if class_ids is not None and i < len(class_ids) else None
                cls_name = state.class_names.get(int(cls_id), "obj") if cls_id is not None else "obj"
                conf = float(confs[i]) if confs is not None and i < len(confs) else 0.0
                if bbox_label_mode == "speed":
                    if vel_samples < 1:
                        label = f"#{int(tid)} {cls_name} TRACKING {conf:.2f}"
                    else:
                        label = f"#{int(tid)} {cls_name} {_SPEED_LABELS[spd_cls]} {conf:.2f}"
                elif bbox_label_mode == "class":
                    label = f"#{int(tid)} {cls_name} {conf:.2f}"
                else:
                    label = f"#{int(tid)} {conf:.2f}"

                # Clean ByteTrack-like style: thin box + small tag.
                box_color = (240, 210, 0) if (mode == "flow" and vel_samples < 1) else (spd_color if mode == "flow" else (0, 240, 220))
                cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                ty1 = max(0, y1 - th - 6)
                cv2.rectangle(out, (x1, ty1), (min(out.shape[1] - 1, x1 + tw + 4), y1), box_color, -1)
                cv2.putText(out, label, (x1 + 2, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1, cv2.LINE_AA)

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

        # Publish frame as BGR; server handles one-time JPEG encode for API stream.
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Exception:
                    pass
            self.frame_queue.put_nowait((state.name, out))
        except Exception:
            pass

    def _process_crowd_frame(self, state: StreamState, frame, result, sv):
        now = time.time()
        self._refresh_overlay(state, now)
        overlay = state.cached_overlay
        heatmap_enabled = bool(overlay.get("heatmap", True))
        full_heatmap_enabled = heatmap_enabled and bool(overlay.get("heatmap_full", False))
        bboxes_enabled = bool(overlay.get("bboxes", True))

        out = frame.copy()
        detections = None
        tracked = None
        if full_heatmap_enabled or bboxes_enabled:
            detections = sv.Detections.from_ultralytics(result)
            tracked = detections

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
            if heatmap_enabled and not full_heatmap_enabled:
                heatmap = state.cached_ccn_result['heatmap']
                heat_alpha = state.cached_ccn_result.get('heat_alpha')
                if heat_alpha is not None:
                    a = np.clip(heat_alpha, 0.0, 1.0)[..., None].astype(np.float32)
                    out_f = out.astype(np.float32)
                    hm_f = heatmap.astype(np.float32)
                    # hotspot-weighted blend (prevents global blue wash)
                    out = np.clip(out_f * (1.0 - 0.55 * a) + hm_f * (0.90 * a), 0, 255).astype(np.uint8)
                else:
                    out = cv2.addWeighted(out, 0.65, heatmap, 0.35, 0)
        else:
            crowd_count = 0
            h, w = out.shape[:2]
            ccn_density = np.zeros((h, w), dtype=np.float32)

        if full_heatmap_enabled and state.full_heatmap_renderer:
            state.full_heatmap_renderer.update(tracked, frame.shape)
            state.full_heatmap_renderer.render(out)

        if bboxes_enabled and tracked is not None:
            for i in range(len(tracked.xyxy)):
                x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                cv2.rectangle(out, (x1, y1), (x2, y2), (40, 40, 220), 1)
            if not state.cached_ccn_result:
                crowd_count = len(tracked.xyxy)

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
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Exception:
                    pass
            self.frame_queue.put_nowait((state.name, out))
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
