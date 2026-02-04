"""YOLO + ByteTrack inference pipeline for vehicle and crowd analytics."""

import time
import gc
import threading
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

from helpers import (
    create_capture, FrameCapture, BboxSmoother, raw_loader_worker,
    INFERENCE_SIZE, SKIP_FRAMES, JPEG_QUALITY, MAX_DET,
    BBOX_SMOOTH_ALPHA, YOLO_GPU_MEMORY_FRACTION,
)
from overlays import TrailRenderer, HeatmapRenderer, CrowdHeatmapRenderer, FullHeatmapRenderer
from crowd import CrowdAnalyticsState, CrowdCounter

# Model paths
MODEL_PATH_VEHICLE = "models/yolov11n-visdrone.pt"
MODEL_PATH_CROWD_YOLO = "models/best_head.pt"
MODEL_PATH_CROWD_CCN = "models/crowd-model.pth"
USE_CCN_FOR_CROWD = True

# Drones that use crowd counting model
CROWD_DRONES = {"bcpdrone10", "bcpdrone12"}

# Alert settings
CONGESTION_ALERT_THRESHOLD = 40

# Speed thresholds in pixels/second
THRESH_STALLED = 8.0
THRESH_SLOW = 50.0
THRESH_MEDIUM = 160.0

EMA_ALPHA = 0.2

# Hot region grid settings
HOT_GRID_COLS = 6
HOT_GRID_ROWS = 4
HOT_THRESHOLD = 0.4
HOT_SEVERITY_MOD = 0.6
HOT_SEVERITY_HIGH = 0.8


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
        "class_names",
        "track_first_seen", "track_last_seen", "track_class", "track_bbox_dims",
        "track_speed_history", "track_position_history",
        "track_state", "track_behavior", "track_impact_score",
        "grid_congestion_ema", "grid_first_hot", "grid_hot_history",
        "prev_region_cell_count", "hot_regions_cache"
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
        self.track_first_seen = {}
        self.track_last_seen = {}
        self.track_class = {}
        self.track_bbox_dims = {}
        self.track_speed_history = {}
        self.track_position_history = {}
        self.track_state = {}
        self.track_behavior = {}
        self.track_impact_score = {}
        self.grid_congestion_ema = [0.0] * (HOT_GRID_COLS * HOT_GRID_ROWS)
        self.grid_first_hot = {}
        self.grid_hot_history = {}
        self.prev_region_cell_count = 0
        self.hot_regions_cache = []

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
                self.track_class[tid] = cls_id
                self.track_bbox_dims[tid] = (float(x2 - x1), float(y2 - y1))

            speed_px_s = 0.0
            if tid in self.track_positions:
                old_cx, old_cy, old_speed = self.track_positions[tid]
                dx, dy = cx - old_cx, cy - old_cy
                raw_speed = np.sqrt(dx * dx + dy * dy) * self.fps
                speed_px_s = old_speed * 0.4 + raw_speed * 0.6 if old_speed > 0 else raw_speed

            new_positions[tid] = (cx, cy, speed_px_s)
            speed_counts[classify_speed(speed_px_s)] += 1

        self.track_positions = new_positions
        self._update_vehicle_analytics()
        return self._build_metrics(len(tracked_detections), speed_counts, total_area, class_counts)

    def _update_vehicle_analytics(self):
        now = time.time()
        active_tids = set(self.track_positions.keys())

        expired = [tid for tid, ts in self.track_last_seen.items() if now - ts > 2.0]
        for tid in expired:
            for d in (self.track_first_seen, self.track_last_seen, self.track_class,
                      self.track_bbox_dims, self.track_speed_history,
                      self.track_position_history, self.track_state,
                      self.track_behavior, self.track_impact_score):
                d.pop(tid, None)

        for tid in active_tids:
            cx, cy, speed = self.track_positions[tid]

            if tid not in self.track_first_seen:
                self.track_first_seen[tid] = now
            self.track_last_seen[tid] = now

            if tid not in self.track_speed_history:
                self.track_speed_history[tid] = deque(maxlen=12)
            self.track_speed_history[tid].append(speed)

            if tid not in self.track_position_history:
                self.track_position_history[tid] = deque(maxlen=12)
            self.track_position_history[tid].append((cx, cy))

            self.track_state[tid] = self._classify_state(tid, speed)
            self.track_behavior[tid] = self._classify_behavior(tid)

            bbox_w, bbox_h = self.track_bbox_dims.get(tid, (0.0, 0.0))
            self.track_impact_score[tid] = self._compute_impact(tid, speed, bbox_w, bbox_h)

    def _classify_state(self, tid, speed):
        if speed < THRESH_STALLED:
            return "stopped"
        hist = self.track_speed_history.get(tid)
        if hist and len(hist) >= 4:
            arr = np.array(hist)
            if arr.std() > 40.0 and arr.mean() > 15.0:
                return "abnormal"
        return "moving"

    def _classify_behavior(self, tid):
        pos_hist = self.track_position_history.get(tid)
        if not pos_hist or len(pos_hist) < 3:
            return "stable"

        pts = list(pos_hist)

        spd_hist = self.track_speed_history.get(tid)
        if spd_hist and len(spd_hist) >= 4:
            transitions = 0
            was_stalled = spd_hist[0] < THRESH_STALLED
            for s in list(spd_hist)[1:]:
                is_stalled = s < THRESH_STALLED
                if is_stalled != was_stalled:
                    transitions += 1
                    was_stalled = is_stalled
            if transitions >= 3:
                return "start_stop"

        if len(pts) >= 3:
            fx, fy = pts[0]
            lx, ly = pts[-1]
            dx, dy = lx - fx, ly - fy
            line_len = np.sqrt(dx * dx + dy * dy)
            if line_len > 1.0:
                max_dev = 0.0
                for px, py in pts[1:-1]:
                    dev = abs(dy * (px - fx) - dx * (py - fy)) / line_len
                    if dev > max_dev:
                        max_dev = dev
                if max_dev > 30.0:
                    return "erratic"

        return "stable"

    _SIZE_WEIGHTS = {
        "car": 1.0, "van": 1.3, "truck": 2.0, "bus": 2.5,
        "motor": 0.4, "bicycle": 0.3,
    }

    def _compute_impact(self, tid, speed, bbox_w, bbox_h):
        cls_id = self.track_class.get(tid)
        cls_name = self.class_names.get(cls_id, "car") if cls_id is not None else "car"
        size_w = self._SIZE_WEIGHTS.get(cls_name, 1.0)

        dwell = time.time() - self.track_first_seen.get(tid, time.time())
        dwell_factor = 1.0 + min(dwell / 10.0, 3.0)

        state = self.track_state.get(tid, "moving")
        if state == "stopped":
            blockage = 2.0
        elif speed < THRESH_SLOW:
            blockage = 1.2
        else:
            blockage = 0.3

        pos_hist = self.track_position_history.get(tid)
        lane_factor = 0.0
        if pos_hist:
            cx = pos_hist[-1][0]
            center_dist = abs(cx - self.width / 2.0) / (self.width / 2.0)
            lane_factor = max(0.0, 0.5 - center_dist)

        return round(size_w * dwell_factor * blockage * (1.0 + lane_factor), 2)

    def _compute_hot_regions(self):
        now = time.time()
        ncells = HOT_GRID_COLS * HOT_GRID_ROWS
        cell_w = self.width / HOT_GRID_COLS
        cell_h = self.height / HOT_GRID_ROWS

        cell_total = [0] * ncells
        cell_stalled = [0] * ncells
        cell_slow = [0] * ncells

        for tid, (cx, cy, speed) in self.track_positions.items():
            col = min(int(cx / cell_w), HOT_GRID_COLS - 1)
            row = min(int(cy / cell_h), HOT_GRID_ROWS - 1)
            idx = row * HOT_GRID_COLS + col
            cell_total[idx] += 1
            if speed < THRESH_STALLED:
                cell_stalled[idx] += 1
            elif speed < THRESH_SLOW:
                cell_slow[idx] += 1

        for i in range(ncells):
            tot = max(1, cell_total[i])
            ratio = (cell_stalled[i] + cell_slow[i]) / tot if cell_total[i] > 0 else 0.0
            self.grid_congestion_ema[i] = self.grid_congestion_ema[i] * 0.7 + ratio * 0.3

        hot_set = set()
        for i in range(ncells):
            if self.grid_congestion_ema[i] >= HOT_THRESHOLD and cell_total[i] >= 1:
                hot_set.add(i)
                if i not in self.grid_first_hot:
                    self.grid_first_hot[i] = now
                if i not in self.grid_hot_history:
                    self.grid_hot_history[i] = deque(maxlen=20)
                self.grid_hot_history[i].append(now)
            else:
                self.grid_first_hot.pop(i, None)

        visited = set()
        regions = []
        for cell in hot_set:
            if cell in visited:
                continue
            region_cells = []
            queue = [cell]
            while queue:
                c = queue.pop(0)
                if c in visited:
                    continue
                visited.add(c)
                region_cells.append(c)
                r, co = divmod(c, HOT_GRID_COLS)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc_ = r + dr, co + dc
                    if 0 <= nr < HOT_GRID_ROWS and 0 <= nc_ < HOT_GRID_COLS:
                        ni = nr * HOT_GRID_COLS + nc_
                        if ni in hot_set and ni not in visited:
                            queue.append(ni)
            if region_cells:
                regions.append(region_cells)

        result = []
        total_hot_cells = sum(len(r) for r in regions)

        for ri, region_cells in enumerate(regions):
            avg_ema = sum(self.grid_congestion_ema[c] for c in region_cells) / len(region_cells)
            if avg_ema >= HOT_SEVERITY_HIGH:
                severity = "HIGH"
            elif avg_ema >= HOT_SEVERITY_MOD:
                severity = "MODERATE"
            else:
                severity = "LOW"

            size_pct = round(len(region_cells) / ncells * 100, 1)

            rows_in = [c // HOT_GRID_COLS for c in region_cells]
            cols_in = [c % HOT_GRID_COLS for c in region_cells]
            min_row, max_row = min(rows_in), max(rows_in)
            min_col, max_col = min(cols_in), max(cols_in)

            center_row = (min_row + max_row) / 2 / HOT_GRID_ROWS
            center_col = (min_col + max_col) / 2 / HOT_GRID_COLS

            r_stalled = sum(cell_stalled[c] for c in region_cells)
            r_slow = sum(cell_slow[c] for c in region_cells)
            r_total = sum(cell_total[c] for c in region_cells)

            if r_total > 0 and r_stalled / max(1, r_total) > 0.6:
                cause = "lane_blockage"
            elif r_total >= 4 and r_stalled + r_slow >= r_total * 0.5:
                cause = "vehicle_accumulation"
            elif center_row > 0.7:
                cause = "downstream_spillback"
            elif 0.3 < center_col < 0.7 and center_row > 0.5:
                cause = "signal_choke"
            else:
                cause = "vehicle_accumulation"

            earliest_hot = min((self.grid_first_hot.get(c, now) for c in region_cells), default=now)
            duration = now - earliest_hot

            recurring = False
            for c in region_cells:
                hist = self.grid_hot_history.get(c)
                if hist and len(hist) >= 8:
                    times = list(hist)
                    for j in range(1, len(times)):
                        if times[j] - times[j - 1] > 5.0:
                            recurring = True
                            break
                if recurring:
                    break

            if recurring:
                persistence = "recurring"
            elif duration < 5.0:
                persistence = "new"
            else:
                persistence = "ongoing"

            if total_hot_cells > self.prev_region_cell_count + 1:
                spread = "expanding"
            elif total_hot_cells < self.prev_region_cell_count - 1:
                spread = "clearing"
            else:
                spread = "localized"

            if 0.25 < center_col < 0.75 and 0.25 < center_row < 0.75:
                impact = "blocking_junctions"
            elif center_col < 0.15 or center_col > 0.85:
                impact = "affecting_service_roads"
            elif center_row > 0.8:
                impact = "affecting_crossings"
            else:
                impact = "blocking_junctions"

            if severity == "HIGH" and persistence in ("ongoing", "recurring"):
                action_priority = "immediate"
            elif severity in ("HIGH", "MODERATE"):
                action_priority = "monitor"
            else:
                action_priority = "no_action"

            if cause == "lane_blockage" and severity == "HIGH":
                suggested = "clear_stalled_vehicle"
            elif cause == "lane_blockage":
                suggested = "deploy_personnel"
            elif cause == "downstream_spillback":
                suggested = "redirect_upstream"
            elif cause == "signal_choke":
                suggested = "open_close_lanes"
            elif action_priority == "immediate":
                suggested = "deploy_personnel"
            elif action_priority == "monitor":
                suggested = "monitor"
            else:
                suggested = "no_action"

            result.append({
                "id": ri,
                "severity": severity,
                "size_pct": size_pct,
                "cells": len(region_cells),
                "cause": cause,
                "persistence": persistence,
                "duration": round(duration, 1),
                "spread": spread,
                "recurring": recurring,
                "impact": impact,
                "action_priority": action_priority,
                "suggested_action": suggested,
                "stalled": r_stalled,
                "slow": r_slow,
                "total_vehicles": r_total,
                "avg_congestion": round(avg_ema * 100),
                "grid_bounds": {
                    "min_row": min_row, "max_row": max_row,
                    "min_col": min_col, "max_col": max_col,
                },
            })

        self.prev_region_cell_count = total_hot_cells

        sev_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
        result.sort(key=lambda x: (sev_order.get(x["severity"], 3), -x["cells"]))

        self.hot_regions_cache = result
        return result

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

        state_counts = {"moving": 0, "stopped": 0, "abnormal": 0}
        behavior_counts = {"stable": 0, "start_stop": 0, "erratic": 0}
        for tid in self.track_positions:
            st = self.track_state.get(tid, "moving")
            state_counts[st] = state_counts.get(st, 0) + 1
            bh = self.track_behavior.get(tid, "stable")
            behavior_counts[bh] = behavior_counts.get(bh, 0) + 1

        now = time.time()
        scored = []
        for tid, impact in self.track_impact_score.items():
            cls_id = self.track_class.get(tid)
            cls_name = self.class_names.get(cls_id, "unknown") if cls_id is not None else "unknown"
            dwell = round(now - self.track_first_seen.get(tid, now), 1)
            scored.append({
                "tid": int(tid),
                "class": cls_name,
                "impact": impact,
                "dwell": dwell,
                "state": self.track_state.get(tid, "moving"),
                "behavior": self.track_behavior.get(tid, "stable"),
            })
        scored.sort(key=lambda x: x["impact"], reverse=True)
        high_impact = scored[:5]

        dwells = [now - self.track_first_seen.get(tid, now) for tid in self.track_positions]
        avg_dwell = round(sum(dwells) / max(1, len(dwells)), 1)
        max_dwell = round(max(dwells, default=0.0), 1)
        over_threshold = sum(1 for d in dwells if d > 10.0)
        dwell_sorted = sorted(
            [(tid, now - self.track_first_seen.get(tid, now)) for tid in self.track_positions],
            key=lambda x: x[1], reverse=True
        )[:3]
        longest_vehicles = []
        for tid, d in dwell_sorted:
            cls_id = self.track_class.get(tid)
            cls_name = self.class_names.get(cls_id, "unknown") if cls_id is not None else "unknown"
            longest_vehicles.append({"tid": int(tid), "class": cls_name, "dwell": round(d, 1)})

        type_impact = {}
        total_impact = 0.0
        for tid, impact in self.track_impact_score.items():
            cls_id = self.track_class.get(tid)
            cls_name = self.class_names.get(cls_id, "unknown") if cls_id is not None else "unknown"
            type_impact[cls_name] = type_impact.get(cls_name, 0.0) + impact
            total_impact += impact
        type_influence = {}
        if total_impact > 0:
            for cls_name, imp in type_impact.items():
                type_influence[cls_name] = round(imp / total_impact, 2)

        hot_regions = self._compute_hot_regions()
        hot_region_summary = {
            "active_count": len(hot_regions),
            "severity_counts": {
                "HIGH": sum(1 for r in hot_regions if r["severity"] == "HIGH"),
                "MODERATE": sum(1 for r in hot_regions if r["severity"] == "MODERATE"),
                "LOW": sum(1 for r in hot_regions if r["severity"] == "LOW"),
            },
            "regions": hot_regions[:8],
        }

        attention_list = []
        for v in scored[:5]:
            reasons = []
            if v["state"] == "stopped":
                reasons.append("stalled")
            if v["state"] == "abnormal":
                reasons.append("abnormal speed")
            if v["behavior"] == "erratic":
                reasons.append("erratic path")
            if v["behavior"] == "start_stop":
                reasons.append("start-stop")
            if v["dwell"] > 10.0:
                reasons.append("long dwell")
            if v["impact"] > 5.0:
                reasons.append("high impact")
            if reasons:
                attention_list.append({**v, "reasons": reasons})
        attention_list = attention_list[:5]

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
            "state_counts": state_counts,
            "behavior_counts": behavior_counts,
            "high_impact_vehicles": high_impact,
            "dwell_stats": {
                "avg_dwell": avg_dwell,
                "max_dwell": max_dwell,
                "over_threshold": over_threshold,
                "longest_vehicles": longest_vehicles,
            },
            "type_influence": type_influence,
            "attention_list": attention_list,
            "hot_regions": hot_region_summary,
        }


def process_stream(index, name, url, stop_event, f_q, m_q, a_q, rf_q, overlay_dict):
    """Main RTSP inference loop for a single source."""
    from server import update_frame, update_raw_frame, update_metrics, add_alert

    print(f"[+] Starting optimized inference: {name}")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    # 1. Connect to RTSP immediately and start pushing raw frames
    cap = create_capture(url)
    if not cap.isOpened():
        print(f"[!] Failed to open source: {name}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # 2. Start background thread to stream raw frames at 5 FPS while model loads
    model_loading = threading.Event()
    model_loading.set()

    loader_thread = threading.Thread(
        target=raw_loader_worker,
        args=(cap, stop_event, model_loading, name, f_q, encode_params),
        daemon=True,
    )
    loader_thread.start()

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
    use_ccn = is_crowd and USE_CCN_FOR_CROWD

    tracker = None
    trail_renderer = None
    heatmap_renderer = None
    full_heatmap_renderer = None
    bbox_smoother = None
    crowd_analytics = None
    analytics = None
    ccn_counter = None

    if is_crowd:
        model_path = MODEL_PATH_CROWD_YOLO
        model = YOLO(model_path, task="detect")
        if device == "cuda":
            model.to(device)

        target_classes = [0]
        CLASS_NAMES = {0: "head"}
        conf_thresh = 0.25
        tracker = None
        trail_renderer = None
        heatmap_renderer = CrowdHeatmapRenderer(accumulate_frames=8)
        bbox_smoother = None
        crowd_analytics = CrowdAnalyticsState(w, h, src_fps)
        analytics = None

        last_ccn_time = 0.0
        ccn_interval = 1.0 / 5.0
        cached_ccn_result = None

        if use_ccn:
            ccn_counter = CrowdCounter(MODEL_PATH_CROWD_CCN, device)
            print(f"[+] {name}: CCN crowd counting model loaded")

        print(f"[+] {name}: YOLO head detection model loaded for crowd analysis")

    else:
        model_path = MODEL_PATH_VEHICLE
        model = YOLO(model_path, task="detect")
        if device == "cuda":
            model.to(device)

        target_classes = [3, 4, 5, 7, 8, 9]
        CLASS_NAMES = {3: "car", 4: "van", 5: "truck", 7: "bus", 8: "motor", 9: "bicycle"}
        conf_thresh = 0.15
        tracker = sv.ByteTrack(
            frame_rate=int(src_fps),
            track_activation_threshold=0.35,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            minimum_consecutive_frames=4,
        )
        trail_renderer = TrailRenderer(max_len=15)
        heatmap_renderer = HeatmapRenderer(max_len=8)
        full_heatmap_renderer = FullHeatmapRenderer(decay=0.90)
        bbox_smoother = BboxSmoother(alpha=BBOX_SMOOTH_ALPHA)
        crowd_analytics = None
        analytics = AnalyticsState(w, h, src_fps, CLASS_NAMES)
        print(f"[+] {name}: YOLO vehicle detection model loaded with ByteTrack")

    # Stop loader thread
    model_loading.clear()
    loader_thread.join(timeout=2.0)
    print(f"[+] {name}: ready for inference")

    last_metrics_time = time.time()
    frame_count = 0

    fps_times = []
    actual_fps = 0.0

    cached_overlay = {
        "heatmap": True,
        "heatmap_full": True,
        "heatmap_trails": True,
        "trails": True,
        "bboxes": True,
        "confidence": conf_thresh,
    }
    last_overlay_check = 0

    # 3. Main inference loop
    while cap.isOpened() and not stop_event.is_set():
        frame_start = time.time()
        tracked = None

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        out = frame.copy()

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

        if frame_start - last_overlay_check > 0.05:
            last_overlay_check = frame_start
            try:
                if name in overlay_dict:
                    raw = overlay_dict[name]
                    if isinstance(raw, dict):
                        cached_overlay = {
                            "heatmap": raw.get("heatmap", True),
                            "heatmap_full": raw.get("heatmap_full", raw.get("heatmap", True)),
                            "heatmap_trails": raw.get("heatmap_trails", raw.get("heatmap", True)),
                            "trails": raw.get("trails", True),
                            "bboxes": raw.get("bboxes", True),
                            "confidence": raw.get("confidence", conf_thresh),
                            "bbox_label": str(raw.get("bbox_label", "speed")),
                        }
                    else:
                        cached_overlay = {
                            "heatmap": bool(raw.get("heatmap", True)),
                            "heatmap_full": bool(raw.get("heatmap_full", raw.get("heatmap", True))),
                            "heatmap_trails": bool(raw.get("heatmap_trails", raw.get("heatmap", True))),
                            "trails": bool(raw.get("trails", True)),
                            "bboxes": bool(raw.get("bboxes", True)),
                            "confidence": float(raw.get("confidence", conf_thresh)),
                            "bbox_label": str(raw.get("bbox_label", "speed")),
                        }
            except Exception:
                pass

        overlay = cached_overlay
        current_conf = overlay.get("confidence", conf_thresh)

        if is_crowd:
            if use_ccn and ccn_counter:
                now = time.time()
                if now - last_ccn_time >= ccn_interval:
                    last_ccn_time = now
                    try:
                        cached_ccn_result = ccn_counter.count(frame)
                    except Exception as e:
                        print(f"CCN Error: {e}")

            if cached_ccn_result:
                crowd_count = cached_ccn_result['count']
                ccn_density = cached_ccn_result['density_map']

                if overlay.get("heatmap", True):
                    heatmap = cached_ccn_result['heatmap']
                    out = cv2.addWeighted(out, 0.6, heatmap, 0.4, 0)

            else:
                crowd_count = 0
                ccn_density = np.zeros((h, w), dtype=np.float32)

            if overlay.get("bboxes", True):
                results = model.predict(
                    frame,
                    imgsz=INFERENCE_SIZE,
                    conf=current_conf,
                    max_det=MAX_DET,
                    verbose=False,
                    classes=target_classes,
                    half=half_precision and device == "cuda",
                    device=device,
                )[0]

                detections = sv.Detections.from_ultralytics(results)
                tracked = detections

                for i in range(len(tracked.xyxy)):
                    x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)

            if not use_ccn or not cached_ccn_result:
                if overlay.get("bboxes", True):
                    crowd_count = len(tracked.xyxy)
                else:
                    results = model.predict(frame, imgsz=INFERENCE_SIZE, conf=current_conf, max_det=MAX_DET, verbose=False, classes=target_classes, half=half_precision and device == "cuda", device=device)[0]
                    crowd_count = len(results)

        else:
            results = model.predict(
                frame,
                imgsz=INFERENCE_SIZE,
                conf=current_conf,
                max_det=MAX_DET,
                verbose=False,
                classes=target_classes,
                half=half_precision and device == "cuda",
                device=device,
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            tracked = tracker.update_with_detections(detections)

            current_ids = trail_renderer.update(tracked) if trail_renderer else set()
            if heatmap_renderer:
                heatmap_renderer.update(tracked)
            if full_heatmap_renderer:
                full_heatmap_renderer.update(tracked, frame.shape)
            crowd_count = None
            ccn_heatmap = None

        now = time.time()
        if now - last_metrics_time >= 0.2:
            if is_crowd:
                if ccn_density is None:
                    ccn_density = np.zeros((h, w), dtype=np.float32)
                metrics = crowd_analytics.update(crowd_count, ccn_density)
            else:
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
        if is_crowd:
            pass
        else:
            if overlay.get("heatmap_full", overlay.get("heatmap", True)) and full_heatmap_renderer:
                full_heatmap_renderer.render(out)

            if overlay.get("heatmap_trails", overlay.get("heatmap", True)) and heatmap_renderer:
                heatmap_renderer.render(out, current_ids)

            if overlay.get("trails", True) and trail_renderer:
                trail_renderer.render(out, current_ids)

            if overlay.get("bboxes", True) and tracked is not None and tracked.tracker_id is not None and bbox_smoother:
                _SPEED_LABELS = {0: "STALLED", 1: "SLOW", 2: "MEDIUM", 3: "FAST"}
                _SPEED_COLORS = {
                    0: (0, 0, 220),      # red
                    1: (0, 165, 255),     # amber/orange
                    2: (0, 210, 130),     # green
                    3: (210, 180, 0),     # cyan-ish
                }
                bbox_label_mode = overlay.get("bbox_label", "speed")
                smoothed_xyxy = bbox_smoother.smooth(tracked)
                tids = tracked.tracker_id
                class_ids = tracked.class_id

                for i in range(len(tids)):
                    x1, y1, x2, y2 = map(int, smoothed_xyxy[i])
                    tid = tids[i]

                    # Determine speed for color coding
                    pos = analytics.track_positions.get(tid)
                    spd_px = pos[2] if pos else 0.0
                    spd_cls = classify_speed(spd_px)
                    spd_color = _SPEED_COLORS[spd_cls]

                    if bbox_label_mode == "speed":
                        # Flow mode: show speed band, color-coded bbox
                        label = _SPEED_LABELS[spd_cls]
                        box_color = spd_color
                    else:
                        # Traffic/vehicle mode: show class name, neutral bbox
                        cls_name = CLASS_NAMES.get(int(class_ids[i]), "") if class_ids is not None else ""
                        label = cls_name.upper() if cls_name else f"#{tid}"
                        box_color = (0, 255, 255)

                    cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 1)

                    ly = y1 - 4 if y1 > 15 else y2 + 12
                    cv2.putText(out, label, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            elif bbox_smoother:
                bbox_smoother.smooth(tracked)

        encode_params_out = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        ret_enc, buffer = cv2.imencode(".jpg", out, encode_params_out)
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

    cap.release()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def process_upload_stream(name, file_path, stop_event, f_q, m_q, a_q, rf_q, overlay_dict, is_crowd=False):
    """Process an uploaded video file with inference - loops the video."""
    print(f"[+] Starting upload inference: {name} from {file_path}, is_crowd={is_crowd}")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    # 1. Open file and push raw frames instantly
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

    # 2. Load model
    device = "cpu"
    half_precision = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(YOLO_GPU_MEMORY_FRACTION)
        device = "cuda"
        half_precision = True
        torch.backends.cudnn.benchmark = True

    use_ccn = is_crowd and USE_CCN_FOR_CROWD
    ccn_counter = None
    model = None

    tracker = None
    trail_renderer = None
    heatmap_renderer = None
    full_heatmap_renderer = None
    bbox_smoother = None
    crowd_analytics = None
    analytics = None

    if is_crowd:
        model_path = MODEL_PATH_CROWD_YOLO
        model = YOLO(model_path, task="detect")
        if device == "cuda":
            model.to(device)

        target_classes = [0]
        CLASS_NAMES = {0: "head"}
        conf_thresh = 0.25
        tracker = None
        trail_renderer = None
        heatmap_renderer = CrowdHeatmapRenderer(accumulate_frames=8)
        bbox_smoother = None
        crowd_analytics = CrowdAnalyticsState(w, h, src_fps)
        analytics = None

        last_ccn_time = 0.0
        ccn_interval = 1.0 / 5.0
        cached_ccn_result = None

        if use_ccn:
            ccn_counter = CrowdCounter(MODEL_PATH_CROWD_CCN, device)
            print(f"[+] {name}: CCN crowd counting model loaded")

        print(f"[+] {name}: YOLO head detection model loaded for crowd analysis")

    else:
        model_path = MODEL_PATH_VEHICLE
        model = YOLO(model_path, task="detect")
        if device == "cuda":
            model.to(device)

        target_classes = [3, 4, 5, 7, 8, 9]
        CLASS_NAMES = {3: "car", 4: "van", 5: "truck", 7: "bus", 8: "motor", 9: "bicycle"}
        conf_thresh = 0.25
        tracker = sv.ByteTrack(
            frame_rate=int(src_fps),
            track_activation_threshold=0.35,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            minimum_consecutive_frames=4,
        )
        trail_renderer = TrailRenderer(max_len=15)
        heatmap_renderer = HeatmapRenderer(max_len=8)
        full_heatmap_renderer = FullHeatmapRenderer(decay=0.90)
        bbox_smoother = BboxSmoother(alpha=BBOX_SMOOTH_ALPHA)
        crowd_analytics = None
        analytics = AnalyticsState(w, h, src_fps, CLASS_NAMES)
        print(f"[+] {name}: YOLO vehicle detection model loaded with ByteTrack")

    last_metrics_time = time.time()

    frame_interval = 1.0 / src_fps
    fps_times = []
    actual_fps = 0.0

    cached_overlay = {
        "heatmap": True,
        "heatmap_full": True,
        "heatmap_trails": True,
        "trails": True,
        "bboxes": True,
        "confidence": conf_thresh,
    }
    last_overlay_check = 0

    while not stop_event.is_set():
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if not use_ccn:
                tracker = sv.ByteTrack(frame_rate=int(src_fps))
                trail_renderer = TrailRenderer(max_len=15)
            heatmap_renderer = HeatmapRenderer(max_len=8)
            full_heatmap_renderer = FullHeatmapRenderer(decay=0.90)
            continue

        out = frame.copy()

        fps_times.append(frame_start)
        if len(fps_times) > 30:
            fps_times.pop(0)
        if len(fps_times) >= 2:
            elapsed = fps_times[-1] - fps_times[0]
            if elapsed > 0:
                actual_fps = (len(fps_times) - 1) / elapsed

        if frame_start - last_overlay_check > 0.05:
            last_overlay_check = frame_start
            try:
                if name in overlay_dict:
                    raw = overlay_dict[name]
                    if isinstance(raw, dict):
                        cached_overlay = {
                            "heatmap": raw.get("heatmap", True),
                            "heatmap_full": raw.get("heatmap_full", raw.get("heatmap", True)),
                            "heatmap_trails": raw.get("heatmap_trails", raw.get("heatmap", True)),
                            "trails": raw.get("trails", True),
                            "bboxes": raw.get("bboxes", True),
                            "confidence": raw.get("confidence", conf_thresh),
                            "bbox_label": str(raw.get("bbox_label", "speed")),
                        }
                    else:
                        cached_overlay = {
                            "heatmap": bool(raw.get("heatmap", True)),
                            "heatmap_full": bool(raw.get("heatmap_full", raw.get("heatmap", True))),
                            "heatmap_trails": bool(raw.get("heatmap_trails", raw.get("heatmap", True))),
                            "trails": bool(raw.get("trails", True)),
                            "bboxes": bool(raw.get("bboxes", True)),
                            "confidence": float(raw.get("confidence", conf_thresh)),
                            "bbox_label": str(raw.get("bbox_label", "speed")),
                        }
            except Exception:
                pass

        overlay = cached_overlay
        current_conf = overlay.get("confidence", conf_thresh)

        if is_crowd:
            if use_ccn and ccn_counter:
                now = time.time()
                if now - last_ccn_time >= ccn_interval:
                    last_ccn_time = now
                    try:
                        cached_ccn_result = ccn_counter.count(frame)
                    except Exception as e:
                        print(f"CCN Error: {e}")

            if cached_ccn_result:
                crowd_count = cached_ccn_result['count']
                ccn_density = cached_ccn_result['density_map']

                if overlay.get("heatmap", True):
                    heatmap = cached_ccn_result['heatmap']
                    out = cv2.addWeighted(out, 0.6, heatmap, 0.4, 0)
            else:
                crowd_count = 0
                ccn_density = np.zeros((h, w), dtype=np.float32)

            if overlay.get("bboxes", True):
                results = model.predict(
                    frame,
                    imgsz=INFERENCE_SIZE,
                    conf=current_conf,
                    max_det=MAX_DET,
                    verbose=False,
                    classes=target_classes,
                    half=half_precision and device == "cuda",
                    device=device,
                )[0]

                detections = sv.Detections.from_ultralytics(results)
                tracked = detections

                for i in range(len(tracked.xyxy)):
                    x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)

            if not use_ccn or not cached_ccn_result:
                if overlay.get("bboxes", True):
                    crowd_count = len(tracked.xyxy)
                else:
                    results = model.predict(frame, imgsz=INFERENCE_SIZE, conf=current_conf, max_det=MAX_DET, verbose=False, classes=target_classes, half=half_precision and device == "cuda", device=device)[0]
                    crowd_count = len(results)

        else:
            results = model.predict(
                frame,
                imgsz=INFERENCE_SIZE,
                conf=current_conf,
                max_det=MAX_DET,
                verbose=False,
                classes=target_classes,
                half=half_precision and device == "cuda",
                device=device,
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            tracked = tracker.update_with_detections(detections)
            current_ids = trail_renderer.update(tracked) if trail_renderer else set()
            if heatmap_renderer:
                heatmap_renderer.update(tracked)
            if full_heatmap_renderer:
                full_heatmap_renderer.update(tracked, frame.shape)

            if overlay.get("heatmap_full", overlay.get("heatmap", True)) and full_heatmap_renderer:
                full_heatmap_renderer.render(out)

            if overlay.get("heatmap_trails", overlay.get("heatmap", True)) and heatmap_renderer:
                heatmap_renderer.render(out, current_ids)

            if overlay.get("trails", True) and trail_renderer:
                trail_renderer.render(out, current_ids)

            # Compute and send metrics
            now = time.time()
            if now - last_metrics_time >= 0.5:  # Send metrics every 0.5s
                if analytics:
                    metrics = analytics.update(tracked)
                    metrics["fps"] = round(actual_fps, 1)
                    try:
                        if not m_q.full():
                            m_q.put_nowait((name, metrics))
                    except:
                        pass
                last_metrics_time = now

        encode_params_out = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        ret_enc, buffer = cv2.imencode(".jpg", out, encode_params_out)
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

        elapsed = time.time() - frame_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

    cap.release()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[+] Upload inference stopped: {name}")
