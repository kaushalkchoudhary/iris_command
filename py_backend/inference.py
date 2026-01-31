import os
import cv2
import time
import threading
import numpy as np
import gc
import multiprocessing as mp
from collections import deque

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

# Hot region grid settings
HOT_GRID_COLS = 6
HOT_GRID_ROWS = 4
HOT_THRESHOLD = 0.4       # cell congestion EMA > 40% = hot
HOT_SEVERITY_MOD = 0.6    # > 60% = MODERATE
HOT_SEVERITY_HIGH = 0.8   # > 80% = HIGH


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

            # Count per class and store per-vehicle info
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

        # Purge expired tracks (not seen for > 2s)
        expired = [tid for tid, ts in self.track_last_seen.items() if now - ts > 2.0]
        for tid in expired:
            for d in (self.track_first_seen, self.track_last_seen, self.track_class,
                      self.track_bbox_dims, self.track_speed_history,
                      self.track_position_history, self.track_state,
                      self.track_behavior, self.track_impact_score):
                d.pop(tid, None)

        # Update per-vehicle persistent state
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

            # Classify state
            self.track_state[tid] = self._classify_state(tid, speed)

            # Classify behavior
            self.track_behavior[tid] = self._classify_behavior(tid)

            # Compute impact
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

        # Check start-stop: count transitions across stalled threshold
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

        # Check erratic: max perpendicular deviation from first-to-last line
        if len(pts) >= 3:
            fx, fy = pts[0]
            lx, ly = pts[-1]
            dx, dy = lx - fx, ly - fy
            line_len = np.sqrt(dx * dx + dy * dy)
            if line_len > 1.0:
                max_dev = 0.0
                for px, py in pts[1:-1]:
                    # perpendicular distance from point to line
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

        # Simple lane factor based on position in frame (center = higher impact)
        pos_hist = self.track_position_history.get(tid)
        lane_factor = 0.0
        if pos_hist:
            cx = pos_hist[-1][0]
            center_dist = abs(cx - self.width / 2.0) / (self.width / 2.0)
            lane_factor = max(0.0, 0.5 - center_dist)  # 0 to 0.5

        return round(size_w * dwell_factor * blockage * (1.0 + lane_factor), 2)

    def _compute_hot_regions(self):
        now = time.time()
        ncells = HOT_GRID_COLS * HOT_GRID_ROWS
        cell_w = self.width / HOT_GRID_COLS
        cell_h = self.height / HOT_GRID_ROWS

        # Count vehicles per cell
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

        # Update congestion EMA per cell
        for i in range(ncells):
            tot = max(1, cell_total[i])
            ratio = (cell_stalled[i] + cell_slow[i]) / tot if cell_total[i] > 0 else 0.0
            self.grid_congestion_ema[i] = self.grid_congestion_ema[i] * 0.7 + ratio * 0.3

        # Identify hot cells
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

        # Flood-fill merge adjacent hot cells into regions
        visited = set()
        regions = []
        for cell in hot_set:
            if cell in visited:
                continue
            # BFS
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

        # Build region descriptors
        result = []
        total_hot_cells = sum(len(r) for r in regions)

        for ri, region_cells in enumerate(regions):
            # Severity: average congestion EMA in region
            avg_ema = sum(self.grid_congestion_ema[c] for c in region_cells) / len(region_cells)
            if avg_ema >= HOT_SEVERITY_HIGH:
                severity = "HIGH"
            elif avg_ema >= HOT_SEVERITY_MOD:
                severity = "MODERATE"
            else:
                severity = "LOW"

            # Size: fraction of frame
            size_pct = round(len(region_cells) / ncells * 100, 1)

            # Bounding box of region in grid coordinates
            rows_in = [c // HOT_GRID_COLS for c in region_cells]
            cols_in = [c % HOT_GRID_COLS for c in region_cells]
            min_row, max_row = min(rows_in), max(rows_in)
            min_col, max_col = min(cols_in), max(cols_in)

            # Region center in normalized coords (0-1)
            center_row = (min_row + max_row) / 2 / HOT_GRID_ROWS
            center_col = (min_col + max_col) / 2 / HOT_GRID_COLS

            # Stalled vs total in region
            r_stalled = sum(cell_stalled[c] for c in region_cells)
            r_slow = sum(cell_slow[c] for c in region_cells)
            r_total = sum(cell_total[c] for c in region_cells)

            # Cause classification
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

            # Persistence
            earliest_hot = min((self.grid_first_hot.get(c, now) for c in region_cells), default=now)
            duration = now - earliest_hot

            # Check recurring: any cell had multiple hot episodes
            recurring = False
            for c in region_cells:
                hist = self.grid_hot_history.get(c)
                if hist and len(hist) >= 8:
                    # Check if there's a gap (was cold then hot again)
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

            # Spread direction
            if total_hot_cells > self.prev_region_cell_count + 1:
                spread = "expanding"
            elif total_hot_cells < self.prev_region_cell_count - 1:
                spread = "clearing"
            else:
                spread = "localized"

            # Impact level
            if 0.25 < center_col < 0.75 and 0.25 < center_row < 0.75:
                impact = "blocking_junctions"
            elif center_col < 0.15 or center_col > 0.85:
                impact = "affecting_service_roads"
            elif center_row > 0.8:
                impact = "affecting_crossings"
            else:
                impact = "blocking_junctions"

            # Action priority
            if severity == "HIGH" and persistence in ("ongoing", "recurring"):
                action_priority = "immediate"
            elif severity in ("HIGH", "MODERATE"):
                action_priority = "monitor"
            else:
                action_priority = "no_action"

            # Suggested action
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

        # Sort by severity (HIGH first) then by size
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

        # Vehicle state and behavior counts
        state_counts = {"moving": 0, "stopped": 0, "abnormal": 0}
        behavior_counts = {"stable": 0, "start_stop": 0, "erratic": 0}
        for tid in self.track_positions:
            st = self.track_state.get(tid, "moving")
            state_counts[st] = state_counts.get(st, 0) + 1
            bh = self.track_behavior.get(tid, "stable")
            behavior_counts[bh] = behavior_counts.get(bh, 0) + 1

        # High-impact vehicles (top 5)
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

        # Dwell stats
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

        # Type influence (fraction of total impact by class)
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

        # Hot regions
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

        # Attention list (top 5 vehicles needing attention)
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
    analytics = AnalyticsState(w, h, src_fps, CLASS_NAMES)

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

                # Per-vehicle state tag
                v_state = analytics.track_state.get(tid, "moving")
                v_impact = analytics.track_impact_score.get(tid, 0.0)
                if v_state == "abnormal":
                    tag_color = (0, 0, 255)  # Red
                    tag_text = f"#{tid} ABNORMAL"
                elif v_impact > 5.0:
                    tag_color = (0, 165, 255)  # Orange
                    tag_text = f"#{tid} HIGH-IMPACT"
                else:
                    tag_color = (0, 200, 0)  # Green
                    tag_text = f"#{tid} NORMAL"
                tag_y = y1 - 16 if y1 > 30 else y2 + 24
                cv2.putText(out, tag_text, (x1, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, tag_color, 1, cv2.LINE_AA)
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


def start_backend(idx, url, name, overlay_config=None):
    overlay_shared_dict[name] = overlay_config if overlay_config else get_overlay_state(name)
    stop = spawn_ctx.Event()
    p = spawn_ctx.Process(
        target=process_stream,
        args=(idx, name, url, stop, frame_queue, metrics_queue, alert_queue, raw_frame_queue, overlay_shared_dict),
        daemon=True,
    )
    p.start()
    return p, stop


def start_upload_backend(file_path, name, overlay_config=None):
    """Start inference on an uploaded video file."""
    overlay_shared_dict[name] = overlay_config if overlay_config else get_overlay_state(name)
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
    analytics = AnalyticsState(w, h, src_fps, CLASS_NAMES)

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

                # Per-vehicle state tag
                v_state = analytics.track_state.get(tid, "moving")
                v_impact = analytics.track_impact_score.get(tid, 0.0)
                if v_state == "abnormal":
                    tag_color = (0, 0, 255)  # Red
                    tag_text = f"#{tid} ABNORMAL"
                elif v_impact > 5.0:
                    tag_color = (0, 165, 255)  # Orange
                    tag_text = f"#{tid} HIGH-IMPACT"
                else:
                    tag_color = (0, 200, 0)  # Green
                    tag_text = f"#{tid} NORMAL"
                tag_y = y1 - 16 if y1 > 30 else y2 + 24
                cv2.putText(out, tag_text, (x1, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, tag_color, 1, cv2.LINE_AA)
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
