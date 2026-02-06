"""Overlay renderers for video analytics streams."""

import os
import cv2
import numpy as np

_COLORMAP_BY_NAME = {
    "AUTUMN": cv2.COLORMAP_AUTUMN,
    "BONE": cv2.COLORMAP_BONE,
    "JET": cv2.COLORMAP_JET,
    "WINTER": cv2.COLORMAP_WINTER,
    "RAINBOW": cv2.COLORMAP_RAINBOW,
    "OCEAN": cv2.COLORMAP_OCEAN,
    "SUMMER": cv2.COLORMAP_SUMMER,
    "SPRING": cv2.COLORMAP_SPRING,
    "COOL": cv2.COLORMAP_COOL,
    "HSV": cv2.COLORMAP_HSV,
    "PINK": cv2.COLORMAP_PINK,
    "HOT": cv2.COLORMAP_HOT,
    "PARULA": cv2.COLORMAP_PARULA,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "CIVIDIS": cv2.COLORMAP_CIVIDIS,
    "TWILIGHT": cv2.COLORMAP_TWILIGHT,
    "TWILIGHT_SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED,
    "TURBO": cv2.COLORMAP_TURBO,
}


def _resolve_colormap(env_key: str, default_name: str):
    name = os.environ.get(env_key, default_name).strip().upper()
    return _COLORMAP_BY_NAME.get(name, _COLORMAP_BY_NAME[default_name])


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


class CrowdHeatmapRenderer:
    """Renders crowd density heatmaps from head detections."""

    def __init__(self, accumulate_frames=10):
        self.accumulate_frames = accumulate_frames
        self.density_accumulator = None
        self.frame_count = 0
        self.colormap = _resolve_colormap("IRIS_CROWD_HEATMAP_COLORMAP", "JET")

    def update(self, tracked_detections):
        """Update density accumulator from head detections."""
        if tracked_detections is None or tracked_detections.tracker_id is None:
            return

        xyxys = tracked_detections.xyxy
        tids = tracked_detections.tracker_id

        if len(xyxys) == 0:
            return

        h, w = 480, 640  # Assuming standard resolution
        if self.density_accumulator is None:
            self.density_accumulator = np.zeros((h, w), dtype=np.float32)

        # Add Gaussian blobs for each detected head
        for i in range(len(tids)):
            x1, y1, x2, y2 = xyxys[i]
            cx = int((x1 + x2) * 0.5)
            cy = int((y1 + y2) * 0.5)

            # Head radius based on bbox size
            head_radius = max(8, int(min(x2 - x1, y2 - y1) * 0.3))

            # Create Gaussian kernel for this head
            y_grid, x_grid = np.ogrid[:h, :w]
            dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
            gaussian = np.exp(-dist_sq / (2 * (head_radius**2)))

            # Add to accumulator
            self.density_accumulator += gaussian * 2.0

        self.frame_count += 1

    def render(self, frame):
        """Render crowd density heatmap on frame."""
        if self.density_accumulator is None:
            return frame

        h, w = frame.shape[:2]

        # Normalize accumulator
        if self.density_accumulator.max() > 0:
            # Use percentile-based normalization for better contrast
            nonzero_vals = self.density_accumulator[self.density_accumulator > 0]
            if len(nonzero_vals) > 0:
                p95 = np.percentile(nonzero_vals, 95)
                max_val = max(self.density_accumulator.max(), p95)
                density_norm = np.clip(self.density_accumulator / max_val, 0, 1)
            else:
                density_norm = self.density_accumulator / self.density_accumulator.max()
        else:
            density_norm = self.density_accumulator

        # Resize density to match frame if needed
        if density_norm.shape != (h, w):
            density_norm = cv2.resize(density_norm, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply Gaussian blur for smooth heatmap
        density_blurred = cv2.GaussianBlur(density_norm, (25, 25), 0)

        # Apply colormap
        heatmap = cv2.applyColorMap((density_blurred * 255).astype(np.uint8), self.colormap)

        # Blend with original frame
        result = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # Decay accumulator over time (for real-time effect)
        self.density_accumulator *= 0.92

        # Reset accumulator periodically to prevent buildup
        if self.frame_count >= self.accumulate_frames:
            self.density_accumulator *= 0.5
            self.frame_count = 0

        return result


class HeatmapRenderer:
    """Per-vehicle motion-tail heatmap similar to traffic streak visuals."""

    def __init__(self, max_len=16, scale=0.5, blur=7):
        self.scale = float(os.environ.get("IRIS_HEATMAP_SCALE", str(scale)))
        self.blur = int(os.environ.get("IRIS_HEATMAP_BLUR", str(blur)))
        self.blur = self.blur if self.blur % 2 == 1 else self.blur + 1
        self.colormap = _resolve_colormap("IRIS_HEATMAP_COLORMAP", "JET")
        self.overlay_alpha = float(os.environ.get("IRIS_HEATMAP_ALPHA", "0.50"))
        self.max_len = int(os.environ.get("IRIS_HEATMAP_TRAIL_LEN", str(max_len)))
        self.max_missing = int(os.environ.get("IRIS_HEATMAP_TRAIL_MISSING", "5"))
        self.match_dist = float(os.environ.get("IRIS_HEATMAP_MATCH_DIST", "80"))
        self.trails = {}          # track_id -> [(x, y), ...]
        self.last_seen = {}       # track_id -> missed frame count
        self._next_id = 1

    def _assign_ids_without_tracker(self, points):
        assigned = []
        used = set()
        live_ids = [tid for tid, miss in self.last_seen.items() if miss <= self.max_missing]

        for px, py in points:
            best_id = None
            best_d2 = self.match_dist * self.match_dist
            for tid in live_ids:
                if tid in used:
                    continue
                trail = self.trails.get(tid) or []
                if not trail:
                    continue
                tx, ty = trail[-1]
                d2 = float((px - tx) * (px - tx) + (py - ty) * (py - ty))
                if d2 < best_d2:
                    best_d2 = d2
                    best_id = tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self.trails[best_id] = []
                self.last_seen[best_id] = 0
            assigned.append(best_id)
            used.add(best_id)

        return assigned

    def update(self, tracked_detections, frame_shape):
        if tracked_detections is None or not hasattr(tracked_detections, "xyxy"):
            return set()

        xyxys = tracked_detections.xyxy
        if xyxys is None or len(xyxys) == 0:
            for tid in list(self.last_seen.keys()):
                self.last_seen[tid] = self.last_seen.get(tid, 0) + 1
                if self.last_seen[tid] > self.max_missing:
                    self.last_seen.pop(tid, None)
                    self.trails.pop(tid, None)
            return set()

        points = []
        for i in range(len(xyxys)):
            x1, y1, x2, y2 = xyxys[i]
            # Lower-center anchor gives road-attached tails.
            points.append((int((x1 + x2) * 0.5), int(y1 * 0.35 + y2 * 0.65)))

        tids = getattr(tracked_detections, "tracker_id", None)
        if tids is None or len(tids) != len(points):
            tids = self._assign_ids_without_tracker(points)

        current_ids = set()
        for (px, py), tid in zip(points, tids):
            tid = int(tid)
            current_ids.add(tid)
            trail = self.trails.setdefault(tid, [])
            trail.append((px, py))
            if len(trail) > self.max_len:
                trail.pop(0)
            self.last_seen[tid] = 0

        for tid in list(self.last_seen.keys()):
            if tid not in current_ids:
                self.last_seen[tid] = self.last_seen.get(tid, 0) + 1
                if self.last_seen[tid] > self.max_missing:
                    self.last_seen.pop(tid, None)
                    self.trails.pop(tid, None)

        return current_ids

    def render(self, frame, current_ids=None):
        if frame is None:
            return

        h, w = frame.shape[:2]
        sh, sw = max(1, int(h * self.scale)), max(1, int(w * self.scale))
        mask = np.zeros((sh, sw), dtype=np.uint8)

        for tid, trail in self.trails.items():
            if len(trail) < 2:
                continue
            n = len(trail)
            for i in range(1, n):
                p0 = trail[i - 1]
                p1 = trail[i]
                x0, y0 = int(p0[0] * self.scale), int(p0[1] * self.scale)
                x1, y1 = int(p1[0] * self.scale), int(p1[1] * self.scale)
                progress = i / max(1, n - 1)
                intensity = int(35 + 220 * progress)   # blue tail -> red head in JET
                thickness = max(1, int(2 + 4 * progress))
                cv2.line(mask, (x0, y0), (x1, y1), intensity, thickness, cv2.LINE_AA)

            # Bright head point.
            hx, hy = trail[-1]
            cv2.circle(mask, (int(hx * self.scale), int(hy * self.scale)), 3, 255, -1)

        if mask.max() <= 0:
            return

        mask_blur = cv2.GaussianBlur(mask, (self.blur, self.blur), 0)
        heatmap = cv2.applyColorMap(mask_blur, self.colormap)
        heatmap_full = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_full = cv2.resize(mask_blur, (w, h), interpolation=cv2.INTER_LINEAR) > 8
        if not mask_full.any():
            return

        a = min(max(self.overlay_alpha, 0.05), 0.95)
        frame[mask_full] = cv2.addWeighted(frame[mask_full], 1.0 - a, heatmap_full[mask_full], a, 0)


class FullHeatmapRenderer:
    """Full-frame light-blue wash with congestion hotspots."""

    def __init__(self, scale=0.24, decay=0.985, blur=21, update_stride=2):
        self.scale = float(os.environ.get("IRIS_FULL_HEATMAP_SCALE", str(scale)))
        self.decay = float(os.environ.get("IRIS_FULL_HEATMAP_DECAY", str(decay)))
        self.blur = int(os.environ.get("IRIS_FULL_HEATMAP_BLUR", str(blur)))
        self.blur = self.blur if self.blur % 2 == 1 else self.blur + 1
        self.accumulator = None
        self.update_stride = max(1, int(os.environ.get("IRIS_FULL_HEATMAP_UPDATE_STRIDE", str(update_stride))))
        self.colormap = _resolve_colormap("IRIS_FULL_HEATMAP_COLORMAP", "JET")
        self.overlay_alpha = float(os.environ.get("IRIS_FULL_HEATMAP_ALPHA", "0.55"))
        self.base_blue_alpha = float(os.environ.get("IRIS_FULL_HEATMAP_BASE_BLUE_ALPHA", "0.16"))
        self.no_det_decay = float(os.environ.get("IRIS_FULL_HEATMAP_NO_DET_DECAY", "0.80"))
        self.no_det_clear_frames = int(os.environ.get("IRIS_FULL_HEATMAP_NO_DET_CLEAR_FRAMES", "20"))
        self.hotspot_min_activation = float(os.environ.get("IRIS_FULL_HEATMAP_MIN_ACTIVATION", "0.18"))
        self._no_det_frames = 0
        self._step = 0

    def _ensure_accumulator(self, frame_shape):
        h, w = frame_shape[:2]
        sh, sw = int(h * self.scale), int(w * self.scale)
        if self.accumulator is None or self.accumulator.shape != (sh, sw):
            self.accumulator = np.zeros((sh, sw), dtype=np.float32)

    def update(self, tracked_detections, frame_shape):
        if tracked_detections is None:
            self._no_det_frames += 1
            if self.accumulator is not None:
                self.accumulator *= self.no_det_decay
                if self._no_det_frames >= self.no_det_clear_frames:
                    self.accumulator.fill(0.0)
            return
        self._step += 1
        if (self._step % self.update_stride) != 0:
            return

        self._ensure_accumulator(frame_shape)

        xyxys = tracked_detections.xyxy if hasattr(tracked_detections, "xyxy") else None
        if xyxys is None or len(xyxys) == 0:
            self._no_det_frames += 1
            self.accumulator *= self.no_det_decay
            if self._no_det_frames >= self.no_det_clear_frames:
                self.accumulator.fill(0.0)
            return
        self._no_det_frames = 0

        for i in range(len(xyxys)):
            x1, y1, x2, y2 = xyxys[i]
            # CENTERED: center of bbox for better visualization
            gx = int((x1 + x2) * 0.5)
            gy = int((y1 + y2) * 0.5)
            r = int(max(6, min(x2 - x1, y2 - y1) * 0.4))

            sx = int(gx * self.scale)
            sy = int(gy * self.scale)
            sr = max(3, int(r * self.scale))

            if 0 <= sx < self.accumulator.shape[1] and 0 <= sy < self.accumulator.shape[0]:
                cv2.circle(self.accumulator, (sx, sy), sr, 1.0, -1)

    def render(self, frame):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return

        # 1) Apply light blue global overlay always in full mode.
        blue_overlay = np.zeros_like(frame, dtype=np.uint8)
        blue_overlay[:, :] = (220, 120, 45)  # BGR light blue tint
        b = min(max(self.base_blue_alpha, 0.02), 0.5)
        frame[:] = cv2.addWeighted(frame, 1.0 - b, blue_overlay, b, 0)

        if self.accumulator is None:
            return

        acc = self.accumulator
        if acc.max() <= 0:
            return

        nonzero = acc[acc > 0]
        if len(nonzero) == 0:
            return
        p90 = np.percentile(nonzero, 90)
        max_val = max(acc.max(), p90, 1e-6)
        norm = np.clip(acc / max_val, 0, 1)

        # Emphasize high-density zones so congestion appears red/yellow.
        hotspot = np.power(norm, 1.35)
        hotspot = cv2.GaussianBlur(hotspot, (self.blur, self.blur), 0)
        if float(hotspot.max()) < self.hotspot_min_activation:
            # Ignore weak residual energy to avoid ghost dots when scene is empty.
            self.accumulator *= self.no_det_decay
            return
        heatmap = cv2.applyColorMap((hotspot * 255).astype(np.uint8), self.colormap)

        heatmap_full = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        if heatmap_full is None:
            return
        mask = cv2.resize(hotspot, (w, h), interpolation=cv2.INTER_LINEAR) > 0.08
        if not mask.any():
            return

        a = min(max(self.overlay_alpha, 0.05), 0.9)
        frame[mask] = cv2.addWeighted(frame[mask], 1.0 - a, heatmap_full[mask], a, 0)

        self.accumulator *= self.decay
