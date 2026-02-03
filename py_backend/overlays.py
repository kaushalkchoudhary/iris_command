"""Overlay renderers for video analytics streams."""

import cv2
import numpy as np


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
        heatmap = cv2.applyColorMap((density_blurred * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

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


class FullHeatmapRenderer:
    """Renders a full-scene heatmap from vehicle ground contact points."""

    def __init__(self, scale=0.5, decay=0.94, blur=21):
        self.scale = scale
        self.decay = decay
        self.blur = blur if blur % 2 == 1 else blur + 1
        self.accumulator = None

    def _ensure_accumulator(self, frame_shape):
        h, w = frame_shape[:2]
        sh, sw = int(h * self.scale), int(w * self.scale)
        if self.accumulator is None or self.accumulator.shape != (sh, sw):
            self.accumulator = np.zeros((sh, sw), dtype=np.float32)

    def update(self, tracked_detections, frame_shape):
        if tracked_detections is None or tracked_detections.tracker_id is None:
            return

        self._ensure_accumulator(frame_shape)

        xyxys = tracked_detections.xyxy
        tids = tracked_detections.tracker_id
        if len(tids) == 0:
            return

        for i in range(len(tids)):
            x1, y1, x2, y2 = xyxys[i]
            gx = int((x1 + x2) * 0.5)
            gy = int(y2)
            r = int(max(6, min(x2 - x1, y2 - y1) * 0.35))

            sx = int(gx * self.scale)
            sy = int(gy * self.scale)
            sr = max(2, int(r * self.scale))

            if 0 <= sx < self.accumulator.shape[1] and 0 <= sy < self.accumulator.shape[0]:
                cv2.circle(self.accumulator, (sx, sy), sr, 1.0, -1)

    def render(self, frame):
        if self.accumulator is None:
            return
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return

        if frame.dtype == object or not np.issubdtype(frame.dtype, np.number):
            return

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return
        acc = self.accumulator

        if acc.max() > 0:
            nonzero = acc[acc > 0]
            if len(nonzero) > 0:
                p95 = np.percentile(nonzero, 95)
                max_val = max(acc.max(), p95)
                norm = np.clip(acc / max_val, 0, 1)
            else:
                norm = acc / acc.max()
        else:
            norm = acc

        blur = cv2.GaussianBlur(norm, (self.blur, self.blur), 0)
        heatmap = cv2.applyColorMap((blur * 255).astype(np.uint8), cv2.COLORMAP_JET)

        heatmap_full = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        if heatmap_full is None:
            return
        mask = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR) > 0.05
        if mask is None or not mask.any():
            return

        frame[mask] = cv2.addWeighted(frame[mask], 0.55, heatmap_full[mask], 0.45, 0)

        self.accumulator *= self.decay
