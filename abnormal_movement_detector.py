"""Robust abnormal-movement detector using smoothed, normalized trajectories.

Version 2: the original implementation compared two adjacent raw history
points to estimate acceleration. In dense/CCTV footage, a few pixels of
tracker jitter divided by a tiny time delta produces enormous apparent
"acceleration" values that have nothing to do with real sudden movement --
this was confirmed by running the detector against real footage, where it
fired 25 times in 6 seconds on ordinary pedestrian traffic.

This version instead:
  1. Smooths velocity over short sub-windows (not single adjacent points).
  2. Normalizes displacement by the person's body height (perspective-safe).
  3. Requires the CURRENT smoothed speed to itself be meaningfully fast
     before "sudden acceleration" can fire -- a jump from near-zero to
     "still slow" is normal walking, not abnormal movement, even if the
     ratio between the two looks large.
"""
from collections import defaultdict, deque
import math
from safety_config import (
    ABNORMAL_SPEED_BODY_HEIGHTS_SEC,
    ABNORMAL_ACCEL_BODY_HEIGHTS_SEC2,
    ABNORMAL_REVERSE_ANGLE_DEG,
    ABNORMAL_MIN_SPEED_PX_SEC,
    ABNORMAL_CONFIRM_FRAMES,
    MOVEMENT_WINDOW,
    MAX_HISTORY_FRAMES,
)
from safety_geometry import angle_between, center


class AbnormalMovementDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = defaultdict(lambda: deque(maxlen=max(20, MOVEMENT_WINDOW * 2)))
        self.alert_state = defaultdict(int)
        self.alerted = set()

    @staticmethod
    def _windowed_velocity(points):
        """Average velocity vector + normalized speed across a list of
        (t, (x, y), body_h) samples, using the first/last sample only
        (span-based, not per-step), which smooths out per-frame jitter."""
        if len(points) < 2:
            return (0.0, 0.0), 0.0, 0.0
        t0, p0, h0 = points[0]
        t1, p1, h1 = points[-1]
        dt = max(t1 - t0, 1e-3)
        vx = (p1[0] - p0[0]) / dt
        vy = (p1[1] - p0[1]) / dt
        scale = max(30.0, (h0 + h1) * 0.5)
        speed_px = math.hypot(vx, vy)
        return (vx, vy), speed_px, speed_px / scale

    def update(self, people, frame_index, fps, roi_polygon=None):
        fps = max(float(fps), 1.0)
        now = frame_index / fps
        events = []
        active_ids = set()

        # Sub-window length in seconds: short enough to catch real sudden
        # movement, long enough to average out tracker jitter.
        sub_window_sec = 0.35

        for person in people:
            pid = person.get("stable_id")
            if pid is None:
                continue
            pid = int(pid)
            active_ids.add(pid)

            bbox = person["bbox"]
            p = center(bbox)
            body_h = max(30.0, float(bbox[3] - bbox[1]))
            hist = self.history[pid]
            hist.append((now, p, body_h))

            # Need enough real time span, not just enough samples -- this
            # matters at both low and high fps.
            if len(hist) < 6 or (hist[-1][0] - hist[0][0]) < sub_window_sec * 1.6:
                continue

            points = list(hist)

            # "Current" window: most recent sub_window_sec of samples.
            current_pts = [pt for pt in points if now - pt[0] <= sub_window_sec]
            # "Baseline" window: the sub_window_sec immediately before that.
            baseline_pts = [
                pt for pt in points
                if sub_window_sec < now - pt[0] <= sub_window_sec * 2.2
            ]

            if len(current_pts) < 3 or len(baseline_pts) < 3:
                continue

            cur_vec, cur_speed_px, cur_speed_norm = self._windowed_velocity(current_pts)
            base_vec, base_speed_px, base_speed_norm = self._windowed_velocity(baseline_pts)

            # A jump is only "sudden acceleration" if the CURRENT speed is
            # itself already meaningfully fast -- going from standing still
            # to a normal walking pace is not abnormal, even though the
            # ratio between the two looks dramatic.
            accel_delta_norm = cur_speed_norm - base_speed_norm
            sudden_accel = (
                cur_speed_px >= ABNORMAL_MIN_SPEED_PX_SEC
                and cur_speed_norm >= ABNORMAL_SPEED_BODY_HEIGHTS_SEC * 0.55
                and accel_delta_norm >= ABNORMAL_ACCEL_BODY_HEIGHTS_SEC2
            )

            high_speed = (
                cur_speed_px >= ABNORMAL_MIN_SPEED_PX_SEC
                and cur_speed_norm >= ABNORMAL_SPEED_BODY_HEIGHTS_SEC
            )

            reverse = (
                cur_speed_px >= ABNORMAL_MIN_SPEED_PX_SEC
                and base_speed_px >= ABNORMAL_MIN_SPEED_PX_SEC
                and angle_between(base_vec, cur_vec) >= ABNORMAL_REVERSE_ANGLE_DEG
            )

            abnormal = high_speed or sudden_accel or reverse

            if abnormal:
                self.alert_state[pid] += 1
            else:
                self.alert_state[pid] = max(0, self.alert_state[pid] - 1)

            if self.alert_state[pid] >= ABNORMAL_CONFIRM_FRAMES and pid not in self.alerted:
                self.alerted.add(pid)
                reason = (
                    "high_speed" if high_speed
                    else "sudden_acceleration" if sudden_accel
                    else "direction_reversal"
                )
                events.append({
                    "event_type": "ABNORMAL_MOVEMENT",
                    "severity": "MEDIUM",
                    "person_id": pid,
                    "confidence": min(
                        0.95,
                        0.55 + min(0.30, cur_speed_norm / max(ABNORMAL_SPEED_BODY_HEIGHTS_SEC, 0.1) * 0.15)
                    ),
                    "frame": frame_index,
                    "timestamp": now,
                    "details": {
                        "reason": reason,
                        "current_speed_body_heights_sec": round(cur_speed_norm, 2),
                        "baseline_speed_body_heights_sec": round(base_speed_norm, 2),
                        "speed_px_sec": round(cur_speed_px, 1),
                    },
                })

            if not abnormal and self.alert_state[pid] == 0:
                self.alerted.discard(pid)

        for pid in list(self.history):
            if pid not in active_ids:
                if self.history[pid] and (now - self.history[pid][-1][0]) > (MAX_HISTORY_FRAMES / fps):
                    self.history.pop(pid, None)
                    self.alert_state.pop(pid, None)
                    self.alerted.discard(pid)

        return events
