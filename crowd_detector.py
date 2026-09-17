"""Crowd count, density and rapid crowd-buildup detector."""
from collections import deque
from safety_config import (
    CROWD_COUNT_THRESHOLD, CROWD_DENSITY_THRESHOLD,
    CROWD_BUILDUP_DELTA, CROWD_BUILDUP_WINDOW_SEC,
    CROWD_CONFIRM_FRAMES,
)
from safety_geometry import normalized_density, point_in_polygon


class CrowdDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count_history = deque(maxlen=120)
        self.high_count_streak = 0
        self.active = False
        self.last_event_time = -999.0

    def update(self, people, frame_index, fps, frame_shape, roi_polygon=None):
        height, width = frame_shape[:2]
        fps = max(float(fps), 1.0)
        now = frame_index / fps

        if roi_polygon:
            roi_people = [
                p for p in people
                if point_in_polygon(p["foot_point"], roi_polygon)
            ]
        else:
            roi_people = list(people)

        count = len(roi_people)
        density = normalized_density(
            [p["bbox"] for p in roi_people], roi_polygon, width, height
        )

        self.count_history.append((now, count))

        old_count = count
        for timestamp, previous_count in self.count_history:
            if now - timestamp >= CROWD_BUILDUP_WINDOW_SEC:
                old_count = previous_count
                break

        rapid_growth = count - old_count >= CROWD_BUILDUP_DELTA
        crowded = count >= CROWD_COUNT_THRESHOLD or density >= CROWD_DENSITY_THRESHOLD
        condition = crowded or rapid_growth

        if condition:
            self.high_count_streak += 1
        else:
            self.high_count_streak = 0
            self.active = False

        new_event = None
        if self.high_count_streak >= CROWD_CONFIRM_FRAMES and not self.active:
            self.active = True
            kind = "CROWD_BUILDUP" if rapid_growth else "HIGH_DENSITY"
            severity = "HIGH" if density >= CROWD_DENSITY_THRESHOLD * 1.5 else "MEDIUM"
            new_event = {
                "event_type": kind,
                "severity": severity,
                "person_id": None,
                "confidence": min(0.99, 0.55 + min(0.35, count / max(CROWD_COUNT_THRESHOLD, 1) * 0.2)),
                "frame": frame_index,
                "timestamp": now,
                "details": {
                    "count": count,
                    "density": round(density, 4),
                    "growth": count - old_count,
                },
            }

        return {
            "count": count,
            "density": density,
            "rapid_growth": rapid_growth,
            "event": new_event,
        }
