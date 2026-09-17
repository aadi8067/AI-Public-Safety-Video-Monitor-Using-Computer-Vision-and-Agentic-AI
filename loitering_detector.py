"""Robust temporal loitering detector for tracked people.

A person is considered loitering only when all of these hold:
1) they remain in the configured loitering area for the dwell threshold;
2) their movement stays consistently small relative to their body height;
3) their net displacement from the entry anchor remains small.

Using body-height normalization plus an anchor-displacement test prevents a
moving pedestrian who merely remains visible in the camera for several
seconds from being labelled as loitering.
"""
from collections import defaultdict, deque
from statistics import median

from safety_config import (
    LOITER_SECONDS,
    LOITER_MAX_SPEED_BODY_HEIGHTS_SEC,
    LOITER_MAX_DISPLACEMENT_BODY_HEIGHTS,
    LOITER_SMOOTH_WINDOW_SEC,
    LOITER_MIN_SLOW_RATIO,
    LOITER_CONFIRM_FRAMES,
    MAX_HISTORY_FRAMES,
)
from safety_geometry import distance, point_in_polygon


class LoiteringDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = defaultdict(dict)

    @staticmethod
    def _body_height(person):
        bbox = person.get("bbox") or [0, 0, 0, 0]
        try:
            return max(20.0, float(bbox[3]) - float(bbox[1]))
        except Exception:
            return 20.0

    def _reset_inside_state(self, state, now, point, body_height):
        state.clear()
        state.update({
            "entered": now,
            "anchor": tuple(point),
            "last_point": tuple(point),
            "last_time": now,
            "history": deque(maxlen=max(12, MAX_HISTORY_FRAMES)),
            "slow_streak": 0,
            "alerted": False,
            "speed_norm": 0.0,
            "displacement_norm": 0.0,
            "body_height": body_height,
        })
        state["history"].append((now, tuple(point), body_height, 0.0))

    def update(self, people, frame_index, fps, zone_polygon):
        fps = max(float(fps), 1.0)
        now = frame_index / fps
        active_ids = set()
        events = []

        for person in people:
            pid = person.get("stable_id")
            if pid is None:
                continue
            pid = int(pid)
            active_ids.add(pid)

            point = tuple(person["foot_point"])
            inside = point_in_polygon(point, zone_polygon)
            state = self.state.setdefault(pid, {})
            body_height = self._body_height(person)

            if not inside:
                state.clear()
                continue

            if "entered" not in state:
                self._reset_inside_state(state, now, point, body_height)
                continue

            last_point = state.get("last_point", point)
            last_time = float(state.get("last_time", now))
            dt = max(1.0 / fps, now - last_time)
            moved = distance(point, last_point)
            raw_speed = moved / dt
            speed_norm = raw_speed / max(body_height, 20.0)

            history = state.setdefault(
                "history", deque(maxlen=max(12, MAX_HISTORY_FRAMES))
            )
            history.append((now, point, body_height, speed_norm))

            state["last_point"] = point
            state["last_time"] = now
            state["body_height"] = body_height

            dwell = now - float(state["entered"])
            window_start = now - LOITER_SMOOTH_WINDOW_SEC
            recent = [item for item in history if item[0] >= window_start]

            if recent:
                recent_speeds = [item[3] for item in recent]
                median_speed_norm = float(median(recent_speeds))
                slow_ratio = sum(
                    s <= LOITER_MAX_SPEED_BODY_HEIGHTS_SEC
                    for s in recent_speeds
                ) / len(recent_speeds)
            else:
                median_speed_norm = speed_norm
                slow_ratio = 0.0

            anchor = state.get("anchor", point)
            displacement_px = distance(point, anchor)
            displacement_norm = displacement_px / max(body_height, 20.0)

            state["speed_norm"] = median_speed_norm
            state["displacement_norm"] = displacement_norm
            state["slow_ratio"] = slow_ratio

            # Require several consecutive qualifying frames.  This is a
            # temporal confirmation on top of the dwell and spatial tests.
            qualifies = (
                median_speed_norm <= LOITER_MAX_SPEED_BODY_HEIGHTS_SEC
                and displacement_norm <= LOITER_MAX_DISPLACEMENT_BODY_HEIGHTS
                and slow_ratio >= LOITER_MIN_SLOW_RATIO
            )
            if qualifies:
                state["slow_streak"] = state.get("slow_streak", 0) + 1
            else:
                # Do not immediately forget a brief tracker jitter spike, but
                # do reset enough to prevent a walking person from accumulating
                # a false confirmation streak.
                state["slow_streak"] = max(
                    0, state.get("slow_streak", 0) - 2
                )

            if (
                dwell >= LOITER_SECONDS
                and state.get("slow_streak", 0) >= LOITER_CONFIRM_FRAMES
                and not state.get("alerted", False)
            ):
                state["alerted"] = True
                confidence = min(
                    0.98,
                    0.65
                    + min(0.12, dwell / (LOITER_SECONDS * 10.0))
                    + min(0.12, max(0.0, 1.0 - median_speed_norm) * 0.12),
                )
                events.append({
                    "event_type": "LOITERING",
                    "severity": "MEDIUM",
                    "person_id": pid,
                    "confidence": confidence,
                    "frame": frame_index,
                    "timestamp": now,
                    "details": {
                        "dwell_seconds": round(dwell, 2),
                        "speed_body_heights_sec": round(median_speed_norm, 3),
                        "displacement_body_heights": round(displacement_norm, 3),
                        "slow_ratio": round(slow_ratio, 2),
                        "method": "body_height_normalized_smoothed_anchor",
                    },
                })

        for pid in list(self.state):
            if pid not in active_ids:
                self.state[pid]["missing"] = self.state[pid].get("missing", 0) + 1
                if self.state[pid]["missing"] > MAX_HISTORY_FRAMES:
                    self.state.pop(pid, None)
            else:
                self.state[pid]["missing"] = 0

        return events
