"""Report Agent for AI Public Safety Video Monitor.

Sits at the end of the pipeline after Risk Agent and Response Agent
have already scored and actioned a finalized incident:

  EventCorrelator -> RiskAgent -> ResponseAgent -> ReportAgent

The Report Agent does NOT re-score, re-classify, or re-decide anything.
It translates the already-scored incident into plain English that any
operator can read at a glance.
"""
import os
import re


# ---------------------------------------------------------------------------
# Human-readable label maps
# ---------------------------------------------------------------------------

_EVENT_LABELS = {
    "WEAPON_CONFIRMED":  "A weapon was confirmed",
    "POSSIBLE_FALL":     "A person may have fallen",
    "ZONE_INTRUSION":    "Someone entered a restricted area",
    "LOITERING":         "A person was loitering",
    "ABNORMAL_MOVEMENT": "Unusual movement was detected",
    "HIGH_DENSITY":      "A large crowd gathered",
    "CROWD_BUILDUP":     "The crowd was growing rapidly",
}

# Same labels without the leading capital — used when joining mid-sentence
_EVENT_LABELS_LOWER = {k: v[0].lower() + v[1:] for k, v in _EVENT_LABELS.items()}

_SEVERITY_LABELS = {
    "CRITICAL": "Immediate attention required",
    "HIGH":     "Likely safety concern",
    "MEDIUM":   "Needs monitoring",
    "LOW":      "Minor observation",
}


def _readable_event(event_type: str, capitalize: bool = True) -> str:
    key = str(event_type).upper()
    if capitalize:
        return _EVENT_LABELS.get(key, str(event_type).replace("_", " ").title())
    return _EVENT_LABELS_LOWER.get(key, str(event_type).replace("_", " ").lower())


def _safe_event_type(event_type: str) -> str:
    """Same sanitisation used in detect_video() when saving evidence files."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(event_type))


def _person_label(pid) -> str:
    if pid is not None:
        return f"Person {int(pid):03d}"
    return "everyone in the scene"


def _person_label_short(pid) -> str:
    if pid is not None:
        return f"PERSON-{int(pid):03d}"
    return "SCENE-WIDE"


def _evidence_filename_for_event(event: dict) -> str | None:
    """Reconstruct the filename that detect_video() would have saved."""
    frame = event.get("frame")
    if frame is None:
        return None
    event_type = str(event.get("event_type", "EVENT"))
    if event_type == "WEAPON_CONFIRMED":
        wid = event.get("details", {}).get("weapon_id")
        if wid is not None:
            return f"evidence_WEAPON-{int(wid):03d}_frame_{int(frame):06d}.jpg"
    safe_type = _safe_event_type(event_type)
    pid = event.get("person_id")
    pid_text = f"PERSON-{int(pid):03d}" if pid is not None else "ALL"
    return f"evidence_{safe_type}_{pid_text}_frame_{int(frame):06d}.jpg"


# ---------------------------------------------------------------------------
# Plain-English summary builder
# ---------------------------------------------------------------------------

def _build_summary(event_types: list, person_ids: list,
                   start_time: float, end_time: float,
                   risk_severity: str, risk_score: float) -> str:
    """One plain sentence a non-technical operator can read instantly."""
    if not event_types:
        return "An unclassified safety incident was detected."

    readable = [_readable_event(e) for e in event_types]
    if len(readable) == 1:
        what = readable[0]
    elif len(readable) == 2:
        what = f"{readable[0]}, and {_readable_event(event_types[1], capitalize=False)}"
    else:
        first = readable[0]
        rest = [_readable_event(e, capitalize=False) for e in event_types[1:]]
        what = first + ", " + ", ".join(rest[:-1]) + f", and {rest[-1]}"

    if not person_ids:
        who = "in the scene"
    elif len(person_ids) == 1:
        who = f"involving Person {int(person_ids[0]):03d}"
    else:
        ids = ", ".join(f"{int(p):03d}" for p in sorted(person_ids))
        who = f"involving Persons {ids}"

    when = f"at {start_time:.1f}s"
    if end_time > start_time + 0.5:
        when = f"between {start_time:.1f}s and {end_time:.1f}s"

    sev_note = _SEVERITY_LABELS.get(risk_severity.upper(), "")

    return f"{what} {who} {when}. {sev_note}."


def _build_what_happened(events: list, fps: float) -> list:
    """Plain-English bullet points, one per unique event type+person."""
    fps = max(float(fps), 1.0)
    seen = set()
    bullets = []
    for ev in sorted(events, key=lambda e: e.get("frame") or 0):
        et = str(ev.get("event_type", "EVENT"))
        pid = ev.get("person_id")
        key = (et, pid)
        if key in seen:
            continue
        seen.add(key)
        frame = ev.get("frame")
        t = f"{int(frame) / fps:.1f}s" if frame is not None else None
        conf = ev.get("confidence")
        person_text = f"Person {int(pid):03d}" if pid is not None else "the scene"
        label = _readable_event(et)
        # Build a clean readable line with no brackets or symbols
        if t:
            line = f"{label} — {person_text}, spotted at {t}"
        else:
            line = f"{label} — {person_text}"
        if conf is not None and conf > 0:
            line += f" with {int(conf * 100)}% confidence"
        bullets.append(line)
    return bullets


# ---------------------------------------------------------------------------
# ReportAgent
# ---------------------------------------------------------------------------

class ReportAgent:

    def generate(
        self,
        incident: dict,
        fps: float,
        capture_dir: str,
        capture_dir_name: str,
    ) -> dict:
        fps = max(float(fps), 1.0)

        incident_id      = incident.get("incident_id", "INC-????")
        start_time       = float(incident.get("start_time", 0.0))
        end_time         = float(incident.get("end_time", 0.0))
        duration         = float(incident.get("duration_seconds", max(0.0, end_time - start_time)))
        event_types      = incident.get("event_types", [])
        event_count      = int(incident.get("event_count", 0))
        person_ids       = incident.get("person_ids", [])
        risk_score       = float(incident.get("risk_score", 0.0))
        risk_severity    = str(incident.get("risk_severity", "LOW"))
        response_actions = incident.get("response_actions", [])
        response_labels  = incident.get("response_action_labels", [])
        events           = incident.get("events", [])

        # Persons display
        if person_ids:
            persons_str = ", ".join(f"PERSON-{int(p):03d}" for p in sorted(person_ids))
        else:
            persons_str = "SCENE-WIDE"

        # Plain-English summary sentence
        summary = _build_summary(
            event_types, person_ids, start_time, end_time, risk_severity, risk_score
        )

        # Plain-English bullet points of what happened
        what_happened = _build_what_happened(events, fps)

        # Plain-English risk reason (replaces raw risk_explanation)
        risk_reason = _build_risk_reason(event_types, risk_score, risk_severity)

        # Evidence images
        evidence = []
        seen_files: set = set()
        for ev in events:
            filename = _evidence_filename_for_event(ev)
            if filename is None or filename in seen_files:
                continue
            full_path = os.path.join(capture_dir, filename)
            frame = ev.get("frame")
            t = round(int(frame) / fps, 2) if frame is not None else 0.0
            exists = os.path.isfile(full_path)
            evidence.append({
                "event_type":   str(ev.get("event_type", "EVENT")),
                "event_label":  _readable_event(str(ev.get("event_type", "EVENT"))),
                "person_id":    ev.get("person_id"),
                "frame":        frame,
                "time_seconds": t,
                "image_url":    f"/outputs/{capture_dir_name}/{filename}" if exists else None,
                "filename":     filename,
                "exists":       exists,
            })
            seen_files.add(filename)
        evidence.sort(key=lambda e: e.get("frame") or 0)

        return {
            "incident_id":            incident_id,
            "summary":                summary,
            "what_happened":          what_happened,
            "risk_reason":            risk_reason,
            "risk_score":             risk_score,
            "risk_severity":          risk_severity,
            "start_time":             start_time,
            "end_time":               end_time,
            "duration_seconds":       duration,
            "persons":                persons_str,
            "event_types":            event_types,
            "event_count":            event_count,
            "response_actions":       response_actions,
            "response_action_labels": response_labels,
            "evidence":               evidence,
        }


def _build_risk_reason(event_types: list, risk_score: float, risk_severity: str) -> str:
    """One plain sentence explaining why this severity was assigned."""
    if not event_types:
        return "No specific events were detected."
    score_band = _SEVERITY_LABELS.get(risk_severity.upper(), "requires attention")
    if len(event_types) >= 2:
        parts = [_readable_event(e, capitalize=False) for e in event_types]
        if len(parts) == 2:
            events_text = f"{parts[0]} and {parts[1]}"
        else:
            events_text = ", ".join(parts[:-1]) + f" and {parts[-1]}"
        return (
            f"Several things happened at once — {events_text}. "
            f"When multiple signals occur together the overall risk score goes up, "
            f"reaching {risk_score:.0f} out of 100. {score_band}."
        )
    label = _readable_event(event_types[0], capitalize=True)
    return (
        f"{label}. The risk score for this event is {risk_score:.0f} out of 100. "
        f"{score_band}."
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_incident_reports(
    final_incidents: list,
    fps: float,
    capture_dir: str,
    capture_dir_name: str,
) -> list:
    agent = ReportAgent()
    return [
        agent.generate(inc, fps, capture_dir, capture_dir_name)
        for inc in final_incidents
    ]
