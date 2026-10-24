"""
pipeline_db.py — Database persistence layer for the AI Public Safety pipeline.

This module is the ONLY place that calls db.py functions from pipeline code.
app.py calls save_session_start() and save_pipeline_results() only.
All other pipeline code (detectors, agents, report_agent) is untouched.

Insertion order respects foreign keys:
  sessions -> persons -> events -> incidents
          -> incident_events (junction)
          -> response_actions
          -> reports
          -> evidence_frames

All writes are wrapped so a DB failure never crashes the video pipeline.
The caller always gets back a result dict describing what was saved (or
what failed) so app.py can log it without caring about DB internals.
"""

import sys
import traceback
from typing import Optional

import db


# ---------------------------------------------------------------------------
# Internal safe wrapper
# ---------------------------------------------------------------------------

def _safe(label: str, fn, *args, **kwargs):
    """Call fn(*args, **kwargs). On any exception, print and return None."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"[pipeline_db] {label} failed: {exc!r}", file=sys.stderr)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Step 1 — called at the START of video/live processing
# ---------------------------------------------------------------------------

def save_session_start(
    filename: str,
    file_path: Optional[str] = None,
) -> Optional[int]:
    """
    Insert a session row with status='processing' as soon as a video
    is accepted. Returns the session_id (int) or None if DB is down.
    """
    session_id = _safe(
        "insert_session",
        db.insert_session,
        filename=filename,
        file_path=file_path,
        status="processing",
    )
    if session_id is not None:
        print(f"[pipeline_db] Session {session_id} opened for '{filename}'")
    return session_id


# ---------------------------------------------------------------------------
# Step 2 — called at the END of video/live processing
# ---------------------------------------------------------------------------

def save_pipeline_results(
    session_id: Optional[int],
    *,
    fps: float,
    total_frames: int,
    duration_sec: float,
    output_dir: str,
    capture_dir_name: str,
    final_incidents: list,
    incident_reports: list,
    failed: bool = False,
) -> dict:
    """
    Persist all pipeline results for one completed session.

    Parameters
    ----------
    session_id       : int returned by save_session_start(), or None (skips all writes)
    fps              : video fps
    total_frames     : frames processed
    duration_sec     : video duration in seconds
    output_dir       : absolute path to the captured_frames_* directory
    capture_dir_name : basename of output_dir (for image URLs)
    final_incidents  : list of incident dicts from agentic_supervisor.finalize()
    incident_reports : list of report dicts from generate_incident_reports()
    failed           : True if the pipeline itself failed (marks session as 'failed')

    Returns
    -------
    dict with counts of rows saved and any errors encountered.
    """
    result = {
        "session_id": session_id,
        "persons_saved": 0,
        "events_saved": 0,
        "incidents_saved": 0,
        "actions_saved": 0,
        "reports_saved": 0,
        "evidence_saved": 0,
        "errors": [],
    }

    if session_id is None:
        result["errors"].append("No session_id — DB was unavailable at session start.")
        return result

    # Mark session failed immediately if the pipeline errored
    if failed:
        _safe("update_session failed", db.update_session_status, session_id, "failed")
        result["errors"].append("Pipeline failed — session marked as failed.")
        return result

    # Update session with final metadata
    _safe(
        "update_session_done",
        db.update_session_done,
        session_id,
        fps=fps,
        total_frames=total_frames,
        duration_sec=duration_sec,
        output_dir=output_dir,
    )

    if not final_incidents:
        # No incidents — session is done, nothing else to write
        _safe("update_session_done (no incidents)",
              db.update_session_status, session_id, "done")
        return result

    # Build a lookup: incident_uid -> report dict (from ReportAgent)
    report_by_uid = {}
    for rep in (incident_reports or []):
        uid = rep.get("incident_id")
        if uid:
            report_by_uid[uid] = rep

    # ------------------------------------------------------------------
    # Collect all stable person IDs across all incidents so we can
    # upsert person rows BEFORE inserting events (FK: events.person_id
    # references persons.stable_id via the pipeline's integer IDs —
    # we store the stable_id integer directly; no separate FK column).
    # ------------------------------------------------------------------
    all_person_ids: set = set()
    for incident in final_incidents:
        for pid in (incident.get("person_ids") or []):
            all_person_ids.add(int(pid))
        for ev in (incident.get("events") or []):
            pid = ev.get("person_id")
            if pid is not None:
                all_person_ids.add(int(pid))

    for stable_id in all_person_ids:
        r = _safe(
            f"upsert_person stable_id={stable_id}",
            db.upsert_person,
            session_id=session_id,
            stable_id=stable_id,
        )
        if r is not None:
            result["persons_saved"] += 1

    # ------------------------------------------------------------------
    # Persist each incident and its children
    # ------------------------------------------------------------------
    for incident in final_incidents:
        incident_uid  = incident.get("incident_id", "INC-????")
        start_time    = float(incident.get("start_time", 0.0))
        end_time      = float(incident.get("end_time", 0.0))
        duration      = float(incident.get("duration_seconds", max(0.0, end_time - start_time)))
        event_types   = incident.get("event_types") or []
        person_ids    = [int(p) for p in (incident.get("person_ids") or [])]
        event_count   = int(incident.get("event_count", 0))
        risk_score    = incident.get("risk_score")
        risk_severity = incident.get("risk_severity")
        risk_expl     = incident.get("risk_explanation")
        raw_events    = incident.get("events") or []
        actions       = incident.get("response_actions") or []
        action_labels = incident.get("response_action_labels") or []

        # 1. Insert incident row
        inc_db_id = _safe(
            f"insert_incident {incident_uid}",
            db.insert_incident,
            session_id=session_id,
            incident_uid=incident_uid,
            start_time=start_time,
            end_time=end_time,
            duration_sec=duration,
            event_types=event_types,
            person_ids=person_ids,
            event_count=event_count,
            risk_score=float(risk_score) if risk_score is not None else None,
            risk_severity=risk_severity,
            risk_explanation=risk_expl,
        )
        if inc_db_id is None:
            result["errors"].append(f"Failed to insert incident {incident_uid}")
            continue
        result["incidents_saved"] += 1

        # 2. Insert each event and link it to the incident
        ev_uid_to_db_id: dict = {}  # event_uid -> db event id (for evidence lookup)
        for ev in raw_events:
            event_type  = str(ev.get("event_type", "EVENT"))
            person_id   = ev.get("person_id")
            frame       = ev.get("frame")
            ts          = ev.get("timestamp") or ev.get("time_seconds")
            confidence  = ev.get("confidence")
            severity    = ev.get("severity")
            details     = ev.get("details") if isinstance(ev.get("details"), dict) else None

            # Weapon events carry details inside the event dict itself
            if event_type == "WEAPON_CONFIRMED" and not details:
                details = {
                    k: ev[k] for k in ("weapon_id", "weapon_type")
                    if k in ev
                }

            # Build a unique event_uid.
            # SafetyEventPipeline events carry EVENT-NNNN in event_id.
            # Weapon events from agentic_supervisor have no event_id — generate
            # a deterministic uid from weapon_id + incident so it is always unique.
            raw_uid = ev.get("event_id") or ev.get("event_uid") or ""
            if not raw_uid or raw_uid == "EVENT-????":
                w_id = (details or {}).get("weapon_id")
                if w_id is not None:
                    raw_uid = f"WEAPON-EV-{int(w_id):03d}-{incident_uid}"
                else:
                    raw_uid = f"EV-{event_type}-{incident_uid}-f{frame or 0}"
            event_uid = str(raw_uid)

            ev_db_id = _safe(
                f"insert_event {event_uid}",
                db.insert_event,
                session_id=session_id,
                event_uid=event_uid,
                event_type=event_type,
                person_id=int(person_id) if person_id is not None else None,
                frame=int(frame) if frame is not None else None,
                timestamp_sec=float(ts) if ts is not None else None,
                confidence=float(confidence) if confidence is not None else None,
                severity=severity,
                details=details,
            )
            if ev_db_id is None:
                result["errors"].append(f"Failed to insert event {event_uid}")
                continue
            result["events_saved"] += 1
            ev_uid_to_db_id[event_uid] = ev_db_id

            # Link event -> incident
            _safe(
                f"link_incident_event inc={inc_db_id} ev={ev_db_id}",
                db.link_incident_event,
                inc_db_id,
                ev_db_id,
            )

        # 3. Response actions (one row per action code)
        label_map = dict(zip(actions, action_labels)) if action_labels else {}
        for code in actions:
            r = _safe(
                f"insert_response_action {code}",
                db.insert_response_action,
                incident_id=inc_db_id,
                action_code=code,
                action_label=label_map.get(code),
            )
            if r is not None:
                result["actions_saved"] += 1

        # 4. Report
        rep = report_by_uid.get(incident_uid, {})
        if rep:
            r = _safe(
                f"insert_report {incident_uid}",
                db.insert_report,
                incident_id=inc_db_id,
                summary=rep.get("summary"),
                what_happened=rep.get("what_happened") or [],
                risk_reason=rep.get("risk_reason"),
                persons=rep.get("persons"),
            )
            if r is not None:
                result["reports_saved"] += 1

        # 5. Evidence frames (from ReportAgent output — already resolved paths)
        for evid in (rep.get("evidence") or []):
            filename    = evid.get("filename")
            if not filename:
                continue
            image_url   = evid.get("image_url")
            frame_num   = evid.get("frame")
            ts          = evid.get("time_seconds")
            exists      = bool(evid.get("exists", False))
            ev_type     = evid.get("event_type", "")

            # Resolve the db event id for this evidence item if possible
            # Match by event_type + frame number against what we just inserted
            ev_db_id_for_frame = None
            for raw_ev in raw_events:
                if (str(raw_ev.get("event_type", "")) == ev_type and
                        raw_ev.get("frame") == frame_num):
                    uid = str(raw_ev.get("event_id", ""))
                    ev_db_id_for_frame = ev_uid_to_db_id.get(uid)
                    break

            import os
            file_path = (
                os.path.join(output_dir, filename)
                if output_dir else None
            )

            r = _safe(
                f"insert_evidence_frame {filename}",
                db.insert_evidence_frame,
                incident_id=inc_db_id,
                filename=filename,
                event_id=ev_db_id_for_frame,
                file_path=file_path,
                image_url=image_url,
                frame=int(frame_num) if frame_num is not None else None,
                timestamp_sec=float(ts) if ts is not None else None,
                exists_on_disk=exists,
            )
            if r is not None:
                result["evidence_saved"] += 1

    # Mark session done
    _safe("update_session_done (final)", db.update_session_status, session_id, "done")

    print(
        f"[pipeline_db] Session {session_id} saved — "
        f"incidents={result['incidents_saved']} events={result['events_saved']} "
        f"persons={result['persons_saved']} actions={result['actions_saved']} "
        f"reports={result['reports_saved']} evidence={result['evidence_saved']}"
    )
    if result["errors"]:
        print(f"[pipeline_db] {len(result['errors'])} non-fatal errors: {result['errors']}", file=sys.stderr)

    return result
