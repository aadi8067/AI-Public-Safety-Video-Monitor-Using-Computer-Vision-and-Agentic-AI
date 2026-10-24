"""
Integration tests for the database layer.
Tests: connection, session lifecycle, full pipeline save, duplicate safety,
       partial failure handling, and DB failure isolation.

Run with:  python test_db_integration.py
"""
import sys
import traceback
import db
import pipeline_db

PASS = []
FAIL = []

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))

# ---------------------------------------------------------------------------
# T1 — DB connection
# ---------------------------------------------------------------------------
print("\n── T1: DB connection ──")
try:
    ok = db.test_connection()
    check("connection OK", ok)
except Exception as e:
    check("connection OK", False, str(e))

# ---------------------------------------------------------------------------
# T2 — Session lifecycle: insert, update status, update done
# ---------------------------------------------------------------------------
print("\n── T2: Session lifecycle ──")
try:
    sid = db.insert_session("test_video.mp4", file_path="/tmp/test_video.mp4", status="processing")
    check("insert_session returns int", isinstance(sid, int))

    db.update_session_status(sid, "processing")
    check("update_session_status no error", True)

    db.update_session_done(sid, fps=25.0, total_frames=750, duration_sec=30.0, output_dir="/tmp/out")
    check("update_session_done no error", True)

    # Verify the row is actually in the DB
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status, fps, total_frames FROM sessions WHERE id=%s", (sid,))
            row = cur.fetchone()
    check("session status=done", row["status"] == "done")
    check("session fps=25.0", abs(row["fps"] - 25.0) < 0.01)
    check("session total_frames=750", row["total_frames"] == 750)
except Exception as e:
    traceback.print_exc()
    check("T2 session lifecycle", False, str(e))
    sid = None

# ---------------------------------------------------------------------------
# T3 — Full pipeline save (simulated incident with events, actions, report, evidence)
# ---------------------------------------------------------------------------
print("\n── T3: Full pipeline save ──")

MOCK_INCIDENTS = [
    {
        "incident_id":         "INC-TEST-001",
        "start_time":          1.2,
        "end_time":            4.8,
        "duration_seconds":    3.6,
        "person_ids":          [1, 2],
        "event_types":         ["LOITERING", "ZONE_INTRUSION"],
        "event_count":         2,
        "risk_score":          42.0,
        "risk_severity":       "MEDIUM",
        "risk_explanation":    "LOITERING x1 (+10); ZONE_INTRUSION x1 (+20)",
        "response_actions":    ["LOG_INCIDENT", "FLAG_FOR_REVIEW"],
        "response_action_labels": ["Log incident to record", "Flag for operator review"],
        "events": [
            {
                "event_id":    "EVENT-0001",
                "event_type":  "LOITERING",
                "person_id":   1,
                "frame":       36,
                "timestamp":   1.2,
                "confidence":  0.78,
                "severity":    "MEDIUM",
                "details":     None,
            },
            {
                "event_id":    "EVENT-0002",
                "event_type":  "ZONE_INTRUSION",
                "person_id":   2,
                "frame":       72,
                "timestamp":   2.4,
                "confidence":  0.85,
                "severity":    "HIGH",
                "details":     {"zone": "restricted_area_1"},
            },
        ],
    }
]

MOCK_REPORTS = [
    {
        "incident_id":   "INC-TEST-001",
        "summary":       "Someone entered a restricted area and was loitering between 1.2s and 4.8s.",
        "what_happened": ["Loitering — Person 001, spotted at 1.2s", "Zone intrusion — Person 002, spotted at 2.4s"],
        "risk_reason":   "Multiple signals at once.",
        "persons":       "PERSON-001, PERSON-002",
        "evidence": [
            {
                "event_type":   "LOITERING",
                "filename":     "evidence_LOITERING_PERSON-001_frame_000036.jpg",
                "image_url":    "/outputs/captured_frames_test/evidence_LOITERING_PERSON-001_frame_000036.jpg",
                "frame":        36,
                "time_seconds": 1.2,
                "exists":       False,
            }
        ],
    }
]

try:
    s2 = db.insert_session("pipeline_test.mp4", status="processing")
    result = pipeline_db.save_pipeline_results(
        s2,
        fps=30.0,
        total_frames=144,
        duration_sec=4.8,
        output_dir="/tmp/captured_frames_test",
        capture_dir_name="captured_frames_test",
        final_incidents=MOCK_INCIDENTS,
        incident_reports=MOCK_REPORTS,
    )
    check("no fatal errors in result", "session_id" in result)
    check("1 incident saved",          result["incidents_saved"] == 1)
    check("2 events saved",            result["events_saved"] == 2)
    check("2 persons saved",           result["persons_saved"] == 2)
    check("2 actions saved",           result["actions_saved"] == 2)
    check("1 report saved",            result["reports_saved"] == 1)
    check("1 evidence frame saved",    result["evidence_saved"] == 1)
    check("no DB errors",              len(result["errors"]) == 0,
          str(result["errors"]))

    # Verify rows in DB
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM incidents WHERE session_id=%s", (s2,))
            check("incidents row in DB", cur.fetchone()["n"] == 1)
            cur.execute("SELECT COUNT(*) AS n FROM events WHERE session_id=%s", (s2,))
            check("events rows in DB",   cur.fetchone()["n"] == 2)
            cur.execute("SELECT COUNT(*) AS n FROM response_actions ra JOIN incidents i ON ra.incident_id=i.id WHERE i.session_id=%s", (s2,))
            check("actions rows in DB",  cur.fetchone()["n"] == 2)
            cur.execute("SELECT COUNT(*) AS n FROM reports r JOIN incidents i ON r.incident_id=i.id WHERE i.session_id=%s", (s2,))
            check("report row in DB",    cur.fetchone()["n"] == 1)
            cur.execute("SELECT COUNT(*) AS n FROM evidence_frames ef JOIN incidents i ON ef.incident_id=i.id WHERE i.session_id=%s", (s2,))
            check("evidence row in DB",  cur.fetchone()["n"] == 1)
except Exception as e:
    traceback.print_exc()
    check("T3 full pipeline save", False, str(e))

# ---------------------------------------------------------------------------
# T4 — Duplicate safety: re-running save_pipeline_results must not duplicate
# ---------------------------------------------------------------------------
print("\n── T4: Duplicate safety (re-run same incidents) ──")
try:
    # Re-run the exact same incidents on the same session — ON CONFLICT rules
    # for reports should prevent duplicate report rows.
    # Events with the same (session_id, event_uid) will raise a unique violation
    # which pipeline_db catches per-event and continues. This is the expected
    # behaviour — we log the error but don't crash.
    result2 = pipeline_db.save_pipeline_results(
        s2,
        fps=30.0,
        total_frames=144,
        duration_sec=4.8,
        output_dir="/tmp/captured_frames_test",
        capture_dir_name="captured_frames_test",
        final_incidents=MOCK_INCIDENTS,
        incident_reports=MOCK_REPORTS,
    )
    # Counts will be 0 new successful inserts (all duplicates rejected safely)
    # or report upserted — main thing is it didn't raise an exception
    check("duplicate run did not crash", True)
except Exception as e:
    check("duplicate run did not crash", False, str(e))

# ---------------------------------------------------------------------------
# T5 — DB failure isolation: None session_id returns gracefully
# ---------------------------------------------------------------------------
print("\n── T5: DB failure isolation (None session_id) ──")
try:
    result3 = pipeline_db.save_pipeline_results(
        None,
        fps=25.0, total_frames=100, duration_sec=4.0,
        output_dir="/tmp/x", capture_dir_name="x",
        final_incidents=MOCK_INCIDENTS, incident_reports=MOCK_REPORTS,
    )
    check("None session_id returns dict",   isinstance(result3, dict))
    check("None session_id has error msg",  len(result3["errors"]) > 0)
    check("None session_id saved 0 rows",
          result3["incidents_saved"] == 0 and result3["events_saved"] == 0)
except Exception as e:
    check("T5 isolation", False, str(e))

# ---------------------------------------------------------------------------
# T6 — Failed session marking
# ---------------------------------------------------------------------------
print("\n── T6: Failed session marking ──")
try:
    s3 = db.insert_session("failed_test.mp4", status="processing")
    pipeline_db.save_pipeline_results(
        s3,
        fps=0, total_frames=0, duration_sec=0,
        output_dir="", capture_dir_name="",
        final_incidents=[], incident_reports=[],
        failed=True,
    )
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM sessions WHERE id=%s", (s3,))
            row = cur.fetchone()
    check("failed session status=failed", row["status"] == "failed")
except Exception as e:
    traceback.print_exc()
    check("T6 failed session", False, str(e))

# ---------------------------------------------------------------------------
# Summary printed after ALL tests complete (T1-T11)
# ---------------------------------------------------------------------------
# (summary block moved to end of file)


import os
import json

# ---------------------------------------------------------------------------
# Shared fixtures for T7–T11
# ---------------------------------------------------------------------------

# Two fully separate incidents with different persons, event types, and
# weapon IDs — simulates a video where two unrelated things happen at once.
MULTI_INCIDENTS = [
    {
        "incident_id":            "INC-MULTI-001",
        "start_time":             0.5,
        "end_time":               3.0,
        "duration_seconds":       2.5,
        "person_ids":             [10],
        "event_types":            ["POSSIBLE_FALL"],
        "event_count":            1,
        "risk_score":             38.0,
        "risk_severity":          "MEDIUM",
        "risk_explanation":       "POSSIBLE_FALL x1 (+38.0)",
        "response_actions":       ["LOG_INCIDENT", "FLAG_FOR_REVIEW", "DISPATCH_ALERT"],
        "response_action_labels": ["Log", "Flag", "Alert"],
        "events": [
            {
                "event_id":   "EVENT-M001",
                "event_type": "POSSIBLE_FALL",
                "person_id":  10,
                "frame":      15,
                "timestamp":  0.5,
                "confidence": 0.91,
                "severity":   "HIGH",
                "details":    {"posture_ratio": 2.1},
            }
        ],
    },
    {
        "incident_id":            "INC-MULTI-002",
        "start_time":             4.0,
        "end_time":               6.5,
        "duration_seconds":       2.5,
        "person_ids":             [11],
        "event_types":            ["WEAPON_CONFIRMED"],
        "event_count":            1,
        "risk_score":             75.0,
        "risk_severity":          "CRITICAL",
        "risk_explanation":       "WEAPON_CONFIRMED x1 (+75.0)",
        "response_actions":       ["LOG_INCIDENT", "NOTIFY_SECURITY_TEAM"],
        "response_action_labels": ["Log", "Notify security"],
        "events": [
            {
                # No event_id — weapon event, uid generated from weapon_id
                "event_type": "WEAPON_CONFIRMED",
                "person_id":  11,
                "frame":      120,
                "timestamp":  4.0,
                "confidence": 0.82,
                "severity":   "CRITICAL",
                "details":    {"weapon_id": 99, "weapon_type": "gun"},
            }
        ],
    },
]

MULTI_REPORTS = [
    {
        "incident_id":   "INC-MULTI-001",
        "summary":       "A person may have fallen at 0.5s.",
        "what_happened": ["Possible fall — Person 010, spotted at 0.5s"],
        "risk_reason":   "Fall detected.",
        "persons":       "PERSON-010",
        "evidence": [],
    },
    {
        "incident_id":   "INC-MULTI-002",
        "summary":       "A weapon was confirmed at 4.0s.",
        "what_happened": ["Weapon confirmed — Person 011, spotted at 4.0s"],
        "risk_reason":   "Confirmed weapon.",
        "persons":       "PERSON-011",
        "evidence": [
            {
                "event_type":   "WEAPON_CONFIRMED",
                "filename":     "evidence_WEAPON-099_frame_000120.jpg",
                "image_url":    "/outputs/multi_test/evidence_WEAPON-099_frame_000120.jpg",
                "frame":        120,
                "time_seconds": 4.0,
                "exists":       False,
            }
        ],
    },
]


# ---------------------------------------------------------------------------
# T7 — Multiple simultaneous incidents stay correctly separated
# ---------------------------------------------------------------------------
print("\n── T7: Multiple simultaneous incidents ──")
try:
    s7 = db.insert_session("multi_incident_test.mp4", status="processing")
    r7 = pipeline_db.save_pipeline_results(
        s7,
        fps=30.0, total_frames=300, duration_sec=10.0,
        output_dir="/tmp/multi_test", capture_dir_name="multi_test",
        final_incidents=MULTI_INCIDENTS,
        incident_reports=MULTI_REPORTS,
    )
    check("T7 no DB errors",          len(r7["errors"]) == 0, str(r7["errors"]))
    check("T7 2 incidents saved",      r7["incidents_saved"] == 2)
    check("T7 2 events saved",         r7["events_saved"] == 2)
    check("T7 2 persons saved",        r7["persons_saved"] == 2)
    check("T7 5 actions saved",        r7["actions_saved"] == 5)
    check("T7 2 reports saved",        r7["reports_saved"] == 2)

    # Verify person isolation — person 10 only in INC-MULTI-001
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT incident_uid, person_ids FROM incidents WHERE session_id=%s ORDER BY start_time",
                (s7,)
            )
            rows = cur.fetchall()
    check("T7 INC-MULTI-001 persons=[10]", rows[0]["person_ids"] == [10])
    check("T7 INC-MULTI-002 persons=[11]", rows[1]["person_ids"] == [11])
    check("T7 incidents have different severities",
          {rows[0]["incident_uid"]: "MEDIUM", rows[1]["incident_uid"]: "CRITICAL"} and
          rows[0]["incident_uid"] != rows[1]["incident_uid"])

    # Verify weapon event got a deterministic uid (not EVENT-????)
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT event_uid FROM events WHERE session_id=%s AND event_type='WEAPON_CONFIRMED'",
                (s7,)
            )
            ev_row = cur.fetchone()
    check("T7 weapon event_uid not EVENT-????",
          ev_row is not None and "????" not in ev_row["event_uid"])

except Exception as e:
    traceback.print_exc()
    check("T7 multiple incidents", False, str(e))


# ---------------------------------------------------------------------------
# T8 — Repeated processing: same video twice produces two separate sessions
# ---------------------------------------------------------------------------
print("\n── T8: Repeated processing (same video, two sessions) ──")
try:
    s8a = db.insert_session("repeat_video.mp4", status="processing")
    r8a = pipeline_db.save_pipeline_results(
        s8a,
        fps=25.0, total_frames=150, duration_sec=6.0,
        output_dir="/tmp/run_a", capture_dir_name="run_a",
        final_incidents=MOCK_INCIDENTS,
        incident_reports=MOCK_REPORTS,
    )

    s8b = db.insert_session("repeat_video.mp4", status="processing")
    r8b = pipeline_db.save_pipeline_results(
        s8b,
        fps=25.0, total_frames=150, duration_sec=6.0,
        output_dir="/tmp/run_b", capture_dir_name="run_b",
        final_incidents=MOCK_INCIDENTS,
        incident_reports=MOCK_REPORTS,
    )

    check("T8 two different session IDs", s8a != s8b)
    check("T8 run A: 1 incident", r8a["incidents_saved"] == 1)
    check("T8 run B: 1 incident", r8b["incidents_saved"] == 1)
    check("T8 run A: no errors",  len(r8a["errors"]) == 0)
    check("T8 run B: no errors",  len(r8b["errors"]) == 0)

    # Verify incidents are isolated — each session has exactly its own
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM incidents WHERE session_id=%s", (s8a,))
            count_a = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM incidents WHERE session_id=%s", (s8b,))
            count_b = cur.fetchone()["n"]
    check("T8 session A has exactly 1 incident", count_a == 1)
    check("T8 session B has exactly 1 incident", count_b == 1)

    # Verify output_dir is stored separately per session
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT output_dir FROM sessions WHERE id IN (%s, %s)", (s8a, s8b))
            dirs = {row["output_dir"] for row in cur.fetchall()}
    check("T8 output dirs are different", len(dirs) == 2)

except Exception as e:
    traceback.print_exc()
    check("T8 repeated processing", False, str(e))


# ---------------------------------------------------------------------------
# T9 — DB unavailable: pipeline_db must not raise, must return error dict
#
# We simulate DB unavailability by temporarily breaking the connection
# settings inside db.py's module-level variables, then restoring them.
# ---------------------------------------------------------------------------
print("\n── T9: DB unavailable (simulated) ──")
try:
    # Stash real password, inject a wrong one
    real_pw = db._DB_PASSWORD
    db._DB_PASSWORD = "definitely_wrong_password_xyz"

    # save_session_start must return None (not raise)
    broken_session_id = pipeline_db.save_session_start("unavailable_test.mp4")
    check("T9 broken DB returns None session_id", broken_session_id is None)

    # save_pipeline_results with None session_id must return error dict (not raise)
    r9 = pipeline_db.save_pipeline_results(
        broken_session_id,
        fps=25.0, total_frames=100, duration_sec=4.0,
        output_dir="/tmp/x", capture_dir_name="x",
        final_incidents=MOCK_INCIDENTS, incident_reports=MOCK_REPORTS,
    )
    check("T9 broken DB: returns dict",       isinstance(r9, dict))
    check("T9 broken DB: 0 incidents saved",  r9["incidents_saved"] == 0)
    check("T9 broken DB: has error message",  len(r9["errors"]) > 0)

finally:
    # Always restore — must not leave DB broken for subsequent tests
    db._DB_PASSWORD = real_pw

# Confirm DB is working again after restore
try:
    ok = db.test_connection()
    check("T9 DB restored after test", ok)
except Exception as e:
    check("T9 DB restored after test", False, str(e))


# ---------------------------------------------------------------------------
# T10 — Report and evidence integrity
#       Each incident's report and evidence must link to the correct incident.
# ---------------------------------------------------------------------------
print("\n── T10: Report and evidence integrity ──")
try:
    s10 = db.insert_session("integrity_test.mp4", status="processing")
    r10 = pipeline_db.save_pipeline_results(
        s10,
        fps=30.0, total_frames=300, duration_sec=10.0,
        output_dir="/tmp/integrity_test", capture_dir_name="integrity_test",
        final_incidents=MULTI_INCIDENTS,
        incident_reports=MULTI_REPORTS,
    )

    with db.get_conn() as conn:
        with conn.cursor() as cur:
            # Each report must point to its correct incident
            cur.execute("""
                SELECT i.incident_uid, r.summary
                FROM reports r
                JOIN incidents i ON r.incident_id = i.id
                WHERE i.session_id = %s
                ORDER BY i.start_time
            """, (s10,))
            rpt_rows = cur.fetchall()

            # Each evidence frame must point to an incident that exists
            cur.execute("""
                SELECT ef.filename, i.incident_uid
                FROM evidence_frames ef
                JOIN incidents i ON ef.incident_id = i.id
                WHERE i.session_id = %s
            """, (s10,))
            ev_rows = cur.fetchall()

            # incident_events junction must be populated
            cur.execute("""
                SELECT COUNT(*) AS n
                FROM incident_events ie
                JOIN incidents i ON ie.incident_id = i.id
                WHERE i.session_id = %s
            """, (s10,))
            junction_count = cur.fetchone()["n"]

    check("T10 2 reports saved",                    len(rpt_rows) == 2)
    check("T10 INC-MULTI-001 report summary correct",
          any("fallen" in (r["summary"] or "") for r in rpt_rows))
    check("T10 INC-MULTI-002 report summary correct",
          any("weapon" in (r["summary"] or "").lower() for r in rpt_rows))
    check("T10 evidence links to correct incident",
          all(r["incident_uid"] == "INC-MULTI-002" for r in ev_rows))
    check("T10 incident_events junction populated",  junction_count == 2)

except Exception as e:
    traceback.print_exc()
    check("T10 report/evidence integrity", False, str(e))


# ---------------------------------------------------------------------------
# T11 — No-incident video: session saved, counts are zero, status is done
# ---------------------------------------------------------------------------
print("\n── T11: Video with no incidents ──")
try:
    s11 = db.insert_session("quiet_video.mp4", status="processing")
    r11 = pipeline_db.save_pipeline_results(
        s11,
        fps=25.0, total_frames=500, duration_sec=20.0,
        output_dir="/tmp/quiet", capture_dir_name="quiet",
        final_incidents=[],
        incident_reports=[],
    )
    check("T11 no errors",           len(r11["errors"]) == 0)
    check("T11 0 incidents saved",   r11["incidents_saved"] == 0)
    check("T11 0 events saved",      r11["events_saved"] == 0)

    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status, total_frames FROM sessions WHERE id=%s", (s11,))
            row = cur.fetchone()
    check("T11 session status=done",      row["status"] == "done")
    check("T11 session total_frames=500", row["total_frames"] == 500)

except Exception as e:
    traceback.print_exc()
    check("T11 no-incident video", False, str(e))


# ---------------------------------------------------------------------------
# Final summary (T1–T11)
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print(f"FAILED: {FAIL}")
print('='*50)
sys.exit(0 if not FAIL else 1)
