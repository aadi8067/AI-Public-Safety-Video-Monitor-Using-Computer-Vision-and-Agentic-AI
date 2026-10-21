"""
Database connection layer for AI Public Safety Video Monitor.
Uses psycopg 3 (already installed globally).
Credentials are loaded from .env — never hardcoded here.

Public API
----------
get_conn()                          -> psycopg.Connection  (caller must close)
insert_session(...)                 -> int  session_id
update_session_status(id, status)
update_session_done(id, ...)
insert_person(...)                  -> int  person_id
upsert_person(...)                  -> int  person_id
insert_event(...)                   -> int  event_id
insert_incident(...)                -> int  incident_id
link_incident_event(inc_id, ev_id)
insert_response_action(...)         -> int  action_id
insert_report(...)                  -> int  report_id
insert_evidence_frame(...)          -> int  frame_id
test_connection()                   -> bool
"""

import os
import sys
from pathlib import Path
from typing import Optional

import psycopg
from psycopg.rows import dict_row

# ---------------------------------------------------------------------------
# Load .env manually — no extra dependency needed
# ---------------------------------------------------------------------------
def _load_env(env_path: str = ".env") -> None:
    p = Path(env_path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

_load_env()

# ---------------------------------------------------------------------------
# Connection settings from environment
# ---------------------------------------------------------------------------
_DB_HOST     = os.environ.get("DB_HOST", "localhost")
_DB_PORT     = int(os.environ.get("DB_PORT", "5432"))
_DB_NAME     = os.environ.get("DB_NAME", "public_safety_db")
_DB_USER     = os.environ.get("DB_USER", "safety_app")
_DB_PASSWORD = os.environ.get("DB_PASSWORD", "")


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------
def get_conn() -> psycopg.Connection:
    """Return a new psycopg connection. Caller is responsible for closing it."""
    return psycopg.connect(
        host=_DB_HOST,
        port=_DB_PORT,
        dbname=_DB_NAME,
        user=_DB_USER,
        password=_DB_PASSWORD,
        row_factory=dict_row,
        connect_timeout=10,
    )


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------
def insert_session(
    filename: str,
    file_path: Optional[str] = None,
    fps: Optional[float] = None,
    total_frames: Optional[int] = None,
    duration_sec: Optional[float] = None,
    output_dir: Optional[str] = None,
    status: str = "pending",
) -> int:
    """Insert a new session row and return its id."""
    sql = """
        INSERT INTO sessions
            (filename, file_path, fps, total_frames, duration_sec, output_dir, status)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (filename, file_path, fps, total_frames,
                              duration_sec, output_dir, status))
            row = cur.fetchone()
        conn.commit()
    return row["id"]


def update_session_status(session_id: int, status: str) -> None:
    """Update session status (pending / processing / done / failed)."""
    sql = "UPDATE sessions SET status=%s, updated_at=now() WHERE id=%s"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (status, session_id))
        conn.commit()


def update_session_done(
    session_id: int,
    fps: Optional[float] = None,
    total_frames: Optional[int] = None,
    duration_sec: Optional[float] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Mark a session as done and fill in final metadata."""
    sql = """
        UPDATE sessions
        SET status='done', fps=%s, total_frames=%s,
            duration_sec=%s, output_dir=%s, updated_at=now()
        WHERE id=%s
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (fps, total_frames, duration_sec, output_dir, session_id))
        conn.commit()


# ---------------------------------------------------------------------------
# persons
# ---------------------------------------------------------------------------
def insert_person(
    session_id: int,
    stable_id: int,
    first_frame: Optional[int] = None,
    last_frame: Optional[int] = None,
    is_threat: bool = False,
) -> int:
    """Insert a person row and return its id."""
    sql = """
        INSERT INTO persons (session_id, stable_id, first_frame, last_frame, is_threat)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id, stable_id, first_frame, last_frame, is_threat))
            row = cur.fetchone()
        conn.commit()
    return row["id"]


def upsert_person(
    session_id: int,
    stable_id: int,
    first_frame: Optional[int] = None,
    last_frame: Optional[int] = None,
    is_threat: bool = False,
) -> int:
    """Insert or update a person row (keyed on session_id + stable_id)."""
    sql = """
        INSERT INTO persons (session_id, stable_id, first_frame, last_frame, is_threat)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (session_id, stable_id) DO UPDATE
            SET last_frame = EXCLUDED.last_frame,
                is_threat  = persons.is_threat OR EXCLUDED.is_threat
        RETURNING id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id, stable_id, first_frame, last_frame, is_threat))
            row = cur.fetchone()
        conn.commit()
    return row["id"]


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def insert_event(
    session_id: int,
    event_uid: str,
    event_type: str,
    person_id: Optional[int] = None,
    frame: Optional[int] = None,
    timestamp_sec: Optional[float] = None,
    confidence: Optional[float] = None,
    severity: Optional[str] = None,
    details: Optional[dict] = None,
) -> int:
    """Insert a structured safety event and return its id."""
    import json
    sql = """
        INSERT INTO events
            (session_id, event_uid, event_type, person_id, frame,
             timestamp_sec, confidence, severity, details)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    details_json = json.dumps(details) if details else None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id, event_uid, event_type, person_id,
                              frame, timestamp_sec, confidence, severity, details_json))
            row = cur.fetchone()
        conn.commit()
    return row["id"]


# ---------------------------------------------------------------------------
# incidents
# ---------------------------------------------------------------------------
def insert_incident(
    session_id: int,
    incident_uid: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    duration_sec: Optional[float] = None,
    event_types: Optional[list] = None,
    person_ids: Optional[list] = None,
    event_count: int = 0,
    risk_score: Optional[float] = None,
    risk_severity: Optional[str] = None,
    risk_explanation: Optional[str] = None,
) -> int:
    """Insert a correlated incident and return its id."""
    sql = """
        INSERT INTO incidents
            (session_id, incident_uid, start_time, end_time, duration_sec,
             event_types, person_ids, event_count,
             risk_score, risk_severity, risk_explanation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                session_id, incident_uid, start_time, end_time, duration_sec,
                event_types or [], person_ids or [], event_count,
                risk_score, risk_severity, risk_explanation,
            ))
            row = cur.fetchone()
        conn.commit()
    return row["id"]


def link_incident_event(incident_id: int, event_id: int) -> None:
    """Link an event to an incident in the junction table."""
    sql = """
        INSERT INTO incident_events (incident_id, event_id)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (incident_id, event_id))
        conn.commit()


# ---------------------------------------------------------------------------
# response_actions
# ---------------------------------------------------------------------------
def insert_response_action(
    incident_id: int,
    action_code: str,
    action_label: Optional[str] = None,
) -> int:
    """Insert one response action row and return its id."""
    sql = """
        INSERT INTO response_actions (incident_id, action_code, action_label)
        VALUES (%s, %s, %s)
        RETURNING id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (incident_id, action_code, action_label))
            row = cur.fetchone()
        conn.commit()
    return row["id"]


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def insert_report(
    incident_id: int,
    summary: Optional[str] = None,
    what_happened: Optional[list] = None,
    risk_reason: Optional[str] = None,
    persons: Optional[str] = None,
) -> int:
    """Insert a report row (one per incident) and return its id."""
    sql = """
        INSERT INTO reports
            (incident_id, summary, what_happened, risk_reason, persons)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (incident_id) DO UPDATE
            SET summary       = EXCLUDED.summary,
                what_happened = EXCLUDED.what_happened,
                risk_reason   = EXCLUDED.risk_reason,
                persons       = EXCLUDED.persons,
                generated_at  = now()
        RETURNING id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (incident_id, summary, what_happened or [], risk_reason, persons))
            row = cur.fetchone()
        conn.commit()
    return row["id"]


# ---------------------------------------------------------------------------
# evidence_frames
# ---------------------------------------------------------------------------
def insert_evidence_frame(
    incident_id: int,
    filename: str,
    event_id: Optional[int] = None,
    file_path: Optional[str] = None,
    image_url: Optional[str] = None,
    frame: Optional[int] = None,
    timestamp_sec: Optional[float] = None,
    exists_on_disk: bool = False,
) -> int:
    """Insert an evidence frame record and return its id."""
    sql = """
        INSERT INTO evidence_frames
            (incident_id, event_id, filename, file_path,
             image_url, frame, timestamp_sec, exists_on_disk)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (incident_id, event_id, filename, file_path,
                              image_url, frame, timestamp_sec, exists_on_disk))
            row = cur.fetchone()
        conn.commit()
    return row["id"]


# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------
def test_connection() -> bool:
    """Return True if a connection to the database can be established."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database(), current_user, now()")
                row = cur.fetchone()
        print(f"[db] Connected — database: {row['current_database']}, "
              f"user: {row['current_user']}, server time: {row['now']}")
        return True
    except Exception as exc:
        print(f"[db] Connection failed: {exc}", file=sys.stderr)
        return False


if __name__ == "__main__":
    ok = test_connection()
    sys.exit(0 if ok else 1)
