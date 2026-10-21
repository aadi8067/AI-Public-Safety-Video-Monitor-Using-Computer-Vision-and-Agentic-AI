-- =============================================================================
-- AI Public Safety Video Monitor — Database Schema
-- Target:   public_safety_db (PostgreSQL 18)
-- Strategy: CREATE TABLE IF NOT EXISTS on every object.
--           Safe to re-run — will never drop or overwrite existing data.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. sessions
--    One row per uploaded video file / processing run.
--    Every other table references this via session_id.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id              SERIAL          PRIMARY KEY,
    filename        TEXT            NOT NULL,               -- original upload name
    file_path       TEXT,                                   -- absolute path on disk
    fps             FLOAT,                                  -- frames per second
    total_frames    INTEGER,
    duration_sec    FLOAT,
    status          TEXT            NOT NULL DEFAULT 'pending'
                                    CHECK (status IN ('pending','processing','done','failed')),
    output_dir      TEXT,                                   -- evidence frames directory
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 2. persons
--    Stable tracked person identities resolved by StablePersonIdentity.
--    PERSON-001 etc. live here; one row per unique stable_id per session.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS persons (
    id              SERIAL          PRIMARY KEY,
    session_id      INTEGER         NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    stable_id       INTEGER         NOT NULL,               -- the public PERSON-NNN number
    first_frame     INTEGER,
    last_frame      INTEGER,
    is_threat       BOOLEAN         NOT NULL DEFAULT false, -- ever linked to a confirmed weapon
    UNIQUE (session_id, stable_id)
);

-- ---------------------------------------------------------------------------
-- 3. events
--    One row per structured safety event fired by SafetyEventPipeline.
--    event_type is one of the seven detector outputs:
--      WEAPON_CONFIRMED | POSSIBLE_FALL | ZONE_INTRUSION | LOITERING |
--      ABNORMAL_MOVEMENT | HIGH_DENSITY | CROWD_BUILDUP
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    id              SERIAL          PRIMARY KEY,
    session_id      INTEGER         NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    event_uid       TEXT            NOT NULL,               -- pipeline EVENT-0001 string
    event_type      TEXT            NOT NULL,
    person_id       INTEGER,                                -- NULL for scene-wide events
    frame           INTEGER,
    timestamp_sec   FLOAT,
    confidence      FLOAT,
    severity        TEXT            CHECK (severity IN ('CRITICAL','HIGH','MEDIUM','LOW')),
    details         JSONB,                                  -- weapon_id, weapon_type, extras
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    UNIQUE (session_id, event_uid)
);

-- ---------------------------------------------------------------------------
-- 4. incidents
--    One row per correlated incident produced by EventCorrelator.
--    event_types and person_ids are PostgreSQL native arrays so they can be
--    queried with the ANY / @> operators without JSON parsing.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS incidents (
    id              SERIAL          PRIMARY KEY,
    session_id      INTEGER         NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    incident_uid    TEXT            NOT NULL,               -- INC-0001 string
    start_time      FLOAT,
    end_time        FLOAT,
    duration_sec    FLOAT,
    event_types     TEXT[],                                 -- e.g. {WEAPON_CONFIRMED,LOITERING}
    person_ids      INTEGER[],
    event_count     INTEGER         NOT NULL DEFAULT 0,
    risk_score      FLOAT           CHECK (risk_score >= 0 AND risk_score <= 100),
    risk_severity   TEXT            CHECK (risk_severity IN ('CRITICAL','HIGH','MEDIUM','LOW')),
    risk_explanation TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    UNIQUE (session_id, incident_uid)
);

-- ---------------------------------------------------------------------------
-- 5. incident_events  (junction table)
--    Links each event to the incident it was correlated into.
--    Avoids storing a JSON blob of event IDs inside the incident row.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS incident_events (
    incident_id     INTEGER         NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    event_id        INTEGER         NOT NULL REFERENCES events(id)    ON DELETE CASCADE,
    PRIMARY KEY (incident_id, event_id)
);

-- ---------------------------------------------------------------------------
-- 6. response_actions
--    One row per approved action per incident (from ResponseAgent).
--    Normalized so actions can be queried/counted independently.
--    action_code values: LOG_INCIDENT | FLAG_FOR_REVIEW | DISPATCH_ALERT |
--      RETAIN_EVIDENCE | NOTIFY_SECURITY_TEAM | REQUIRE_HUMAN_VERIFICATION
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS response_actions (
    id              SERIAL          PRIMARY KEY,
    incident_id     INTEGER         NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    action_code     TEXT            NOT NULL,
    action_label    TEXT,
    triggered_at    TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 7. reports
--    One row per incident report generated by ReportAgent.
--    what_happened is a TEXT[] array (one bullet point per element).
--    UNIQUE on incident_id enforces one report per incident.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reports (
    id              SERIAL          PRIMARY KEY,
    incident_id     INTEGER         NOT NULL REFERENCES incidents(id) ON DELETE CASCADE UNIQUE,
    summary         TEXT,                                   -- plain-English one-liner
    what_happened   TEXT[],                                 -- bullet point array
    risk_reason     TEXT,
    persons         TEXT,                                   -- 'PERSON-001, PERSON-002' / 'SCENE-WIDE'
    generated_at    TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 8. evidence_frames
--    One row per saved JPEG evidence screenshot linked to an event.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence_frames (
    id              SERIAL          PRIMARY KEY,
    incident_id     INTEGER         NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    event_id        INTEGER         REFERENCES events(id)  ON DELETE SET NULL,
    filename        TEXT            NOT NULL,
    file_path       TEXT,
    image_url       TEXT,                                   -- '/outputs/<dir>/<filename>'
    frame           INTEGER,
    timestamp_sec   FLOAT,
    exists_on_disk  BOOLEAN         NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- =============================================================================
-- Indexes — targeted only on columns that will be filtered/joined in queries.
-- Kept minimal to avoid write overhead on inserts during video processing.
-- =============================================================================

-- Sessions: look up by status (e.g. find all 'processing' runs)
CREATE INDEX IF NOT EXISTS idx_sessions_status
    ON sessions (status);

-- Persons: look up all persons for a session
CREATE INDEX IF NOT EXISTS idx_persons_session
    ON persons (session_id);

-- Events: the two most common filters — by session and by event_type
CREATE INDEX IF NOT EXISTS idx_events_session
    ON events (session_id);

CREATE INDEX IF NOT EXISTS idx_events_type
    ON events (event_type);

-- Incidents: look up by session and by risk severity
CREATE INDEX IF NOT EXISTS idx_incidents_session
    ON incidents (session_id);

CREATE INDEX IF NOT EXISTS idx_incidents_severity
    ON incidents (risk_severity);

-- Response actions: look up all actions for an incident
CREATE INDEX IF NOT EXISTS idx_response_actions_incident
    ON response_actions (incident_id);

-- Evidence frames: look up all frames for an incident
CREATE INDEX IF NOT EXISTS idx_evidence_incident
    ON evidence_frames (incident_id);

-- =============================================================================
-- End of schema
-- =============================================================================
