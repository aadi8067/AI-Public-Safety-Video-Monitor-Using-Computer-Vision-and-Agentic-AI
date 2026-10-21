"""Offline rule/geometry self-test; does not require YOLO weights."""
import numpy as np, cv2
from restricted_zone_detector import RestrictedZoneDetector
from fall_detector import FallDetector
from event_correlator import EventCorrelator
from agentic_ai import AgenticSupervisor, RiskAgent, ResponseAgent
from agentic_graph import build_agentic_pipeline, backend_status

def make_red_barrier_frame():
    f=np.zeros((720,1280,3),np.uint8)+255
    # red rope/barricade matching the intended geometry
    cv2.line(f,(764,355),(928,262),(0,0,255),8)
    cv2.line(f,(764,355),(1122,435),(0,0,255),8)
    cv2.line(f,(928,262),(1122,435),(0,0,255),8)
    return f

def test_zone():
    # auto_update() only finalizes once frame_index > ZONE_AUTO_WARMUP_FRAMES
    # (96), so the sampling loop must actually cross that threshold, not
    # just approach it.
    d=RestrictedZoneDetector(); found=None
    f=make_red_barrier_frame()
    for i in range(0,104,8):
        found=d.auto_update(f,i)
    assert found and len(found)>=3, "AUTO Restricted Zone failed"
    return found

def test_no_zone():
    # current_zone() returns [] (not None) by design when no zone is set
    # -- normalize_polygon() always coerces a missing polygon to [].
    d=RestrictedZoneDetector(); f=np.zeros((720,1280,3),np.uint8)+255
    for i in range(0,104,8): d.auto_update(f,i)
    assert not d.current_zone(), "False Restricted Zone on blank scene"

def test_fall():
    d=FallDetector(); fps=24.0; pid=1; events=[]
    # upright for ~1 sec, then rapid drop and horizontal posture for >1 sec
    for i in range(0,72):
        if i<24: bbox=[300,100,360,280]
        elif i<36:
            t=(i-24)/12; bbox=[300,100+int(160*t),420,280+int(40*t)]
        else: bbox=[300,360,500,430]
        events += d.update([{"stable_id":pid,"bbox":bbox}],i,fps)
    assert any(e["event_type"]=="POSSIBLE_FALL" for e in events), "Fall detector failed"

def _ev(event_type, t, pid=None, severity="MEDIUM", conf=0.7):
    return {"event_type": event_type, "severity": severity, "person_id": pid,
            "confidence": conf, "timestamp": t, "details": {}}

def test_correlator_groups_close_related_events():
    # A crowd-wide event followed shortly by a movement + fall on the same
    # person, all within the correlation window, must land in ONE incident.
    c = EventCorrelator(window_seconds=6.0)
    c.ingest(_ev("HIGH_DENSITY", 1.0, pid=None, severity="HIGH"))
    c.ingest(_ev("ABNORMAL_MOVEMENT", 2.5, pid=5))
    inc = c.ingest(_ev("POSSIBLE_FALL", 4.0, pid=5, severity="HIGH"))
    assert inc.person_ids == {5}
    assert len(inc.events) == 3, "related events should correlate into one incident"

def test_correlator_splits_unrelated_people():
    # An event on a different, unrelated person should NOT be folded into
    # an incident that is already specifically about someone else.
    c = EventCorrelator(window_seconds=6.0)
    c.ingest(_ev("ABNORMAL_MOVEMENT", 1.0, pid=5))
    c.ingest(_ev("POSSIBLE_FALL", 2.0, pid=5, severity="HIGH"))
    other = c.ingest(_ev("ZONE_INTRUSION", 2.5, pid=9, severity="HIGH"))
    assert other.person_ids == {9}, "a different tracked person must start its own incident"
    assert len(c.open_incidents) == 2

def test_correlator_closes_stale_incidents():
    # Events far apart in time must NOT be merged into the same incident.
    c = EventCorrelator(window_seconds=6.0)
    c.ingest(_ev("LOITERING", 1.0, pid=2))
    c.ingest(_ev("LOITERING", 40.0, pid=2))
    final = c.finalize()
    assert len(final) == 2, "events separated by a long gap must be separate incidents"

def test_risk_agent_weapon_alone_is_always_critical():
    # A CONFIRMED weapon has already passed strict multi-frame confirmation
    # upstream, so on its own it must never be triaged below CRITICAL.
    risk = RiskAgent().assess({"events": [_ev("WEAPON_CONFIRMED", 1.0, pid=1, severity="CRITICAL", conf=0.5)]})
    assert risk["risk_severity"] == "CRITICAL", "lone confirmed weapon must floor to CRITICAL"

def test_risk_agent_scales_with_correlated_signals():
    # The same event set correlated together should score at least as high
    # as any single event in isolation (correlation bonus, not a penalty).
    lone = RiskAgent().assess({"events": [_ev("ABNORMAL_MOVEMENT", 1.0, pid=1)]})
    combined = RiskAgent().assess({"events": [
        _ev("HIGH_DENSITY", 1.0, pid=None, severity="HIGH"),
        _ev("ABNORMAL_MOVEMENT", 2.0, pid=1),
        _ev("POSSIBLE_FALL", 3.0, pid=1, severity="HIGH"),
    ]})
    assert combined["risk_score"] > lone["risk_score"]

def test_response_agent_action_table_is_fixed():
    # ResponseAgent must only ever emit actions from the approved table.
    from safety_config import AGENTIC_BASE_ACTIONS
    approved = {a for actions in AGENTIC_BASE_ACTIONS.values() for a in actions}
    risk = {"risk_severity": "CRITICAL"}
    incident = {"event_types": ["WEAPON_CONFIRMED", "ABNORMAL_MOVEMENT"]}
    response = ResponseAgent().determine(risk, incident)
    assert set(response["response_actions"]).issubset(approved)
    assert "NOTIFY_SECURITY_TEAM" in response["response_actions"]
    assert "REQUIRE_HUMAN_VERIFICATION" in response["response_actions"]

def test_agentic_supervisor_end_to_end():
    # Full Structured Events -> Correlator -> Supervisor -> Risk -> Response
    # pipeline, exactly as app.py drives it frame by frame.
    sup = AgenticSupervisor()
    sup.ingest_event(_ev("HIGH_DENSITY", 1.0, pid=None, severity="HIGH", conf=0.8))
    sup.ingest_event(_ev("ABNORMAL_MOVEMENT", 2.0, pid=3, conf=0.6))
    sup.ingest_event(_ev("POSSIBLE_FALL", 3.5, pid=3, severity="HIGH", conf=0.75))
    sup.ingest_event(_ev("WEAPON_CONFIRMED", 20.0, pid=9, severity="CRITICAL", conf=0.9))
    final = sup.finalize()
    assert len(final) == 2, "expected two distinct incidents (correlated group + isolated weapon)"
    critical = [i for i in final if i["risk_severity"] == "CRITICAL"]
    assert critical and "WEAPON_CONFIRMED" in critical[0]["event_types"]
    assert "REQUIRE_HUMAN_VERIFICATION" in critical[0]["response_actions"]

def test_zone_survives_opencv_line_array_shapes():
    # Regression: HoughLinesP output is normally (N,1,4) but a flat (N,4)
    # array made the old `for line in lines[:,0]: map(int, line)` iterate a
    # bare numpy.int32 -> TypeError("'numpy.int32' object is not iterable")
    # and abort the whole /detect_video request. Both shapes must work.
    real = cv2.HoughLinesP
    f = make_red_barrier_frame()
    try:
        for reshape in (lambda l: l, lambda l: l.reshape(-1, 4)):
            cv2.HoughLinesP = lambda *a, _r=reshape, **k: (lambda l: None if l is None else _r(l))(real(*a, **k))
            cand = RestrictedZoneDetector()._candidate_from_frame(f)
            assert cand and len(cand) >= 3, "zone candidate not found for this Hough shape"
    finally:
        cv2.HoughLinesP = real

def test_pipeline_isolates_detector_failures():
    # A bug inside ONE detector must be logged + counted, never abort the video.
    from safety_event_pipeline import SafetyEventPipeline
    pipe = SafetyEventPipeline()
    def boom(*a, **k): raise TypeError("'numpy.int32' object is not iterable")
    pipe.loitering.update = boom
    people = [{"stable_id": 1, "bbox": np.array([100, 100, 180, 300], dtype=np.int32)}]
    frame = np.zeros((720, 1280, 3), np.uint8)
    res = pipe.update(people, frame_index=5, fps=24.0, frame_shape=frame.shape, frame=frame)
    assert res["detector_errors"].get("loitering") == 1
    assert "crowd_count" in res  # the rest of the frame still processed

def test_agentic_graph_pipeline_matches_direct_calls():
    # The LangGraph pipeline (agentic_graph.py) must produce output
    # byte-identical to calling RiskAgent.assess() / ResponseAgent.determine()
    # directly, whether or not langchain-core/langgraph are installed --
    # the graph only changes how the two calls are sequenced, never what
    # they compute.
    import copy
    incident_template = {
        "incident_id": "INC-TEST", "start_time": 1.0, "end_time": 3.0,
        "person_ids": [1], "event_types": ["WEAPON_CONFIRMED", "ABNORMAL_MOVEMENT"],
        "events": [
            _ev("WEAPON_CONFIRMED", 1.0, pid=1, severity="CRITICAL", conf=0.9),
            _ev("ABNORMAL_MOVEMENT", 2.0, pid=1, severity="MEDIUM", conf=0.6),
        ],
    }
    risk_agent, response_agent = RiskAgent(), ResponseAgent()
    pipeline = build_agentic_pipeline(risk_agent, response_agent)

    via_pipeline = pipeline.run(copy.deepcopy(incident_template))

    direct = copy.deepcopy(incident_template)
    risk = risk_agent.assess(direct)
    response = response_agent.determine(risk, direct)
    direct.update(risk)
    direct.update(response)

    assert via_pipeline == direct, "pipeline output diverged from direct Risk/Response calls"

if __name__=="__main__":
    print("AUTO ZONE:", test_zone())
    test_no_zone(); print("NO-ZONE: PASS")
    test_fall(); print("FALL: PASS")
    test_correlator_groups_close_related_events(); print("CORRELATOR-GROUP: PASS")
    test_correlator_splits_unrelated_people(); print("CORRELATOR-SPLIT: PASS")
    test_correlator_closes_stale_incidents(); print("CORRELATOR-STALE: PASS")
    test_risk_agent_weapon_alone_is_always_critical(); print("RISK-WEAPON-FLOOR: PASS")
    test_risk_agent_scales_with_correlated_signals(); print("RISK-CORRELATION-SCALING: PASS")
    test_response_agent_action_table_is_fixed(); print("RESPONSE-ACTION-TABLE: PASS")
    test_agentic_supervisor_end_to_end(); print("AGENTIC-SUPERVISOR-E2E: PASS")
    test_zone_survives_opencv_line_array_shapes(); print("ZONE-HOUGH-SHAPES: PASS")
    test_pipeline_isolates_detector_failures(); print("PIPELINE-FAULT-ISOLATION: PASS")
    test_agentic_graph_pipeline_matches_direct_calls(); print("AGENTIC-GRAPH-PARITY: PASS")
    print(f"AGENTIC PIPELINE BACKEND: {backend_status()}")
    print("SELF-TEST PASS")


# ============================================================
# REPORT AGENT TESTS
# ============================================================

def _make_incident(incident_id="INC-0001", events=None, risk_score=58.4,
                   risk_severity="HIGH", start_time=1.8, end_time=4.04,
                   person_ids=None):
    """Build a minimal finalized incident dict as AgenticSupervisor.finalize()
    would return it (risk + response already merged in)."""
    if person_ids is None:
        person_ids = [3]
    if events is None:
        events = [
            {"event_type": "HIGH_DENSITY",      "frame": 45,  "timestamp": 1.8,  "person_id": None, "confidence": 0.72, "severity": "HIGH",   "details": {}},
            {"event_type": "ABNORMAL_MOVEMENT",  "frame": 63,  "timestamp": 2.52, "person_id": 3,    "confidence": 0.65, "severity": "MEDIUM", "details": {}},
            {"event_type": "POSSIBLE_FALL",      "frame": 101, "timestamp": 4.04, "person_id": 3,    "confidence": 0.85, "severity": "HIGH",   "details": {}},
        ]
    return {
        "incident_id":     incident_id,
        "start_time":      start_time,
        "end_time":        end_time,
        "duration_seconds": round(end_time - start_time, 2),
        "person_ids":      person_ids,
        "event_types":     sorted({e["event_type"] for e in events}),
        "event_count":     len(events),
        "events":          events,
        # Risk Agent fields
        "risk_score":      risk_score,
        "risk_severity":   risk_severity,
        "risk_explanation": f"{risk_severity} ({risk_score}/100) test",
        "risk_factors":    [],
        "risk_bonuses":    [],
        # Response Agent fields
        "response_actions": ["LOG_INCIDENT", "FLAG_FOR_REVIEW", "DISPATCH_ALERT"],
        "response_action_labels": ["Log incident", "Flag for review", "Dispatch alert"],
    }


def test_report_agent_basic_structure():
    """Generated report must contain all required keys and correct values."""
    from report_agent import ReportAgent
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        inc = _make_incident()
        report = ReportAgent().generate(inc, fps=25.0, capture_dir=tmp, capture_dir_name="captured_frames_TEST")
        assert report["incident_id"] == "INC-0001"
        assert report["risk_score"] == 58.4
        assert report["risk_severity"] == "HIGH"
        assert abs(report["start_time"] - 1.8) < 0.01
        assert abs(report["end_time"] - 4.04) < 0.01
        assert "PERSON-003" in report["persons"]
        assert isinstance(report["summary"], str) and len(report["summary"]) > 10
        assert isinstance(report["evidence"], list)
        assert isinstance(report["response_actions"], list)
        assert "LOG_INCIDENT" in report["response_actions"]


def test_report_agent_evidence_exists():
    """When an evidence file exists, image_url must be a valid path string."""
    from report_agent import ReportAgent, _evidence_filename_for_event
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        # Create the exact file detect_video() would have written.
        ev = {"event_type": "POSSIBLE_FALL", "frame": 101, "person_id": 3, "timestamp": 4.04, "confidence": 0.85, "severity": "HIGH", "details": {}}
        filename = _evidence_filename_for_event(ev)
        open(os.path.join(tmp, filename), "w").close()   # create empty placeholder

        inc = _make_incident(events=[ev], person_ids=[3])
        report = ReportAgent().generate(inc, fps=25.0, capture_dir=tmp, capture_dir_name="captured_frames_TEST")
        assert len(report["evidence"]) == 1
        e = report["evidence"][0]
        assert e["exists"] is True
        assert e["image_url"] == "/outputs/captured_frames_TEST/" + filename
        assert e["time_seconds"] == round(101 / 25.0, 2)


def test_report_agent_missing_evidence():
    """When an evidence file is absent, image_url is None and exists is False."""
    from report_agent import ReportAgent
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        inc = _make_incident()   # files never created
        report = ReportAgent().generate(inc, fps=25.0, capture_dir=tmp, capture_dir_name="captured_frames_TEST")
        # All evidence entries must have exists=False and image_url=None
        for e in report["evidence"]:
            assert e["exists"] is False
            assert e["image_url"] is None


def test_report_agent_weapon_evidence_filename():
    """WEAPON_CONFIRMED events must resolve to the weapon-specific filename."""
    from report_agent import _evidence_filename_for_event
    ev = {"event_type": "WEAPON_CONFIRMED", "frame": 112, "person_id": 7,
          "timestamp": 4.48, "confidence": 0.78, "severity": "CRITICAL",
          "details": {"weapon_id": 33, "weapon_type": "gun"}}
    filename = _evidence_filename_for_event(ev)
    assert filename == "evidence_WEAPON-033_frame_000112.jpg"


def test_report_agent_no_events_no_crash():
    """An incident with an empty events list must not raise."""
    from report_agent import ReportAgent
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        inc = _make_incident(events=[], person_ids=[])
        report = ReportAgent().generate(inc, fps=25.0, capture_dir=tmp, capture_dir_name="x")
        assert report["evidence"] == []
        assert report["persons"] == "SCENE-WIDE"


def test_report_agent_two_incidents_independent():
    """Two incidents must produce two separate reports with no cross-contamination."""
    from report_agent import generate_incident_reports
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        inc1 = _make_incident("INC-0001", person_ids=[3], risk_score=58.4, risk_severity="HIGH")
        inc2 = _make_incident("INC-0002", person_ids=[9], risk_score=75.0, risk_severity="CRITICAL",
                              start_time=20.0, end_time=21.0,
                              events=[{"event_type": "WEAPON_CONFIRMED", "frame": 500,
                                       "person_id": 9, "timestamp": 20.0, "confidence": 0.9,
                                       "severity": "CRITICAL",
                                       "details": {"weapon_id": 1, "weapon_type": "gun"}}])
        reports = generate_incident_reports([inc1, inc2], fps=25.0,
                                            capture_dir=tmp, capture_dir_name="x")
        assert len(reports) == 2
        ids = {r["incident_id"] for r in reports}
        assert ids == {"INC-0001", "INC-0002"}
        r1 = next(r for r in reports if r["incident_id"] == "INC-0001")
        r2 = next(r for r in reports if r["incident_id"] == "INC-0002")
        assert "PERSON-003" in r1["persons"]
        assert "PERSON-009" in r2["persons"]
        assert r1["risk_severity"] == "HIGH"
        assert r2["risk_severity"] == "CRITICAL"


def test_report_agent_does_not_mutate_incident():
    """ReportAgent.generate() must not modify the original incident dict."""
    from report_agent import ReportAgent
    import copy, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        inc = _make_incident()
        original = copy.deepcopy(inc)
        ReportAgent().generate(inc, fps=25.0, capture_dir=tmp, capture_dir_name="x")
        assert inc == original


if __name__ == "__main__":
    print("AUTO ZONE:", test_zone())
    test_no_zone(); print("NO-ZONE: PASS")
    test_fall(); print("FALL: PASS")
    test_correlator_groups_close_related_events(); print("CORRELATOR-GROUP: PASS")
    test_correlator_splits_unrelated_people(); print("CORRELATOR-SPLIT: PASS")
    test_correlator_closes_stale_incidents(); print("CORRELATOR-STALE: PASS")
    test_risk_agent_weapon_alone_is_always_critical(); print("RISK-WEAPON-FLOOR: PASS")
    test_risk_agent_scales_with_correlated_signals(); print("RISK-CORRELATION-SCALING: PASS")
    test_response_agent_action_table_is_fixed(); print("RESPONSE-ACTION-TABLE: PASS")
    test_agentic_supervisor_end_to_end(); print("AGENTIC-SUPERVISOR-E2E: PASS")
    test_zone_survives_opencv_line_array_shapes(); print("ZONE-HOUGH-SHAPES: PASS")
    test_pipeline_isolates_detector_failures(); print("PIPELINE-FAULT-ISOLATION: PASS")
    test_agentic_graph_pipeline_matches_direct_calls(); print("AGENTIC-GRAPH-PARITY: PASS")
    print(f"AGENTIC PIPELINE BACKEND: {backend_status()}")
    test_report_agent_basic_structure(); print("REPORT-AGENT-STRUCTURE: PASS")
    test_report_agent_evidence_exists(); print("REPORT-AGENT-EVIDENCE-EXISTS: PASS")
    test_report_agent_missing_evidence(); print("REPORT-AGENT-MISSING-EVIDENCE: PASS")
    test_report_agent_weapon_evidence_filename(); print("REPORT-AGENT-WEAPON-FILENAME: PASS")
    test_report_agent_no_events_no_crash(); print("REPORT-AGENT-EMPTY-EVENTS: PASS")
    test_report_agent_two_incidents_independent(); print("REPORT-AGENT-TWO-INCIDENTS: PASS")
    test_report_agent_does_not_mutate_incident(); print("REPORT-AGENT-NO-MUTATION: PASS")
    print("SELF-TEST PASS")
