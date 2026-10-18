PUBLIC SAFETY SAFETY-EVENT FIX

Changes:
- Reduced false ABNORMAL_MOVEMENT alerts by smoothing trajectory and normalizing speed by person body height.
- Fall detection now requires a real downward transition plus sustained horizontal/low posture; it no longer requires both conditions in one exact frame.
- Loitering zone defaults to the full camera view instead of silently reusing the restricted zone.
- Restricted zone remains configurable through /safety/config.
- Weapon/person pipeline in app.py is preserved.

Run:
python app.py

The Ultralytics model files (yolov8s.pt and best.pt) must remain beside app.py.


V4 RESTRICTED-ZONE CALIBRATION
- The polygon is stored in normalized 0..1 coordinates, so it is resolution-independent.
- It is configurable through SAFETY_RESTRICTED_ZONE or POST /safety/config.
- The supplied default is calibrated to the supplied metro CCTV camera only.
- For a different camera/view, configure a new polygon; the detector does not automatically infer a restricted area from arbitrary video.


V5 FIX -- "TypeError: 'numpy.int32' object is not iterable" during /detect_video
- Cause: restricted_zone_detector auto-zone code assumed cv2.HoughLinesP always returns (N,1,4)
  and did `for line in lines[:,0]: map(int, line)`. With a flat (N,4) array that yields bare
  numpy.int32 scalars, which are not iterable, and the whole request aborted.
- Fix: line/hull/polygon arrays are reshaped defensively; auto-zone detection is best-effort
  (failure is logged and the video continues without a zone).
- Fault isolation: every detector in SafetyEventPipeline (and the agentic layer) is wrapped so one
  failing detector can no longer abort the video. The FULL traceback is printed once per detector,
  failures are counted in the JSON field `detector_errors`, and the dashboard shows a warning.
- The /detect_video error handler now prints the full traceback and returns error_type + error_location.
- New self-tests: ZONE-HOUGH-SHAPES, PIPELINE-FAULT-ISOLATION.


V6 -- AGENTIC AI NOW RUNS ON LANGCHAIN + LANGGRAPH (behaviour unchanged)
- New file: agentic_graph.py
  - Risk Agent and Response Agent are wrapped as LangChain Runnables
    (RunnableLambda) and executed as two nodes of a compiled LangGraph
    StateGraph: assess_risk -> determine_response.
  - RiskAgent.assess(), ResponseAgent.determine() and EventCorrelator are
    COMPLETELY UNCHANGED -- same weights, same thresholds, same weapon
    floor, same fixed action table. Only the orchestration around them
    changed; AgenticSupervisor's public API (ingest_event/finalize/reset)
    is identical to before.
  - If langchain-core / langgraph are not installed (or fail to import),
    the code automatically falls back to a plain-Python pipeline that
    runs the exact same two calls in the exact same order -- so the app
    keeps working even before `pip install -r requirements.txt` is re-run
    on a machine that doesn't have the two new packages yet.
  - requirements.txt: added langchain-core>=1.0,<2.0 and langgraph>=1.0,<2.0.
  - app.py prints which backend is active on startup and returns it as
    "agentic_backend" in the /detect_video JSON response.
- New self-test: AGENTIC-GRAPH-PARITY -- asserts the LangGraph pipeline's
  output is byte-identical to calling RiskAgent/ResponseAgent directly.
- Verified end-to-end on both sample videos with the real Flask route
  (person/weapon detection, safety detectors, correlation, scoring,
  response) all producing identical incidents whether the LangGraph
  backend or the fallback pipeline is active.
- IMPORTANT: this sandbox has no internet access, so langchain-core and
  langgraph could not actually be pip-installed and exercised here. The
  LangGraph/LangChain code was written against their stable, documented
  1.x API (StateGraph/START/END/add_node/add_edge/compile, RunnableLambda)
  and verified structurally with local stand-in modules that implement
  just that API surface, confirming correct node wiring and output parity
  with the fallback path. Run `pip install -r requirements.txt` and then
  `python3 self_test.py` on your machine to confirm against the real
  libraries -- the self-test prints which backend (langgraph vs fallback)
  actually ran.
