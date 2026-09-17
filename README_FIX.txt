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
