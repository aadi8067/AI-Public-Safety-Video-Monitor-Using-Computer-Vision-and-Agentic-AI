# Final Public Safety Video Monitor

Run:
python app.py

Place beside app.py:
- yolov8s.pt
- gun_knife_yolo11n.pt
- best.pt (fallback only)

Weapon policy:
- gun_knife_yolo11n.pt is primary for GUN/KNIFE.
- best.pt is only fallback if the primary checkpoint is unavailable.
- Every weapon alert requires person association, temporal confirmation and
  geometry checks.
- Confirmed weapon boxes persist on intermediate frames.

Restricted Zone:
- Each video starts with NO restricted zone.
- AUTO mode detects a stable red barrier/ribbon geometry across multiple frames.
- A previous video's zone is never reused.
- If no qualifying barrier is found, there is no RESTRICTED ZONE and no
  ZONE_INTRUSION.
- Manual polygon is available through SAFETY_RESTRICTED_ZONE or /safety/config.

Fall:
- Temporal downward transition + body-height collapse + horizontal/low posture
  + persistence.

Run offline detector tests:
python self_test.py
