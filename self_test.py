"""Offline rule/geometry self-test; does not require YOLO weights."""
import numpy as np, cv2
from restricted_zone_detector import RestrictedZoneDetector
from fall_detector import FallDetector

def make_red_barrier_frame():
    f=np.zeros((720,1280,3),np.uint8)+255
    # red rope/barricade matching the intended geometry
    cv2.line(f,(764,355),(928,262),(0,0,255),8)
    cv2.line(f,(764,355),(1122,435),(0,0,255),8)
    cv2.line(f,(928,262),(1122,435),(0,0,255),8)
    return f

def test_zone():
    d=RestrictedZoneDetector(); found=None
    f=make_red_barrier_frame()
    for i in range(0,96,8):
        found=d.auto_update(f,i)
    assert found and len(found)>=3, "AUTO Restricted Zone failed"
    return found

def test_no_zone():
    d=RestrictedZoneDetector(); f=np.zeros((720,1280,3),np.uint8)+255
    for i in range(0,96,8): d.auto_update(f,i)
    assert d.current_zone() is None, "False Restricted Zone on blank scene"

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

if __name__=="__main__":
    print("AUTO ZONE:", test_zone())
    test_no_zone(); print("NO-ZONE: PASS")
    test_fall(); print("FALL: PASS")
    print("SELF-TEST PASS")
