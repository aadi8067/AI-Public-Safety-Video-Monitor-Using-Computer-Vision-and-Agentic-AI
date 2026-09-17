"""Temporal fall detector based on tracked-person geometry."""
from collections import defaultdict, deque
from statistics import median
from safety_config import (
    FALL_MIN_DROP_RATIO, FALL_HORIZONTAL_RATIO, FALL_MAX_TALL_RATIO,
    FALL_HEIGHT_DROP_RATIO, FALL_VERIFY_SECONDS, FALL_CONFIRM_FRAMES,
    FALL_MIN_PERSON_HEIGHT, MAX_HISTORY_FRAMES,
)
from safety_geometry import center

class FallDetector:
    def __init__(self): self.reset()
    def reset(self): self.state=defaultdict(dict)

    def update(self, people, frame_index, fps):
        fps=max(float(fps),1.0); now=frame_index/fps
        events=[]; active=set()
        for person in people:
            pid=person.get("stable_id")
            if pid is None: continue
            pid=int(pid); active.add(pid)
            x1,y1,x2,y2=map(float,person["bbox"])
            w=max(1.0,x2-x1); h=max(1.0,y2-y1)
            if h<FALL_MIN_PERSON_HEIGHT: continue
            c=center([x1,y1,x2,y2]); ratio=w/h
            s=self.state[pid]
            hist=s.setdefault("history",deque(maxlen=30))
            hist.append((now,c,w,h,ratio))
            if len(hist)<6: continue

            # Estimate the person's recent upright body height. This remains
            # available even when the person has already become horizontal.
            recent=list(hist)
            upright=[z[3] for z in recent[:-4] if z[4] <= FALL_MAX_TALL_RATIO]
            baseline=median(upright[-8:]) if upright else s.get("baseline_height",h)
            if ratio <= FALL_MAX_TALL_RATIO:
                s["baseline_height"]=max(float(baseline),h)

            baseline=max(float(s.get("baseline_height",baseline)),45.0)
            look=[z for z in recent if now-z[0] <= 0.55]
            old=[z for z in recent if 0.35 <= now-z[0] <= 1.20]
            if len(old)<2 or len(look)<2: continue

            old_y=old[0][1][1]
            drop=(c[1]-old_y)/baseline
            height_ratio=h/baseline
            horizontal=ratio>=FALL_HORIZONTAL_RATIO
            collapsed=height_ratio<=FALL_HEIGHT_DROP_RATIO
            recent_down=sum(1 for z in recent[-6:] if z[4]>=FALL_HORIZONTAL_RATIO or z[3]/baseline<=FALL_HEIGHT_DROP_RATIO)

            candidate = drop>=FALL_MIN_DROP_RATIO and (horizontal or collapsed)
            if candidate and s.get("candidate_start") is None:
                s["candidate_start"]=now
                s["candidate_streak"]=0
                s["peak_drop"]=drop

            if s.get("candidate_start") is not None:
                s["peak_drop"]=max(float(s.get("peak_drop",0)),drop)
                if horizontal or collapsed:
                    s["candidate_streak"]=s.get("candidate_streak",0)+1
                else:
                    s["candidate_streak"]=max(0,s.get("candidate_streak",0)-1)
                elapsed=now-float(s["candidate_start"])
                verified=(elapsed>=FALL_VERIFY_SECONDS and
                          s.get("candidate_streak",0)>=FALL_CONFIRM_FRAMES and
                          s.get("peak_drop",0)>=FALL_MIN_DROP_RATIO and
                          recent_down>=3 and (horizontal or collapsed))
                if verified and not s.get("alerted",False):
                    s["alerted"]=True
                    conf=min(0.98,0.62+min(0.18,s["peak_drop"]*0.35)+min(0.12,elapsed/8.0))
                    events.append({
                        "event_type":"POSSIBLE_FALL","severity":"HIGH","person_id":pid,
                        "confidence":conf,"frame":frame_index,"timestamp":now,
                        "details":{
                            "posture_ratio":round(ratio,3),
                            "drop_ratio":round(s["peak_drop"],3),
                            "height_ratio":round(height_ratio,3),
                            "verification_seconds":round(elapsed,2),
                            "method":"temporal_drop_height_collapse"
                        }
                    })
                # Clear only after the person has demonstrably returned upright.
                if ratio<FALL_MAX_TALL_RATIO and height_ratio>0.78 and s.get("candidate_streak",0)==0:
                    s.pop("candidate_start",None); s.pop("candidate_streak",None); s.pop("peak_drop",None); s["alerted"]=False

        for pid in list(self.state):
            if pid not in active:
                self.state[pid]["missing"]=self.state[pid].get("missing",0)+1
                if self.state[pid]["missing"]>MAX_HISTORY_FRAMES:
                    self.state.pop(pid,None)
            else:
                self.state[pid]["missing"]=0
        return events
