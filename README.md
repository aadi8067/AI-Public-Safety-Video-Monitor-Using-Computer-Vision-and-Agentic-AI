# AI Public Safety Video Monitor Using Computer Vision and Agentic AI

An AI-powered public safety video monitoring system designed to analyze **CCTV and recorded video** using **Computer Vision, Object Detection, Object Tracking, Event Analysis, and Agentic AI**.

The system is being developed to assist security personnel by automatically identifying people, detecting potentially dangerous objects, tracking individuals, and generating intelligent safety alerts from video streams.

---

## 🚀 Project Overview

Traditional CCTV monitoring requires continuous human observation, which can make it difficult to identify important incidents quickly.

This project aims to transform conventional CCTV monitoring into an **intelligent, automated, and risk-aware safety monitoring system**.

The current implementation focuses on building a reliable Computer Vision foundation consisting of:

- Real-time video monitoring
- Person detection
- Object detection
- Person tracking
- Stable person identification
- Gun and knife detection
- Weapon-to-person association
- Temporal weapon confirmation
- Bounding boxes and confidence scores
- Automated safety alerts
- Web-based monitoring

Higher-level event correlation, risk assessment, and Agentic AI capabilities are part of the planned system architecture.

---

## 🏗️ System Architecture

```text
CCTV / Recorded Video
        ↓
Video Ingestion
        ↓
General Object Detection
        ↓
Person Tracking
        ↓
Stable Person Identification
        ↓
Weapon Detection
        ↓
Weapon-Person Association
        ↓
Temporal Event Confirmation
        ↓
Safety Alert
        ↓
Event Correlation
        ↓
Agentic AI
        ↓
Risk Assessment
        ↓
Alerts & Incident Reports
        ↓
Web Dashboard
```

---

## 🎯 Project Objectives

The main objectives of the project are:

1. Automatically analyze CCTV and recorded video.
2. Detect and track people in public environments.
3. Detect potentially dangerous weapons such as guns and knives.
4. Maintain stable identities for tracked individuals.
5. Associate detected weapons with nearby tracked persons.
6. Reduce false alerts using temporal confirmation.
7. Detect abnormal or potentially unsafe events.
8. Assess the risk level of detected incidents.
9. Use Agentic AI to analyze events and assist decision-making.
10. Generate automated alerts and structured incident reports.
11. Provide security personnel with an easy-to-use monitoring dashboard.

---

## 🧠 Current Computer Vision Pipeline

The current working Computer Vision pipeline is:

```text
Video Input
    ↓
Frame Extraction
    ↓
General YOLO Detection
    ↓
Person Tracking
    ↓
Stable Person ID
    ↓
Dedicated Weapon Detection
    ↓
Weapon Detection Filtering
    ↓
Weapon-Person Association
    ↓
Temporal Confirmation
    ↓
Safety Alert
```

---

# 🔍 Implemented Features

## 1. Real-Time Video Monitoring

The system supports processing CCTV-style video and displaying the processed output through a web application.

The video pipeline processes frames and overlays detection information directly on the video.

Current monitoring output includes:

- Person bounding boxes
- Person IDs
- Weapon bounding boxes
- Weapon class
- Confidence scores
- Safety alerts

---

## 2. General Object Detection

A general-purpose YOLO model is used for scene understanding and person detection.

The general detection model provides the foundation for:

- Person detection
- Common-object detection
- Person tracking
- Future crowd analysis
- Future movement analysis
- Future zone monitoring

The general model is kept separate from the dedicated weapon model to improve the overall system design.

---

## 3. Person Detection

People are detected using the general YOLO detection pipeline.

Each detected person is represented using a bounding box and tracking identifier.

Example:

```text
PERSON-028
PERSON-005
PERSON-016
```

This allows the system to associate later events with specific tracked individuals.

---

## 4. Person Tracking

Object tracking has been integrated to maintain identities of people across video frames.

The tracking layer allows the system to follow people as they move through the scene.

Example:

```text
Frame 1 → PERSON-028
Frame 2 → PERSON-028
Frame 3 → PERSON-028
Frame 4 → PERSON-028
```

This provides continuity between detections instead of treating every frame as a completely new observation.

---

## 5. Stable Person Identification

An application-level stable identity layer has been implemented on top of object tracking.

This addresses situations where the underlying tracker may temporarily lose a person or assign a different tracking ID.

Instead of relying only on raw tracker IDs, the application maintains stable identifiers such as:

```text
PERSON-028
```

This is especially important for safety events because a detected weapon can be associated with the same person across multiple frames.

Example:

```text
Person detected
      ↓
PERSON-028
      ↓
Weapon detected
      ↓
ALERT gun 0.66 → PERSON-028
```

---

# 🔫 Weapon Detection

## Initial Custom Weapon Model

The project initially used a custom YOLOv8s object detection model trained to detect multiple weapon categories.

The original model contains **11 classes**:

| ID | Class |
|----|-------|
| 0 | Person |
| 1 | Gun |
| 2 | Knife |
| 3 | Sword |
| 4 | Automatic Rifle |
| 5 | Bazooka |
| 6 | Grenade Launcher |
| 7 | Handgun |
| 8 | SMG |
| 9 | Shotgun |
| 10 | Sniper |

### Initial Model Details

- **Architecture:** YOLOv8s
- **Framework:** Ultralytics YOLO
- **Task:** Object Detection
- **Number of Classes:** 11
- **Model:** `best.pt`

### Initial Model Performance

The original model was evaluated on its validation dataset.

| Metric | Score |
|--------|-------|
| Precision | 0.688 |
| Recall | 0.714 |
| mAP@50 | 0.760 |
| mAP@50-95 | 0.488 |

### Per-Class mAP@50-95

| Class | mAP@50-95 |
|-------|-----------|
| Person | 0.747 |
| Gun | 0.335 |
| Knife | 0.193 |
| Sword | 0.434 |
| Automatic Rifle | 0.540 |
| Bazooka | 0.405 |
| Grenade Launcher | 0.625 |
| Handgun | 0.652 |
| SMG | 0.511 |
| Shotgun | 0.590 |
| Sniper | 0.339 |

The initial evaluation showed that performance varied across classes, particularly for small or visually ambiguous objects.

---

# 🎯 Improved Weapon Detection

During testing, false-positive weapon detections were observed with visually similar objects.

To improve practical gun and knife detection, the current implementation uses a **dedicated YOLO11 gun/knife model** together with the general object detection model.

The architecture is:

```text
              CCTV / Video
                   ↓
          ┌────────┴────────┐
          ↓                 ↓
   General YOLO       Weapon YOLO11
          ↓                 ↓
     Person/Object       Gun/Knife
      Detection          Detection
          ↓                 ↓
       Tracking       Weapon Filtering
          ↓                 ↓
          └────────┬────────┘
                   ↓
          Weapon-Person Association
                   ↓
          Temporal Confirmation
                   ↓
              Safety Alert
```

This two-model architecture separates general scene understanding from specialized weapon detection.

---

## 🔧 Weapon Detection Improvements

The current weapon detection pipeline includes:

- Full-frame inference
- Overlapping tiled inference
- Increased inference resolution
- Dedicated gun/knife detection
- Weapon confidence thresholding
- Class filtering
- Duplicate detection merging
- Class-aware Non-Maximum Suppression
- Temporal confirmation
- Weapon-person association

These improvements are intended to make detection more reliable, especially when weapons occupy a small portion of the CCTV frame.

---

## 🧩 Tiled Weapon Detection

Weapons in CCTV footage can be very small compared with the complete video frame.

To improve detection of small objects, the system performs detection on:

```text
Full Frame
     +
Overlapping Region 1
Overlapping Region 2
Overlapping Region 3
Overlapping Region 4
```

The detections from these regions are mapped back to the original frame and duplicate detections are merged.

This allows the dedicated weapon model to inspect smaller areas of the scene at higher effective detail.

---

## ⏱️ Temporal Weapon Confirmation

A single-frame detection can sometimes be uncertain.

The current system therefore uses temporal confirmation before treating a weapon observation as a confirmed event.

Current confirmation logic:

```text
Weapon Observation
       ↓
Weapon Observation
       ↓
Weapon Observation
       ↓
Confirmed Weapon Event
```

The implementation allows multiple observations within a short frame window rather than requiring every individual frame to contain a perfect detection.

This helps reduce isolated one-frame detections.

---

# 🚨 Weapon-Person Association

The system does not only detect the weapon.

It attempts to associate the detected weapon with the nearest relevant tracked person.

Example:

```text
Gun detected
     ↓
Find nearby tracked person
     ↓
PERSON-028
     ↓
Generate alert
```

Example output:

```text
ALERT gun 0.66 → PERSON-028
```

This provides more useful information to security personnel than simply reporting:

```text
Gun detected
```

---

# 🔴 Safety Alerts

The system generates visual safety alerts when a confirmed weapon detection is produced.

Example:

```text
ALERT gun 0.66 → PERSON-028
```

The alert can contain:

- Weapon type
- Confidence score
- Associated person ID
- Detection location
- Bounding box

The event is also displayed directly on the processed CCTV video.

---

# 🧪 Testing & Results

The Computer Vision system has been tested using CCTV-style video scenarios containing multiple people and public-space environments.

Testing has included:

- Railway-station-style CCTV scenes
- Public promenade environments
- Multiple-person scenes
- Normal pedestrian activity
- Weapon-present scenarios
- Screen-recorded application testing

---

## ✅ Demonstrated Result

The current implementation successfully produces weapon alerts in the monitoring interface.

Example:

```text
ALERT gun 0.66 → PERSON-028
```

The corresponding weapon bounding box is displayed in the CCTV video.

At the same time, the system maintains person detections and tracking identifiers.

Example monitoring output:

```text
PERSON-028
PERSON-005
PERSON-016

        +

ALERT gun 0.66 → PERSON-028
```

This demonstrates the integration of:

```text
Person Detection
       +
Person Tracking
       +
Stable Identity
       +
Weapon Detection
       +
Weapon-Person Association
       +
Safety Alert
```

---

# 🌐 Web Application

The Computer Vision pipeline is integrated into a web-based monitoring application.

The current application supports:

- Video processing
- Image detection
- Webcam/live detection
- Person detection
- Weapon detection
- Bounding boxes
- Class labels
- Confidence scores
- Person tracking IDs
- Weapon-person association
- Safety alerts
- Browser-based monitoring

The application can be accessed locally through:

```text
http://127.0.0.1:8000
```

---

# 💻 Technology Stack

## Computer Vision

- Python
- OpenCV
- Ultralytics YOLO
- YOLO-based Object Detection
- Object Tracking

## AI Models

- General YOLO model
- Dedicated YOLO11 Gun/Knife Detection Model
- Original custom YOLOv8s weapon model

## Web Application

- Flask
- HTML
- CSS
- JavaScript
- JSON-based communication
- CORS

---

# 📂 Project Structure

```text
AI-Public-Safety-Video-Monitor-Using-Computer-Vision-and-Agentic-AI/
│
├── app.py
├── best.pt
├── gun_knife_yolo11n.pt
├── yolov8s.pt
├── data.yaml
│
├── templates/
│   └── ...
│
├── static/
│   └── ...
│
└── README.md
```

> Model filenames may vary depending on the local configuration used for testing.

---

# 📊 Current Project Status

| Module | Status |
|--------|--------|
| Project Architecture | ✅ Completed |
| Video Input | ✅ Completed |
| Video Frame Processing | ✅ Completed |
| General Object Detection | ✅ Implemented |
| Person Detection | ✅ Implemented |
| Person Tracking | ✅ Implemented |
| Stable Person IDs | ✅ Implemented |
| Gun Detection | ✅ Working |
| Knife Detection | ✅ Implemented |
| Weapon Filtering | ✅ Implemented |
| Weapon-Person Association | ✅ Working |
| Temporal Weapon Confirmation | ✅ Implemented |
| Safety Alerts | ✅ Working |
| Web Application | ✅ Working |
| CCTV Video Testing | ✅ Completed |
| Crowd Analysis | 🔄 Planned |
| Abnormal Movement Detection | 🔄 Planned |
| Restricted-Zone Detection | 🔄 Planned |
| Loitering Detection | 🔄 Planned |
| Fall Detection | 🔄 Planned |
| Event Correlation | 🔄 Planned |
| Risk Assessment | 🔄 Planned |
| Agentic AI Supervisor | 🔄 Planned |
| Incident Reports | 🔄 Planned |
| Database Integration | 🔄 Planned |
| Final Monitoring Dashboard | 🔄 Planned |

---

# 🔮 Future Development

The next stage of the project will extend the existing Computer Vision foundation with higher-level event intelligence.

## 1. Crowd Detection

The system will analyze the number and density of people within monitored areas.

Potential events include:

```text
Normal Crowd
     ↓
Increasing Density
     ↓
High Crowd Density
     ↓
Crowd Risk Alert
```

---

## 2. Abnormal Movement Detection

The system will analyze movement patterns to identify potentially unusual behavior.

Possible indicators include:

- Sudden movement
- Rapid movement
- Running
- Unusual group movement
- Sudden crowd movement

---

## 3. Restricted-Zone Intrusion

Specific areas of the CCTV scene can be defined as restricted zones.

Example:

```text
Person
   ↓
Enters Restricted Zone
   ↓
Intrusion Event
   ↓
Alert
```

---

## 4. Loitering Detection

The system will monitor how long a person remains within a defined area.

Example:

```text
Person enters zone
       ↓
Timer starts
       ↓
Person remains for extended period
       ↓
Possible Loitering Event
```

---

## 5. Possible Fall Detection

The system is planned to identify possible falling events using person position, posture, and movement changes.

Example:

```text
Standing Person
       ↓
Rapid Position Change
       ↓
Horizontal/Low Posture
       ↓
Possible Fall
       ↓
Alert
```

---

# 🔗 Event Correlation

Individual detections will eventually be combined into higher-level events.

For example:

```text
Person Detected
       +
Weapon Detected
       +
Person Enters Restricted Zone
       +
Abnormal Movement
       ↓
High-Risk Incident
```

This is important because a public safety system should not treat every individual detection as an independent emergency.

---

# 🤖 Agentic AI

The planned Agentic AI layer will operate above the Computer Vision detection system.

The Agentic AI Supervisor will receive structured events from the Computer Vision pipeline and reason about the overall situation.

Proposed architecture:

```text
Computer Vision
       ↓
Event Detector
       ↓
Event Correlator
       ↓
Agentic AI Supervisor
       ↓
┌──────────────┬──────────────┬──────────────┐
│              │              │
Risk Agent   Response Agent  Report Agent
│              │              │
↓              ↓              ↓
Risk Score   Recommended     Incident
             Action          Report
```

---

# ⚠️ Risk Assessment

The future risk-assessment module will classify incidents into different severity levels.

Example:

```text
LOW
MEDIUM
HIGH
CRITICAL
```

The risk score can consider multiple factors such as:

- Type of detected event
- Weapon presence
- Number of people involved
- Location/zone
- Movement behavior
- Duration
- Multiple simultaneous events

---

# 📝 Incident Reports

The planned incident-reporting module will generate structured information for detected incidents.

A future incident report may contain:

```text
Incident ID
Timestamp
Location / Zone
Person ID
Detected Event
Detected Object
Confidence
Risk Level
Evidence Frame / Video
Recommended Response
```

Example:

```text
Incident: INC-001

Time: 15:42:18
Person: PERSON-028
Event: Weapon Detection
Object: Gun
Confidence: 0.66
Risk Level: HIGH

Recommended Action:
Security personnel review required.
```

---

# 🖥️ Future Monitoring Dashboard

The final dashboard is planned to provide security personnel with:

- Live CCTV feed
- Active alerts
- Person tracking
- Event timeline
- Risk level
- Incident history
- Evidence frames
- Event details
- AI-generated incident summaries

A possible dashboard structure:

```text
┌───────────────────────────────────────────────┐
│          AI PUBLIC SAFETY MONITOR             │
├───────────────────────┬───────────────────────┤
│                       │ Active Alerts         │
│                       │                       │
│     CCTV FEED         │ 🔴 HIGH - Gun        │
│                       │ 🟠 MEDIUM - Crowd    │
│                       │                       │
├───────────────────────┴───────────────────────┤
│ Event Timeline                                 │
├───────────────────────────────────────────────┤
│ Risk Level | Person | Event | Time | Status   │
└───────────────────────────────────────────────┘
```

---

# 📈 Evaluation Plan

The final system will be evaluated using multiple Computer Vision and system-level metrics.

### Object Detection

- Precision
- Recall
- mAP@50
- mAP@50-95

### Tracking

- Identity consistency
- ID switches
- Tracking continuity

### Event Detection

- True positives
- False positives
- False negatives
- Event detection accuracy

### System Performance

- Processing FPS
- Detection latency
- Alert latency
- Resource utilization

### AI Layer

- Risk classification accuracy
- Event-correlation accuracy
- Incident-report quality
- Response recommendation quality

---

# 🔐 Privacy & Responsible Use

This project is intended as an **academic and research prototype** for intelligent public safety monitoring.

The system should be deployed responsibly and should follow applicable privacy, surveillance, and data-protection requirements.

Computer Vision predictions can be affected by:

- Lighting conditions
- Camera angle
- Object size
- Occlusion
- Image quality
- Distance from camera
- Background similarity
- Motion blur
- Training data distribution

Therefore, automated detections should be treated as **decision-support signals rather than definitive conclusions**.

Human verification should remain part of the response process for safety-critical incidents.

---

# 🎯 Project Goal

The ultimate goal of this project is to transform traditional CCTV monitoring into an **intelligent, automated, and risk-aware public safety monitoring system**.

The system aims to progress from:

```text
DETECT
   ↓
TRACK
   ↓
UNDERSTAND
   ↓
CORRELATE
   ↓
ASSESS RISK
   ↓
ALERT
   ↓
ASSIST RESPONSE
```

Instead of requiring security personnel to continuously monitor every CCTV feed, the proposed system will automatically identify potentially important situations and provide structured information for faster decision-making.

---

# 📌 Current Milestone

## Computer Vision Foundation + Weapon Monitoring — Completed

The current implementation demonstrates:

```text
CCTV / Recorded Video
        ↓
Person Detection
        ↓
Person Tracking
        ↓
Stable Person Identification
        ↓
Gun / Knife Detection
        ↓
Weapon-Person Association
        ↓
Temporal Confirmation
        ↓
Safety Alert
        ↓
Web Monitoring
```

### Example Demonstrated Alert

```text
ALERT gun 0.66 → PERSON-028
```

The next major milestone is:

```text
Event Detection
        ↓
Event Correlation
        ↓
Risk Assessment
        ↓
Agentic AI
        ↓
Incident Reporting
        ↓
Intelligent Monitoring Dashboard
```

---

## 👥 Contributors

- [@aadi8067](https://github.com/aadi8067)
- [@AryaShah07](https://github.com/AryaShah07)
- [@siony-chaudhari](https://github.com/siony-chaudhari)
- [@ShreyaA0105](https://github.com/ShreyaA0105)
- [@vaibhavmhetre17](https://github.com/vaibhavmhetre17)

---

## ⭐ Project Status

**Status: Active Development**

**Current Phase: Computer Vision Foundation & Weapon Monitoring**

**Next Phase: Event Intelligence & Agentic AI**
