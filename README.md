# AI Public Safety Video Monitor Using Computer Vision and Agentic AI

An AI-powered video monitoring system designed to detect potentially unsafe or abnormal situations from CCTV and recorded video using **Computer Vision and Agentic AI**.

The system combines real-time video analysis, object detection, object tracking, event detection, risk assessment, and automated alerts to support intelligent public safety monitoring.

## 🚀 Features

- Real-time video monitoring
- Object detection using YOLO
- Object tracking
- Crowd detection
- Abnormal movement detection
- Restricted-zone intrusion detection
- Loitering detection
- Possible fall detection
- Weapon detection
- AI-based risk assessment
- Automated alerts and incident reports
- Web-based monitoring dashboard

## 🔫 Weapon Detection

A custom **YOLOv8s** object detection model has been developed as part of the public safety system to detect people and different types of weapons.

The model is trained to detect **11 classes**:

| ID | Class |
|---:|---|
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

### Model Details

- **Architecture:** YOLOv8s
- **Framework:** Ultralytics YOLO
- **Task:** Object Detection
- **Number of Classes:** 11
- **Model:** `best.pt`

### Model Performance

The model was evaluated on the validation dataset.

| Metric | Score |
|---|---:|
| Precision | 0.688 |
| Recall | 0.714 |
| mAP@50 | 0.760 |
| mAP@50-95 | 0.488 |

### Per-Class mAP@50-95

| Class | mAP@50-95 |
|---|---:|
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

The model can perform differently depending on lighting, object size, camera angle, visibility, occlusion, and similarity to the training data.

## 🌐 Web Application

The weapon detection module is integrated into a **Flask-based web application**.

It supports:

- Image upload and detection
- Real-time webcam detection
- Bounding boxes
- Class names
- Confidence scores
- JSON-based API communication
- CORS support

The trained model is loaded from:

```text
best.pt


Video Input
    ↓
Object Detection
    ↓
Object Tracking
    ↓
Event Detection
    ↓
Event Correlation
    ↓
Agentic AI
    ↓
Risk Assessment
    ↓
Alerts & Reports
    ↓
Web Dashboard


🎯 Goal

The goal of this project is to transform traditional CCTV monitoring into an intelligent, automated, and risk-aware safety monitoring system that helps security personnel identify important incidents faster.

The system aims to combine computer vision with Agentic AI to detect events, assess potential risks, and assist in generating alerts and reports.