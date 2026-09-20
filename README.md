# CampusGuard Vision

Privacy-preserving anomaly detection for campus safety.
Smart India Hackathon 2026 | Problem Statement SW-63 | Theme: Smart Automation

## Overview
CampusGuard Vision detects fights, falls, and unattended bags from video **without facial recognition or identity tracking**. It analyzes body pose keypoints instead of identities.

## Features
- **Fight detection**: rapid limb motion and close-proximity interaction
- **Fall detection**: sudden change in body orientation and posture
- **Unattended bag detection**: object persists with no person nearby
- **Web interface** for running detection and viewing results

## Tech Stack
- Python, YOLOv8, OpenCV
- Flask (web app)
- HTML, CSS, JavaScript (frontend)

## Project Structure
```
campusguard/
├── app.py                    # Web app entry point
├── features.py               # Pose and motion feature extraction
├── batch_extractor.py        # Batch feature extraction from video data
├── train_models.py           # Model training
├── debug_fight_detection.py  # Debugging and testing fight detection
├── templates/                # HTML templates
├── static/                   # CSS and JS
└── requirements.txt
```

## How It Works
1. Read video frames with OpenCV
2. Detect people and objects with YOLOv8
3. Extract body keypoints (pose estimation)
4. Compute motion and posture features (`features.py`)
5. Classify or flag fights, falls, and abandoned objects
6. Display alerts in the web interface

## Privacy by Design
- No face recognition
- No identity tracking or storage
- Only skeleton keypoints are analyzed

## Installation
```bash
git clone https://github.com/sohandev9/campusguard.git
cd campusguard
pip install -r requirements.txt
```

## Usage
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

To train models:
```bash
python train_models.py
```

## Status
Work in progress. We did not qualify in SIH 2026, and we are continuing to improve accuracy and robustness.

## Roadmap
- [ ] Improve detection accuracy
- [ ] Reduce false positives
- [ ] Test on more real-world footage

## Datasets and References
- [Add datasets used]
- YOLOv8: Ultralytics

## Team
- [Name] ([GitHub link])
