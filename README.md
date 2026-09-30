# CampusGuard Vision

Privacy-preserving anomaly detection for campus safety.
Smart India Hackathon 2026 | Problem Statement SW-63 | Theme: Smart Automation

## Overview

CampusGuard Vision detects fights, falls, and unattended bags from video
without facial recognition or identity tracking. It tracks people using
short-lived tracking IDs only (never linked to identity) and analyzes body
pose keypoints and scene-level motion instead.

This is a full rebuild (v2) of the original SIH submission. v1's heuristic-
based fall/fight detection did not work reliably even on clean test footage,
so v2 was built from scratch using trained models instead of hand-tuned 
thresholds.

## Features

- **Person tracking**: BoT-SORT with appearance re-identification, keeping
  stable short-term IDs across occlusion and crowding — no identity storage.
- **Fall detection**: per-person pose-keypoint sequences fed through an LSTM,
  trained on the UR Fall Detection Dataset. Keypoints are normalized
  (position/scale-invariant) so it generalizes across different camera
  distances and angles rather than overfitting to one dataset's setup.
- **Fight detection**: scene-level clip classification using a
  Kinetics-pretrained R(2+1)D-18 video model, fine-tuned on RWF-2000.
- **Unattended bag detection**: rule-based — an object (backpack/handbag/
  suitcase) that stays in roughly the same spot with no person nearby for a
  set duration is flagged.

## Tech Stack

- Python, PyTorch, torchvision
- Ultralytics YOLOv8 (person detection, pose estimation, tracking)
- OpenCV

## Project Structure

```
campusguard/
├── main.py                          # Combined end-to-end pipeline (all 4 modules)
├── modules/
│   ├── tracker/
│   │   └── track_test.py            # Standalone BoT-SORT tracking test
│   ├── fall/
│   │   ├── extract_keypoints.py     # Pose keypoint extraction from UR Fall dataset
│   │   ├── build_windows.py         # Builds normalized, labeled training windows
│   │   ├── train_model.py           # Trains the fall LSTM
│   │   ├── pose_utils.py            # Shared keypoint normalization
│   │   └── visualize_fall_detection.py  # Real-footage test/visualization
│   ├── fight/
│   │   ├── preprocess_rwf.py        # Caches RWF-2000 clips as frame arrays
│   │   ├── train_fight.py           # Fine-tunes R(2+1)D-18 on RWF-2000
│   │   └── visualize_fight_detection.py  # Real-footage test/visualization
│   └── bags/
│       └── detect_bags.py           # Rule-based unattended bag detection
├── models/                          # Trained model weights (not tracked in git)
└── data/                            # Datasets, cached features (not tracked in git)
```

## How It Works

1. Read video frames with OpenCV.
2. Track people with YOLOv8 + BoT-SORT (short-lived IDs, no identity storage).
3. For each tracked person, extract pose keypoints, normalize them, and run
   a sliding-window LSTM to flag falls.
4. In parallel, run a scene-level sliding window through the fight model to
   flag fights.
5. Detect bag-class objects with YOLOv8 and track whether each one has a
   person nearby; flag as unattended past a time threshold.
6. Draw all alerts on the video (bounding boxes, labels, banners).

## Privacy by Design

- No facial recognition
- No persistent identity tracking or storage — tracking IDs are short-lived
  and reset, never linked to a real identity
- Only pose keypoints and bounding boxes are analyzed

## Installation

```bash
git clone https://github.com/sohandev9/campusguard.git
cd campusguard
pip install -r requirements.txt
```

**GPU note:** if you're on an RTX 50-series (Blackwell) GPU, you need
PyTorch 2.7.1+ with CUDA 12.8 wheels specifically — install with:
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

## Usage

```bash
python main.py
```
Update the video path at the top of `main.py` to point to your input clip.
Individual modules can also be run/tested standalone from their respective
folders under `modules/`.

## Status

Core pipeline (tracker + fall + fight + bags) is built and validated on real
recorded footage. Currently building a web dashboard front-end
(specification and phased build plan drafted, implementation in progress).

## Roadmap

- Web dashboard for live viewing and alerts
- Real-time performance optimization (currently runs two separate YOLO
  passes per frame — tracker/pose and object detection — which could be
  combined)
- Broader real-world testing, especially for fight detection edge cases

## Datasets and References

- **UR Fall Detection Dataset** — fall detection training data
- **RWF-2000** — fight/violence detection training data
- **YOLOv8 / YOLOv8-pose** (Ultralytics) — person detection, pose estimation
- **BoT-SORT** — multi-object tracking with appearance re-identification
- **R(2+1)D-18** (Kinetics-400 pretrained, torchvision) — fight action recognition backbone

## Team

Team EVO!