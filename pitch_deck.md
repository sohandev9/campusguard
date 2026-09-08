# CampusGuard AI -- Pitch Deck

## Slide 1: Title
**CampusGuard AI**
Real-Time Vision-Based Campus Security
SIH Hackathon 2026

Tech: OpenCV, MediaPipe Pose, YOLOv8, Flask, RTX 5060 GPU

---

## Slide 2: Problem
University campuses generate thousands of hours of security footage daily:
- Security teams cannot watch every screen - human fatigue causes missed incidents
- Falls go undetected for minutes - 30% of campus medical calls are falls
- Fights escalate from seconds of tension to violence
- Unattended bags are the #1 security concern but impossible to track manually
- Cloud AI costs +/month and raises privacy concerns
- Centralized systems require expensive server rooms

Current campus security is REACTIVE. We make it PROACTIVE and LOCAL.

---

## Slide 3: Solution
**CampusGuard AI** - an all-local, real-time vision pipeline on a single GPU.

Three tasks, one pipeline:
- Fight Detection: arm velocity + acceleration + bilateral coordination
- Fall Detection: hip descent + torso tilt + impact deceleration
- Unattended Bag Detection: YOLOv8 tracking with abandonment timer

**Runs entirely on-device** - no cloud, no privacy risk, no recurring fees.

Architecture:
Camera -> YOLOv8n -> MediaPipe PoseLandmarker -> PersonTracker -> Velocity/Accel/Tilt Features -> 0.6s Window Stats -> XGBoost -> Alerts + Dashboard
---

## Slide 4: Technology Stack
| Layer | Tech | Why |
|-------|------|-----|
| Pose Estimation | MediaPipe PoseLandmarker (heavy.task) | 33 keypoints, robust to occlusion |
| Object Detection | YOLOv8n (yolov8n.pt) | Fast bag/person detection at 10+ FPS |
| Motion Features | Custom Python (features.py) | Time-normalized velocity, signed max/min stats |
| Classification | XGBoost (planned) / RandomForest (current) | Handles small datasets, interpretable |
| Backend | Flask (threaded=True) | Concurrent video stream + dashboard polling |
| Frontend | Vanilla JS + CSS | 500ms live stats polling, no frameworks |
| Hardware | RTX 5060 | Runs everything locally, ~2.5GB VRAM |

Key: features.py is the SINGLE SOURCE OF TRUTH. batch_extractor.py, train_models.py, and app.py all import from it.

---

## Slide 5: Key Innovations

### 1. Train/Serve Skew Elimination
Both training and inference use features.compute_velocity() with real elapsed seconds.
Training: dt = frame_idx / true_video_fps
Live: dt = time.time() - prev_timestamp
Same math, zero skew.

### 2. Signed Velocity Statistics
Before (bug): max(abs(v)) - fall down looks same as stand up
After (fix): signed max + min - preserves direction

### 3. Per-Landmark Velocity Rejection
If one landmark glitches, zero only that landmark - keep the rest. 2.6% more training data.

### 4. Nested Cross-Validation
5-fold GroupShuffleSplit by clip, inner val for threshold, outer test for honest scoring.

### 5. Rising-Edge Alert Logging
Log only on False->True transition - no alert spam during sustained detections.
---

## Slide 6: Diagnostic Results
debug_fight_detection.py - read-only diagnostics on 5 clips:

| Setting | Fight avg people/frame | Zero-detect frames | Window invocations |
|---------|----------------------|-------------------|-------------------|
| num_poses=4, conf=0.5 (old) | 0.61 | 48% (72/150) | 21 |
| num_poses=8, conf=0.3 (new) | 0.91 | 19% (28/150) | 111 |

**Key finding**: Detection count is the dominant failure mode - NOT tracking.
- When people ARE detected: 100% window fill rate (tracking is solid)
- 48% of fight frames had ZERO detections -> classifier never ran
- Relaxed settings: 5x more classifier invocations

**CSV verification**: All 22 columns present, 169,392 rows, 1 bad line skipped.
**Model loading**: Both .pkl files load with thresholds (0.386 fight, 0.432 fall).

---

## Slide 7: Performance Metrics (5-fold CV)
| Model | Macro F1 | Recall | Precision | Clips |
|-------|----------|--------|-----------|-------|
| Fight | 0.739 +/- 0.016 | 0.83 | 0.70 | 1,153 |
| Fall | 0.685 +/- 0.041 | 0.85 | 0.60 | 169 |

Inference: 88-119ms/frame, ~5 FPS effective (FRAME_SKIP=2), 15-20 FPS display.

---

## Slide 8: Dashboard
Ops-Room Interface:
- Dark charcoal-navy theme (#0B0F14 to #121821)
- Cyan/amber/red status colors (security HUD aesthetic)
- Viewfinder corner brackets around video feed
- Stat cards with pulsing LED indicators
- Rolling alert log (newest first, thread-safe)
- Live stats polling (500ms), alert polling (1s)
- Health check endpoint, graceful error handling

---

## Slide 9: Demo Walkthrough
1. Start: python app.py (loads in ~10s)
2. Open: http://localhost:5000
3. Upload or switch to webcam
4. Watch real-time detection:
   - Person circles with stable tracking IDs
   - RED = fight alert, ORANGE = fall, RED box = unattended bag
   - Live stats update, alert log fills
5. Toggle between video sources

---

## Slide 10: Lessons Learned
1. Detection count was the DOMINANT failure mode, not tracking continuity
2. Signed max/min velocity matters - direction is signal
3. Per-landmark rejection saves 2.6% of training data
4. Flask threaded=True was critical - video stream blocked all other requests
5. Single train/test split was lucky - 5-fold CV revealed high fall-model variance

---

## Slide 11: Competitive Advantages
| Feature | Us | AWS Rekognition | Cisco Meraki |
|---------|----|-----------------|--------------|
| Price | Free (local) | +/mo | +/yr |
| Privacy | 100% on-device | Cloud upload | Cloud |
| Fight Detection | Specialized | Generic | Basic |
| Fall Detection | Specialized | Basic | Basic |
| Deploy | Plug & play | API integration | Vendor lockin |

---

## Slide 12: Roadmap
Phase 1 (Next 48h): Fine-tune thresholds, optimize FPS, polish dashboard
Phase 2 (Post-Hack): XGBoost, multi-scale windows, 2-person features
Phase 3 (Market): ONNX export, multi-camera, alert escalation, SaaS

---

## Slide 13: Thank You
CampusGuard AI - Proactive. Local. Affordable.
Questions? http://localhost:5000
