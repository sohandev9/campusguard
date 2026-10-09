"""
CampusGuard Vision - Backend Pipeline Runner

Runs the combined pipeline (tracking + fall + fight + bags) as a background worker
that pushes annotated frames into a frame_queue (MJPEG stream) and alert events
into an alert_queue (WebSocket).

Fall detection now uses the rule-based detector in modules/fall/fall_rule.py
(torso flat + wide box, held ~0.5 s) - same as main.py and run_video.py.
The old LSTM fall model is no longer used here.

Fight / bag logic and constants are imported from main.py (source of truth).

Interface used by backend/app.py (unchanged):
    CampusGuardPipeline(video_path, frame_queue=, alert_queue=, stop_event=)
    .start() .stop() .is_running() .processed_frames .alerts
"""

import os
import sys
import threading
import queue as queue_module
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "modules", "fall"))
from fall_rule import FallRule  # noqa: E402

# Fight / bag components and constants come from main.py (the source of truth).
from main import (  # noqa: E402
    predict_fight,
    BagWatcher,
    FIGHT_MODEL_PATH,
    TRACK_CONF_THRESHOLD,
    FIGHT_WINDOW_SECONDS,
    FIGHT_CHECK_EVERY_SECONDS,
    FIGHT_THRESHOLD,
    FIGHT_CONSECUTIVE,
    FIGHT_NUM_FRAMES,
    FIGHT_SIZE,
    BAG_CLASSES,
    ABANDON_SECONDS,
    BAG_CONF_THRESHOLD,
    device,
)
import torch  # noqa: E402


def _find_weights(name):
    for p in (os.path.join(BASE_DIR, "models", name), os.path.join(BASE_DIR, name)):
        if os.path.exists(p):
            return p
    return name                                   # let ultralytics download it


POSE_MODEL_PATH = _find_weights("yolov8n-pose.pt")
OBJECT_MODEL_PATH = _find_weights("yolov8n.pt")

# Cap streamed frame width (presentation only, applied after all detection/annotation).
_MAX_STREAM_WIDTH = 1920
# A bag counts as "already alerted" if a new unattended bag is within this many pixels.
_BAG_ALERT_RADIUS_PX = 80


class CampusGuardPipeline:
    def __init__(self, video_path, frame_queue=None, alert_queue=None, stop_event=None):
        self.video_path = video_path
        self.frame_queue = frame_queue or queue_module.Queue(maxsize=4)
        self.alert_queue = alert_queue or queue_module.Queue()
        self._stop_event = stop_event or threading.Event()
        self._thread = None
        self.processed_frames = 0
        self.alerts = []

    def _emit_alert(self, alert_type, detail, confidence=None):
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": alert_type,
            "detail": detail,
            "confidence": round(float(confidence), 4) if confidence is not None else None,
        }
        self.alerts.append(alert)
        try:
            self.alert_queue.put_nowait(alert)
        except queue_module.Full:
            pass

    def _run(self):
        try:
            print("[Pipeline] Loading models...", flush=True)
            pose_model = YOLO(POSE_MODEL_PATH)
            object_model = YOLO(OBJECT_MODEL_PATH)

            fight_model = r2plus1d_18(weights=None)
            fight_model.fc = nn.Linear(fight_model.fc.in_features, 2)
            fight_model.load_state_dict(torch.load(FIGHT_MODEL_PATH, map_location=device))
            fight_model = fight_model.to(device).eval()

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self._emit_alert("error", f"Could not open video: {self.video_path}")
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            print(f"[Pipeline] Video: {width}x{height} @ {fps:.1f}fps on device={device}", flush=True)

            # ---- fall state lives inside FallRule ----
            fall_rule = FallRule(fps=fps)

            # ---- fight state (same as main.py) ----
            fight_window_frames = int(round(FIGHT_WINDOW_SECONDS * fps))
            fight_min_frames = max(FIGHT_NUM_FRAMES, int(fight_window_frames * 0.5))
            fight_check_every = max(1, int(round(FIGHT_CHECK_EVERY_SECONDS * fps)))
            fight_buffer = deque(maxlen=fight_window_frames)
            fight_prob, fight_streak, fight_alert = 0.0, 0, False
            fight_emitted = False

            # ---- bags state (same as main.py) ----
            bag_watcher = BagWatcher(fps)
            bag_alert_centers = []

            frame_idx = 0
            print("[Pipeline] Processing combined pipeline...", flush=True)

            results_stream = pose_model.track(
                source=self.video_path,
                tracker="botsort.yaml",
                persist=True,
                conf=TRACK_CONF_THRESHOLD,
                classes=[0],
                stream=True,
                verbose=False,
            )

            for r in results_stream:
                if self._stop_event.is_set():
                    break

                frame = r.orig_img.copy()
                person_boxes = []

                # ---- Person tracking + fall detection ----
                if r.boxes is not None and r.boxes.id is not None:
                    track_ids = r.boxes.id.cpu().numpy().astype(int)
                    boxes = r.boxes.xyxy.cpu().numpy().astype(int)

                    for i, track_id in enumerate(track_ids):
                        x1, y1, x2, y2 = boxes[i]
                        person_boxes.append((x1, y1, x2, y2))

                        has_kp = r.keypoints is not None and i < len(r.keypoints.xy)
                        kp = r.keypoints.xy[i].cpu().numpy() if has_kp else np.zeros((17, 2), np.float32)
                        kc_all = getattr(r.keypoints, "conf", None) if r.keypoints is not None else None
                        if kc_all is not None and i < len(kc_all):
                            kc = kc_all[i].cpu().numpy()
                        else:
                            kc = np.ones(17, np.float32)

                        fall_on, fall_new, fall_score, _ = fall_rule.update(
                            track_id, boxes[i], kp, kc, frame_idx, (width, height)
                        )
                        if fall_new:
                            self._emit_alert(
                                "fall", f"Person ID:{track_id} fall detected", confidence=fall_score
                            )

                        box_color, label = (0, 255, 0), f"ID:{track_id}"
                        if fall_on:
                            box_color, label = (0, 0, 255), f"ID:{track_id} FALL DETECTED"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                fall_rule.prune(frame_idx)

                # ---- Fight detection (scene-level) ----
                fight_buffer.append(
                    cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (FIGHT_SIZE, FIGHT_SIZE))
                )
                if len(fight_buffer) >= fight_min_frames and frame_idx % fight_check_every == 0:
                    fight_prob = predict_fight(fight_model, list(fight_buffer))
                    fight_streak = fight_streak + 1 if fight_prob >= FIGHT_THRESHOLD else 0
                    fight_alert = fight_streak >= FIGHT_CONSECUTIVE

                if fight_alert:
                    if not fight_emitted:
                        self._emit_alert("fight", "Fight detected in scene", confidence=fight_prob)
                        fight_emitted = True
                    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 12)
                    cv2.putText(frame, "FIGHT DETECTED", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                else:
                    fight_emitted = False
                cv2.putText(frame, f"fight_prob: {fight_prob:.2f}", (20, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255) if fight_alert else (0, 255, 0), 2)

                # ---- Unattended bags ----
                obj_results = object_model(frame, conf=BAG_CONF_THRESHOLD, verbose=False)[0]
                bag_boxes = []
                if obj_results.boxes is not None:
                    for box, cls in zip(obj_results.boxes.xyxy.cpu().numpy(),
                                        obj_results.boxes.cls.cpu().numpy()):
                        if int(cls) in BAG_CLASSES:
                            bag_boxes.append(tuple(box))

                for (x1, y1, x2, y2), is_unattended, seconds_alone in bag_watcher.update(
                    bag_boxes, person_boxes
                ):
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    color = (0, 0, 255) if is_unattended else (0, 200, 255)
                    label = (f"UNATTENDED BAG ({seconds_alone:.0f}s)" if is_unattended
                             else f"Bag ({seconds_alone:.0f}s)")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if is_unattended and seconds_alone >= ABANDON_SECONDS:
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        if all(np.hypot(cx - ax, cy - ay) > _BAG_ALERT_RADIUS_PX
                               for ax, ay in bag_alert_centers):
                            self._emit_alert(
                                "bag", f"Unattended bag detected ({seconds_alone:.0f}s alone)",
                                confidence=1.0,
                            )
                            bag_alert_centers.append((cx, cy))

                # ---- Push annotated frame to the stream queue ----
                if width > _MAX_STREAM_WIDTH:
                    ratio = _MAX_STREAM_WIDTH / width
                    frame = cv2.resize(frame, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

                self.processed_frames += 1
                try:
                    self.frame_queue.put_nowait(frame)
                except queue_module.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue_module.Empty:
                        pass

                frame_idx += 1

            print(f"[Pipeline] Done. Processed {self.processed_frames} frames.", flush=True)

        except Exception as e:
            print(f"[Pipeline] Error: {e}", flush=True)
            self._emit_alert("error", f"Pipeline error: {str(e)}")

    def start(self):
        self._stop_event.clear()
        self.processed_frames = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()