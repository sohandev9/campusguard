"""
app.py
------
CampusGuard AI live server.

Shares its velocity + windowing math with train_models.py via
features.compute_velocity(), features.update_window(), and
features.compute_window_stats() - dt is a real time.time() delta here,
exactly the same functions used in training (where dt came from the
video's true FPS and the window used CSV timestamps), so a model trained
on real-seconds windowed statistics is fed real-seconds windowed
statistics at inference too. A rolling WINDOW_SECONDS of motion history
is aggregated (mean/std/max) into the actual model input - not a single
frame's instantaneous velocity - so the models can tell a punch from a
wave, or a fall from a fast sit-down.

Bag detection (YOLO classes 24/26/28) is completely independent of the
fight/fall ML models - it is never turned into a model feature, so there is
no data leakage and bag logic can't distort the pose classifiers.
"""

import os
import time
import uuid
from collections import deque, defaultdict

import cv2
import joblib
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

import mediapipe as mp_mediapipe
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

import features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UPLOAD_DIR = "uploads"
POSE_MODEL_PATH = "pose_landmarker_heavy.task"
YOLO_MODEL_PATH = "yolov8n.pt"

BAG_CLASSES = {24, 26, 28}   # COCO: backpack, handbag, suitcase
PERSON_CLASS = 0
UNATTENDED_SECONDS = 15.0
BAG_NEAR_PIXELS_FRACTION = 0.12   # "near a person" = within 12% of frame width

FRAME_SKIP = 2  # process every 2nd frame; redraw cached overlay on the rest

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load models once at boot (NOT per-request - keeps GPU/VRAM usage sane)
# ---------------------------------------------------------------------------
print("Loading fight/fall models ...")
fight_bundle = joblib.load("fight_detector_model.pkl")
fall_bundle = joblib.load("fall_detector_model.pkl")
FIGHT_MODEL, FIGHT_COLUMNS = fight_bundle["model"], fight_bundle["feature_columns"]
FALL_MODEL, FALL_COLUMNS = fall_bundle["model"], fall_bundle["feature_columns"]

print("Loading YOLOv8 ...")
yolo_model = YOLO(YOLO_MODEL_PATH)

print("Loading MediaPipe PoseLandmarker ...")
_pose_options = mp_vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_poses=4,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose_landmarker = mp_vision.PoseLandmarker.create_from_options(_pose_options)


# ---------------------------------------------------------------------------
# Lightweight pixel-space tracker for bags (separate from the pose tracker,
# which lives in normalized 0-1 coordinate space).
# ---------------------------------------------------------------------------
class BagTracker:
    def __init__(self, max_distance_px):
        self.max_distance = max_distance_px
        self.tracks = {}   # id -> {"centroid": (x,y), "first_seen": t, "last_near_person": t}
        self._next_id = 0

    def update(self, bag_centroids, person_centroids, now):
        near_threshold = self.max_distance
        matched_ids = set()

        for cx, cy in bag_centroids:
            best_id, best_dist = None, None
            for tid, t in self.tracks.items():
                if tid in matched_ids:
                    continue
                d = np.hypot(cx - t["centroid"][0], cy - t["centroid"][1])
                if d <= near_threshold and (best_dist is None or d < best_dist):
                    best_id, best_dist = tid, d

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self.tracks[best_id] = {"centroid": (cx, cy), "first_seen": now, "last_near_person": now}

            self.tracks[best_id]["centroid"] = (cx, cy)
            matched_ids.add(best_id)

            is_near_person = any(
                np.hypot(cx - px, cy - py) <= near_threshold for px, py in person_centroids
            )
            if is_near_person:
                self.tracks[best_id]["last_near_person"] = now

        # drop tracks not seen this frame
        for tid in list(self.tracks.keys()):
            if tid not in matched_ids:
                del self.tracks[tid]

        unattended = []
        for tid in matched_ids:
            t = self.tracks[tid]
            if now - t["last_near_person"] > UNATTENDED_SECONDS:
                unattended.append(tid)
        return unattended


# ---------------------------------------------------------------------------
# Core per-stream processing state (fresh instance per /video_feed call)
# ---------------------------------------------------------------------------
class StreamState:
    def __init__(self):
        self.person_tracker = features.PersonTracker()
        self.bag_tracker = BagTracker(max_distance_px=None)  # set once we know frame width
        self.prev_coords = {}       # person_id -> (coords_dict, timestamp)
        self.velocity_windows = defaultdict(list)  # person_id -> [(timestamp, velocity_dict), ...]
        self.fight_buffers = defaultdict(lambda: deque(maxlen=features.ALERT_WINDOW))
        self.fall_buffers = defaultdict(lambda: deque(maxlen=features.ALERT_WINDOW))
        self.last_overlay_boxes = []   # cached draw instructions for skipped frames
        self.frame_idx = 0


def run_yolo_bags(frame, state, now):
    h, w = frame.shape[:2]
    if state.bag_tracker.max_distance is None:
        state.bag_tracker.max_distance = w * BAG_NEAR_PIXELS_FRACTION

    results = yolo_model.predict(
        frame, classes=list(BAG_CLASSES | {PERSON_CLASS}), verbose=False
    )[0]

    bag_boxes, person_centroids, bag_centroids = [], [], []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if cls_id == PERSON_CLASS:
            person_centroids.append((cx, cy))
        elif cls_id in BAG_CLASSES:
            bag_boxes.append((x1, y1, x2, y2))
            bag_centroids.append((cx, cy))

    unattended_ids = state.bag_tracker.update(bag_centroids, person_centroids, now)
    # unattended_ids are track ids; re-derive which boxes are unattended by
    # matching the just-updated centroids back to tracker state
    unattended_flags = []
    for (cx, cy) in bag_centroids:
        flagged = False
        for tid in unattended_ids:
            t = state.bag_tracker.tracks.get(tid)
            if t and t["centroid"] == (cx, cy):
                flagged = True
                break
        unattended_flags.append(flagged)

    return list(zip(bag_boxes, unattended_flags)), len(bag_boxes)


def run_pose_models(frame, state, now):
    """Returns list of (person_id, hip_pixel_xy, fight_alert, fall_alert)."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp_mediapipe.Image(image_format=mp_mediapipe.ImageFormat.SRGB, data=rgb)
    result = pose_landmarker.detect(mp_image)

    people_overlay = []
    if not result.pose_landmarks:
        return people_overlay

    per_person_coords = [features.extract_landmark_coords(pl) for pl in result.pose_landmarks]
    centroids = [features.hip_centroid(c) for c in per_person_coords]
    person_ids = state.person_tracker.update(centroids)

    for coords, pid, centroid in zip(per_person_coords, person_ids, centroids):
        prev = state.prev_coords.get(pid)
        prev_coords, prev_time = prev if prev else (None, None)
        dt = None if prev_time is None else (now - prev_time)

        velocity = features.compute_velocity(prev_coords, coords, dt)
        state.prev_coords[pid] = (coords, now)

        fight_alert = False
        fall_alert = False

        if velocity is None:
            # discontinuity (gap / teleportation jump) - reset this
            # person's window, exactly like train_models.py does, so we
            # never average across a broken sequence.
            state.velocity_windows[pid] = []
        else:
            velocity[features.TORSO_TILT_COLUMN] = features.compute_torso_tilt_degrees(coords)
            window = state.velocity_windows[pid]
            features.update_window(window, now, velocity)
            stats = features.compute_window_stats(window)

            if stats is not None:
                fight_vec = np.array([features.build_feature_vector(stats, FIGHT_COLUMNS)])
                fall_vec = np.array([features.build_feature_vector(stats, FALL_COLUMNS)])

                fight_pred = int(FIGHT_MODEL.predict(fight_vec)[0])
                fall_pred = int(FALL_MODEL.predict(fall_vec)[0])

                state.fight_buffers[pid].append(fight_pred)
                state.fall_buffers[pid].append(fall_pred)

                if sum(state.fight_buffers[pid]) >= features.ALERT_THRESHOLD:
                    fight_alert = True
                if sum(state.fall_buffers[pid]) >= features.ALERT_THRESHOLD:
                    fall_alert = True

        px, py = int(centroid[0] * w), int(centroid[1] * h)
        people_overlay.append((pid, (px, py), fight_alert, fall_alert))

    return people_overlay


def draw_overlay(frame, people_overlay, bag_overlay, bag_count):
    for pid, (px, py), fight_alert, fall_alert in people_overlay:
        color = (0, 255, 0)
        label = f"person {pid}"
        if fight_alert:
            color = (0, 0, 255)
            label += " FIGHT ALERT"
        if fall_alert:
            color = (0, 140, 255)
            label += " FALL ALERT"
        cv2.circle(frame, (px, py), 6, color, -1)
        cv2.putText(frame, label, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for (x1, y1, x2, y2), unattended in bag_overlay:
        color = (0, 0, 255) if unattended else (255, 200, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if unattended:
            cv2.putText(frame, "UNATTENDED BAG", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"Bags detected: {bag_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def generate_frames(source):
    if source == "webcam":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(os.path.join(UPLOAD_DIR, source))

    if not cap.isOpened():
        return

    state = StreamState()
    cached_people_overlay, cached_bag_overlay, cached_bag_count = [], [], 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()

            if state.frame_idx % FRAME_SKIP == 0:
                cached_people_overlay = run_pose_models(frame, state, now)
                cached_bag_overlay, cached_bag_count = run_yolo_bags(frame, state, now)

            frame = draw_overlay(frame, cached_people_overlay, cached_bag_overlay, cached_bag_count)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

            state.frame_idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", source=request.args.get("source"))


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    file.save(os.path.join(UPLOAD_DIR, unique_name))
    return redirect(url_for("index", source=unique_name))


@app.route("/video_feed")
def video_feed():
    source = request.args.get("source", "webcam")
    return Response(generate_frames(source), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    # use_reloader=False: prevents Flask's debug reloader from loading the
    # YOLO/MediaPipe/RandomForest models onto the GPU twice.
    app.run(debug=True, port=5000, use_reloader=False)