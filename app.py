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

NOTE on fight detection (updated): fight alerting is now SCENE-LEVEL, not
per-identity. The original per-pid buffer approach assumed a person keeps
the same pid long enough to accumulate 5+ history entries, but on real
footage the tracker churns IDs constantly during fast/chaotic motion -
exactly the motion fight detection is supposed to catch - so the per-pid
buffer could structurally starve and never fire. Fall detection keeps its
original per-pid buffer, since it isn't affected by this churn problem.
"""

import os
import time
import uuid
import threading
from collections import deque, defaultdict

import cv2
import joblib
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify
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

BAG_CLASSES = {24, 26, 28}   # COCO: backpack (24), handbag (26), suitcase (28)
PERSON_CLASS = 0
UNATTENDED_SECONDS = 15.0
BAG_NEAR_PIXELS_FRACTION = 0.12

FRAME_SKIP = 2

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load models once at boot with graceful error handling.
# ---------------------------------------------------------------------------
_model_errors = []
try:
    print("Loading fight/fall models ...")
    fight_bundle = joblib.load(os.path.join(os.path.dirname(__file__), "fight_detector_model.pkl"))
    fall_bundle = joblib.load(os.path.join(os.path.dirname(__file__), "fall_detector_model.pkl"))
    FIGHT_MODEL = fight_bundle["model"]
    FALL_MODEL = fall_bundle["model"]
    FIGHT_COLUMNS = fight_bundle["feature_columns"]
    FALL_COLUMNS = fall_bundle["feature_columns"]
    FIGHT_THRESHOLD = fight_bundle.get("threshold", 0.5)
    FALL_THRESHOLD = fall_bundle.get("threshold", 0.5)
    print(f"  Fight threshold: {FIGHT_THRESHOLD:.3f}")
    print(f"  Fall threshold:  {FALL_THRESHOLD:.3f}")
except Exception as e:
    _model_errors.append(f"Fight/fall models: {e}")
    FIGHT_MODEL = FALL_MODEL = None
    FIGHT_COLUMNS = FALL_COLUMNS = []
    FIGHT_THRESHOLD = FALL_THRESHOLD = 0.5

try:
    print("Loading YOLOv8 ...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    _model_errors.append(f"YOLOv8: {e}")
    yolo_model = None

try:
    print("Loading MediaPipe PoseLandmarker ...")
    # num_poses=8 + conf=0.3: diagnostic testing (debug_fight_detection.py)
    # showed default 4/0.5 drops 48% of frames to zero detections on fight
    # clips. Relaxed settings raise avg detection 0.61 -> 0.91 people/frame
    # and increase window fill rate 5x (21 -> 111 classifier invocations).
    # min_tracking_confidence is omitted: it only applies to VIDEO/LIVE_STREAM
    # mode, not IMAGE mode (where we run), so setting it has no effect.
    _pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=8,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
    )
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(_pose_options)
except Exception as e:
    _model_errors.append(f"MediaPipe: {e}")
    pose_landmarker = None

if _model_errors:
    print("WARNING - model load errors:")
    for err in _model_errors:
        print(f"  {err}")
else:
    print("All models loaded successfully.")


# ---------------------------------------------------------------------------
# Global alert log and live stats (thread-safe via _stats_lock).
# The /alerts and /stats endpoints poll these; run_pose_models and
# run_yolo_bags update them. With threaded=True on app.run(), these
# shared structures need a lock to avoid corruption from concurrent
# /video_feed streams.
# ---------------------------------------------------------------------------
MAX_ALERT_HISTORY = 100
_alert_log = deque(maxlen=MAX_ALERT_HISTORY)
_stats_lock = threading.Lock()
_live_stats = {
    "people_tracked": 0,
    "bags_detected": 0,
    "active_fight_alerts": 0,
    "active_fall_alerts": 0,
    "unattended_bags": 0,
    "total_fight_alerts": 0,
    "total_fall_alerts": 0,
    "total_unattended_bags": 0,
    "last_update": None,
    "models_loaded": len(_model_errors) == 0,
    "model_errors": _model_errors,
}


def _log_alert(alert_type, person_id, timestamp, extra=""):
    """Append an alert to the in-memory log and bump the matching cumulative
    total (thread-safe)."""
    with _stats_lock:
        entry = {
            "type": alert_type,
            "person_id": person_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "extra": extra,
        }
        _alert_log.appendleft(entry)
        if alert_type == "fight":
            _live_stats["total_fight_alerts"] += 1
        elif alert_type == "fall":
            _live_stats["total_fall_alerts"] += 1
        elif alert_type == "unattended_bag":
            _live_stats["total_unattended_bags"] += 1
        _live_stats["last_update"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )


def _update_live_stats(**kwargs):
    """Update one or more fields in _live_stats (thread-safe)."""
    with _stats_lock:
        for k, v in kwargs.items():
            if k in _live_stats:
                _live_stats[k] = v
        _live_stats["last_update"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )


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
        matched_ids = set()

        for cx, cy in bag_centroids:
            best_id, best_dist = None, None
            for tid, t in self.tracks.items():
                if tid in matched_ids:
                    continue
                d = np.hypot(cx - t["centroid"][0], cy - t["centroid"][1])
                if d <= self.max_distance and (best_dist is None or d < best_dist):
                    best_id, best_dist = tid, d

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self.tracks[best_id] = {"centroid": (cx, cy), "first_seen": now, "last_near_person": now}

            self.tracks[best_id]["centroid"] = (cx, cy)
            matched_ids.add(best_id)

            is_near_person = any(
                np.hypot(cx - px, cy - py) <= self.max_distance
                for px, py in person_centroids
            )
            if is_near_person:
                self.tracks[best_id]["last_near_person"] = now

        # Age out: tracks whose centroid moved far from any detection this frame
        # are likely gone (not just briefly occluded).
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
        self.bag_tracker = BagTracker(max_distance_px=None)
        self.prev_coords = {}       # person_id -> (coords_dict, timestamp)
        self.velocity_windows = defaultdict(list)
        self.fight_buffers = defaultdict(lambda: deque(maxlen=features.ALERT_WINDOW))
        self.fall_buffers = defaultdict(lambda: deque(maxlen=features.ALERT_WINDOW))
        self.prev_fight_alert = {}  # person_id -> bool (rising edge detection) [unused by fight now, kept for compat]
        self.prev_fall_alert = {}   # person_id -> bool
        self.frame_idx = 0
        # Scene-level fight detection: fight is treated as a property of the
        # scene (is someone showing fight-like motion right now), not of a
        # single tracked identity, since real footage churns pids too fast
        # for a per-pid sustained buffer to ever fill.
        self.scene_fight_buffer = deque(maxlen=10)
        self.prev_scene_fight_alert = False


def run_yolo_bags(frame, state, now):
    """Run YOLOv8 bag + person detection. Returns (bag_overlay, bag_count)."""
    h, w = frame.shape[:2]
    if state.bag_tracker.max_distance is None:
        state.bag_tracker.max_distance = w * BAG_NEAR_PIXELS_FRACTION

    if yolo_model is None:
        return [], 0

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

    # Re-derive which boxes are unattended by matching centroids back to tracker
    unattended_flags = []
    for (cx, cy) in bag_centroids:
        flagged = False
        for tid in unattended_ids:
            t = state.bag_tracker.tracks.get(tid)
            if t and t["centroid"] == (cx, cy):
                flagged = True
                break
        unattended_flags.append(flagged)

    # Rising-edge alert logging for unattended bags
    for (cx, cy), flagged in zip(bag_centroids, unattended_flags):
        if flagged:
            bag_id_str = f"bag@{cx:.0f},{cy:.0f}"
            _log_alert("unattended_bag", bag_id_str, now, "Abandoned near " + str(len(person_centroids)) + " people")

    return list(zip(bag_boxes, unattended_flags)), len(bag_boxes)


def run_pose_models(frame, state, now):
    """Returns list of (person_id, hip_pixel_xy, fight_alert, fall_alert).

    fight_alert is SCENE-LEVEL: every entry that is currently above the
    per-frame fight threshold shares the same scene-wide alert decision
    (state.scene_fight_buffer >= 3-of-10 recent frames had someone above
    threshold). This replaces the old per-pid sustained-buffer approach,
    which assumed a person keeps the same pid long enough to accumulate
    5+ history entries - untrue on real footage where fast/chaotic motion
    (i.e. exactly what we're trying to detect) causes constant tracker
    churn. fall_alert is unchanged: still a per-pid sustained buffer,
    since fall isn't affected by this churn problem.
    """
    h, w = frame.shape[:2]
    people_overlay = []

    if pose_landmarker is None or FIGHT_MODEL is None:
        return people_overlay

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp_mediapipe.Image(image_format=mp_mediapipe.ImageFormat.SRGB, data=rgb)
    result = pose_landmarker.detect(mp_image)

    if not result.pose_landmarks:
        _update_live_stats(people_tracked=0)
        return people_overlay

    per_person_coords = [features.extract_landmark_coords(pl) for pl in result.pose_landmarks]
    centroids = [features.hip_centroid(c) for c in per_person_coords]
    person_ids = state.person_tracker.update(centroids)

    frame_high_fight_probas = []   # (pid, proba) pairs that crossed FIGHT_THRESHOLD this frame

    for coords, pid, centroid in zip(per_person_coords, person_ids, centroids):
        prev = state.prev_coords.get(pid)
        prev_coords, prev_time = prev if prev else (None, None)
        dt = None if prev_time is None else (now - prev_time)

        velocity = features.compute_velocity(prev_coords, coords, dt)
        state.prev_coords[pid] = (coords, now)

        fall_alert = False

        if velocity is None:
            state.velocity_windows[pid] = []
        else:
            velocity[features.TORSO_TILT_COLUMN] = features.compute_torso_tilt_degrees(coords)
            window = state.velocity_windows[pid]
            features.update_window(window, now, velocity)
            stats = features.compute_window_stats(window)

            if stats is not None:
                fight_vec = np.array([features.build_feature_vector(stats, FIGHT_COLUMNS)])
                fall_vec = np.array([features.build_feature_vector(stats, FALL_COLUMNS)])

                fight_proba = FIGHT_MODEL.predict_proba(fight_vec)[0, 1]
                fall_proba = FALL_MODEL.predict_proba(fall_vec)[0, 1]
                print(f"person {pid}: fight_proba={fight_proba:.3f} (threshold={FIGHT_THRESHOLD:.3f})")

                if fight_proba >= FIGHT_THRESHOLD:
                    frame_high_fight_probas.append((pid, fight_proba))

                fall_pred = int(fall_proba >= FALL_THRESHOLD)
                state.fall_buffers[pid].append(fall_pred)
                if sum(state.fall_buffers[pid]) >= features.ALERT_THRESHOLD:
                    fall_alert = True

                prev_fall = state.prev_fall_alert.get(pid, False)
                if fall_alert and not prev_fall:
                    _log_alert("fall", pid, now, f"confidence={fall_proba:.2f}")
                state.prev_fall_alert[pid] = fall_alert

        px, py = int(centroid[0] * w), int(centroid[1] * h)
        # fight_alert filled in below, after the scene-level decision is made
        people_overlay.append([pid, (px, py), False, fall_alert])

    # --- Scene-level fight decision -----------------------------------
    state.scene_fight_buffer.append(1 if frame_high_fight_probas else 0)
    scene_fight_alert = sum(state.scene_fight_buffer) >= 3

    if scene_fight_alert and not state.prev_scene_fight_alert:
        if frame_high_fight_probas:
            top_pid, top_proba = max(frame_high_fight_probas, key=lambda x: x[1])
            _log_alert(
                "fight", "scene", now,
                f"triggered by pid={top_pid} confidence={top_proba:.2f}, "
                f"{len(frame_high_fight_probas)} person(s) above threshold this frame",
            )
        else:
            _log_alert("fight", "scene", now, "scene-level trigger")
    state.prev_scene_fight_alert = scene_fight_alert

    # Only the people who actually crossed threshold THIS frame get flagged
    # visually, even though the underlying alert is scene-wide, so bystanders
    # who happen to share a frame with a genuine fight aren't painted red.
    flagged_pids = {pid for pid, _ in frame_high_fight_probas}
    for row in people_overlay:
        row[2] = scene_fight_alert and (row[0] in flagged_pids)
    people_overlay = [tuple(row) for row in people_overlay]

    # active_fight_alerts is 0/1 (scene-wide), not a per-person count -
    # this mirrors the new semantics: "is a fight happening", not "how many
    # people are individually flagged".
    active_fight = 1 if scene_fight_alert else 0
    active_fall = sum(1 for _, _, _, fl in people_overlay if fl)
    _update_live_stats(
        people_tracked=len(people_overlay),
        active_fight_alerts=active_fight,
        active_fall_alerts=active_fall,
    )

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
        # Yield a visible error frame instead of silently dropping the stream
        # connection (which leaves the dashboard hanging on a loading spinner).
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, f"Camera/Video unavailable: {source}",
                    (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ok, buffer = cv2.imencode(".jpg", blank)
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
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


@app.route("/stats")
def stats():
    """Live dashboard stats polled every 500ms via JS.

    Returns current people/bag counts, active alerts, and cumulative totals.
    """
    with _stats_lock:
        snap = dict(_live_stats)
    snap["alert_history_count"] = min(len(_alert_log), MAX_ALERT_HISTORY)
    return jsonify(snap)


@app.route("/alerts")
def alerts():
    """Recent alert history polled every 1s via JS.

    Newest alerts first. Each entry has type, person_id, timestamp, extra.
    """
    with _stats_lock:
        entries = list(_alert_log)
    return jsonify({"alerts": entries})


@app.route("/health")
def health():
    """Quick health check — whether all models loaded cleanly."""
    return jsonify({
        "models_loaded": _live_stats["models_loaded"],
        "errors": _live_stats["model_errors"],
    })


if __name__ == "__main__":
    # threaded=True: Flask dev server handles concurrent requests in threads.
    #   Without this, /video_feed (an infinite generator) blocks the entire
    #   server — no other request (dashboard polling, upload, health check)
    #   can proceed while the stream is open.
    # use_reloader=False: prevents Flask's debug reloader from loading the
    #   YOLO/MediaPipe/RF models twice (double VRAM + double init time).
    app.run(debug=True, port=5000, use_reloader=False, threaded=True)