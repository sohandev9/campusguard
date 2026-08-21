"""
features.py
------------
Single source of truth for:
  1. Which MediaPipe landmarks we care about.
  2. How raw landmark coordinates are turned into TIME-NORMALIZED velocity
     (v = dy/dt, using real elapsed seconds, never a fixed-FPS row diff).
  3. How per-frame velocity is aggregated into a TIME-BASED WINDOW of
     mean/std/max statistics - this is what the models actually train and
     predict on, since a single instantaneous velocity reading can't tell
     a punch from a wave or a fall from sitting down quickly. A window is
     needed to see the *sustained pattern*.
  4. Which windowed-stat columns feed the FIGHT model vs the FALL model.
  5. A frame-to-frame person tracker that assigns a STABLE person_id, so a
     MediaPipe list-order swap between two people is never mistaken for one
     person teleporting (the "identity switch" bug).

batch_extractor.py, train_models.py and app.py all import from this file.
If you ever change a threshold or a landmark list, change it here ONLY.
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# 1. Landmarks we track (MediaPipe Pose landmark indices - identical across
#    the lite/full/heavy pose_landmarker.task variants)
# ---------------------------------------------------------------------------
LANDMARK_INDEX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}
LANDMARK_NAMES = list(LANDMARK_INDEX.keys())

# Raw (x, y) columns stored by batch_extractor.py in the CSV.
RAW_COLUMNS = []
for _name in LANDMARK_NAMES:
    RAW_COLUMNS.append(f"{_name}_x")
    RAW_COLUMNS.append(f"{_name}_y")

# Every per-frame velocity column that compute_velocity() can produce.
ALL_VELOCITY_COLUMNS = [f"{n}_vx" for n in LANDMARK_NAMES] + [f"{n}_vy" for n in LANDMARK_NAMES]

# Torso tilt: a POSTURE column (not a velocity) - see compute_torso_tilt_degrees.
TORSO_TILT_COLUMN = "torso_tilt_deg"
ALL_SAMPLE_COLUMNS = ALL_VELOCITY_COLUMNS + [TORSO_TILT_COLUMN]

# ---------------------------------------------------------------------------
# 2. Which landmarks each model cares about.
#    Fight = arm/upper-body kinematics. Fall = torso collapse kinematics
#    (downward velocity) PLUS torso tilt (final posture) - velocity alone
#    can't tell a fall from a fast controlled sit-down/kneel, since both
#    involve fast downward hip motion. Tilt distinguishes them: a real fall
#    ends with the torso near-horizontal; sitting/kneeling ends upright.
#    NOTE: bag/YOLO features are intentionally NEVER added here (data leakage).
# ---------------------------------------------------------------------------
FIGHT_LANDMARKS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]
FALL_LANDMARKS = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
]

FIGHT_VELOCITY_COLUMNS = [f"{n}_vx" for n in FIGHT_LANDMARKS] + [f"{n}_vy" for n in FIGHT_LANDMARKS]
FALL_VELOCITY_COLUMNS = [f"{n}_vx" for n in FALL_LANDMARKS] + [f"{n}_vy" for n in FALL_LANDMARKS]

# Any single-landmark velocity above this (normalized-coord units / second)
# is physically impossible for a real human and is treated as a MediaPipe
# glitch or a residual identity switch -> the whole row is dropped (and,
# since it breaks continuity, the rolling window is reset - see below).
MAX_PLAUSIBLE_VELOCITY = 10.0

# ---------------------------------------------------------------------------
# 3. Time-based rolling window for aggregating velocity into model features.
#    Deliberately measured in SECONDS, not a frame count, so it stays
#    correct regardless of a video's real FPS, an image sequence's assumed
#    FPS, or live inference frame-skipping / GPU lag - the exact same
#    "no train/serve skew" principle as compute_velocity() itself.
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 0.6        # how much recent motion history to aggregate
MIN_WINDOW_SAMPLES = 5      # need at least this many samples before trusting the stats

# Rolling ALERT stability (separate from the feature window above): this is
# a post-prediction smoother - N-out-of-WINDOW positive classifications
# required before the on-screen alert fires, so one noisy prediction can't
# flash an alert on its own.
ALERT_WINDOW = 15
ALERT_THRESHOLD = 10

# Person tracker tuning
MAX_TRACK_DISTANCE = 0.25   # normalized-coord hip-centroid distance to match
MAX_MISSED_FRAMES = 10      # frames a track can go undetected before it dies


def _window_stat_columns(base_columns):
    cols = []
    for col in base_columns:
        cols += [f"{col}_mean", f"{col}_std", f"{col}_max"]
    return cols


# These are the ACTUAL columns each .pkl model is trained/predicted on.
FIGHT_FEATURE_COLUMNS = _window_stat_columns(FIGHT_VELOCITY_COLUMNS)
FALL_FEATURE_COLUMNS = _window_stat_columns(FALL_VELOCITY_COLUMNS) + _window_stat_columns([TORSO_TILT_COLUMN])


# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------
def extract_landmark_coords(pose_landmarks):
    """
    pose_landmarks: a single person's list of MediaPipe NormalizedLandmark
    (indexable, e.g. result.pose_landmarks[i] from PoseLandmarker).

    Returns dict {"left_shoulder_x": ..., "left_shoulder_y": ..., ...}
    Coordinates are MediaPipe's own normalized 0-1 image-space values, so
    the extraction step is identical whether the frame came from a stored
    video file or a live webcam.
    """
    coords = {}
    for name, idx in LANDMARK_INDEX.items():
        lm = pose_landmarks[idx]
        coords[f"{name}_x"] = float(lm.x)
        coords[f"{name}_y"] = float(lm.y)
    return coords


def hip_centroid(coords):
    """Midpoint of the two hips - used as the tracking anchor point."""
    x = (coords["left_hip_x"] + coords["right_hip_x"]) / 2.0
    y = (coords["left_hip_y"] + coords["right_hip_y"]) / 2.0
    return x, y


def compute_torso_tilt_degrees(coords):
    """Angle, in degrees, between the hip->shoulder line and VERTICAL.
    0 = upright torso (standing, sitting, kneeling - shoulders directly
    above hips). 90 = horizontal torso (lying down - a real fall's
    end-state). This is a POSTURE snapshot from a single frame's raw
    coordinates - not a velocity, no dt involved - which is exactly what
    velocity-only features are missing: a fast controlled sit-down and an
    uncontrolled collapse can have near-identical downward hip velocity,
    but only the collapse ends up horizontal.

    abs() on both components so the result doesn't care about left/right
    lean or which way the shoulders/hips are offset - just how far the
    torso has tilted away from vertical, so forward, backward, and
    sideways falls all read the same way.
    """
    shoulder_x = (coords["left_shoulder_x"] + coords["right_shoulder_x"]) / 2.0
    shoulder_y = (coords["left_shoulder_y"] + coords["right_shoulder_y"]) / 2.0
    hip_x = (coords["left_hip_x"] + coords["right_hip_x"]) / 2.0
    hip_y = (coords["left_hip_y"] + coords["right_hip_y"]) / 2.0

    dx = abs(shoulder_x - hip_x)
    dy = abs(shoulder_y - hip_y)
    angle_rad = math.atan2(dx, dy + 1e-9)  # +epsilon: avoid divide-by-zero if dy==0
    return math.degrees(angle_rad)


# ---------------------------------------------------------------------------
# Velocity: the ONE function both training and inference call.
# ---------------------------------------------------------------------------
def compute_velocity(prev_coords, curr_coords, dt_seconds):
    """
    v = dy/dt using REAL elapsed seconds (dt_seconds), never a fixed-FPS
    row index diff. This is what eliminates train/serve skew:
      - In batch_extractor/train_models, dt_seconds = (curr_timestamp -
        prev_timestamp), where timestamp = frame_idx / true_fps (the
        video's real FPS, or the assumed IMAGE_SEQUENCE_FPS for image
        sequences - not a hardcoded 30fps guess baked into the diff itself).
      - In app.py (live), dt_seconds = time.time() - prev_timestamp, so
        GPU lag / fluctuating FPS is naturally absorbed into dt instead of
        distorting velocity.

    Returns None if dt is invalid (<=0, missing, or too large a gap i.e.
    the person likely left and re-entered frame) or if any landmark's
    velocity exceeds MAX_PLAUSIBLE_VELOCITY (teleportation / identity
    switch residue) -> caller should drop the sample AND reset any rolling
    window (see update_window below), since continuity is broken.
    """
    if prev_coords is None or dt_seconds is None:
        return None
    if dt_seconds <= 0 or dt_seconds > 1.0:
        # >1s gap: person was lost and reappeared: don't diff across the gap
        return None

    velocity = {}
    for name in LANDMARK_NAMES:
        dx = curr_coords[f"{name}_x"] - prev_coords[f"{name}_x"]
        dy = curr_coords[f"{name}_y"] - prev_coords[f"{name}_y"]
        vx = dx / dt_seconds
        vy = dy / dt_seconds
        if abs(vx) > MAX_PLAUSIBLE_VELOCITY or abs(vy) > MAX_PLAUSIBLE_VELOCITY:
            return None  # teleportation jump -> reject whole row
        velocity[f"{name}_vx"] = vx
        velocity[f"{name}_vy"] = vy
    return velocity


# ---------------------------------------------------------------------------
# Windowing: turns a short history of per-frame velocity into the actual
# feature vector the models see. Called identically by train_models.py
# (using CSV timestamps) and app.py (using time.time()).
# ---------------------------------------------------------------------------
def update_window(window, timestamp, sample):
    """
    window: a list of (timestamp, sample_dict) tuples, chronological.
    sample_dict holds this frame's per-landmark velocity (from
    compute_velocity) plus any extra per-frame columns the caller wants
    aggregated the same way, e.g. TORSO_TILT_COLUMN from
    compute_torso_tilt_degrees - it's just a flat dict of named values.
    Appends the new sample and evicts anything older than WINDOW_SECONDS.
    Mutates and returns `window` for convenience.

    Callers MUST reset the window (window.clear(), or reassign to []) when
    compute_velocity() returns None for this person, since that signals a
    gap or a rejected teleportation jump - continuing to average across
    that discontinuity would corrupt the window's statistics.
    """
    window.append((timestamp, sample))
    cutoff = timestamp - WINDOW_SECONDS
    while window and window[0][0] < cutoff:
        window.pop(0)
    return window


def compute_window_stats(window, base_columns=ALL_SAMPLE_COLUMNS):
    """
    window: list of (timestamp, velocity_dict) as built by update_window().
    Returns None if the window doesn't yet have MIN_WINDOW_SAMPLES entries
    (caller should skip prediction/training for this row - not enough
    history yet to trust the statistics).

    Otherwise returns a flat dict {"col_mean":..., "col_std":..., "col_max":...}
    for every column in base_columns. mean/std capture sustained motion
    level; max (of |v|) captures a single sharp peak (a punch, a collapse)
    even if the rest of the window was calm.
    """
    if len(window) < MIN_WINDOW_SAMPLES:
        return None

    velocities = [v for _, v in window]
    stats = {}
    for col in base_columns:
        vals = np.array([v[col] for v in velocities], dtype=float)
        stats[f"{col}_mean"] = float(vals.mean())
        stats[f"{col}_std"] = float(vals.std())
        stats[f"{col}_max"] = float(np.max(np.abs(vals)))
    return stats


def build_feature_vector(stats, columns):
    """Order a window-stats dict into a flat list matching a trained
    model's expected column order. Missing keys default to 0.0."""
    return [stats.get(col, 0.0) for col in columns]


# ---------------------------------------------------------------------------
# 4. Person tracker: decouples person_id from MediaPipe's per-frame list
#    order, which is what actually caused the identity-switch bug.
# ---------------------------------------------------------------------------
class PersonTracker:
    def __init__(self, max_distance=MAX_TRACK_DISTANCE, max_missed=MAX_MISSED_FRAMES):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks = {}       # person_id -> (x, y) last known hip centroid
        self.missed = {}       # person_id -> consecutive frames missed
        self._next_id = 0

    def update(self, detections):
        """
        detections: list of (x, y) hip centroids for this frame, in
        whatever order MediaPipe returned them (that order is NOT trusted).

        Returns: list of person_id, same length/order as `detections`.
        """
        assigned = [None] * len(detections)
        used_tracks = set()

        # Greedy nearest-neighbour matching: closest pairs first.
        candidate_pairs = []
        for i, det in enumerate(detections):
            for pid, last in self.tracks.items():
                dist = math.hypot(det[0] - last[0], det[1] - last[1])
                if dist <= self.max_distance:
                    candidate_pairs.append((dist, i, pid))
        candidate_pairs.sort(key=lambda p: p[0])

        used_dets = set()
        for dist, i, pid in candidate_pairs:
            if i in used_dets or pid in used_tracks:
                continue
            assigned[i] = pid
            used_dets.add(i)
            used_tracks.add(pid)

        # Unmatched detections become new tracks.
        for i, det in enumerate(detections):
            if assigned[i] is None:
                pid = self._next_id
                self._next_id += 1
                assigned[i] = pid
            self.tracks[assigned[i]] = det
            self.missed[assigned[i]] = 0

        # Age out tracks that weren't seen this frame.
        for pid in list(self.tracks.keys()):
            if pid not in used_tracks and pid not in assigned:
                self.missed[pid] = self.missed.get(pid, 0) + 1
                if self.missed[pid] > self.max_missed:
                    del self.tracks[pid]
                    del self.missed[pid]

        return assigned

    def __init__(self, max_distance=MAX_TRACK_DISTANCE, max_missed=MAX_MISSED_FRAMES):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks = {}       # person_id -> (x, y) last known hip centroid
        self.missed = {}       # person_id -> consecutive frames missed
        self._next_id = 0

    def update(self, detections):
        """
        detections: list of (x, y) hip centroids for this frame, in
        whatever order MediaPipe returned them (that order is NOT trusted).

        Returns: list of person_id, same length/order as `detections`.
        """
        assigned = [None] * len(detections)
        used_tracks = set()

        # Greedy nearest-neighbour matching: closest pairs first.
        candidate_pairs = []
        for i, det in enumerate(detections):
            for pid, last in self.tracks.items():
                dist = math.hypot(det[0] - last[0], det[1] - last[1])
                if dist <= self.max_distance:
                    candidate_pairs.append((dist, i, pid))
        candidate_pairs.sort(key=lambda p: p[0])

        used_dets = set()
        for dist, i, pid in candidate_pairs:
            if i in used_dets or pid in used_tracks:
                continue
            assigned[i] = pid
            used_dets.add(i)
            used_tracks.add(pid)

        # Unmatched detections become new tracks.
        for i, det in enumerate(detections):
            if assigned[i] is None:
                pid = self._next_id
                self._next_id += 1
                assigned[i] = pid
            self.tracks[assigned[i]] = det
            self.missed[assigned[i]] = 0

        # Age out tracks that weren't seen this frame.
        for pid in list(self.tracks.keys()):
            if pid not in used_tracks and pid not in assigned:
                self.missed[pid] = self.missed.get(pid, 0) + 1
                if self.missed[pid] > self.max_missed:
                    del self.tracks[pid]
                    del self.missed[pid]

        return assigned