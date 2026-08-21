"""
batch_extractor.py
-------------------
Walks the raw video datasets, runs MediaPipe PoseLandmarker (num_poses=4) on
every frame, assigns a STABLE person_id via features.PersonTracker, and dumps
one row per (video, frame, person) with RAW landmark coordinates only.

Velocity is deliberately NOT computed here. It is computed later in
train_models.py from (frame_idx / true_fps), grouped by (video_name,
person_id) - see features.compute_velocity. Keeping raw extraction and
velocity math in separate, shared-logic steps is what guarantees training
and live inference use identical math.

Output: master_training_data.csv in the project root (appended to, not
overwritten - see main()).

------------------------------------------------------------------------
EDIT THIS SECTION to match your folder layout before running.
------------------------------------------------------------------------
Three dataset sources are wired up:

  1. RWF-2000 style fight videos (DATASET_CONFIG, item_type="video")
       data/fight/Fight/*.mp4, data/fight/NonFight/*.mp4

  2. UR Fall Dataset - PNG frame folders, binary Fall/ADL split
     (DATASET_CONFIG, item_type="image_sequence")
       data/UR_data/Fall/fall-XX-camY-rgb/*.png
       data/UR_data/ADL/adl-XX-camY-rgb/*.png

  3. CAUCAFall - PNG frame folders, one folder PER SUBJECT PER ACTIVITY,
     where the activity folder name itself encodes the label (any folder
     starting with "Fall" -> label 1, everything else -> label 0). Built
     dynamically by build_caucafall_items() since there's no fixed glob
     that separates fall/not-fall like the other datasets - see
     CAUCAFALL_ROOT below.
       data/CAUCAFall/Subject.1/Fall backwards/*.png   (+ .txt bbox files,
                                                          ignored - we run
                                                          our own pose model)
       data/CAUCAFall/Subject.1/Walk/*.png
       ... Subject.2 .. Subject.10, same activity folders each

IMPORTANT (why this file changed): CAUCAFall reuses the SAME activity
folder name ("Fall backwards", "Walk", etc.) under every subject. If we
identified a sequence only by its folder's basename, Subject.1's "Fall
backwards" and Subject.2's "Fall backwards" would collide onto the same
video_name and get merged into one bogus grouped sequence during
training. get_item_name() below fixes this by including the subject
folder in the identifier for image sequences.

Two item types are supported:
  "video"          -> pattern matches individual .mp4/.avi FILES. Real FPS
                       is read from the file itself.
  "image_sequence" -> matches FOLDERS, each folder = one ordered sequence
                       of frames. No file carries a real per-frame
                       timestamp, so we assume a constant FPS per dataset
                       (IMAGE_SEQUENCE_FPS / CAUCAFALL_FPS below) - edit
                       these to match how each source was actually sampled.
"""

import os
import re
import glob
import csv
import multiprocessing as mp

import cv2
import mediapipe as mp_mediapipe
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import features

# ---------------------------------------------------------------------------
# CONFIG - edit to match your dataset folders
# ---------------------------------------------------------------------------
POSE_MODEL_PATH = "pose_landmarker_heavy.task"
OUTPUT_CSV = "master_training_data.csv"
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# UR Fall Dataset's RGB camera frames are sampled at 30fps.
IMAGE_SEQUENCE_FPS = 30.0

# (glob pattern, task, label, item_type, fps_or_None)
# task in {"fight","fall"}   label in {0,1}   item_type in {"video","image_sequence"}
# fps_or_None: ignored for "video" (real fps is read from the file); for
# "image_sequence" it's the assumed sampling rate (falls back to
# IMAGE_SEQUENCE_FPS if None).
DATASET_CONFIG = [
    ("data/fight/Fight/*.mp4",     "fight", 1, "video", None),
    ("data/fight/Fight/*.avi",     "fight", 1, "video", None),
    ("data/fight/NonFight/*.mp4",  "fight", 0, "video", None),
    ("data/fight/NonFight/*.avi",  "fight", 0, "video", None),
    ("data/UR_data/Fall/*",        "fall",  1, "image_sequence", IMAGE_SEQUENCE_FPS),
    ("data/UR_data/ADL/*",         "fall",  0, "image_sequence", IMAGE_SEQUENCE_FPS),
]

# --- CAUCAFall: built dynamically, not via DATASET_CONFIG - see docstring ---
CAUCAFALL_ROOT = "data2/Dataset CAUCAFall/CAUCAFall"   # edit to wherever you extracted it
# HIKVISION IP camera footage. Confirmed by the dataset's own Mendeley
# page: captured at 23fps, 1080x960.
CAUCAFALL_FPS = 23.0
CAUCAFALL_FALL_PREFIX = "fall"      # activity folder names starting with this -> label 1

CSV_COLUMNS = ["video_name", "task", "label", "frame_idx", "timestamp", "person_id"] + features.RAW_COLUMNS

_TRAILING_NUMBER = re.compile(r"(\d+)(?=\.\w+$)")


def _frame_sort_key(filename):
    """Sorts fall-01-cam0-rgb-007.png before -010.png (or cas100007.png
    before cas100012.png) even without consistent zero-padding, by
    extracting the trailing frame number. Returns a tuple, never a bare
    int/str, so sort() never has to compare an int to a str if some file
    in the folder doesn't end in digits (e.g. a stray non-frame image)."""
    match = _TRAILING_NUMBER.search(filename)
    if match:
        return (0, int(match.group(1)), filename)
    return (1, 0, filename)  # no trailing number - sorts after numbered frames


def get_item_name(path, item_type):
    """Unique identifier used as video_name in the CSV.

    Videos: filename alone (assumed unique, e.g. RWF-2000 clip names) -
    SANITIZED to strip anything that isn't alphanumeric/dash/underscore/dot.
    Some RWF-2000 downloads have corrupted/mojibake filenames (leftover
    "_urlgot_NNN" artifacts from a failed scrape) containing stray bytes
    that decode to characters like literal commas or quotes, which breaks
    the CSV's column structure when written raw. Sanitizing here prevents
    that at the source, rather than needing pandas to skip bad rows later.

    Image sequences: last TWO path components (parent + folder), so
    per-subject folders that reuse the same activity name across subjects
    (CAUCAFall's "Fall backwards" appearing under every Subject.N) don't
    collide into a single fake merged sequence.
    """
    norm = os.path.normpath(path)
    if item_type == "video":
        name = os.path.basename(norm)
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)
    parts = norm.split(os.sep)
    tail = parts[-2:] if len(parts) >= 2 else parts
    return "_".join(tail).replace(" ", "_")


def build_caucafall_items():
    """Scans CAUCAFALL_ROOT/Subject.*/<activity>/ and auto-labels each
    activity folder by whether its name starts with 'fall' - no static
    glob-per-label needed since CAUCAFall has many differently-named
    activity folders (Fall backwards, Fall forward, Hop, Kneel, Walk, ...)
    rather than a simple Fall/NotFall split."""
    items = []
    if not os.path.isdir(CAUCAFALL_ROOT):
        return items

    for subject_dir in sorted(glob.glob(os.path.join(CAUCAFALL_ROOT, "Subject.*"))):
        if not os.path.isdir(subject_dir):
            continue
        for activity_dir in sorted(glob.glob(os.path.join(subject_dir, "*"))):
            if not os.path.isdir(activity_dir):
                continue
            activity_name = os.path.basename(activity_dir)
            label = 1 if activity_name.lower().startswith(CAUCAFALL_FALL_PREFIX) else 0
            items.append((activity_dir, "fall", label, "image_sequence", CAUCAFALL_FPS))

    return items


def build_item_list():
    items = []
    for pattern, task, label, item_type, fps in DATASET_CONFIG:
        for path in glob.glob(pattern):
            if item_type == "image_sequence" and not os.path.isdir(path):
                continue
            if item_type == "video" and not os.path.isfile(path):
                continue
            items.append((path, task, label, item_type, fps))

    caucafall_items = build_caucafall_items()
    items.extend(caucafall_items)
    if caucafall_items:
        n_fall = sum(1 for it in caucafall_items if it[2] == 1)
        print(f"CAUCAFall: found {len(caucafall_items)} sequences "
              f"({n_fall} fall, {len(caucafall_items) - n_fall} not-fall)")

    return items


def _run_pose_loop(frame_iterator, video_name, task, label, fps):
    """Shared pose-tracking loop. frame_iterator yields BGR numpy frames in
    order. fps is the (real or assumed) time base used to build timestamps -
    this is the ONLY thing that differs between a real video and an image
    sequence; everything downstream (extraction, tracking, CSV schema) is
    identical, so train_models.py treats both sources exactly the same way.
    """
    rows = []
    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=4,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    tracker = features.PersonTracker()
    frame_idx = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame in frame_iterator:
            timestamp_ms = int((frame_idx / fps) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_mediapipe.Image(image_format=mp_mediapipe.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                per_person_coords = [features.extract_landmark_coords(pl) for pl in result.pose_landmarks]
                centroids = [features.hip_centroid(c) for c in per_person_coords]
                person_ids = tracker.update(centroids)

                # elapsed seconds on whatever time base this source uses
                # (real video FPS, or the assumed sequence FPS) - this is
                # what train_models.py diffs against, so it must be the
                # same unit (seconds) in both cases.
                timestamp_sec = frame_idx / fps

                for coords, pid in zip(per_person_coords, person_ids):
                    row = {
                        "video_name": video_name,
                        "task": task,
                        "label": label,
                        "frame_idx": frame_idx,
                        "timestamp": timestamp_sec,
                        "person_id": pid,
                    }
                    row.update(coords)
                    rows.append(row)

            frame_idx += 1

    print(f"[done] {video_name}: {frame_idx} frames, {len(rows)} pose rows")
    return rows


def _video_frame_iterator(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[skip] could not open {video_path}")
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _image_sequence_frame_iterator(folder_path):
    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
    filenames.sort(key=_frame_sort_key)

    unnumbered = [f for f in filenames if not _TRAILING_NUMBER.search(f)]
    if unnumbered:
        print(f"[warn] {folder_path}: {len(unnumbered)} file(s) with no trailing "
              f"frame number, sorted last: {unnumbered[:3]}{'...' if len(unnumbered) > 3 else ''}")

    if not filenames:
        print(f"[skip] no images found in {folder_path}")
        return
    for filename in filenames:
        frame = cv2.imread(os.path.join(folder_path, filename))
        if frame is not None:
            yield frame


def process_item(args):
    """Runs in a worker process. Dispatches to a video file or an image
    sequence folder based on item_type, then runs the shared pose loop."""
    path, task, label, item_type, fps = args
    item_name = get_item_name(path, item_type)

    if item_type == "video":
        cap = cv2.VideoCapture(path)
        real_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
        cap.release()
        if not real_fps or real_fps <= 1:
            real_fps = 30.0  # sane fallback only if the file has no valid FPS header
        return _run_pose_loop(_video_frame_iterator(path), item_name, task, label, real_fps)

    elif item_type == "image_sequence":
        seq_fps = fps if fps else IMAGE_SEQUENCE_FPS
        return _run_pose_loop(
            _image_sequence_frame_iterator(path), item_name, task, label, seq_fps
        )

    else:
        print(f"[skip] unknown item_type '{item_type}' for {path}")
        return []


def _already_extracted_names(csv_path):
    """video_name values already present in an existing CSV, so re-running
    main() after adding new DATASET_CONFIG rows / CAUCAFall doesn't
    reprocess (or duplicate) items you already extracted."""
    if not os.path.exists(csv_path):
        return set()
    names = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.add(row["video_name"])
    return names


def main():
    all_items = build_item_list()
    if not all_items:
        print("No videos/image sequences found. Check DATASET_CONFIG / CAUCAFALL_ROOT in batch_extractor.py")
        return

    done = _already_extracted_names(OUTPUT_CSV)
    items = [it for it in all_items if get_item_name(it[0], it[3]) not in done]
    skipped = len(all_items) - len(items)
    if skipped:
        print(f"Skipping {skipped} item(s) already present in {OUTPUT_CSV}")

    if not items:
        print("Nothing new to extract.")
        return

    print(f"Found {len(items)} new items. Extracting with {NUM_WORKERS} worker processes...")

    with mp.Pool(processes=NUM_WORKERS) as pool:
        all_results = pool.map(process_item, items)

    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        total_rows = 0
        for rows in all_results:
            for row in rows:
                writer.writerow(row)
                total_rows += 1

    print(f"Appended {total_rows} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # required for MediaPipe + multiprocessing on Windows
    main()