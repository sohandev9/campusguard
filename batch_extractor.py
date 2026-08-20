"""
batch_extractor.py
-------------------
Walks the raw video datasets, runs MediaPipe PoseLandmarker (num_poses=4) on
every frame, assigns a STABLE person_id via features.PersonTracker, and dumps
one row per (video, frame, person) with RAW landmark coordinates only.

Velocity is deliberately NOT computed here. It is computed later in
train_models.py from (frame_idx / true_video_fps), grouped by
(video_name, person_id) - see features.compute_velocity. Keeping raw
extraction and velocity math in separate, shared-logic steps is what
guarantees training and live inference use identical math.

Output: master_training_data.csv in the project root.

------------------------------------------------------------------------
EDIT THIS SECTION to match your folder layout before running.
------------------------------------------------------------------------
Expected layout (edit DATASET_CONFIG below if yours differs):

    data/
      fight/Fight/*.mp4              RWF-2000 style
      fight/NonFight/*.mp4
      UR_data/Fall/fall-XX-camY-rgb/     UR Fall Dataset - PNG frame folders
      UR_data/ADL/adl-XX-camY-rgb/       UR Fall Dataset "normal activity" frame folders

Two dataset item types are supported per DATASET_CONFIG row:
  "video"          -> pattern matches individual .mp4/.avi FILES
  "image_sequence" -> pattern matches FOLDERS, each folder = one ordered
                       sequence of frames (fall-01-cam0-rgb/*.png etc.)
                       There is no real per-frame timestamp for image
                       sequences, so we assume a constant FPS (see
                       IMAGE_SEQUENCE_FPS below) - edit it to match how
                       the source dataset's frames were actually sampled.
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

# UR Fall Dataset's RGB camera frames are sampled at 30fps. If your image
# sequences came from a different source, change this to match.
IMAGE_SEQUENCE_FPS = 30.0

# (glob pattern, task, label, item_type)
# task in {"fight","fall"}   label in {0,1}   item_type in {"video","image_sequence"}
DATASET_CONFIG = [
    # Updated to match your RWF-2000 folders
    ("data/RWF-2000/train/Fight/*.mp4",     "fight", 1, "video"),
    ("data/RWF-2000/train/Fight/*.avi",     "fight", 1, "video"),
    ("data/RWF-2000/train/NonFight/*.mp4",  "fight", 0, "video"),
    ("data/RWF-2000/train/NonFight/*.avi",  "fight", 0, "video"),
    
    # Updated with /*/* to reach your nested image sequence folders
    ("data/UR_data/Fall/*/*",               "fall",  1, "image_sequence"),
    ("data/UR_data/ADL/*/*",                "fall",  0, "image_sequence"),
]

CSV_COLUMNS = ["video_name", "task", "label", "frame_idx", "timestamp", "person_id"] + features.RAW_COLUMNS

_TRAILING_NUMBER = re.compile(r"(\d+)(?=\.\w+$)")


def _frame_sort_key(filename):
    """Sorts fall-01-cam0-rgb-007.png before -010.png even without
    consistent zero-padding, by extracting the trailing frame number."""
    match = _TRAILING_NUMBER.search(filename)
    return int(match.group(1)) if match else filename


def build_item_list():
    items = []
    for pattern, task, label, item_type in DATASET_CONFIG:
        for path in glob.glob(pattern):
            if item_type == "image_sequence" and not os.path.isdir(path):
                continue
            if item_type == "video" and not os.path.isfile(path):
                continue
            items.append((path, task, label, item_type))
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
                # (real video FPS, or the assumed IMAGE_SEQUENCE_FPS) - this
                # is what train_models.py diffs against, so it must be the
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
    path, task, label, item_type = args
    item_name = os.path.basename(os.path.normpath(path))

    if item_type == "video":
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
        cap.release()
        if not fps or fps <= 1:
            fps = 30.0  # sane fallback only if the file has no valid FPS header
        return _run_pose_loop(_video_frame_iterator(path), item_name, task, label, fps)

    elif item_type == "image_sequence":
        return _run_pose_loop(
            _image_sequence_frame_iterator(path), item_name, task, label, IMAGE_SEQUENCE_FPS
        )

    else:
        print(f"[skip] unknown item_type '{item_type}' for {path}")
        return []


def _already_extracted_names(csv_path):
    """video_name values already present in an existing CSV, so re-running
    main() after adding new DATASET_CONFIG rows doesn't reprocess (or
    duplicate) items you already extracted, e.g. the fight dataset."""
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
        print("No videos/image sequences found. Check DATASET_CONFIG paths in batch_extractor.py")
        return

    done = _already_extracted_names(OUTPUT_CSV)
    items = [it for it in all_items if os.path.basename(os.path.normpath(it[0])) not in done]
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