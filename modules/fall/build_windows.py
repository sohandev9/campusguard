"""
CampusGuard Vision - Fall Detection Module
Step 3 (updated): Build fixed-length labeled training windows, with
normalized keypoints to fix the domain-gap issue found during testing.

Save as: modules/fall/build_windows.py
Run as:  python build_windows.py
"""

import os
import csv
import numpy as np
from collections import defaultdict
from pose_utils import normalize_keypoints

KEYPOINTS_DIR = r"D:\hackathon\campusguard\data\keypoints"
OUTPUT_DIR = r"D:\hackathon\campusguard\data\windows"
FALL_LABELS_CSV = r"D:\hackathon\campusguard\data\urfall-cam0-falls.csv"
WINDOW_SIZE = 30
STRIDE = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_frame_labels(csv_path):
    labels = defaultdict(dict)
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            seq_name, frame_num, label = row[0], int(row[1]), int(row[2])
            labels[seq_name][frame_num] = label
    return labels


def clean_seq_name(npy_filename, prefix):
    name = npy_filename.replace(prefix, "").replace(".npy", "")
    name = name.replace("-cam0-rgb", "")
    return name


def normalize_sequence(seq):
    """Applies normalize_keypoints to every frame in a (num_frames, 34) sequence."""
    return np.array([normalize_keypoints(frame) for frame in seq], dtype=np.float32)


def main():
    fall_labels = load_frame_labels(FALL_LABELS_CSV)

    X, y, sequence_ids = [], [], []

    for filename in sorted(os.listdir(KEYPOINTS_DIR)):
        if not filename.startswith("fall_"):
            continue

        seq_name = clean_seq_name(filename, "fall_")
        seq = np.load(os.path.join(KEYPOINTS_DIR, filename))
        seq = normalize_sequence(seq)

        if seq_name not in fall_labels:
            print(f"[WARNING] No label data for {seq_name}, skipping")
            continue

        frame_label_map = fall_labels[seq_name]

        if len(seq) < WINDOW_SIZE:
            print(f"[SKIPPED] {seq_name}: shorter than window size")
            continue

        for start in range(0, len(seq) - WINDOW_SIZE + 1, STRIDE):
            window = seq[start:start + WINDOW_SIZE]
            window_frame_nums = range(start + 1, start + WINDOW_SIZE + 1)
            window_labels = [frame_label_map.get(fn, -1) for fn in window_frame_nums]
            is_fall_window = 1 if 1 in window_labels else 0

            X.append(window)
            y.append(is_fall_window)
            sequence_ids.append(seq_name)

    for filename in sorted(os.listdir(KEYPOINTS_DIR)):
        if not filename.startswith("adl_"):
            continue

        seq_name = clean_seq_name(filename, "adl_")
        seq = np.load(os.path.join(KEYPOINTS_DIR, filename))
        seq = normalize_sequence(seq)

        if len(seq) < WINDOW_SIZE:
            print(f"[SKIPPED] {seq_name}: shorter than window size")
            continue

        for start in range(0, len(seq) - WINDOW_SIZE + 1, STRIDE):
            window = seq[start:start + WINDOW_SIZE]
            X.append(window)
            y.append(0)
            sequence_ids.append(seq_name)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    sequence_ids = np.array(sequence_ids)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "sequence_ids.npy"), sequence_ids)

    print(f"\nTotal windows: {len(X)}")
    print(f"Fall windows: {(y == 1).sum()}")
    print(f"Not-fall windows: {(y == 0).sum()}")
    print(f"Unique sequences: {len(np.unique(sequence_ids))}")


if __name__ == "__main__":
    main()