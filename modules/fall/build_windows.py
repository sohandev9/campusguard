"""
CampusGuard Vision - Fall Detection
Step 3 (fixed labelling): build training windows from raw keypoint sequences.

Save as: modules/fall/build_windows.py   (replaces the old file)
Run as:  python build_windows.py

WHAT WAS WRONG BEFORE
UR Fall's per-frame labels are:   -1 = upright,  0 = the fall itself (30 frames),
                                   1 = lying on the floor (until the clip ends).
The old script marked a window as "fall" if it contained any label-1 frame,
so the model was taught that LYING STILL = fall, while the actual falling
motion (label 0) was taught as "not a fall".

NOW
  positive  = window contains >= MIN_TRANSITION_FRAMES frames of the fall itself
  negative  = no transition frames (upright, or already lying still), all ADL
              windows, plus any hard negatives mined from your own clips
  ambiguous = 1..MIN_TRANSITION_FRAMES-1 transition frames -> dropped
Windows are saved RAW (pixel keypoints); normalisation/augmentation happen in
train_model.py via pose_utils so training and inference share the same code.
"""

import os
import csv
import glob
import numpy as np
from collections import defaultdict

KEYPOINTS_DIR = r"D:\hackathon\campusguard\data\keypoints"
HARD_NEG_DIR = r"D:\hackathon\campusguard\data\hard_negatives"
OUTPUT_DIR = r"D:\hackathon\campusguard\data\windows"
FALL_LABELS_CSV = r"D:\hackathon\campusguard\data\urfall-cam0-falls.csv"

WINDOW_SIZE = 30
STRIDE = 5
MIN_TRANSITION_FRAMES = 15


def load_frame_labels(csv_path):
    labels = defaultdict(dict)
    with open(csv_path, "r") as f:
        for row in csv.reader(f):
            labels[row[0]][int(row[1])] = int(row[2])
    return labels


def clean_seq_name(npy_filename, prefix):
    return npy_filename.replace(prefix, "").replace(".npy", "").replace("-cam0-rgb", "")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fall_labels = load_frame_labels(FALL_LABELS_CSV)
    X, y, ids = [], [], []
    n_fall_pos = n_fall_neg = n_dropped = n_adl = n_hard = 0

    for filename in sorted(os.listdir(KEYPOINTS_DIR)):
        if not filename.endswith(".npy"):
            continue
        seq = np.load(os.path.join(KEYPOINTS_DIR, filename)).astype(np.float32)
        if len(seq) < WINDOW_SIZE:
            continue

        if filename.startswith("fall_"):
            name = clean_seq_name(filename, "fall_")
            if name not in fall_labels:
                print(f"[WARNING] no labels for {name}, skipped")
                continue
            lab = fall_labels[name]
            for s in range(0, len(seq) - WINDOW_SIZE + 1, STRIDE):
                n_trans = sum(1 for fn in range(s + 1, s + WINDOW_SIZE + 1) if lab.get(fn, -1) == 0)
                if n_trans >= MIN_TRANSITION_FRAMES:
                    X.append(seq[s:s + WINDOW_SIZE]); y.append(1); ids.append(name); n_fall_pos += 1
                elif n_trans == 0:
                    X.append(seq[s:s + WINDOW_SIZE]); y.append(0); ids.append(name); n_fall_neg += 1
                else:
                    n_dropped += 1

        elif filename.startswith("adl_"):
            name = clean_seq_name(filename, "adl_")
            for s in range(0, len(seq) - WINDOW_SIZE + 1, STRIDE):
                X.append(seq[s:s + WINDOW_SIZE]); y.append(0); ids.append(name); n_adl += 1

    # hard negatives mined from your own non-fall clips (evaluate_fall.py --mine)
    for path in sorted(glob.glob(os.path.join(HARD_NEG_DIR, "hn_*.npy"))):
        wins = np.load(path).astype(np.float32)          # (M, 30, 34)
        name = os.path.splitext(os.path.basename(path))[0]
        for w in wins:
            X.append(w); y.append(0); ids.append(name); n_hard += 1

    X, y, ids = np.array(X, dtype=np.float32), np.array(y), np.array(ids)
    np.save(os.path.join(OUTPUT_DIR, "X_raw.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "sequence_ids.npy"), ids)

    print(f"Positive (fall motion) windows : {n_fall_pos}")
    print(f"Negative from fall clips       : {n_fall_neg}  (upright / already lying still)")
    print(f"Negative from ADL clips        : {n_adl}")
    print(f"Negative from your hard-negs   : {n_hard}")
    print(f"Dropped as ambiguous           : {n_dropped}")
    print(f"TOTAL {len(X)} windows, {len(np.unique(ids))} source clips -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
