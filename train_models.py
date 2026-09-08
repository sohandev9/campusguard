"""
train_models.py
----------------
Reads master_training_data.csv (raw landmark coords, one row per
video/frame/person) and:

  1. Groups rows by (video_name, person_id) and sorts by frame_idx, so
     velocity is only ever diffed between two rows that are actually the
     SAME tracked person in the SAME video (fixes identity-switch bleed
     and prevents diffing across a gap where a person left frame).
  2. Computes dt = curr_timestamp - prev_timestamp (real seconds, derived
     from the video's true FPS in batch_extractor.py) and calls
     features.compute_velocity() - the EXACT same function app.py calls
     at inference time. This is what eliminates train/serve skew.
  3. Feeds each valid velocity sample into a WINDOW_SECONDS-wide rolling
     window (features.update_window) and, once the window has enough
     history, computes mean/std/max statistics over it
     (features.compute_window_stats) - this is the actual training row.
     A single instantaneous velocity reading can't tell a punch from a
     wave, or a fall from sitting down fast; the windowed statistics can.
  4. Resets the window whenever compute_velocity() returns None
     (teleportation jump, missing prior frame, or >1s gap), since
     continuing to average across that discontinuity would corrupt the
     window's statistics - the SAME reset rule app.py's live loop follows.
  5. Trains two independent Random Forest classifiers using NESTED
     cross-validation, split by VIDEO (GroupShuffleSplit on video_name)
     at every level, never by row. This matters because the sliding window
     advances one frame at a time, so consecutive windowed rows from the
     same clip overlap almost entirely - a plain row-level split would leak
     near-duplicate windows across train/val/test and report inflated
     accuracy that doesn't reflect performance on genuinely unseen footage.
     For each of N_CV_SPLITS outer splits: fit on an inner-train slice,
     pick the F1-optimal decision threshold on an inner-val slice (never
     the same data used to report the score), then score on the outer-test
     slice using that threshold. Repeating across multiple splits (rather
     than trusting one split) turns a single possibly-lucky/unlucky number
     into a mean +/- std you can actually trust - especially with the fall
     model's small clip count. The deployed model is then refit on ALL
     available clips (more data is strictly better for production; the CV
     loop already gave the honest generalization estimate) with a decision
     threshold averaged across the per-fold tuned thresholds.
       - fight_detector_model.pkl  (arm/body kinematics)
       - fall_detector_model.pkl   (downward hip/shoulder kinematics)
     Bag/YOLO data is never touched here - no data leakage.

Each .pkl stores {"model": clf, "feature_columns": [...], "threshold": ...}
so app.py can build its feature vector in the exact right order and apply
the right decision threshold without hardcoding either.
"""

import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, precision_recall_curve, f1_score

import features

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "master_training_data.csv")

N_CV_SPLITS = 5  # outer splits, each an independent train/val/test partition by CLIP


def compute_windowed_dataset(df):
    """Returns a DataFrame of windowed-stat feature rows (one per frame
    once each person's rolling window has enough history), each tagged
    with task/label/video_name."""
    stat_rows = []

    df = df.sort_values(["video_name", "person_id", "frame_idx"])
    grouped = df.groupby(["video_name", "person_id"], sort=False)

    for (video_name, person_id), group in grouped:
        prev_coords = None
        prev_time = None
        window = []  # list of (timestamp, velocity_dict), see features.update_window

        for _, row in group.iterrows():
            curr_coords = {col: row[col] for col in features.RAW_COLUMNS}
            curr_time = row["timestamp"]

            dt = None if prev_time is None else (curr_time - prev_time)
            velocity = features.compute_velocity(prev_coords, curr_coords, dt)

            if velocity is None:
                window = []  # discontinuity - don't average across it
            else:
                velocity[features.TORSO_TILT_COLUMN] = features.compute_torso_tilt_degrees(curr_coords)
                features.update_window(window, curr_time, velocity)
                stats = features.compute_window_stats(window)
                if stats is not None:
                    stats["video_name"] = video_name
                    stats["person_id"] = person_id
                    stats["task"] = row["task"]
                    stats["label"] = row["label"]
                    stat_rows.append(stats)

            prev_coords = curr_coords
            prev_time = curr_time

    return pd.DataFrame(stat_rows)


def _make_classifier():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def _best_threshold(y_val, val_proba):
    """F1-optimal decision threshold, found on a VALIDATION slice that is
    never the same data used for final reporting - tuning on the same set
    you report on is a leak, just like a row-level train/test split was."""
    if len(np.unique(y_val)) < 2:
        return 0.5  # degenerate val fold (rare with small clip counts) - fall back to default
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def train_one_model(stats_df, task_name, feature_columns, output_path):
    task_df = stats_df[stats_df["task"] == task_name].reset_index(drop=True)
    if task_df.empty:
        print(f"[skip] no rows for task='{task_name}' - check DATASET_CONFIG / CSV")
        return

    X = task_df[feature_columns].values
    y = task_df["label"].values
    groups = task_df["video_name"].values  # split by CLIP, not by row - see module docstring
    n_videos = task_df["video_name"].nunique()

    # Outer loop: N_CV_SPLITS independent train+val / test partitions, split
    # by clip. Test is only ever touched once, at the very end of each fold,
    # after the threshold has already been chosen from val - never the
    # reverse. Repeating this across multiple splits (Fix 8) turns a single
    # possibly-lucky/unlucky number into a mean +/- std you can trust.
    outer = GroupShuffleSplit(n_splits=N_CV_SPLITS, test_size=0.2, random_state=42)
    fold_f1s, fold_thresholds = [], []
    last_fold_diagnostics = None

    for fold_i, (trainval_idx, test_idx) in enumerate(outer.split(X, y, groups=groups)):
        X_trainval, y_trainval, groups_trainval = X[trainval_idx], y[trainval_idx], groups[trainval_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Inner split: carve the threshold-tuning validation slice out of
        # trainval only - test_idx above is never seen until final scoring.
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=100 + fold_i)
        train_idx, val_idx = next(inner.split(X_trainval, y_trainval, groups=groups_trainval))
        X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
        X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

        clf = _make_classifier()
        clf.fit(X_train, y_train)

        threshold = _best_threshold(y_val, clf.predict_proba(X_val)[:, 1])

        test_proba = clf.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= threshold).astype(int)
        fold_f1 = f1_score(y_test, test_pred)
        fold_f1s.append(fold_f1)
        fold_thresholds.append(threshold)
        print(f"  [{task_name}] fold {fold_i + 1}/{N_CV_SPLITS}: "
              f"threshold={threshold:.3f}  test F1={fold_f1:.3f}")

        if fold_i == N_CV_SPLITS - 1:
            last_fold_diagnostics = (y_test, test_pred, task_df.iloc[test_idx]["video_name"].values)
            print(f"\n=== {task_name} model report (fold {fold_i + 1}, held-out CLIPS) ===")
            print(classification_report(y_test, test_pred, digits=3))

    mean_f1, std_f1 = float(np.mean(fold_f1s)), float(np.std(fold_f1s))
    print(f"[{task_name}] {n_videos} clips total -> "
          f"{N_CV_SPLITS}-fold F1 = {mean_f1:.3f} +/- {std_f1:.3f}")

    # Which specific clips does the last fold still get wrong? (one fold
    # shown as a representative sample, not aggregated across folds, since
    # GroupShuffleSplit folds can overlap and double-counting would mislead)
    if last_fold_diagnostics is not None:
        y_test, test_pred, test_video_names = last_fold_diagnostics
        wrong_mask = test_pred != y_test
        if wrong_mask.any():
            counts = Counter(test_video_names[wrong_mask])
            print(f"Top misclassified clips, last fold ({wrong_mask.sum()} wrong windows "
                  f"across {len(counts)} clips):")
            for name, cnt in counts.most_common(10):
                print(f"  {name}: {cnt} misclassified windows")

    # Deployed model: refit on ALL available clips (not held back), since
    # more data is strictly better for the production model - the CV loop
    # above already gave us an honest estimate of how it generalizes, we
    # don't need to sacrifice training data to prove that again. Deployed
    # threshold is the average of the per-fold tuned thresholds, which is
    # less sensitive to any single val split's noise than any one fold's.
    final_threshold = float(np.mean(fold_thresholds))
    final_clf = _make_classifier()
    final_clf.fit(X, y)

    joblib.dump(
        {"model": final_clf, "feature_columns": feature_columns, "threshold": final_threshold},
        output_path,
    )
    print(f"Saved {output_path} (threshold={final_threshold:.3f}, fit on all {n_videos} clips)\n")


def main():
    print(f"Loading {INPUT_CSV} ...")
    # latin-1: some source video filenames (corrupted RWF-2000 downloads)
    # contain non-UTF8 bytes. on_bad_lines='skip': a handful of those same
    # corrupted filenames contain stray comma/quote characters that break
    # the CSV's column structure - those rows are junk from broken source
    # files anyway (0 pose rows in practice), so skipping them is safe.
    raw_df = pd.read_csv(INPUT_CSV, encoding='latin-1', on_bad_lines='skip')
    print(f"Loaded {len(raw_df)} rows")

    print(f"Computing windowed velocity statistics "
          f"(window={features.WINDOW_SECONDS}s, grouped by video + person_id) ...")
    stats_df = compute_windowed_dataset(raw_df)
    print(f"{len(stats_df)} windowed feature rows "
          f"(each needs >= {features.MIN_WINDOW_SAMPLES} samples of continuous motion history)")

    train_one_model(stats_df, "fight", features.FIGHT_FEATURE_COLUMNS, os.path.join(BASE_DIR, "fight_detector_model.pkl"))
    train_one_model(stats_df, "fall", features.FALL_FEATURE_COLUMNS, os.path.join(BASE_DIR, "fall_detector_model.pkl"))


if __name__ == "__main__":
    main()