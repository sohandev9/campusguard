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
  5. Trains two independent Random Forest classifiers, splitting train/test
     by VIDEO (GroupShuffleSplit on video_name), never by row. This matters
     because the sliding window advances one frame at a time, so
     consecutive windowed rows from the same clip overlap almost entirely -
     a plain row-level random split would leak near-duplicate windows into
     both train and test and report inflated accuracy that doesn't reflect
     performance on genuinely unseen footage.
       - fight_detector_model.pkl  (arm/body kinematics)
       - fall_detector_model.pkl   (downward hip/shoulder kinematics)
     Bag/YOLO data is never touched here - no data leakage.

Each .pkl stores {"model": clf, "feature_columns": [...]} so app.py can
build its feature vector in the exact right order without hardcoding it
twice.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report

import features

# --- UPDATED ABSOLUTE PATHS ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "master_training_data.csv")


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


def train_one_model(stats_df, task_name, feature_columns, output_path):
    task_df = stats_df[stats_df["task"] == task_name]
    if task_df.empty:
        print(f"[skip] no rows for task='{task_name}' - check DATASET_CONFIG / CSV")
        return

    X = task_df[feature_columns].values
    y = task_df["label"].values
    groups = task_df["video_name"].values  # split by CLIP, not by row - see module docstring

    n_videos = task_df["video_name"].nunique()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_videos = set(groups[train_idx])
    test_videos = set(groups[test_idx])
    print(f"[{task_name}] {n_videos} clips total -> {len(train_videos)} train clips, "
          f"{len(test_videos)} test clips (no overlap: "
          f"{len(train_videos & test_videos) == 0})")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\n=== {task_name} model report (held-out CLIPS, never seen during training) ===")
    print(classification_report(y_test, y_pred, digits=3))

    # Which specific clips is it still getting wrong? video_name encodes the
    # activity (e.g. "Subject.3_Sit_down", "Subject.3_Fall_backwards"), so
    # this shows you exactly which activities the model confuses - e.g.
    # whether "sit down" / "kneel" / "pick up object" are still being
    # mistaken for falls, or whether that's actually resolved.
    test_video_names = task_df.iloc[test_idx]["video_name"].values
    wrong_mask = y_pred != y_test
    if wrong_mask.any():
        from collections import Counter
        counts = Counter(test_video_names[wrong_mask])
        print(f"Top misclassified clips ({wrong_mask.sum()} wrong windows across "
              f"{len(counts)} clips):")
        for name, cnt in counts.most_common(10):
            print(f"  {name}: {cnt} misclassified windows")

    joblib.dump({"model": clf, "feature_columns": feature_columns}, output_path)
    print(f"Saved {output_path}")


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