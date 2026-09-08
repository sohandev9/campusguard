"""
debug_fight_detection.py
-------------------------
Run this against ONE known fight video to see, frame by frame:
  - how many people MediaPipe actually detected (avg, min, max)
  - whether each tracked person's rolling window ever reached
    MIN_WINDOW_SAMPLES (i.e. whether the fight classifier was even
    INVOKED for them), or kept resetting before it could
  - per-frame processing time -> effective FPS after inference

Usage:
    python debug_fight_detection.py path/to/one_fight_clip.mp4
    python debug_fight_detection.py path/to/clip.mp4 --num_poses 8 --conf 0.3

This does NOT touch your trained models or app.py - it is read-only
diagnostics using the same features.py your real pipeline uses.
"""

import sys
import time
import argparse
from collections import defaultdict, Counter

import cv2
import mediapipe as mp_mediapipe
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import features

POSE_MODEL_PATH = "pose_landmarker_heavy.task"


def parse_args():
    p = argparse.ArgumentParser(description="Debug fight detection pipeline on one clip")
    p.add_argument("video_path", help="Path to a single fight or normal clip")
    p.add_argument("--num_poses", type=int, default=4,
                   help="MediaPipe num_poses (default: 4, try 8 for crowded scenes)")
    p.add_argument("--conf", type=float, default=0.5,
                   help="min_pose_detection/presence confidence (default: 0.5)")
    return p.parse_args()


def main():
    args = parse_args()

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=args.num_poses,
        min_pose_detection_confidence=args.conf,
        min_pose_presence_confidence=args.conf,
        min_tracking_confidence=args.conf,
    )
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open {args.video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = features.PersonTracker()
    prev_coords = {}
    windows = defaultdict(list)

    window_reached_count = defaultdict(int)
    window_reset_count = defaultdict(int)
    pose_counts = []
    frame_times = []

    print(f"Reading {args.video_path}")
    print(f"  Reported FPS: {fps:.1f}, Total frames: {total_frames}")
    print(f"  num_poses={args.num_poses}, min_confidence={args.conf}")
    print()

    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        timestamp = frame_idx / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_mediapipe.Image(image_format=mp_mediapipe.ImageFormat.SRGB, data=rgb)
        result = pose_landmarker.detect(mp_image)

        t1 = time.perf_counter()
        frame_times.append(t1 - t0)

        n_detected = len(result.pose_landmarks) if result.pose_landmarks else 0
        pose_counts.append(n_detected)

        if n_detected > 0:
            coords_list = [features.extract_landmark_coords(pl) for pl in result.pose_landmarks]
            centroids = [features.hip_centroid(c) for c in coords_list]
            person_ids = tracker.update(centroids)

            for coords, pid in zip(coords_list, person_ids):
                prev = prev_coords.get(pid)
                prev_c, prev_t = prev if prev else (None, None)
                dt = None if prev_t is None else (timestamp - prev_t)

                velocity = features.compute_velocity(prev_c, coords, dt)
                prev_coords[pid] = (coords, timestamp)

                if velocity is None:
                    if windows[pid]:
                        window_reset_count[pid] += 1
                    windows[pid] = []
                else:
                    velocity[features.TORSO_TILT_COLUMN] = features.compute_torso_tilt_degrees(coords)
                    features.update_window(windows[pid], timestamp, velocity)
                    stats = features.compute_window_stats(windows[pid])
                    if stats is not None:
                        window_reached_count[pid] += 1

        frame_idx += 1

    cap.release()

    # --- Report ---
    if not pose_counts:
        print("No frames processed.")
        return

    avg_pose = sum(pose_counts) / len(pose_counts)
    max_pose = max(pose_counts)
    min_pose = min(pose_counts)
    hist = Counter(pose_counts)

    print(f"Frames processed: {len(pose_counts)}")
    print(f"Pose detection per frame: min={min_pose}, max={max_pose}, avg={avg_pose:.2f}")
    print(f"Detection histogram (count -> #frames): {dict(sorted(hist.items()))}")

    avg_proc_ms = (sum(frame_times) / len(frame_times)) * 1000
    effective_fps = 1.0 / (sum(frame_times) / len(frame_times))
    print(f"Per-frame inference: avg={avg_proc_ms:.1f}ms, effective FPS={effective_fps:.1f}")

    all_pids = set(list(window_reached_count.keys()) + list(window_reset_count.keys()))
    print(f"\nUnique person_ids tracked: {len(all_pids)}")

    print("\nPer-person window status (was the classifier invoked?):")
    for pid in sorted(all_pids):
        reached = window_reached_count.get(pid, 0)
        reset = window_reset_count.get(pid, 0)
        total = reached + reset
        pct = (reached / total * 100) if total > 0 else 0
        status = "OK" if pct > 50 else "PROBLEM"
        print(f"  person {pid}: reached_full={reached}, reset_early={reset}, "
              f"hit_rate={pct:.1f}% [{status}]")

    # Summary diagnosis
    print("\n--- DIAGNOSIS ---")
    if avg_pose < 2:
        print("LOW DETECTION: avg people detected below 2. "
              "Try --num_poses 8 --conf 0.3 to catch more people in dense scenes.")
    else:
        print(f"Detection looks OK (avg={avg_pose:.1f} people/frame).")

    problem_pids = [
        pid for pid in sorted(all_pids)
        if window_reached_count.get(pid, 0) == 0 and window_reset_count.get(pid, 0) > 0
    ]
    if problem_pids:
        print(f"TRACKING BREAKS ON {len(problem_pids)} person(s) - window never reached "
              f"full before resetting. These people were never classified by the fight model.")
    else:
        print("At least some windows reached full status - classifier was invoked.")


if __name__ == "__main__":
    main()
