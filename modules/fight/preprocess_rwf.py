"""
CampusGuard Vision - Fight Detection Module
Step 1: Cache RWF-2000 clips as fixed-size frame arrays (16 frames, 128x128).

Save as: modules/fight/preprocess_rwf.py
Run as:  python preprocess_rwf.py

Expected dataset layout (folder-name case doesn't matter):
    RWF-2000/train/Fight/*.avi
    RWF-2000/train/NonFight/*.avi
    RWF-2000/val/Fight/*.avi
    RWF-2000/val/NonFight/*.avi
Resumable: already-cached clips are skipped, so you can stop and re-run.
"""

import os
import cv2
import numpy as np

RWF_DIR = r"D:\hackathon\campusguard_old\data\RWF-2000"
OUT_DIR = r"D:\hackathon\campusguard\data\rwf_cache"
NUM_FRAMES = 16
SIZE = 128
VIDEO_EXT = (".avi", ".mp4", ".mkv", ".mov")


def find_dir(parent, name):
    """Case-insensitive subfolder lookup."""
    for d in os.listdir(parent):
        if d.lower() == name.lower() and os.path.isdir(os.path.join(parent, d)):
            return os.path.join(parent, d)
    return None


def load_clip(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return None
    idx = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
    out = []
    for i in idx:
        f = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        out.append(cv2.resize(f, (SIZE, SIZE)))
    return np.stack(out).astype(np.uint8)  # (16, 128, 128, 3)


def main():
    if not os.path.isdir(RWF_DIR):
        print(f"[ERROR] Dataset folder not found: {RWF_DIR}")
        return

    print("Top-level contents of dataset folder:", os.listdir(RWF_DIR))

    total = 0
    for split in ["train", "val"]:
        split_dir = find_dir(RWF_DIR, split)
        if split_dir is None:
            print(f"[ERROR] No '{split}' folder inside {RWF_DIR}")
            return
        for cls in ["Fight", "NonFight"]:
            cls_dir = find_dir(split_dir, cls)
            if cls_dir is None:
                print(f"[ERROR] No '{cls}' folder inside {split_dir}")
                return

            out_cls = os.path.join(OUT_DIR, split, cls)
            os.makedirs(out_cls, exist_ok=True)

            files = [f for f in os.listdir(cls_dir) if f.lower().endswith(VIDEO_EXT)]
            print(f"\n{split}/{cls}: {len(files)} videos")

            done, failed = 0, 0
            for n, fname in enumerate(files):
                save_path = os.path.join(out_cls, os.path.splitext(fname)[0] + ".npy")
                if os.path.exists(save_path):
                    done += 1
                    continue
                clip = load_clip(os.path.join(cls_dir, fname))
                if clip is None:
                    failed += 1
                    continue
                np.save(save_path, clip)
                done += 1
                if (n + 1) % 100 == 0:
                    print(f"  {n + 1}/{len(files)}")
            print(f"  cached: {done}, failed: {failed}")
            total += done

    print(f"\nDone. Total cached clips: {total} -> {OUT_DIR}")


if __name__ == "__main__":
    main()