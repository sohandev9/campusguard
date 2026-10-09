r"""
CampusGuard Vision - turn the UR Fall dataset (folders of PNG frames) into video files.

Save as: modules\fall\ur_to_video.py   (NEW file)
Run as:  python ur_to_video.py

Writes:
  D:\clips\ur\falls\fall-01.mp4 ... fall-30.mp4   (30 real fall sequences)
  D:\clips\ur\adl\adl-01.mp4  ... adl-40.mp4      (40 normal-activity sequences: walking, sitting, bending, lying down on purpose)
Existing videos are skipped, so it is safe to run twice.
"""

import os
import cv2

UR_DATA_DIR = r"D:\hackathon\campusguard_old\data\UR_data"
OUT_DIR = r"D:\clips\ur"
FPS = 30
IMAGE_EXT = (".png", ".jpg", ".jpeg")
CATEGORIES = {"Fall": "falls", "ADL": "adl"}


def frame_folders(category_dir):
    """Every folder that directly contains image frames (handles the doubly-nested UR layout)."""
    found = []
    for root, _, files in os.walk(category_dir):
        if any(f.lower().endswith(IMAGE_EXT) for f in files):
            found.append(root)
    return sorted(found)


def main():
    total = 0
    for cat, out_name in CATEGORIES.items():
        cat_dir = os.path.join(UR_DATA_DIR, cat)
        if not os.path.isdir(cat_dir):
            print(f"[WARNING] not found: {cat_dir}")
            continue
        out_dir = os.path.join(OUT_DIR, out_name)
        os.makedirs(out_dir, exist_ok=True)
        folders = frame_folders(cat_dir)
        print(f"{cat}: {len(folders)} sequences -> {out_dir}")
        for folder in folders:
            name = os.path.basename(folder).replace("-cam0-rgb", "")
            out_path = os.path.join(out_dir, name + ".mp4")
            if os.path.exists(out_path):
                continue
            frames = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT))
            first = cv2.imread(os.path.join(folder, frames[0]))
            if first is None:
                print(f"  [skipped] {name}: cannot read frames")
                continue
            h, w = first.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            for fn in frames:
                img = cv2.imread(os.path.join(folder, fn))
                if img is not None:
                    writer.write(cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img)
            writer.release()
            total += 1
            print(f"  {name}: {len(frames)} frames")
    print(f"\nDone. {total} videos written under {OUT_DIR}")


if __name__ == "__main__":
    main()