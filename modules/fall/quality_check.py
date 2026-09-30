import numpy as np
import os

keypoints_dir = r"D:\hackathon\campusguard\data\keypoints"

for filename in sorted(os.listdir(keypoints_dir)):
    if filename.startswith("fall_"):
        seq = np.load(os.path.join(keypoints_dir, filename))
        valid = np.any(seq != 0, axis=1).sum()
        total = len(seq)
        pct = 100 * valid / total if total > 0 else 0
        print(f"{filename}: {valid}/{total} frames valid ({pct:.0f}%)")
