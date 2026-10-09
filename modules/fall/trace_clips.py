r"""
CampusGuard Vision - dump raw tracking + pose data from your clips.

Save as: modules\fall\trace_clips.py   (NEW file)

Writes ONE CSV per video: every tracked person on every frame (box, pose
keypoints, keypoint confidences). No detection or decisions are made here -
the CSVs are only used to design and test a fall rule on your real footage.

Usage (inside modules\fall, campusguard environment active):
  python trace_clips.py --dir D:\clips\falls --out D:\clips\trace\falls
  python trace_clips.py --dir "D:\hackathon\campusguard_old\data\RWF-2000\val\NonFight" --limit 30 --out D:\clips\trace\normal
--dir can be a folder of videos or a single video file.
"""

import os
import csv
import argparse
import numpy as np
import cv2
from ultralytics import YOLO

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
TRACK_CONF = 0.4          # same as the live pipeline, so the traces match what it sees
HERE = os.path.dirname(os.path.abspath(__file__))


def find_pose_weights():
    for p in (os.path.join(HERE, "..", "..", "models", "yolov8n-pose.pt"),
              os.path.join(HERE, "..", "..", "yolov8n-pose.pt"), "yolov8n-pose.pt"):
        if os.path.exists(p):
            return p
    return "yolov8n-pose.pt"


HEADER = (["frame", "t", "id", "x1", "y1", "x2", "y2", "W", "H", "fps", "box_conf"]
          + [f"kx{i}" for i in range(17)] + [f"ky{i}" for i in range(17)] + [f"kc{i}" for i in range(17)])


def trace_video(path, out_csv, weights):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    model = YOLO(weights)                              # fresh model = fresh tracker per clip
    rows = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        stream = model.track(source=path, tracker="botsort.yaml", persist=True, conf=TRACK_CONF,
                             classes=[0], stream=True, verbose=False)
        for idx, r in enumerate(stream):
            if r.boxes is None or r.boxes.id is None:
                continue
            H, W = r.orig_shape
            ids = r.boxes.id.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            bconf = r.boxes.conf.cpu().numpy()
            kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
            kc = None
            if r.keypoints is not None and getattr(r.keypoints, "conf", None) is not None:
                kc = r.keypoints.conf.cpu().numpy()
            for i, tid in enumerate(ids):
                xy = kxy[i] if kxy is not None and i < len(kxy) else np.zeros((17, 2), np.float32)
                c = kc[i] if kc is not None and i < len(kc) else np.zeros(17, np.float32)
                w.writerow([idx, f"{idx / fps:.3f}", int(tid)] + [f"{v:.1f}" for v in xyxy[i]]
                           + [W, H, f"{fps:.2f}", f"{bconf[i]:.3f}"]
                           + [f"{v:.1f}" for v in xy[:, 0]] + [f"{v:.1f}" for v in xy[:, 1]]
                           + [f"{v:.2f}" for v in c])
                rows += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of videos, or one video file")
    ap.add_argument("--out", required=True, help="folder for the CSV files")
    ap.add_argument("--limit", type=int, help="only the first N videos")
    a = ap.parse_args()

    if os.path.isdir(a.dir):
        vids = [os.path.join(a.dir, f) for f in sorted(os.listdir(a.dir)) if f.lower().endswith(VIDEO_EXT)]
    else:
        vids = [a.dir]
    if a.limit:
        vids = vids[:a.limit]
    os.makedirs(a.out, exist_ok=True)

    weights = find_pose_weights()
    for n, p in enumerate(vids, 1):
        stem = os.path.splitext(os.path.basename(p))[0]
        rows = trace_video(p, os.path.join(a.out, stem + ".csv"), weights)
        print(f"[{n}/{len(vids)}] {os.path.basename(p)}: {rows} rows", flush=True)
    print(f"\nDone. CSV files are in {a.out}")


if __name__ == "__main__":
    main()