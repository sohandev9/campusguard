r"""
CampusGuard Vision - Fall Detection
Measure false alarms and detections on YOUR clips, instead of eyeballing.

Save as: modules/fall/evaluate_fall.py   (NEW file; replaces visualize_fall_detection.py)

Usage (run inside modules/fall):
  python evaluate_fall.py --normal_dir D:\clips\normal --fall_dir D:\clips\falls
  python evaluate_fall.py --normal_dir D:\clips\normal --save_videos D:\clips\out
  python evaluate_fall.py --normal_dir D:\clips\normal --mine        # save false alarms as hard negatives

normal_dir = clips where NOBODY falls (walking, sitting, fights, crowds, overhead views...)
fall_dir   = clips with a real fall
"""

import os
import argparse
import numpy as np
import cv2
from ultralytics import YOLO

from fall_detector import FallDetector

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
DEFAULT_MODEL = r"D:\hackathon\campusguard\models\fall_lstm.pt"
HARD_NEG_DIR = r"D:\hackathon\campusguard\data\hard_negatives"
TRACK_CONF = 0.4
MINE_PROB = 0.5
MINE_GAP_FRAMES = 10


def list_videos(folder, limit=None):
    if not folder:
        return []
    vids = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(VIDEO_EXT)]
    return vids[:limit] if limit else vids


def run_video(path, model_path, save_dir, mine):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    detector = FallDetector(model_path, fps=fps, diagnostic=True)
    pose = YOLO("yolov8n-pose.pt")                      # fresh model = fresh tracker per clip
    stem = os.path.splitext(os.path.basename(path))[0]
    writer = None
    events, mined, last_mined, max_raw, n_frames, tracks = [], [], {}, 0.0, 0, {}

    stream = pose.track(source=path, tracker="botsort.yaml", persist=True, conf=TRACK_CONF,
                        classes=[0], stream=True, verbose=False)
    for idx, r in enumerate(stream):
        n_frames = idx + 1
        H, W = r.orig_shape
        frame = r.orig_img.copy() if save_dir else None

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            boxes = r.boxes.xyxy.cpu().numpy()
            kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
            kconf = None
            if r.keypoints is not None and getattr(r.keypoints, "conf", None) is not None:
                kconf = r.keypoints.conf.cpu().numpy()

            for i, tid in enumerate(ids):
                kp = kxy[i] if kxy is not None and i < len(kxy) else np.zeros((17, 2), np.float32)
                kc = kconf[i] if kconf is not None and i < len(kconf) else None
                res = detector.update(tid, boxes[i], kp, kc, idx, (W, H))
                st = tracks.setdefault(int(tid), {"first": idx / fps, "last": idx / fps, "n": 0,
                                                  "max": 0.0, "max_t": 0.0, "reasons": {}})
                st["last"] = idx / fps
                if res["prob"] is not None:
                    st["n"] += 1
                    st["reasons"][res["reason"]] = st["reasons"].get(res["reason"], 0) + 1
                    if res["prob"] > st["max"]:
                        st["max"], st["max_t"] = res["prob"], idx / fps

                if res["prob"] is not None:
                    max_raw = max(max_raw, res["prob"])
                    if mine and res["prob"] >= MINE_PROB and idx - last_mined.get(tid, -10**9) >= MINE_GAP_FRAMES:
                        mined.append(res["raw_window"]); last_mined[tid] = idx
                if res["new_alert"]:
                    events.append((idx / fps, int(tid), res["last_prob"]))

                if frame is not None:
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    color = (0, 0, 255) if res["alert"] else (0, 200, 0)
                    txt = f"ID:{tid} FALL {res['last_prob']:.2f}" if res["alert"] else f"ID:{tid} p={res['last_prob']:.2f} {res['reason']}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, txt, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        detector.prune(idx)

        if frame is not None:
            if writer is None:
                os.makedirs(save_dir, exist_ok=True)
                writer = cv2.VideoWriter(os.path.join(save_dir, stem + "_fall.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            writer.write(frame)
    if writer is not None:
        writer.release()

    if mine and mined:
        os.makedirs(HARD_NEG_DIR, exist_ok=True)
        np.save(os.path.join(HARD_NEG_DIR, f"hn_{stem}.npy"), np.stack(mined))

    return {"name": os.path.basename(path), "dur": n_frames / fps, "events": events,
            "max_raw": max_raw, "gates": dict(detector.gate_counts), "mined": len(mined), "tracks": tracks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_dir")
    ap.add_argument("--fall_dir")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--save_videos")
    ap.add_argument("--tracks", action="store_true", help="also print per-person detail for every clip")
    ap.add_argument("--limit", type=int, help="only the first N videos of each folder")
    ap.add_argument("--mine", action="store_true", help="save high-probability windows from NORMAL clips as hard negatives")
    a = ap.parse_args()

    rows = []
    for kind, folder in (("NORMAL", a.normal_dir), ("FALL", a.fall_dir)):
        for p in list_videos(folder, a.limit):
            print(f"processing {kind}: {os.path.basename(p)} ...", flush=True)
            r = run_video(p, a.model, a.save_videos, a.mine and kind == "NORMAL")
            r["kind"] = kind
            rows.append(r)

    print("\n=== RESULTS (paste this) ===")
    print(f"{'clip':<30}{'type':<8}{'sec':>6}{'alerts':>8}{'first@s':>9}{'maxProb':>9}  gates/mined")
    for r in rows:
        first = f"{r['events'][0][0]:.1f}" if r["events"] else "-"
        extra = ", ".join(f"{k}={v}" for k, v in r["gates"].items())
        if r["mined"]:
            extra += f"  mined={r['mined']}"
        print(f"{r['name'][:29]:<30}{r['kind']:<8}{r['dur']:>6.1f}{len(r['events']):>8}{first:>9}{r['max_raw']:>9.2f}  {extra}")

    if a.tracks:
        print("\n=== PER-PERSON DETAIL ===")
        for r in rows:
            print(f"{r['name']} ({r['kind']})")
            for tid, st in sorted(r["tracks"].items()):
                why = ", ".join(f"{k}={v}" for k, v in st["reasons"].items())
                print(f"   ID:{tid:<3} seen {st['first']:.1f}-{st['last']:.1f}s  checks={st['n']:<4} "
                      f"maxProb={st['max']:.2f} at {st['max_t']:.1f}s   {why}")

    normal = [r for r in rows if r["kind"] == "NORMAL"]
    fall = [r for r in rows if r["kind"] == "FALL"]
    if normal:
        fp = sum(len(r["events"]) for r in normal)
        mins = sum(r["dur"] for r in normal) / 60.0
        bad = sum(1 for r in normal if r["events"])
        print(f"\nNORMAL clips: {bad}/{len(normal)} had false alerts | {fp} false alerts in {mins:.1f} min "
              f"({fp / max(mins, 1e-6):.2f} per minute)")
    if fall:
        hit = sum(1 for r in fall if r["events"])
        print(f"FALL clips  : fall detected in {hit}/{len(fall)}")


if __name__ == "__main__":
    main()
