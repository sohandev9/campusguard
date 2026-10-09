r"""
CampusGuard Vision - run the whole pipeline on a video and SEE when alarms fire.

Save as: D:\hackathon\campusguard\run_video.py      (repo root, next to main.py)

Usage (from the repo root, with the campusguard environment active):
  python run_video.py --video D:\clips\falls\clip1.mp4
  python run_video.py --video D:\clips\normal                 (every video in a folder)
  python run_video.py --video clip.mp4 --no-bags              (switch a module off)

For every video it writes two files into  data\out\ :
  <clip>_alerts.mp4  annotated video: clock in the corner, a live "ALERT LOG" panel,
                     red boxes on fallers, orange frame + banner during a fight,
                     boxes on bags that are left alone
  <clip>_alerts.csv  every alarm with its time in seconds
and it prints the same list in the terminal, e.g.
  00:04.9   FALL    ID:3  p=0.97

Standalone: does not import main.py or the backend. Fall detection uses the rule-based
detector in modules\fall\fall_rule.py (torso flat + wide box, held ~0.5 s).
Under every person the video shows:  ang = torso angle (0 upright, 90 flat),
ar = box width/height, s = how "lying" the last 0.5 s were (alert fires at s >= 0.8).
"""

import os
import sys
import csv
import argparse
from collections import deque

import numpy as np
import cv2

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "modules", "fall"))
from fall_rule import FallRule  # noqa: E402

MODELS_DIR = os.path.join(REPO_ROOT, "models")
FIGHT_MODEL_PATH = os.path.join(MODELS_DIR, "fight_r2plus1d.pt")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data", "out")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")

# ---------------- settings ----------------
TRACK_CONF = 0.4
# fight (same values as the validated main.py)
FIGHT_WINDOW_SECONDS = 5.0
FIGHT_CHECK_EVERY_SECONDS = 0.5
FIGHT_THRESHOLD = 0.5
FIGHT_CONSECUTIVE = 2
FIGHT_NUM_FRAMES = 16
FIGHT_SIZE = 128
FIGHT_CROP = 112
# bags
BAG_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}
BAG_CONF = 0.35
STATIONARY_PIXELS = 25
NEARBY_PERSON_PX = 150
ABANDON_SECONDS = 8.0
# colours (BGR)
RED, ORANGE, YELLOW, GREEN, WHITE = (0, 0, 255), (0, 140, 255), (0, 215, 255), (0, 200, 0), (255, 255, 255)
KIND_COLOR = {"FALL": RED, "FIGHT": ORANGE, "BAG": YELLOW}
# ------------------------------------------


def find_weights(name):
    for p in (os.path.join(MODELS_DIR, name), os.path.join(REPO_ROOT, name)):
        if os.path.exists(p):
            return p
    return name                                  # let ultralytics download it


def fmt_time(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


# ---------------- alert log ----------------
class AlertLog:
    def __init__(self):
        self.items = []                          # (time_s, kind, detail)

    def add(self, t, kind, detail):
        self.items.append((float(t), kind, detail))
        print(f"  {fmt_time(t)}   {kind:<6} {detail}", flush=True)

    def latest(self, n):
        return self.items[-n:]

    def save_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "type", "detail"])
            for t, k, d in self.items:
                w.writerow([f"{t:.2f}", k, d])

    def counts(self):
        out = {}
        for _, k, _ in self.items:
            out[k] = out.get(k, 0) + 1
        return out


# ---------------- fight ----------------
def make_fight_predictor():
    """Loads the R(2+1)D fight model once; returns predict(frames_small_rgb_list) -> prob."""
    import torch
    import torch.nn as nn
    from torchvision.models.video import r2plus1d_18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(FIGHT_MODEL_PATH, map_location=device))
    model = model.to(device).eval()
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)

    def predict(frames_small):
        idx = np.linspace(0, len(frames_small) - 1, FIGHT_NUM_FRAMES).astype(int)
        x = torch.from_numpy(np.stack([frames_small[i] for i in idx])).float() / 255.0
        x = x.permute(3, 0, 1, 2)
        off = (FIGHT_SIZE - FIGHT_CROP) // 2
        x = x[:, :, off:off + FIGHT_CROP, off:off + FIGHT_CROP]
        x = ((x - mean) / std).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            return torch.softmax(model(x).float(), dim=1)[0, 1].item()

    return predict


class FightDetector:
    def __init__(self, predict_fn, fps):
        self.predict = predict_fn
        win = int(round(FIGHT_WINDOW_SECONDS * fps))
        self.buf = deque(maxlen=win)
        self.min_frames = max(FIGHT_NUM_FRAMES, int(win * 0.5))
        self.every = max(1, int(round(FIGHT_CHECK_EVERY_SECONDS * fps)))
        self.prob, self.streak, self.active = 0.0, 0, False

    def update(self, frame_bgr, idx):
        """returns (prob, active, started, ended)"""
        self.buf.append(cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (FIGHT_SIZE, FIGHT_SIZE)))
        started = ended = False
        if len(self.buf) >= self.min_frames and idx % self.every == 0:
            self.prob = self.predict(list(self.buf))
            self.streak = self.streak + 1 if self.prob >= FIGHT_THRESHOLD else 0
            now = self.streak >= FIGHT_CONSECUTIVE
            started, ended = now and not self.active, self.active and not now
            self.active = now
        return self.prob, self.active, started, ended


# ---------------- bags ----------------
class BagWatcher:
    """A bag is 'unattended' once nobody has been near it AND it has not moved for ABANDON_SECONDS."""

    def __init__(self, fps):
        self.fps, self.bags, self.next_id, self.frame = fps, {}, 0, 0

    def update(self, bag_boxes, person_boxes):
        """returns list of (bag_id, box, is_unattended, seconds_alone)"""
        pcs = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in person_boxes]
        out, seen = [], set()
        for box in bag_boxes:
            x1, y1, x2, y2 = box
            c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            bid = None
            for k, b in self.bags.items():
                if k not in seen and np.linalg.norm(c - b["c"]) < STATIONARY_PIXELS * 3:
                    bid = k
                    break
            if bid is None:
                bid, self.next_id = self.next_id, self.next_id + 1
                self.bags[bid] = {"c": c, "still": self.frame, "owner": self.frame}
            seen.add(bid)
            b = self.bags[bid]
            if np.linalg.norm(c - b["c"]) > STATIONARY_PIXELS:
                b["still"] = self.frame
            b["c"] = c
            if any(np.linalg.norm(c - np.array(pc)) < NEARBY_PERSON_PX for pc in pcs):
                b["owner"] = self.frame
            alone = (self.frame - b["owner"]) / self.fps
            still = (self.frame - b["still"]) / self.fps
            out.append((bid, box, alone >= ABANDON_SECONDS and still >= ABANDON_SECONDS, alone))
        for k in [k for k in self.bags if k not in seen]:
            del self.bags[k]
        self.frame += 1
        return out


# ---------------- drawing ----------------
def draw_panel(frame, t, log, fight_prob, any_active):
    H, W = frame.shape[:2]
    if any_active:
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1), RED, max(6, W // 160))
    lines = [(f"t = {fmt_time(t)}", WHITE)]
    recent = log.latest(5)
    lines.append(("ALERT LOG" if recent else "no alerts yet", (200, 200, 200)))
    for at, kind, detail in recent:
        lines.append((f"{fmt_time(at)}  {kind}  {detail}", KIND_COLOR.get(kind, WHITE)))
    scale = max(0.5, W / 1400)
    lh = int(30 * scale) + 6
    pw = int(min(W - 10, 560 * scale))
    ph = lh * len(lines) + 10
    roi = frame[5:5 + ph, 5:5 + pw]
    if roi.size:
        frame[5:5 + ph, 5:5 + pw] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (12, 5 + lh * (i + 1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, col, 2 if i else 1)
    cv2.putText(frame, f"fight_prob {fight_prob:.2f}", (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * scale, ORANGE if fight_prob >= FIGHT_THRESHOLD else GREEN, 2)


# ---------------- main per-video routine ----------------
def process_video(path, out_dir, yolo_cls, pose_weights, object_model, fight_predict,
                  use_fall, use_fight, use_bags):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stem = os.path.splitext(os.path.basename(path))[0]
    print(f"\n=== {os.path.basename(path)}  ({fps:.0f} fps) ===", flush=True)

    fall = FallRule(fps=fps) if use_fall else None
    fight = FightDetector(fight_predict, fps) if use_fight else None
    bags = BagWatcher(fps) if use_bags else None
    pose = yolo_cls(pose_weights)                       # fresh model = fresh tracker per clip
    log, writer, alerted_bags, n = AlertLog(), None, set(), 0
    os.makedirs(out_dir, exist_ok=True)
    out_mp4 = os.path.join(out_dir, stem + "_alerts.mp4")

    stream = pose.track(source=path, tracker="botsort.yaml", persist=True, conf=TRACK_CONF,
                        classes=[0], stream=True, verbose=False)
    for idx, r in enumerate(stream):
        n = idx + 1
        frame = r.orig_img.copy()
        H, W = frame.shape[:2]
        t = idx / fps
        person_boxes, any_active = [], False

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            boxes = r.boxes.xyxy.cpu().numpy()
            kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
            kconf = None
            if r.keypoints is not None and getattr(r.keypoints, "conf", None) is not None:
                kconf = r.keypoints.conf.cpu().numpy()
            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = boxes[i].astype(int)
                person_boxes.append((x1, y1, x2, y2))
                color, label = GREEN, f"ID:{tid}"
                if fall is not None:
                    kp = kxy[i] if kxy is not None and i < len(kxy) else np.zeros((17, 2), np.float32)
                    kc = kconf[i] if kconf is not None and i < len(kconf) else None
                    if kc is None:
                        kc = np.ones(17, np.float32)
                    alert, new_alert, score, why = fall.update(tid, boxes[i], kp, kc, idx, (W, H))
                    ang, ar, _ = fall.debug.get(tid, (float("nan"), 0.0, 0.0))
                    if alert:
                        color, label, any_active = RED, f"ID:{tid} FALL", True
                    if new_alert:
                        log.add(t, "FALL", f"ID:{tid}  torso {ang:.0f}deg")
                    cv2.putText(frame, f"ang {ang:.0f} ar {ar:.2f} s {score:.2f}", (x1, min(y2 + 22, H - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if fall is not None:
            fall.prune(idx)

        fight_prob = 0.0
        if fight is not None:
            fight_prob, active, started, ended = fight.update(frame, idx)
            if started:
                log.add(t, "FIGHT", f"p={fight_prob:.2f}")
            if ended:
                print(f"  {fmt_time(t)}   (fight alarm cleared)", flush=True)
            if active:
                any_active = True
                cv2.rectangle(frame, (0, 0), (W - 1, H - 1), ORANGE, max(10, W // 100))
                cv2.putText(frame, "FIGHT DETECTED", (W // 2 - int(150 * max(0.5, W / 1400)), 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2 * max(0.5, W / 1400), ORANGE, 3)

        if bags is not None:
            res = object_model(frame, conf=BAG_CONF, verbose=False)[0]
            bag_boxes = []
            if res.boxes is not None:
                for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                    if int(c) in BAG_CLASSES:
                        bag_boxes.append(tuple(b))
            for bid, (bx1, by1, bx2, by2), unattended, alone in bags.update(bag_boxes, person_boxes):
                bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)
                col = RED if unattended else YELLOW
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 2)
                cv2.putText(frame, f"UNATTENDED BAG {alone:.0f}s" if unattended else f"bag {alone:.0f}s",
                            (bx1, max(by1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                if unattended:
                    any_active = True
                    if bid not in alerted_bags:
                        alerted_bags.add(bid)
                        log.add(t, "BAG", f"bag#{bid} alone {alone:.0f}s")

        draw_panel(frame, t, log, fight_prob, any_active)
        if writer is None:
            writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        writer.write(frame)

    if writer is not None:
        writer.release()
    log.save_csv(os.path.join(out_dir, stem + "_alerts.csv"))
    if not log.items:
        print("  (no alarms in this video)")
    print(f"  -> {out_mp4}")
    return {"name": os.path.basename(path), "seconds": n / fps, "counts": log.counts()}


def list_videos(p):
    if os.path.isdir(p):
        return [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith(VIDEO_EXT)]
    return [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="a video file, or a folder of videos")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-fall", action="store_true")
    ap.add_argument("--no-fight", action="store_true")
    ap.add_argument("--no-bags", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    pose_weights = find_weights("yolov8n-pose.pt")
    object_model = YOLO(find_weights("yolov8n.pt")) if not a.no_bags else None
    fight_predict = make_fight_predictor() if not a.no_fight else None

    results = [process_video(p, a.out, YOLO, pose_weights, object_model, fight_predict,
                             not a.no_fall, not a.no_fight, not a.no_bags) for p in list_videos(a.video)]
    if len(results) > 1:
        print("\n=== SUMMARY ===")
        for r in results:
            c = ", ".join(f"{k}={v}" for k, v in r["counts"].items()) or "no alarms"
            print(f"{r['name'][:40]:<42}{r['seconds']:>6.1f}s   {c}")


if __name__ == "__main__":
    main()