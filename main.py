"""
CampusGuard Vision - Combined Pipeline
Runs all three detection modules together on one video:
  - Person tracking (BoT-SORT) + Fall detection (per-person pose LSTM)
  - Fight detection (scene-level R(2+1)D sliding window)
  - Unattended bag detection (rule-based, stationary object + no owner nearby)

Save as: main.py  (repo root: D:\\hackathon\\campusguard\\main.py)
Run as:  python main.py

Self-contained - no imports from modules/ subfolders, so it can run from
the repo root without path issues. Uses TWO YOLO models:
  - yolov8n-pose.pt  -> person tracking + keypoints (for fall)
  - yolov8n.pt       -> bag-class objects only (for unattended bags)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import deque, defaultdict
from ultralytics import YOLO
from torchvision.models.video import r2plus1d_18

# ==================== CONFIG ====================
VIDEO_PATH = r"D:\hackathon\campusguard\data\test_combined.mp4"   # update to your test clip
OUTPUT_PATH = r"D:\hackathon\campusguard\data\combined_output.mp4"

FALL_MODEL_PATH = r"D:\hackathon\campusguard\models\fall_lstm.pt"
FIGHT_MODEL_PATH = r"D:\hackathon\campusguard\models\fight_r2plus1d.pt"

# Fall settings
FALL_WINDOW_SIZE = 30
FALL_CHECK_EVERY_N_FRAMES = 5
FALL_HOLD_FRAMES = 15
TRACK_CONF_THRESHOLD = 0.4

# Fight settings
FIGHT_WINDOW_SECONDS = 5.0
FIGHT_CHECK_EVERY_SECONDS = 0.5
FIGHT_THRESHOLD = 0.5
FIGHT_CONSECUTIVE = 2
FIGHT_NUM_FRAMES = 16
FIGHT_SIZE = 128
FIGHT_CROP = 112

# Bag settings
BAG_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}
STATIONARY_PIXEL_THRESHOLD = 25
NEARBY_PERSON_DISTANCE = 150
ABANDON_SECONDS = 8.0
BAG_CONF_THRESHOLD = 0.35
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Fall detection ----------------
class FallLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def normalize_keypoints(kpts_flat):
    """Same normalization used in training - see modules/fall/pose_utils.py."""
    kpts = np.array(kpts_flat, dtype=np.float32).reshape(17, 2)
    valid_mask = np.any(kpts != 0, axis=1)
    if valid_mask.sum() == 0:
        return kpts_flat
    valid_pts = kpts[valid_mask]
    center = valid_pts.mean(axis=0)
    min_xy = valid_pts.min(axis=0)
    max_xy = valid_pts.max(axis=0)
    scale = np.linalg.norm(max_xy - min_xy)
    if scale < 1e-3:
        scale = 1.0
    normalized = np.where(kpts != 0, (kpts - center) / scale, 0)
    return normalized.flatten().astype(np.float32)


def predict_fall(model, window):
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    return probs.argmax(dim=1).item(), probs[0, 1].item()


# ---------------- Fight detection ----------------
FIGHT_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
FIGHT_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


def predict_fight(model, frames_small):
    idx = np.linspace(0, len(frames_small) - 1, FIGHT_NUM_FRAMES).astype(int)
    x = torch.from_numpy(np.stack([frames_small[i] for i in idx])).float() / 255.0
    x = x.permute(3, 0, 1, 2)
    off = (FIGHT_SIZE - FIGHT_CROP) // 2
    x = x[:, :, off:off + FIGHT_CROP, off:off + FIGHT_CROP]
    x = ((x - FIGHT_MEAN) / FIGHT_STD).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
        probs = torch.softmax(model(x).float(), dim=1)
    return probs[0, 1].item()


# ---------------- Unattended bags ----------------
class BagWatcher:
    def __init__(self, fps):
        self.fps = fps
        self.tracked_bags = {}
        self.next_id = 0
        self.frame_idx = 0

    def _match_or_create(self, center):
        for bag_id, info in self.tracked_bags.items():
            if np.linalg.norm(np.array(center) - np.array(info["center"])) < STATIONARY_PIXEL_THRESHOLD * 3:
                return bag_id
        bag_id = self.next_id
        self.next_id += 1
        self.tracked_bags[bag_id] = {
            "center": center, "still_since_frame": self.frame_idx, "owner_seen_frame": self.frame_idx,
        }
        return bag_id

    def update(self, bag_boxes, person_boxes):
        person_centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in person_boxes]
        results, seen_ids = [], set()
        for box in bag_boxes:
            x1, y1, x2, y2 = box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            bag_id = self._match_or_create(center)
            seen_ids.add(bag_id)
            info = self.tracked_bags[bag_id]
            moved = np.linalg.norm(np.array(center) - np.array(info["center"])) > STATIONARY_PIXEL_THRESHOLD
            info["center"] = center
            if moved:
                info["still_since_frame"] = self.frame_idx
            has_owner = any(np.linalg.norm(np.array(center) - np.array(pc)) < NEARBY_PERSON_DISTANCE
                             for pc in person_centers)
            if has_owner:
                info["owner_seen_frame"] = self.frame_idx
            seconds_alone = (self.frame_idx - info["owner_seen_frame"]) / self.fps
            results.append((box, seconds_alone >= ABANDON_SECONDS, seconds_alone))
        for bag_id in list(self.tracked_bags.keys()):
            if bag_id not in seen_ids:
                del self.tracked_bags[bag_id]
        self.frame_idx += 1
        return results


# ==================== MAIN ====================
def main():
    print("Loading models...")
    pose_model = YOLO(r"D:\hackathon\campusguard\models\yolov8n-pose.pt")
    object_model = YOLO("yolov8n.pt")

    fall_model = FallLSTM().to(device)
    fall_model.load_state_dict(torch.load(FALL_MODEL_PATH, map_location=device))
    fall_model.eval()

    fight_model = r2plus1d_18(weights=None)
    fight_model.fc = nn.Linear(fight_model.fc.in_features, 2)
    fight_model.load_state_dict(torch.load(FIGHT_MODEL_PATH, map_location=device))
    fight_model = fight_model.to(device).eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {VIDEO_PATH}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # per-person fall state
    keypoint_buffers = defaultdict(lambda: deque(maxlen=FALL_WINDOW_SIZE))
    fall_hold_counters = defaultdict(int)

    # fight state
    fight_window_frames = int(round(FIGHT_WINDOW_SECONDS * fps))
    fight_min_frames = max(FIGHT_NUM_FRAMES, int(fight_window_frames * 0.5))
    fight_check_every = max(1, int(round(FIGHT_CHECK_EVERY_SECONDS * fps)))
    fight_buffer = deque(maxlen=fight_window_frames)
    fight_prob, fight_streak, fight_alert = 0.0, 0, False

    # bags state
    bag_watcher = BagWatcher(fps)

    frame_idx = 0
    print("Processing combined pipeline...")

    results_stream = pose_model.track(
        source=VIDEO_PATH, tracker="botsort.yaml", persist=True,
        conf=TRACK_CONF_THRESHOLD, classes=[0], stream=True, verbose=False,
    )

    for r in results_stream:
        frame = r.orig_img.copy()
        person_boxes = []

        # ---- Person tracking + fall detection ----
        if r.boxes is not None and r.boxes.id is not None:
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)

            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = boxes[i]
                person_boxes.append((x1, y1, x2, y2))

                kpts = (r.keypoints.xy[i].cpu().numpy().flatten()
                        if r.keypoints is not None and i < len(r.keypoints.xy) else np.zeros(34))
                keypoint_buffers[track_id].append(normalize_keypoints(kpts))

                box_color, label = (0, 255, 0), f"ID:{track_id}"
                buf = keypoint_buffers[track_id]
                if len(buf) == FALL_WINDOW_SIZE and frame_idx % FALL_CHECK_EVERY_N_FRAMES == 0:
                    pred, _ = predict_fall(fall_model, np.array(buf, dtype=np.float32))
                    if pred == 1:
                        fall_hold_counters[track_id] = FALL_HOLD_FRAMES

                if fall_hold_counters[track_id] > 0:
                    box_color, label = (0, 0, 255), f"ID:{track_id} FALL DETECTED"
                    fall_hold_counters[track_id] -= 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # ---- Fight detection (scene-level) ----
        fight_buffer.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (FIGHT_SIZE, FIGHT_SIZE)))
        if len(fight_buffer) >= fight_min_frames and frame_idx % fight_check_every == 0:
            fight_prob = predict_fight(fight_model, list(fight_buffer))
            fight_streak = fight_streak + 1 if fight_prob >= FIGHT_THRESHOLD else 0
            fight_alert = fight_streak >= FIGHT_CONSECUTIVE

        if fight_alert:
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 12)
            cv2.putText(frame, "FIGHT DETECTED", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        cv2.putText(frame, f"fight_prob: {fight_prob:.2f}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if fight_alert else (0, 255, 0), 2)

        # ---- Unattended bags ----
        obj_results = object_model(frame, conf=BAG_CONF_THRESHOLD, verbose=False)[0]
        bag_boxes = []
        if obj_results.boxes is not None:
            for box, cls in zip(obj_results.boxes.xyxy.cpu().numpy(), obj_results.boxes.cls.cpu().numpy()):
                if int(cls) in BAG_CLASSES:
                    bag_boxes.append(tuple(box))

        for (x1, y1, x2, y2), is_unattended, seconds_alone in bag_watcher.update(bag_boxes, person_boxes):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            color = (0, 0, 255) if is_unattended else (0, 200, 255)
            label = f"UNATTENDED BAG ({seconds_alone:.0f}s)" if is_unattended else f"Bag ({seconds_alone:.0f}s)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        writer.write(frame)
        frame_idx += 1

    writer.release()
    print(f"\nDone. Combined output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()