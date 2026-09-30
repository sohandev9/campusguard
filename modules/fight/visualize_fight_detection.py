"""
CampusGuard Vision - Fight Detection Module
Step 3 (fixed): Evaluate the fight model on real clips.

Save as: modules/fight/visualize_fight_detection.py
Run as:  python visualize_fight_detection.py

Fixes vs. the first version:
 - Starts predicting once the buffer is half full (old version waited for a
   full window, so a 5 s clip with a 5 s window never got a prediction).
 - Tries several window lengths on every clip and prints ONE summary table,
   so you can paste just the table instead of hundreds of lines.
 - An alert needs CONSECUTIVE checks in a row above THRESHOLD, which filters
   single-frame spikes.
Annotated videos are written to OUTPUT_DIR for the PRIMARY_WINDOW only.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torchvision.models.video import r2plus1d_18

# ---- CONFIG: put ALL your test clips here (fight AND non-fight) ----
VIDEO_PATHS = [
    # r"D:\hackathon\campusguard\data\test_fight.mp4",
    # r"D:\hackathon\campusguard\data\test_fight2.mp4",
    r"D:\hackathon\campusguard\data\test_fight2.avi",
]
MODEL_PATH = r"D:\hackathon\campusguard\models\fight_r2plus1d.pt"
OUTPUT_DIR = r"D:\hackathon\campusguard\data\fight_outputs"

WINDOWS = [2.0, 3.0, 5.0]     # window lengths (seconds) to compare
PRIMARY_WINDOW = 3.0          # the one used for the saved annotated video
CHECK_EVERY_SECONDS = 0.25
MIN_WINDOW_FRACTION = 0.5     # start predicting when buffer is this full
THRESHOLD = 0.5
CONSECUTIVE = 2               # checks in a row above THRESHOLD to raise an alert
# ---------------------------------------------------------------------

NUM_FRAMES = 16
SIZE = 128
CROP = 112
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


def build_model():
    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model.to(device).eval()


def predict(model, frames_small):
    """frames_small: list of 128x128 RGB uint8 frames -> fight probability."""
    idx = np.linspace(0, len(frames_small) - 1, NUM_FRAMES).astype(int)
    x = torch.from_numpy(np.stack([frames_small[i] for i in idx])).float() / 255.0
    x = x.permute(3, 0, 1, 2)
    off = (SIZE - CROP) // 2
    x = x[:, :, off:off + CROP, off:off + CROP]
    x = ((x - MEAN) / STD).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
        probs = torch.softmax(model(x).float(), dim=1)
    return probs[0, 1].item()


def run_video(model, path, window_seconds, write_video):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window_frames = int(round(window_seconds * fps))
    min_frames = max(NUM_FRAMES, int(window_frames * MIN_WINDOW_FRACTION))
    check_every = max(1, int(round(CHECK_EVERY_SECONDS * fps)))

    writer = None
    if write_video:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        name = os.path.splitext(os.path.basename(path))[0] + "_fight.mp4"
        writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR, name),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    buffer = deque(maxlen=window_frames)
    probs, streak, alert, prob = [], 0, False, None
    ever_alert, frame_idx = False, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buffer.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (SIZE, SIZE)))

        if len(buffer) >= min_frames and frame_idx % check_every == 0:
            prob = predict(model, list(buffer))
            probs.append(prob)
            streak = streak + 1 if prob >= THRESHOLD else 0
            alert = streak >= CONSECUTIVE
            ever_alert = ever_alert or alert

        if writer is not None:
            if alert:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 14)
                cv2.putText(frame, "FIGHT DETECTED", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            txt = "warming up..." if prob is None else f"fight_prob: {prob:.2f}"
            cv2.putText(frame, txt, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255) if alert else (0, 255, 0), 2)
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    return probs, ever_alert, frame_idx / fps


def main():
    model = build_model()
    rows = []
    for path in VIDEO_PATHS:
        name = os.path.basename(path)
        for win in WINDOWS:
            res = run_video(model, path, win, write_video=(win == PRIMARY_WINDOW))
            if res is None:
                continue
            probs, ever_alert, dur = res
            if len(probs) == 0:
                rows.append((name, dur, win, 0, None, None, None, False))
                continue
            p = np.array(probs)
            rows.append((name, dur, win, len(p), p.max(), p.mean(),
                         100.0 * (p >= THRESHOLD).mean(), ever_alert))

    print("\n=== SUMMARY (paste this) ===")
    print(f"threshold={THRESHOLD}, alert = {CONSECUTIVE} consecutive checks above threshold")
    print(f"{'clip':<28}{'dur(s)':>7}{'win(s)':>7}{'checks':>7}{'max':>6}{'mean':>6}{'%>thr':>7}{'ALERT':>7}")
    for name, dur, win, n, mx, mean, pct, alert in rows:
        if n == 0:
            print(f"{name[:27]:<28}{dur:>7.1f}{win:>7.1f}{0:>7}   (no checks: clip too short for this window)")
        else:
            print(f"{name[:27]:<28}{dur:>7.1f}{win:>7.1f}{n:>7}{mx:>6.2f}{mean:>6.2f}{pct:>7.0f}{'YES' if alert else 'no':>7}")
    print(f"\nAnnotated videos (window={PRIMARY_WINDOW}s) saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()