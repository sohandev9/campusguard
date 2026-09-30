"""
CampusGuard Vision - Fall Detection Module
Step 6 (updated): Multi-person visual fall detection with normalized keypoints.

Save as: modules/fall/visualize_fall_detection.py
Run as:  python visualize_fall_detection.py

Uses the SAME normalization (pose_utils.normalize_keypoints) as training, so
the model sees geometrically consistent input regardless of camera distance
or angle - this fixes the domain-gap issue where the model never triggered
on real footage despite good validation scores on the training dataset.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import deque, defaultdict
from ultralytics import YOLO
from pose_utils import normalize_keypoints

VIDEO_PATH = r"D:\hackathon\campusguard\data\test_video.mp4"   # update to your test clip
MODEL_PATH = r"D:\hackathon\campusguard\models\fall_lstm.pt"
OUTPUT_PATH = r"D:\hackathon\campusguard\data\fall_output.mp4"
WINDOW_SIZE = 30
CHECK_EVERY_N_FRAMES = 5
FALL_HOLD_FRAMES = 15
CONF_THRESHOLD = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FallLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def predict_fall(model, window):
    """window: np.array shape (WINDOW_SIZE, 34), ALREADY NORMALIZED -> returns (pred, confidence)"""
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0, 1].item()
    return pred, confidence


def main():
    pose_model = YOLO("yolov8n-pose.pt")

    model = FallLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    keypoint_buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
    fall_hold_counters = defaultdict(int)

    frame_idx = 0
    print("Processing video...")

    results_stream = pose_model.track(
        source=VIDEO_PATH,
        tracker="botsort.yaml",
        persist=True,
        conf=CONF_THRESHOLD,
        classes=[0],
        stream=True,
        verbose=False,
    )

    for r in results_stream:
        frame = r.orig_img.copy()

        if r.boxes is not None and r.boxes.id is not None:
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)

            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = boxes[i]

                if r.keypoints is not None and i < len(r.keypoints.xy):
                    kpts = r.keypoints.xy[i].cpu().numpy().flatten()
                else:
                    kpts = np.zeros(34)

                kpts_normalized = normalize_keypoints(kpts)
                keypoint_buffers[track_id].append(kpts_normalized)

                box_color = (0, 255, 0)
                label = f"ID:{track_id} Person"

                buf = keypoint_buffers[track_id]
                if len(buf) == WINDOW_SIZE and frame_idx % CHECK_EVERY_N_FRAMES == 0:
                    window = np.array(buf, dtype=np.float32)
                    pred, confidence = predict_fall(model, window)
                    print(f"Frame {frame_idx} | ID:{track_id} | fall_prob={confidence:.3f} | pred={pred}")
                    if pred == 1:
                        fall_hold_counters[track_id] = FALL_HOLD_FRAMES

                if fall_hold_counters[track_id] > 0:
                    box_color = (0, 0, 255)
                    label = f"ID:{track_id} FALL DETECTED"
                    fall_hold_counters[track_id] -= 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        writer.write(frame)
        frame_idx += 1

    writer.release()
    print(f"\nDone. Annotated video saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()