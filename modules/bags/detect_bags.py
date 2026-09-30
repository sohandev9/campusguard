"""
CampusGuard Vision - Unattended Bags Module
Rule-based detection: an object (backpack/handbag/suitcase) that stays in
roughly the same spot with no person nearby for ABANDON_SECONDS is flagged.

Save as: modules/bags/detect_bags.py
Run as:  python detect_bags.py    (standalone test)

Also exposed as a class (BagWatcher) for import into main.py.
"""

import os
import time
import numpy as np
import cv2
from ultralytics import YOLO

# COCO class ids for "bag-like" objects
BAG_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}
PERSON_CLASS = 0

VIDEO_PATH = r"D:\hackathon\campusguard\data\test_bags.mp4"
OUTPUT_PATH = r"D:\hackathon\campusguard\data\bags_output.mp4"

STATIONARY_PIXEL_THRESHOLD = 25     # bag center can drift this many px and still count as "same spot"
NEARBY_PERSON_DISTANCE = 150        # px - a person within this distance "owns" the bag
ABANDON_SECONDS = 8.0               # how long alone before flagged unattended
CONF_THRESHOLD = 0.35


class BagWatcher:
    """
    Call update(boxes, classes, fps) once per frame with YOLO detections for
    THIS frame (bag classes + person class). Returns list of (bbox, is_unattended)
    for every currently-tracked bag.
    """

    def __init__(self, fps):
        self.fps = fps
        self.tracked_bags = {}  # id -> {center, first_seen_frame, still_since_frame, owner_seen_frame}
        self.next_id = 0
        self.frame_idx = 0

    def _match_or_create(self, center):
        for bag_id, info in self.tracked_bags.items():
            dist = np.linalg.norm(np.array(center) - np.array(info["center"]))
            if dist < STATIONARY_PIXEL_THRESHOLD * 3:  # loose match to same physical bag
                return bag_id
        bag_id = self.next_id
        self.next_id += 1
        self.tracked_bags[bag_id] = {
            "center": center,
            "still_since_frame": self.frame_idx,
            "owner_seen_frame": self.frame_idx,
        }
        return bag_id

    def update(self, bag_boxes, person_boxes):
        """
        bag_boxes: list of (x1,y1,x2,y2)
        person_boxes: list of (x1,y1,x2,y2)
        returns: list of (bbox, is_unattended, seconds_alone)
        """
        person_centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in person_boxes]

        results = []
        seen_ids = set()

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

            has_owner_nearby = any(
                np.linalg.norm(np.array(center) - np.array(pc)) < NEARBY_PERSON_DISTANCE
                for pc in person_centers
            )
            if has_owner_nearby:
                info["owner_seen_frame"] = self.frame_idx

            seconds_alone = (self.frame_idx - info["owner_seen_frame"]) / self.fps
            is_unattended = seconds_alone >= ABANDON_SECONDS
            results.append((box, is_unattended, seconds_alone))

        # drop bags not seen this frame (left with owner, or detection missed)
        for bag_id in list(self.tracked_bags.keys()):
            if bag_id not in seen_ids:
                del self.tracked_bags[bag_id]

        self.frame_idx += 1
        return results


def main():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {VIDEO_PATH}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    watcher = BagWatcher(fps)
    print("Processing video...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        bag_boxes, person_boxes = [], []
        if results.boxes is not None:
            for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                cls = int(cls)
                if cls in BAG_CLASSES:
                    bag_boxes.append(tuple(box))
                elif cls == PERSON_CLASS:
                    person_boxes.append(tuple(box))

        bag_results = watcher.update(bag_boxes, person_boxes)

        for (x1, y1, x2, y2), is_unattended, seconds_alone in bag_results:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            color = (0, 0, 255) if is_unattended else (0, 255, 0)
            label = f"UNATTENDED ({seconds_alone:.0f}s)" if is_unattended else f"Bag ({seconds_alone:.0f}s)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for x1, y1, x2, y2 in person_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()