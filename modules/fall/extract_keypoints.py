"""
CampusGuard Vision - Fall Detection Module
Step 2: Extract pose keypoint sequences from the UR Fall Dataset.

UR Fall Dataset structure is FRAME IMAGES, not video files:
    UR_data/
        Fall/
            fall-03-cam0-rgb/
                fall-03-cam0-rgb/       <- frames live in this doubly-nested folder
                    fall-03-cam0-rgb-001.png
                    fall-03-cam0-rgb-002.png
                    ...
        ADL/
            adl-01-cam0-rgb/
                adl-01-cam0-rgb/
                    adl-01-cam0-rgb-001.png
                    ...

This script walks both Fall/ and ADL/, treats each innermost image folder as
one sequence, runs pose estimation frame-by-frame in correct order, and saves
one .npy file per sequence (shape: num_frames x 34).

Run this from anywhere - paths below are absolute.
"""

import os
import numpy as np
from ultralytics import YOLO

# ---- CONFIG - edit these paths if your folders differ ----
UR_DATA_DIR = r"D:\hackathon\campusguard_old\data\UR_data"
OUTPUT_DIR = r"D:\hackathon\campusguard\data\keypoints"
# Subfolders under UR_DATA_DIR and the label each one gets in the output filename
CATEGORIES = {
    "Fall": "fall",
    "ADL": "adl",
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
# ------------------------------------------------------------

pose_model = YOLO("yolov8n-pose.pt")  # auto-downloads on first run


def find_sequence_folders(category_dir):
    """
    Finds every innermost folder that actually contains frame images.
    Handles the doubly-nested UR Fall structure (outer/outer/inner/*.png).
    Returns list of (sequence_name, folder_path).
    """
    sequences = []
    for root, dirs, files in os.walk(category_dir):
        images = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
        if images:
            seq_name = os.path.basename(root)
            sequences.append((seq_name, root))
    return sequences


def extract_sequence_from_frames(folder_path):
    """Runs pose estimation on an ordered sequence of frame images in a folder."""
    frame_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)
    )

    sequence = []
    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        results = pose_model(frame_path, verbose=False)
        r = results[0]
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            kpts = r.keypoints.xy[0].cpu().numpy()  # first detected person, shape (17, 2)
            sequence.append(kpts.flatten())          # shape (34,)
        else:
            # No person detected in this frame - pad with zeros to keep frame alignment
            sequence.append(np.zeros(34))

    return np.array(sequence)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    extracted_count = 0
    skipped_count = 0

    for category_folder, label in CATEGORIES.items():
        category_dir = os.path.join(UR_DATA_DIR, category_folder)
        if not os.path.isdir(category_dir):
            print(f"[WARNING] Category folder not found, skipping: {category_dir}")
            continue

        sequences = find_sequence_folders(category_dir)
        print(f"\n=== {category_folder}: found {len(sequences)} sequences ===\n")

        for seq_name, folder_path in sequences:
            sequence = extract_sequence_from_frames(folder_path)

            # Count how many frames actually had a detected person (non-zero rows)
            valid_frames = int(np.any(sequence != 0, axis=1).sum()) if len(sequence) > 0 else 0

            if len(sequence) > 0 and valid_frames > 0:
                save_name = f"{label}_{seq_name}.npy"
                save_path = os.path.join(OUTPUT_DIR, save_name)
                np.save(save_path, sequence)
                print(f"[OK]      {seq_name}: {len(sequence)} frames total, {valid_frames} with pose detected")
                extracted_count += 1
            else:
                print(f"[SKIPPED] {seq_name}: no pose detected in any frame")
                skipped_count += 1

    print("\n--- Summary ---")
    print(f"Extracted: {extracted_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Saved to:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()