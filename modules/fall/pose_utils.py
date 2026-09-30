"""
CampusGuard Vision - Fall Detection Module
Shared utility: keypoint normalization.

Raw x,y pixel keypoints are tied to camera distance/resolution/angle - a
model trained on one camera setup's raw pixels often fails to generalize to
a different setup, even for the same physical motion. This normalizes each
frame's keypoints to be centered on the person and scaled by their own
visible size, so "falling" looks geometrically similar regardless of how
close/far/angled the camera is.

Save as: modules/fall/pose_utils.py
Imported by: build_windows.py, visualize_fall_detection.py
"""

import numpy as np


def normalize_keypoints(kpts_flat):
    """
    kpts_flat: array of 34 values (17 keypoints x,y, flattened).
    Returns: normalized array of the same shape.

    Centers on the centroid of all VALID (non-zero) keypoints, and scales by
    the diagonal of their bounding box. Frames with no detection (all zero)
    are returned unchanged - there's nothing to normalize.
    """
    kpts = np.array(kpts_flat, dtype=np.float32).reshape(17, 2)
    valid_mask = np.any(kpts != 0, axis=1)

    if valid_mask.sum() == 0:
        return kpts_flat  # no detection this frame, leave as zeros

    valid_pts = kpts[valid_mask]
    center = valid_pts.mean(axis=0)

    min_xy = valid_pts.min(axis=0)
    max_xy = valid_pts.max(axis=0)
    scale = np.linalg.norm(max_xy - min_xy)
    if scale < 1e-3:
        scale = 1.0  # avoid divide-by-zero on degenerate single-point detections

    normalized = np.where(kpts != 0, (kpts - center) / scale, 0)
    return normalized.flatten().astype(np.float32)