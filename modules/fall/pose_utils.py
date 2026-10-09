"""
CampusGuard Vision - Fall Detection
Shared keypoint utilities. ONE source of truth for training AND inference.

Save as: modules/fall/pose_utils.py   (replaces the old file)

featurize_window(): raw keypoints (T, 34) -> model input (T, 37)
    34 values  = pose, centred on the person and scaled by their own size
                 (position / distance invariant)
     3 values  = trajectory: how far the body centre moved since the start of
                 the window (in body sizes) + how much its size changed.
                 This is the "body is dropping" signal that per-frame
                 normalisation alone throws away.
augment_raw():    training-time augmentation on raw keypoints.
"""

import numpy as np

NUM_KPTS = 17
FEATURE_DIM = 37
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
LEG_IDS = [11, 12, 13, 14, 15, 16]


def featurize_window(raw_window):
    """raw_window: (T, 34) pixel x,y keypoints, 0 = missing -> (T, 37) float32."""
    raw = np.asarray(raw_window, dtype=np.float32).reshape(-1, NUM_KPTS, 2)
    T = raw.shape[0]
    out = np.zeros((T, FEATURE_DIM), dtype=np.float32)
    centers = np.zeros((T, 2), dtype=np.float32)
    scales = np.zeros(T, dtype=np.float32)
    ok = np.zeros(T, dtype=bool)

    for t in range(T):
        k = raw[t]
        valid = np.any(k != 0, axis=1)
        if valid.sum() < 2:
            continue
        pts = k[valid]
        c = pts.mean(axis=0)
        s = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        if s < 1e-3:
            continue
        centers[t], scales[t], ok[t] = c, s, True
        out[t, :34] = np.where(valid[:, None], (k - c) / s, 0.0).reshape(-1)

    if not ok.any():
        return out

    ref_scale = float(np.median(scales[ok]))
    ref_center = centers[int(np.argmax(ok))]
    for t in np.where(ok)[0]:
        out[t, 34:36] = (centers[t] - ref_center) / ref_scale
        out[t, 36] = np.log(scales[t] / ref_scale)
    return out


def augment_raw(raw_window, rng):
    """Random, physically plausible changes to a raw (T, 34) window."""
    k = np.array(raw_window, dtype=np.float32).reshape(-1, NUM_KPTS, 2)
    valid = np.any(k != 0, axis=2)
    if not valid.any():
        return k.reshape(len(k), -1)
    c = k[valid].mean(axis=0)

    # 1) mirror image (swap left/right joints)
    if rng.random() < 0.5:
        k[..., 0] = 2 * c[0] - k[..., 0]
        for a, b in FLIP_PAIRS:
            k[:, [a, b]] = k[:, [b, a]]
            valid[:, [a, b]] = valid[:, [b, a]]

    # 2) different camera angle: squash/stretch vertically + small tilt
    sx, sy = rng.uniform(0.9, 1.1), rng.uniform(0.7, 1.3)
    th = np.deg2rad(rng.uniform(-20, 20))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    k = ((k - c) * np.array([sx, sy], dtype=np.float32)) @ R.T + c

    # 3) pose-estimator jitter
    diag = float(np.linalg.norm(k[valid].max(axis=0) - k[valid].min(axis=0)))
    k = k + rng.normal(0, rng.uniform(0, 0.015) * diag, size=k.shape).astype(np.float32)

    # 4) missing keypoints / frames
    for t in range(len(k)):
        if rng.random() < 0.5:
            valid[t, rng.choice(NUM_KPTS, size=int(rng.integers(1, 5)), replace=False)] = False
        if rng.random() < 0.03:
            valid[t, :] = False
    if rng.random() < 0.15:          # person cut off by the frame edge: legs unseen
        valid[:, LEG_IDS] = False

    k[~valid] = 0
    return k.reshape(len(k), -1)
