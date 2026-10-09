"""
CampusGuard Vision - Fall Detection
Inference: per-person fall detector with safety gates. Used by evaluate_fall.py,
main.py and the dashboard backend.

Save as: modules/fall/fall_detector.py   (NEW file)

An alert needs ALL of these (this is what removes false positives):
  1. the track has a full 1-second history
  2. (edge gate - currently off, see MAX_CLIPPED_FRAC)
  3. enough keypoints are actually visible (low-confidence ones are ignored)
  4. the body actually moved/changed shape during the window
  5. the model's fall probability >= threshold ... on CONSECUTIVE checks in a row
"""

import os
import json
from collections import deque, defaultdict
import numpy as np

from pose_utils import featurize_window, FEATURE_DIM

# ---------------- tunables (all in one place) ----------------
WINDOW = 30               # frames the model sees (1 s at 30 fps)
CHECK_EVERY = 3           # run the model every N buffered frames
KP_CONF = 0.25            # ignore keypoints the pose model is unsure about
MIN_VALID_KPTS = 8        # per frame ...
MIN_VALID_FRAC = 0.6      # ... in at least this share of the window
EDGE_MARGIN_PX = 3
MAX_CLIPPED_FRAC = 1.01   # >1 = edge gate OFF (it blocked every window on close-up phone footage;
                          # the keypoint-visibility gate already covers cut-off bodies)
MIN_DYNAMICS = 0.25       # body box must change by >=25% of its height in the window
CONSECUTIVE = 2           # checks in a row above threshold
MIN_THRESHOLD = 0.70      # never go below this, whatever the config file says
HOLD_SECONDS = 3.0        # how long the alert stays on after it fires
STALE_SECONDS = 5.0
# -------------------------------------------------------------


class _Track:
    def __init__(self):
        self.kp = deque(maxlen=WINDOW)
        self.box = deque(maxlen=WINDOW)
        self.clip = deque(maxlen=WINDOW)
        self.n_added = 0
        self.hits = 0
        self.hold = 0
        self.last_seen = 0
        self.last_prob = 0.0
        self.reason = "warming_up"


class FallDetector:
    def __init__(self, model_path, fps=30.0, diagnostic=False, threshold=None, predictor=None):
        """diagnostic=True also runs the model on gated windows (for evaluation/mining)."""
        self.fps = float(fps) if fps and fps > 1 else 30.0
        self.stride = max(1, int(round(self.fps / 30.0)))     # 60 fps video -> every 2nd frame
        self.hold_frames = int(round(HOLD_SECONDS * self.fps))
        self.stale_frames = int(round(STALE_SECONDS * self.fps))
        self.diagnostic = diagnostic
        self.tracks = {}
        self.gate_counts = defaultdict(int)

        if predictor is not None:                              # used for testing
            self._predict = predictor
            self.threshold = threshold if threshold is not None else 0.9
            return

        import torch
        from fall_model import FallLSTM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FallLSTM(input_size=FEATURE_DIM).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        thr = 0.9
        cfg = os.path.splitext(model_path)[0] + "_config.json"
        if os.path.exists(cfg):
            with open(cfg) as f:
                thr = float(json.load(f).get("threshold", thr))
        self.threshold = threshold if threshold is not None else max(thr, MIN_THRESHOLD)

        def predict(feats):
            x = torch.from_numpy(feats).unsqueeze(0).to(device)
            with torch.no_grad():
                return float(torch.softmax(model(x), dim=1)[0, 1])
        self._predict = predict

    # ------------------------------------------------------------------
    @staticmethod
    def _clipped(box, frame_wh):
        x1, y1, x2, y2 = box
        W, H = frame_wh
        m = EDGE_MARGIN_PX
        return x1 <= m or y1 <= m or x2 >= W - m or y2 >= H - m

    def update(self, track_id, box, kpts_xy, kpts_conf, frame_idx, frame_wh):
        """Call once per person per frame.
        kpts_xy: (17,2) pixels, kpts_conf: (17,) or None, box: x1,y1,x2,y2.
        Returns dict: alert, new_alert, prob (this frame's check or None),
                      last_prob, reason, raw_window (diagnostic only)."""
        t = self.tracks.setdefault(int(track_id), _Track())
        t.last_seen = frame_idx
        res = {"alert": False, "new_alert": False, "prob": None,
               "last_prob": t.last_prob, "reason": t.reason, "raw_window": None}
        hold_before = t.hold

        if frame_idx % self.stride == 0:
            kp = np.asarray(kpts_xy, dtype=np.float32).reshape(17, 2).copy()
            if kpts_conf is not None:
                kp[np.asarray(kpts_conf).reshape(17) < KP_CONF] = 0
            t.kp.append(kp.reshape(-1))
            t.box.append(np.asarray(box, dtype=np.float32))
            t.clip.append(self._clipped(box, frame_wh))
            t.n_added += 1
            if len(t.kp) == WINDOW and t.n_added % CHECK_EVERY == 0:
                self._evaluate(t, res)

        res["alert"] = t.hold > 0
        res["new_alert"] = hold_before == 0 and t.hold > 0
        res["last_prob"], res["reason"] = t.last_prob, t.reason
        if t.hold > 0:
            t.hold -= 1
        return res

    def _evaluate(self, t, res):
        kp = np.array(t.kp, dtype=np.float32)                     # (W, 34)
        boxes = np.array(t.box, dtype=np.float32)
        valid_counts = np.any(kp.reshape(WINDOW, 17, 2) != 0, axis=2).sum(axis=1)
        vis_frac = float((valid_counts >= MIN_VALID_KPTS).mean())
        clip_frac = float(np.mean(t.clip))
        h = boxes[:, 3] - boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        ref = max(float(np.median(h)), 1.0)
        dyn = max(h.max() - h.min(), w.max() - w.min(), cy.max() - cy.min()) / ref

        reason = None
        if clip_frac > MAX_CLIPPED_FRAC:
            reason = "cut_off_by_frame_edge"
        elif vis_frac < MIN_VALID_FRAC:
            reason = "keypoints_not_visible"
        elif dyn < MIN_DYNAMICS:
            reason = "no_body_motion"

        if reason is not None and not self.diagnostic:
            self.gate_counts[reason] += 1
            t.hits, t.reason = 0, reason
            return

        prob = self._predict(featurize_window(kp))
        t.last_prob = prob
        res["prob"] = prob
        if self.diagnostic:
            res["raw_window"] = kp.copy()

        if reason is not None:                                    # diagnostic + gated
            self.gate_counts[reason] += 1
            t.hits, t.reason = 0, reason
            return

        t.reason = "ok"
        t.hits = t.hits + 1 if prob >= self.threshold else 0
        if t.hits >= CONSECUTIVE and t.hold == 0:
            t.hold = self.hold_frames

    def prune(self, frame_idx):
        for tid in [k for k, v in self.tracks.items() if frame_idx - v.last_seen > self.stale_frames]:
            del self.tracks[tid]
