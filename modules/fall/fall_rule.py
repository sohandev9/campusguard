r"""
CampusGuard Vision - rule-based "person on floor" fall detector.

Save as: modules\fall\fall_rule.py   (NEW file, does not touch anything else)

Idea: a person is "lying" when their torso (shoulder-mid -> hip-mid) is close to
horizontal AND the body box is at least as wide as ~0.85 x its height (this rejects
someone bending over to pick something up). If that holds for ~0.5 s, raise an alert. If the pose model loses the
torso for a short moment (common right after a person lands) but the box is still wide,
the last valid torso angle is carried forward for up to 0.8 s.

API (same as the old FallDetector, plus fps in the constructor):
    det = FallRule(fps=30)
    alert, new_alert, last_prob, reason = det.update(track_id, box, kpts_xy, kpts_conf, frame_idx, (W, H))
    det.prune(frame_idx)

alert      -> True while the alert is held on screen (3 s after it fires)
new_alert  -> True only on the frame the alert fires (log it once)
last_prob  -> 0..1 "how lying" score of the last frames (for display only)
reason     -> short text
"""

from collections import deque
import numpy as np

KP_CONF = 0.25
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


class _Track:
    def __init__(self, maxlen):
        self.raw = deque(maxlen=5)        # last raw angles (nan allowed) for causal median
        self.ok = deque(maxlen=maxlen)    # last N "lying" flags
        self.last_valid = np.nan
        self.last_valid_frame = -10 ** 9
        self.last_seen = 0
        self.cooldown_until = -1
        self.hold_until = -1
        self.score = 0.0


class FallRule:
    def __init__(self, fps=30.0, ang_thr=55.0, hold_s=0.5, frac=0.8,
                 fill_s=0.8, ar_fill=0.8, min_vis=6, min_ar=0.85,
                 alert_hold_s=3.0, cooldown_s=3.0):
        self.fps = float(fps)
        self.ang_thr = ang_thr
        self.hold_n = max(2, int(hold_s * self.fps))
        self.frac = frac
        self.fill_n = int(fill_s * self.fps)
        self.ar_fill = ar_fill
        self.min_vis = min_vis
        self.min_ar = min_ar          # box must be at least this wide/tall: rejects bending over
        self.alert_hold_n = int(alert_hold_s * self.fps)
        self.cooldown_n = int(cooldown_s * self.fps)
        self.tracks = {}
        self.debug = {}               # track_id -> (torso_angle, box_w_over_h, lying_score) for drawing

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _torso_angle(xy, conf):
        """0 = upright, 90 = flat. NaN if shoulders or hips are not both visible."""
        def mid(a, b):
            if conf[a] >= KP_CONF and conf[b] >= KP_CONF:
                return (xy[a] + xy[b]) / 2.0
            return None
        s, h = mid(L_SH, R_SH), mid(L_HIP, R_HIP)
        if s is None or h is None:
            return np.nan
        dx, dy = abs(s[0] - h[0]), abs(s[1] - h[1])
        return float(np.degrees(np.arctan2(dx, dy + 1e-6)))

    # ---- main API ------------------------------------------------------
    def update(self, track_id, box, kpts_xy, kpts_conf, frame_idx, frame_size=None):
        t = self.tracks.get(track_id)
        if t is None:
            t = self.tracks[track_id] = _Track(self.hold_n)
        t.last_seen = frame_idx

        xy = np.asarray(kpts_xy, dtype=float)
        conf = np.asarray(kpts_conf, dtype=float)
        x1, y1, x2, y2 = [float(v) for v in box]
        ar = (x2 - x1) / max(y2 - y1, 1.0)
        vis = int((conf >= KP_CONF).sum())

        ang = self._torso_angle(xy, conf)
        t.raw.append(ang)
        valid = [v for v in t.raw if not np.isnan(v)]
        smooth = float(np.median(valid)) if len(valid) >= 3 or (valid and not np.isnan(ang)) else np.nan

        if not np.isnan(ang):
            t.last_valid, t.last_valid_frame = smooth, frame_idx

        # carry the last torso angle forward briefly if pose was lost but the box is still wide
        used = smooth
        filled = False
        if np.isnan(ang) and (frame_idx - t.last_valid_frame) <= self.fill_n \
                and ar >= self.ar_fill and not np.isnan(t.last_valid) \
                and t.last_valid >= self.ang_thr - 10:
            used, filled = t.last_valid, True

        lying = ((not np.isnan(used)) and used >= self.ang_thr and ar >= self.min_ar
                 and (vis >= self.min_vis or filled))
        t.ok.append(1.0 if lying else 0.0)
        t.score = float(np.mean(t.ok)) if t.ok else 0.0
        self.debug[track_id] = (used, ar, t.score)

        new_alert = False
        reason = "upright / ok"
        if lying:
            reason = "torso flat"
        if (len(t.ok) >= self.hold_n and t.score >= self.frac and frame_idx > t.cooldown_until):
            new_alert = True
            t.cooldown_until = frame_idx + self.cooldown_n
            t.hold_until = frame_idx + self.alert_hold_n
            reason = "person on floor (torso flat for %.1fs)" % (self.hold_n / self.fps)

        alert = frame_idx <= t.hold_until
        if alert and not new_alert:
            reason = "person on floor"
        return alert, new_alert, t.score, reason

    def prune(self, frame_idx, max_age_s=5.0):
        limit = int(max_age_s * self.fps)
        for k in [k for k, t in self.tracks.items() if frame_idx - t.last_seen > limit]:
            del self.tracks[k]
            self.debug.pop(k, None)