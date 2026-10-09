"""
CampusGuard Vision - Fall Detection
Step 4 (rewritten): train the fall-MOTION classifier.

Save as: modules/fall/train_model.py   (replaces the old file)
Run as:  python train_model.py     (after build_windows.py)

Changes vs the old version
 - trains on the corrected labels (fall motion, not "lying on the floor")
 - 37 features per frame (pose + trajectory), see pose_utils.py
 - augmentation: mirror, camera-angle jitter, noise, missing keypoints, cut-off legs
 - gentler class weighting (sqrt) so it stops over-predicting "fall"
 - validation split is per clip AND stratified, so fall clips appear in val
 - picks the alert threshold from validation (precision >= 0.90) and saves it
   next to the model as fall_lstm_config.json
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from fall_model import FallLSTM
from pose_utils import featurize_window, augment_raw, FEATURE_DIM

WINDOWS_DIR = r"D:\hackathon\campusguard\data\windows"
MODEL_SAVE_PATH = r"D:\hackathon\campusguard\models\fall_lstm.pt"
CONFIG_SAVE_PATH = MODEL_SAVE_PATH.replace(".pt", "_config.json")
VAL_FRACTION = 0.2
SEED = 42
EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3
PRECISION_TARGET = 0.90
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


class WindowDataset(Dataset):
    def __init__(self, X_raw, y, augment, seed=0):
        self.X, self.y, self.augment = X_raw, y, augment
        self.rng = np.random.default_rng(seed)
        self.cached = None if augment else np.stack([featurize_window(w) for w in X_raw])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        feats = featurize_window(augment_raw(self.X[i], self.rng)) if self.augment else self.cached[i]
        return torch.from_numpy(feats), int(self.y[i])


def pr_at(probs, labels, thr):
    pred = probs >= thr
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, fp


def predict_probs(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            out.append(torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X = np.load(os.path.join(WINDOWS_DIR, "X_raw.npy"))
    y = np.load(os.path.join(WINDOWS_DIR, "y.npy"))
    ids = np.load(os.path.join(WINDOWS_DIR, "sequence_ids.npy"), allow_pickle=True)

    # split by clip, stratified so validation contains fall clips
    rng = np.random.default_rng(SEED)
    pos_clips = sorted({s for s, l in zip(ids, y) if l == 1})
    other_clips = sorted(set(ids) - set(pos_clips))

    def split(lst):
        lst = list(lst); rng.shuffle(lst)
        n = max(1, int(round(VAL_FRACTION * len(lst))))
        return set(lst[:n]), set(lst[n:])

    val_a, train_a = split(pos_clips)
    val_b, train_b = split(other_clips)
    val_clips, train_clips = val_a | val_b, train_a | train_b
    tr = np.array([s in train_clips for s in ids])
    va = np.array([s in val_clips for s in ids])

    print(f"Train: {tr.sum()} windows ({int(y[tr].sum())} fall) from {len(train_clips)} clips")
    print(f"Val  : {va.sum()} windows ({int(y[va].sum())} fall) from {len(val_clips)} clips "
          f"({len(val_a)} fall clips)")

    train_loader = DataLoader(WindowDataset(X[tr], y[tr], True, SEED), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(X[va], y[va], False), batch_size=BATCH_SIZE)

    model = FallLSTM(input_size=FEATURE_DIM).to(device)
    n_pos, n_neg = int(y[tr].sum()), int((1 - y[tr]).sum())
    w_pos = float(np.sqrt(n_neg / max(n_pos, 1)))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, w_pos], dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1, best_state, y_val = -1.0, None, y[va]
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()

        p, r, f1, fp = pr_at(predict_probs(model, val_loader, device), y_val, 0.5)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | loss {total/len(train_loader):.4f} | "
              f"val@0.5  precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}  false-alarm windows {fp}")
        if f1 >= best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    probs = predict_probs(model, val_loader, device)

    print("\nValidation at different alert thresholds (best epoch):")
    print("  thr   precision  recall   false-alarm windows")
    chosen = THRESHOLDS[-1]
    for t in THRESHOLDS:
        p, r, f1, fp = pr_at(probs, y_val, t)
        print(f"  {t:.2f}   {p:.3f}      {r:.3f}    {fp}")
    for t in THRESHOLDS:
        p, r, _, _ = pr_at(probs, y_val, t)
        if p >= PRECISION_TARGET and r > 0:
            chosen = t
            break
    p, r, _, _ = pr_at(probs, y_val, chosen)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(best_state, MODEL_SAVE_PATH)
    with open(CONFIG_SAVE_PATH, "w") as f:
        json.dump({"threshold": chosen, "window": 30, "feature_dim": FEATURE_DIM,
                   "val_precision": round(p, 3), "val_recall": round(r, 3),
                   "val_fall_clips": len(val_a)}, f, indent=2)
    print(f"\nChosen alert threshold: {chosen}  (val precision {p:.3f}, recall {r:.3f})")
    print(f"Saved model  -> {MODEL_SAVE_PATH}\nSaved config -> {CONFIG_SAVE_PATH}")
    print("NOTE: validation is small - treat these numbers as a sanity check, "
          "and judge the model with evaluate_fall.py on your own clips.")


if __name__ == "__main__":
    main()
