"""
CampusGuard Vision - Fall Detection Module
Step 4 (updated): Train LSTM on normalized pose-sequence windows.

Save as: modules/fall/train_model.py
Run as:  python train_model.py

Keypoints are already normalized (position/scale invariant) by
build_windows.py - no extra normalization needed here.
Train/val split is done by SEQUENCE (whole video), not by window, to avoid
data leakage from overlapping windows.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

WINDOWS_DIR = r"D:\hackathon\campusguard\data\windows"
MODEL_SAVE_PATH = r"D:\hackathon\campusguard\models\fall_lstm.pt"
VAL_FRACTION = 0.2
RANDOM_SEED = 42

X = np.load(f"{WINDOWS_DIR}/X.npy").astype(np.float32)   # already normalized
y = np.load(f"{WINDOWS_DIR}/y.npy")
sequence_ids = np.load(f"{WINDOWS_DIR}/sequence_ids.npy", allow_pickle=True)

unique_seqs = np.unique(sequence_ids)
rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(unique_seqs)

n_val = max(1, int(VAL_FRACTION * len(unique_seqs)))
val_seqs = set(unique_seqs[:n_val])
train_seqs = set(unique_seqs[n_val:])

train_mask = np.array([s in train_seqs for s in sequence_ids])
val_mask = np.array([s in val_seqs for s in sequence_ids])

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"Train sequences: {len(train_seqs)} | Train windows: {len(X_train)} "
      f"(fall: {(y_train == 1).sum()}, not-fall: {(y_train == 0).sum()})")
print(f"Val sequences:   {len(val_seqs)} | Val windows:   {len(X_val)} "
      f"(fall: {(y_val == 1).sum()}, not-fall: {(y_val == 0).sum()})")


class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FallLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = FallDataset(X_train, y_train)
    val_ds = FallDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = FallLSTM().to(device)

    class_counts = np.bincount(y_train, minlength=2)
    class_weights = torch.tensor(
        [1.0 / max(class_counts[0], 1), 1.0 / max(class_counts[1], 1)], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                tp += ((preds == 1) & (yb == 1)).sum().item()
                fp += ((preds == 1) & (yb == 0)).sum().item()
                fn += ((preds == 0) & (yb == 1)).sum().item()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | "
              f"Val Acc: {correct/max(total,1):.3f} | Precision: {precision:.3f} | "
              f"Recall: {recall:.3f} | F1: {f1:.3f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()