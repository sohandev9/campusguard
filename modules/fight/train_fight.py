"""
CampusGuard Vision - Fight Detection Module
Step 2: Fine-tune a Kinetics-pretrained R(2+1)D-18 video model on RWF-2000.

Save as: modules/fight/train_fight.py
Run as:  python train_fight.py

Uses RWF-2000's official train/val split (different videos, so no leakage).
Saves the best checkpoint (by validation F1) to models/fight_r2plus1d.pt
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

CACHE_DIR = r"D:\hackathon\campusguard\data\rwf_cache"
MODEL_SAVE_PATH = r"D:\hackathon\campusguard\models\fight_r2plus1d.pt"
EPOCHS = 8
BATCH_SIZE = 8
LR = 1e-4
CROP = 112

MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


class RWFDataset(Dataset):
    def __init__(self, split, train):
        self.train = train
        self.items = []
        for label, cls in [(0, "NonFight"), (1, "Fight")]:
            files = glob.glob(os.path.join(CACHE_DIR, split, cls, "*.npy"))
            self.items += [(f, label) for f in files]
        print(f"{split}: {len(self.items)} clips "
              f"(fight: {sum(l for _, l in self.items)})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        clip = np.load(path)  # (16, 128, 128, 3) uint8
        x = torch.from_numpy(clip).float() / 255.0
        x = x.permute(3, 0, 1, 2)  # (C, T, H, W)

        H = x.shape[2]
        if self.train:
            top = np.random.randint(0, H - CROP + 1)
            left = np.random.randint(0, H - CROP + 1)
            x = x[:, :, top:top + CROP, left:left + CROP]
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[3])  # horizontal flip
            # brightness/contrast jitter - helps with dim real-world footage
            b = np.random.uniform(-0.15, 0.15)
            c = np.random.uniform(0.8, 1.2)
            x = torch.clamp((x - 0.5) * c + 0.5 + b, 0, 1)
        else:
            off = (H - CROP) // 2
            x = x[:, :, off:off + CROP, off:off + CROP]

        x = (x - MEAN) / STD
        return x, label


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            tp += ((preds == 1) & (yb == 1)).sum().item()
            fp += ((preds == 1) & (yb == 0)).sum().item()
            fn += ((preds == 0) & (yb == 1)).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return correct / max(total, 1), precision, recall, f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = RWFDataset("train", train=True)
    val_ds = RWFDataset("val", train=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("[ERROR] No cached clips found. Run preprocess_rwf.py first.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    best_f1 = -1

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if (step + 1) % 50 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(train_loader)} loss {loss.item():.3f}")
        scheduler.step()

        acc, prec, rec, f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> saved best model (F1 {f1:.3f})")

    print(f"\nDone. Best val F1: {best_f1:.3f}\nModel: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()