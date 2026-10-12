"""Train 3-class surface patch CNN: snow / ice / debris (torch + PIL only).

Reads data_s2/patches.csv (path,label,split). Tiny custom CNN (~150k params,
same family as neuralNetwork.ipynb) with 3 outputs. No torchvision needed.

Usage:
  python3 s2/train_surface_cnn.py --epochs 25 --batch 32 --hidden 128
Output:
  outputs/s2_cnn.pt  outputs/s2_results.json
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score

CLASSES = ["snow", "ice", "debris"]


class PatchSet(torch.utils.data.Dataset):
    def __init__(self, rows, mean, std, train=False):
        self.rows = rows
        self.mean = np.array(mean, dtype="float32")
        self.std = np.array(std, dtype="float32")
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        img = Image.open(path).convert("RGB")
        if self.train and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.train:
            ang = random.uniform(-10, 10)
            img = img.rotate(ang)
        a = np.asarray(img).astype("float32") / 255.0
        a = (a - self.mean) / self.std
        return torch.from_numpy(a.transpose(2, 0, 1)), int(label)


class CNN(nn.Module):
    def __init__(self, hidden=128, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.head(self.features(x))


def evaluate(model, loader, device):
    model.eval()
    yp, yt = [], []
    with torch.no_grad():
        for xb, yb in loader:
            yp += model(xb.to(device)).argmax(1).cpu().tolist()
            yt += yb.tolist()
    acc = 100.0 * sum(p == t for p, t in zip(yp, yt)) / max(len(yt), 1)
    return acc, yp, yt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_s2/patches.csv")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = {s: [] for s in ("train", "val", "test")}
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            rows[r["split"]].append((r["path"], int(r["label"])))
    assert rows["train"] and rows["test"], "empty train/test — run make_patches.py first"
    if not rows["val"]:
        rows["val"] = rows["test"]
    print({k: len(v) for k, v in rows.items()})

    # Train-set channel stats (no leakage).
    pix = np.concatenate([
        (np.asarray(Image.open(p).convert("RGB")).astype("float32") / 255.0).reshape(-1, 3)
        for p, _ in rows["train"][:200]
    ])
    mean, std = pix.mean(0).tolist(), (pix.std(0) + 1e-6).tolist()
    print("mean", [round(m, 4) for m in mean], "std", [round(s, 4) for s in std])

    def loader(split, train=False):
        ds = PatchSet(rows[split], mean, std, train=train)
        return torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=train)

    tr, va, te = loader("train", True), loader("val"), loader("test")
    model = CNN(args.hidden, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    counts = np.bincount([y for _, y in rows["train"]], minlength=3).astype("float32")
    w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1))).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    best = {"f1": -1.0, "state": None, "epoch": -1}
    bad = 0
    for ep in range(args.epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        _, yp, yt = evaluate(model, va, device)
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        print(f"epoch {ep+1}/{args.epochs} val_macro_f1={f1:.4f}")
        if f1 > best["f1"]:
            best = {"f1": f1, "state": {k: v.cpu() for k, v in model.state_dict().items()}, "epoch": ep}
            bad = 0
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    acc, yp, yt = evaluate(model, te, device)
    rep = classification_report(yt, yp, target_names=CLASSES, digits=4, zero_division=0)
    print(f"test acc: {acc:.2f}%\n{rep}")

    Path("outputs").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "outputs/s2_cnn.pt")
    json.dump({
        "classes": CLASSES,
        "config": {"hidden": args.hidden, "dropout": args.dropout, "lr": args.lr,
                   "epochs": args.epochs, "best_epoch": best["epoch"]},
        "norm": {"mean": mean, "std": std},
        "test_accuracy": round(acc, 2),
        "val_macro_f1": round(float(best["f1"]), 4),
        "report": rep,
        "confusion": confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist(),
    }, open("outputs/s2_results.json", "w"), indent=2)
    print("saved outputs/s2_cnn.pt + outputs/s2_results.json")


if __name__ == "__main__":
    main()
