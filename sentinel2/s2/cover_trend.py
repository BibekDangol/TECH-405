"""Cover-trend: % snow/ice/debris per Sentinel-2 scene over time.

Runs the trained patch CNN (outputs/s2_cnn.pt) over every data_s2/S2*.tif
with SCL masking (same logic as infer_map.py) and saves:
  outputs/s2_cover.csv          (scene, date, snow%, ice%, debris%, valid)
  outputs/s2_cover_trend.png    (stacked % cover vs date)

Usage: python3 s2/cover_trend.py [--patch 64]
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

CLASSES = ["snow", "ice", "debris"]
SCL_MASK = {0, 1, 3, 7, 8, 9, 10}


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


def vis_rgb(tif):
    import rasterio

    with rasterio.open(tif) as src:
        names = src.tags(ns="s2").get("bands", "").split(",")
        a = src.read().astype("float32")
        i = {k: names.index(k) for k in ("red", "green", "blue")}
        rgb = np.stack([a[i["red"]], a[i["green"]], a[i["blue"]]])
        return np.clip(rgb / 6000.0 * 255, 0, 255).astype("uint8").transpose(1, 2, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=64)
    args = ap.parse_args()

    res = json.load(open("outputs/s2_results.json"))
    cfg, mean, std = res["config"], res["norm"]["mean"], res["norm"]["std"]
    manifest = {r["id"]: r for r in json.load(open("data_s2/scenes.json"))}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN(cfg["hidden"], cfg["dropout"]).to(device)
    model.load_state_dict(torch.load("outputs/s2_cnn.pt", map_location=device, weights_only=True))
    model.eval()

    rows = []
    P = args.patch
    for tif in sorted(Path("data_s2").glob("S2*.tif")):
        if "_scl" in tif.name:
            continue
        rgb = vis_rgb(str(tif))
        H, W, _ = rgb.shape
        xs = range(0, W - P + 1, P)
        ys = range(0, H - P + 1, P)
        scl = None
        sp = tif.parent / f"{tif.stem}_scl.tif"
        if sp.exists():
            import rasterio

            with rasterio.open(sp) as s:
                scl = s.read(1)
        counts = [0, 0, 0]
        n_valid = 0
        with torch.no_grad():
            for y in ys:
                for x in xs:
                    if scl is not None and np.isin(
                            scl[int(y / H * scl.shape[0]):int((y + P) / H * scl.shape[0]),
                                int(x / W * scl.shape[1]):int((x + P) / W * scl.shape[1])],
                            list(SCL_MASK)).mean() > 0.5:
                        continue
                    a = rgb[y:y + P, x:x + P].astype("float32") / 255.0
                    a = ((a - np.array(mean, dtype="float32")) / np.array(std, dtype="float32")).astype("float32")
                    t = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    counts[int(model(t).argmax(1).item())] += 1
                    n_valid += 1
        frac = [c / max(n_valid, 1) * 100 for c in counts]
        date = manifest.get(tif.stem, {}).get("datetime", "")[:10]
        rows.append((date, tif.stem, *frac, n_valid))
        print(f"{tif.stem[:22]} {date}: snow={frac[0]:.1f}% ice={frac[1]:.1f}% debris={frac[2]:.1f}% (n={n_valid})")

    rows.sort()
    with open("outputs/s2_cover.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "scene", "snow_pct", "ice_pct", "debris_pct", "n_valid"])
        w.writerows(rows)

    dates = [r[0] for r in rows]
    y = np.array([[r[2], r[3], r[4]] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.stackplot(dates, y[:, 0], y[:, 1], y[:, 2],
                 labels=["snow", "ice", "debris"],
                 colors=["#f2f2fa", "#78aadc", "#6e5541"], alpha=0.9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% valid-scene cover (SCL-masked)")
    ax.set_title("Khumbu Icefall surface cover per Sentinel-2 scene (CNN, monitoring — not a forecast)")
    ax.legend(loc="upper left")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig("outputs/s2_cover_trend.png", dpi=130, bbox_inches="tight")
    print("saved outputs/s2_cover.csv + outputs/s2_cover_trend.png")


if __name__ == "__main__":
    main()
