"""Sliding-window surface map for one image/GeoTIFF (snow/ice/debris % cover).

Loads outputs/s2_cnn.pt + outputs/s2_results.json, tiles the input into
64px patches, votes the CNN per patch, and saves:
  outputs/s2_map_<stem>.png  (input + predicted map side-by-side)
  + prints % snow / ice / debris (monitoring signal, NOT a hazard warning).

Usage:
  python3 s2/infer_map.py --input data_s2/patches/test/snow/xxx.png
  python3 s2/infer_map.py --input data_s2/S2A_45RVL_*.tif
"""
import argparse
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
COLORS = np.array([[245, 245, 250], [120, 170, 220], [110, 85, 65]], dtype="uint8")
SCL_MASK = {0, 1, 3, 7, 8, 9, 10}
MASK_COLOR = np.array([60, 60, 70], dtype="uint8")


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


def load_image(path, size=64):
    p = str(path)
    if p.endswith(".tif"):
        import rasterio

        with rasterio.open(p) as src:
            tags = src.tags(ns="s2")
            names = tags.get("bands", "").split(",") if tags.get("bands") else []
            a = src.read().astype("float32")
            if names and {"red", "green", "blue"} <= set(names):
                i = {k: names.index(k) for k in ("red", "green", "blue")}
                rgb = np.stack([a[i["red"]], a[i["green"]], a[i["blue"]]])
                rgb = np.clip(rgb / 6000.0 * 255, 0, 255).astype("uint8").transpose(1, 2, 0)
            else:
                rgb = np.clip(a[:3].transpose(1, 2, 0), 0, 255).astype("uint8")
        return Image.fromarray(rgb)
    return Image.open(p).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--patch", type=int, default=64)
    args = ap.parse_args()

    res = json.load(open("outputs/s2_results.json"))
    cfg, mean, std = res["config"], res["norm"]["mean"], res["norm"]["std"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN(cfg["hidden"], cfg["dropout"]).to(device)
    model.load_state_dict(torch.load("outputs/s2_cnn.pt", map_location=device, weights_only=True))
    model.eval()

    img = load_image(args.input)
    W, H = img.size
    P = args.patch
    xs = list(range(0, W - P + 1, P)) or [0]
    ys = list(range(0, H - P + 1, P)) or [0]
    pmap = np.zeros((len(ys), len(xs)), dtype=np.int64)
    valid_mask = np.ones((len(ys), len(xs)), dtype=bool)
    scl_path = Path(str(args.input).replace(".tif", "_scl.tif"))
    scl = None
    if scl_path.exists():
        import rasterio

        with rasterio.open(scl_path) as s:
            scl = s.read(1)
        sh, sw = scl.shape
    with torch.no_grad():
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                if scl is not None:
                    # proportional window into SCL array (same AOI, same grid)
                    yy = slice(int(y / H * sh), int((y + P) / H * sh))
                    xx = slice(int(x / W * sw), int((x + P) / W * sw))
                    if np.isin(scl[yy, xx], list(SCL_MASK)).mean() > 0.5:
                        valid_mask[j, i] = False
                        pmap[j, i] = 0
                        continue
                a = np.asarray(img.crop((x, y, x + P, y + P))).astype("float32") / 255.0
                a = ((a - np.array(mean, dtype="float32")) / np.array(std, dtype="float32")).astype("float32")
                t = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(device)
                pmap[j, i] = int(model(t).argmax(1).item())

    n_valid = int(valid_mask.sum())
    frac = [((pmap == c) & valid_mask).sum() / max(n_valid, 1) * 100 for c in range(3)]
    print(f"{args.input}: " + ", ".join(f"{k}={v:.1f}%" for k, v in zip(CLASSES, frac)) + f" (valid {n_valid}/{valid_mask.size})")
    print("This is surface cover (monitoring), not an avalanche/flood forecast.")

    big = COLORS[pmap].copy()
    big[~valid_mask] = MASK_COLOR  # masked (cloud/shadow) in gray
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title(Path(args.input).name)
    ax[0].axis("off")
    ax[1].imshow(big, interpolation="nearest")
    ax[1].set_title("CNN: " + ", ".join(f"{k} {v:.0f}%" for k, v in zip(CLASSES, frac)))
    ax[1].axis("off")
    out = Path("outputs") / f"s2_map_{Path(args.input).stem}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
