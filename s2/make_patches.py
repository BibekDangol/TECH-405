"""Cut patches + starter labels for snow / ice / debris.

Two modes (auto-detected):
1. S2 GeoTIFF mode — data_s2/<scene>.tif stacks from download_sentinel2.py
   (bands order recorded in tag: blue,green,red,nir,swir16).
   Label per patch by center-pixel NDSI + brightness:
     NDSI=(green-swir)/(green+swir); snow: NDSI>0.4 & bright;
     debris: dark / low NDSI; else ice. SCL clouds masked when available
     (fetch via s2/fetch_scl.py; classes 0,1,3,7,8,9,10 excluded,
     patches <50% valid skipped).
2. Fallback screenshot mode — no S2 files present (offline demo).
   Uses dataset/avalanche + dataset/no_avalanche RGB PNGs with a pure-RGB
   heuristic (bright->snow, blue-grey->ice, dark/brown->debris) so the CNN
   pipeline still runs end-to-end. Starter labels only, not ground truth.

Split is BY SOURCE SCENE/FILE GROUP (no random patch leakage across
train/val/test when S2 scenes exist).

Usage:
  python3 s2/make_patches.py --size 64 --stride 64 --per-class 400
Output:
  data_s2/patches/<split>/<class>/*.png  +  data_s2/patches.csv
"""
import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["snow", "ice", "debris"]
S2_DIR = Path("data_s2")
SHOT_DIRS = [Path("dataset/avalanche"), Path("dataset/no_avalanche")]


def s2_label_patch(green, swir, red, nir):
    """Return class idx per pixel from S2 bands (uint16 arrays).

    Calibrated on 5 Khumbu crops (Dec + spring): winter low-sun scenes average
    brightness ~2100, so the snow brightness gate is 2000 (not 4000), and
    debris requires darkness AND low NDSI (not OR) to avoid calling all
    shadowed ice 'debris'. Snow gate keeps standard NDSI>0.4.
    """
    g = green.astype("float32")
    s = swir.astype("float32")
    r = red.astype("float32")
    n = nir.astype("float32")
    ndsi = (g - s) / np.maximum(g + s, 1.0)
    bright = (g + r + n) / 3.0
    snow = (ndsi > 0.4) & (bright > 2000)
    debris = (bright < 1200) | ((ndsi < 0.0) & (bright < 3000))
    out = np.ones_like(ndsi, dtype=np.int64)  # default ice=1
    out[snow] = 0
    out[debris] = 2
    return out, ndsi


def rgb_heuristic_label(arr):
    """Fallback RGB heuristic for plain screenshots (H,W,3 uint8)."""
    a = arr.astype("float32")
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    bright = (r + g + b) / 3.0
    snow = bright > 195
    brown = (r > g + 8) & (r > b + 8) & (bright < 195)
    dark = bright < 105
    debris = brown | dark
    blueish = (b > r + 6) & (~snow) & (~debris)
    out = np.where(snow, 0, np.where(debris, 2, np.where(blueish, 1, 1)))
    return out.astype(np.int64)


def save_patch(png_arr, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(png_arr).save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--per-class", type=int, default=400, help="cap kept patches per class total")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    s2tifs = sorted(S2_DIR.glob("S2*.tif"))
    tifs = s2tifs if s2tifs else sorted(S2_DIR.glob("*.tif"))
    tifs = [t for t in tifs if "patches" not in str(t) and "_scl" not in str(t)]
    kept = []  # (png_arr, cls, source)
    SCL_MASK = {0, 1, 3, 7, 8, 9, 10}  # nodata/saturated/shadow/cloud/cirrus

    if tifs:
        import rasterio

        for tif in tifs:
            with rasterio.open(tif) as src:
                tags = src.tags(ns="s2")
                names = tags.get("bands", "blue,green,red,nir,swir16").split(",")
                stack = src.read()  # C,H,W
            idx = {k: i for i, k in enumerate(names)}
            B, G, R, NIR = (stack[idx["blue"]], stack[idx["green"]],
                            stack[idx["red"]], stack[idx["nir"]])
            SWIR = stack[idx["swir16"]] if "swir16" in idx else np.zeros_like(G)
            pix, _ = s2_label_patch(G, SWIR, R, NIR)
            scl_path = tif.parent / f"{tif.stem}_scl.tif"
            if scl_path.exists():
                with rasterio.open(scl_path) as s:
                    scl = s.read(1)
                masked = np.isin(scl, list(SCL_MASK))
                print(f"{tif.stem}: scl-masked frac={masked.mean():.3f}")
            else:
                masked = np.zeros_like(pix, dtype=bool)
            # vis RGB for patch PNGs (red,green,blue -> stretched)
            vis = np.stack([R, G, B]).astype("float32")
            vis = np.clip((vis / 6000.0) * 255, 0, 255).astype("uint8").transpose(1, 2, 0)
            H, W = pix.shape
            for y in range(0, H - args.size + 1, args.stride):
                for x in range(0, W - args.size + 1, args.stride):
                    lab = pix[y:y + args.size, x:x + args.size]
                    valid = ~masked[y:y + args.size, x:x + args.size]
                    if valid.mean() < 0.5:
                        continue  # mostly cloud/shadow/nodata
                    vals, counts = np.unique(lab[valid], return_counts=True)
                    cls = int(vals[int(np.argmax(counts))])
                    if counts.max() < 0.6 * valid.sum():
                        continue  # skip mixed patches
                    kept.append((vis[y:y + args.size, x:x + args.size].copy(), cls, tif.stem))
    else:
        files = []
        for d in SHOT_DIRS:
            if d.exists():
                files += sorted(d.glob("*.png"))
        if not files:  # legacy flat layout
            files = sorted(Path("dataset").glob("*.png"))
        for f in files:
            img = Image.open(f).convert("RGB").resize((512, 512))
            arr = np.asarray(img)
            pix = rgb_heuristic_label(arr)
            H, W = pix.shape
            for y in range(0, H - args.size + 1, args.stride):
                for x in range(0, W - args.size + 1, args.stride):
                    lab = pix[y:y + args.size, x:x + args.size]
                    vals, counts = np.unique(lab, return_counts=True)
                    cls = int(vals[int(np.argmax(counts))])
                    if counts.max() < 0.5 * args.size * args.size:
                        continue
                    kept.append((arr[y:y + args.size, x:x + args.size].copy(), cls, f.stem))

    # Balance per class, then split BY SOURCE (scene/file) to avoid leakage.
    rng.shuffle(kept)
    by_cls = {c: [k for k in kept if k[1] == c] for c in range(3)}
    for c in by_cls:
        by_cls[c] = by_cls[c][: args.per_class]
    print({CLASSES[c]: len(by_cls[c]) for c in by_cls})

    sources = sorted({s for lst in by_cls.values() for (_, _, s) in lst})
    rng.shuffle(sources)
    n = len(sources)
    test_src = set(sources[int(0.7 * n):]) if n >= 3 else set(sources[-1:])
    val_src = set(sources[int(0.5 * n):int(0.7 * n)]) if n >= 5 else set()
    # tiny-source fallback: stratified random so val/test are non-empty
    flat = [k for lst in by_cls.values() for k in lst]
    if not test_src or not val_src:
        rng.shuffle(flat)
        nflat = len(flat)
        splits = (
            [("train", flat[: int(0.7 * nflat)]),
             ("val", flat[int(0.7 * nflat): int(0.85 * nflat)]),
             ("test", flat[int(0.85 * nflat):])]
        )
    else:
        def grp(want):
            return [k for k in flat if k[2] in want]
        splits = [
            ("train", grp(set(sources) - val_src - test_src)),
            ("val", grp(val_src)),
            ("test", grp(test_src)),
        ]

    out_root = S2_DIR / "patches"
    S2_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = S2_DIR / "patches.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "class", "split", "source"])
        for split, items in splits:
            for i, (png_arr, cls, src) in enumerate(items):
                dst = out_root / split / CLASSES[cls] / f"{src}_{i:04d}.png"
                save_patch(png_arr, dst)
                w.writerow([str(dst), cls, CLASSES[cls], split, src])
    for split, items in splits:
        print(f"{split}: n={len(items)}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
