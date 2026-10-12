"""Download Sentinel-2 L2A crops over Khumbu Icefall (no auth).

Queries Element84 Earth Search STAC (open AWS COGs), picks low-cloud scenes,
then reads only the AOI window via rasterio /vsicurl/ and saves a local stack:
  data_s2/<scene_id>.tif  (bands: B02,B03,B04,B08 @10m + B11 resampled to 10m)
  data_s2/<scene_id>.json (meta: date, cloud, COG hrefs)

AOI default: Khumbu Icefall ~ lon 86.68-86.76, lat 27.95-28.02.
Bands used: blue/green/red/nir for CNN + swir16 for NDSI starter labels.
SCL is saved when present for cloud/shadow masking.

Usage:
  python3 s2/download_sentinel2.py --n 3 --bbox 86.68,27.95,86.76,28.02
  python3 s2/download_sentinel2.py --n 3 --seasons 2020,2021,2022
"""
import argparse
import json
import os
import urllib.request

STAC = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
WANT = ["blue", "green", "red", "nir", "swir16", "scl"]  # B02,B03,B04,B08,B11,SCL


def stac_search(bbox, datetime, limit, cloud_lt=20):
    body = {
        "collections": [COLLECTION],
        "bbox": bbox,
        "datetime": datetime,
        "query": {"eo:cloud_cover": {"lt": cloud_lt}},
        "limit": limit,
    }
    req = urllib.request.Request(
        STAC, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        fc = json.load(r)
    # Sort client-side (server-side sortby on eo:cloud_cover is unsupported).
    feats = fc.get("features", [])
    feats.sort(key=lambda f: f.get("properties", {}).get("eo:cloud_cover", 100))
    fc["features"] = feats
    return fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", default="86.68,27.95,86.76,28.02")
    ap.add_argument("--datetime", default="2020-01-01T00:00:00Z/2024-12-31T23:59:59Z")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cloud-lt", type=float, default=20.0)
    ap.add_argument("--out", default="data_s2")
    ap.add_argument("--dry-run", action="store_true", help="only list scenes, no download")
    args = ap.parse_args()

    bbox = [float(x) for x in args.bbox.split(",")]
    os.makedirs(args.out, exist_ok=True)

    fc = stac_search(bbox, args.datetime, args.n, args.cloud_lt)
    feats = fc.get("features", [])
    print(f"matched={fc.get('context', {}).get('matched')} returning={len(feats)}")
    manifest = []
    for f in feats:
        fid = f["id"]
        props = f.get("properties", {})
        assets = f.get("assets", {})
        hrefs = {k: assets[k]["href"] for k in WANT if k in assets}
        row = {
            "id": fid,
            "datetime": props.get("datetime"),
            "cloud_cover": props.get("eo:cloud_cover"),
            "hrefs": hrefs,
        }
        manifest.append(row)
        print(f"- {fid} date={row['datetime']} cloud={row['cloud_cover']} bands={sorted(hrefs)}")

    json.dump(manifest, open(os.path.join(args.out, "scenes.json"), "w"), indent=2)
    print(f"saved {args.out}/scenes.json")
    if args.dry_run or not manifest:
        return

    # Windowed COG download via rasterio (only AOI pixels, not full tiles).
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling
        from rasterio.warp import transform_bounds
    except ImportError:
        print("rasterio not installed; manifest saved. Install rasterio to fetch pixels.")
        return

    for row in manifest:
        sid = row["id"]
        out_tif = os.path.join(args.out, f"{sid}.tif")
        out_meta = os.path.join(args.out, f"{sid}.json")
        if os.path.exists(out_tif):
            print(f"exists, skip {sid}")
            continue
        hrefs = row["hrefs"]
        # bbox is lon/lat (EPSG:4326); COGs are UTM — reproject bounds per band.
        # Use green band as 10m reference grid.
        with rasterio.open(hrefs["green"]) as ref:
            b = transform_bounds("EPSG:4326", ref.crs, *bbox)
            win10 = from_bounds(*b, transform=ref.transform)
            win10 = win10.round_offsets().round_lengths()
            profile = ref.profile.copy()
            crs, transform = ref.crs, rasterio.windows.transform(win10, ref.transform)
        print(f"{sid}: window {int(win10.width)}x{int(win10.height)} in {crs}")
        if win10.width <= 0 or win10.height <= 0:
            print(f"WARNING: {sid} does not intersect AOI, skipping pixel fetch")
            continue
        bands, names = [], []
        for key in ["blue", "green", "red", "nir", "swir16"]:
            if key not in hrefs:
                continue
            with rasterio.open(hrefs[key]) as src:
                b = transform_bounds("EPSG:4326", src.crs, *bbox)
                win = from_bounds(*b, transform=src.transform)
                win = win.round_offsets().round_lengths()
                arr = src.read(
                    1,
                    window=win,
                    out_shape=(win10.height, win10.width),
                    resampling=Resampling.bilinear,
                )
                bands.append(arr)
                names.append(key)
        import numpy as np

        stack = np.stack(bands).astype("uint16")
        profile.update(
            driver="GTiff", height=win10.height, width=win10.width,
            count=len(bands), crs=crs, transform=transform, compress="deflate",
        )
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(stack)
            dst.update_tags(ns="s2", bands=",".join(names), scene=sid)
        json.dump(row, open(out_meta, "w"), indent=2)
        print(f"saved {out_tif} shape={stack.shape}")


if __name__ == "__main__":
    main()
