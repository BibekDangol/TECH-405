"""Fetch SCL (scene classification) windows matching downloaded S2 crops.

Reads data_s2/scenes.json hrefs (scl COGs) and crops to the same AOI window
as data_s2/<scene>.tif, saving data_s2/<scene>_scl.tif (single band).

SCL classes: 0 nodata, 1 saturated, 2 dark, 3 cloud-shadow, 4 veg, 5 soil,
6 water, 7/8/9 cloud, 10 cirrus, 11 snow. make_patches.py masks
{0,1,3,7,8,9,10} (nodata/saturated/shadow/cloud/cirrus).

Usage: python3 s2/fetch_scl.py
"""
import json
import os
from pathlib import Path

OUT = Path("data_s2")


def main():
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    from rasterio.enums import Resampling

    manifest = json.load(open(OUT / "scenes.json"))
    for row in manifest:
        sid = row["id"]
        ref_tif = OUT / f"{sid}.tif"
        out_tif = OUT / f"{sid}_scl.tif"
        if out_tif.exists():
            print(f"exists, skip {sid}")
            continue
        if not ref_tif.exists():
            print(f"missing crop {sid}, skip")
            continue
        with rasterio.open(ref_tif) as ref:
            bounds_ll = transform_bounds(ref.crs, "EPSG:4326", *ref.bounds)
        href = row["hrefs"]["scl"]
        with rasterio.open(href) as src:
            b = transform_bounds("EPSG:4326", src.crs, *bounds_ll)
            win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            with rasterio.open(ref_tif) as ref:
                arr = src.read(1, window=win, out_shape=(ref.height, ref.width),
                               resampling=Resampling.nearest)
        import numpy as np

        arr = np.asarray(arr, dtype="uint8")
        with rasterio.open(ref_tif) as ref:
            profile = ref.profile.copy()
        profile.update(count=1, dtype="uint8", compress="deflate")
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(arr, 1)
            dst.update_tags(ns="scl", scene=sid)
        print(f"saved {out_tif} masked_frac={( (arr==3)|(arr>=7)&(arr<=10) ).mean():.3f}")


if __name__ == "__main__":
    main()
