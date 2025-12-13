import os
import glob
import rasterio
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import pandas as pd
import random

# ---------- USER TWEAKABLE PARAMETERS ----------
DOWNLOAD_DIR = "downloads"            # where your tile_XXXXX_t1/t2.tif files are
OUT_DIR = "dataset"                   # output base folder
NDVI_THRESH = 0.20                    # ndvi_t1 - ndvi_t2 > thresh => deforestation
MIN_COMPONENT_PIXELS = 20            # remove connected components smaller than this
ERODE_PIXELS = 1                      # morphological erosion (pixels). Set 0 to disable.
VAL_FRAC = 0.2                        # fraction for validation split
TEST_FRAC = 0.1                        # fraction for test split
BAND_ORDER = ["B2","B3","B4","B8","B11"]  # order you exported (for clarity only)
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)

def read_tile(path):
    """Return (arr, meta) where arr is (bands, H, W) numpy array."""
    with rasterio.open(path) as src:
        arr = src.read()  # shape: (bands, H, W)
        meta = src.meta.copy()
    return arr, meta

def detect_scale_and_normalize(band_arr):
    """
    band_arr: numpy array of one band (H,W).
    If values appear integer > 1 (e.g., 0-10000 or 0-65535), scale to [0,1].
    Heuristic:
      - if max > 1 and <= 65535 -> assume int reflectance scaled (10000 or 100000), divide by 10000
      - if max between 1 and 100 -> likely already in [0,1] or percent -> leave as is
    """
    mx = float(np.nanmax(band_arr))
    if mx > 1.0:
        # prefer dividing by 10000 if sensible
        if mx <= 65535:
            scale = 10000.0
        else:
            scale = mx  # fallback to scale by max (rare)
        return band_arr.astype("float32") / scale
    else:
        return band_arr.astype("float32")

def compute_ndvi(arr, red_idx, nir_idx):
    """
    arr: (bands, H, W)
    red_idx, nir_idx: indices into arr (0-based)
    returns NDVI (H,W) float32
    """
    red = arr[red_idx].astype("float32")
    nir = arr[nir_idx].astype("float32")

    red = detect_scale_and_normalize(red)
    nir = detect_scale_and_normalize(nir)

    denom = (nir + red)
    # safe divide
    ndvi = np.where(denom == 0, 0.0, (nir - red) / denom)
    return ndvi

def postprocess_mask(mask_bool):
    """Remove small components and optionally erode edges."""
    mask = mask_bool.astype(np.uint8)
    # remove small objects
    labeled, ncomp = ndimage.label(mask)
    if ncomp > 0:
        counts = np.bincount(labeled.ravel())
        # counts[0] is background
        small_components = np.where(counts < MIN_COMPONENT_PIXELS)[0]
        for comp in small_components:
            if comp == 0:
                continue
            mask[labeled == comp] = 0

    # morphological erosion
    if ERODE_PIXELS > 0:
        struct = np.ones((3,3), dtype=np.uint8)
        for _ in range(ERODE_PIXELS):
            mask = ndimage.binary_erosion(mask, structure=struct).astype(np.uint8)

    return mask

def write_mask(mask, meta, out_path):
    """Write single-band uint8 GeoTIFF with same geo metadata as meta."""
    meta_out = meta.copy()
    meta_out.update({
        "count": 1,
        "dtype": rasterio.uint8,
        "compress": "lzw"
    })
    with rasterio.open(out_path, "w", **meta_out) as dst:
        dst.write(mask.astype(rasterio.uint8), 1)

def find_tile_pairs(download_dir):
    """Find pairs by matching tile_<id>_t1.tif and tile_<id>_t2.tif"""
    t1_files = sorted(glob.glob(os.path.join(download_dir, "tile_*_t1.tif")))
    pairs = []
    for t1 in t1_files:
        base = os.path.basename(t1)
        tid = base.replace("_t1.tif","")
        t2 = os.path.join(download_dir, f"{tid}_t2.tif")
        if os.path.exists(t2):
            pairs.append((t1,t2))
    return pairs

def main():
    pairs = find_tile_pairs(DOWNLOAD_DIR)
    print(f"Found {len(pairs)} tile pairs.")
    rows = []
    for t1_path, t2_path in tqdm(pairs):
        tid = os.path.basename(t1_path).replace("_t1.tif","")
        try:
            arr1, meta1 = read_tile(t1_path)
            arr2, meta2 = read_tile(t2_path)
        except Exception as e:
            print(f"Failed reading {tid}: {e}")
            continue

        # Basic check: bands count enough
        if arr1.shape[0] < 4 or arr2.shape[0] < 4:
            print(f"Skipping {tid}: not enough bands (arr1 bands {arr1.shape[0]})")
            continue

        # Assuming BAND_ORDER: B2,B3,B4,B8,B11 -> then red=B4 index=2, nir=B8 index=3 (0-based)
        red_idx = 2
        nir_idx = 3

        ndvi1 = compute_ndvi(arr1, red_idx, nir_idx)
        ndvi2 = compute_ndvi(arr2, red_idx, nir_idx)

        ndvi_diff = ndvi1 - ndvi2  # positive -> loss (greener -> less green)
        mask_raw = ndvi_diff > NDVI_THRESH

        mask = postprocess_mask(mask_raw)

        out_mask_path = os.path.join(OUT_DIR, "masks", f"{tid}_mask.tif")
        write_mask(mask, meta1, out_mask_path)

        # Some stats
        pct_loss = 100.0 * mask.sum() / (mask.size + 1e-9)
        rows.append({"tile_id": tid, "t1": t1_path, "t2": t2_path, "mask": out_mask_path, "pct_loss": pct_loss})
    # save metadata
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "masks", "masks_metadata.csv"), index=False)
    print("Mask creation finished. Metadata saved to dataset/masks/masks_metadata.csv")

    # create splits
    tile_ids = list(df['tile_id'])
    random.shuffle(tile_ids)
    n = len(tile_ids)
    ntest = int(n * TEST_FRAC)
    nval = int(n * VAL_FRAC)
    test_ids = tile_ids[:ntest]
    val_ids = tile_ids[ntest:ntest+nval]
    train_ids = tile_ids[ntest+nval:]

    # create directories
    for split, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for sub in ["before","after","mask"]:
            os.makedirs(os.path.join(OUT_DIR, split, sub), exist_ok=True)
        for tid in ids:
            row = df[df['tile_id']==tid].iloc[0]
            # copy files (fast rename/copy)
            src_t1 = row['t1']
            src_t2 = row['t2']
            src_mask = row['mask']
            dst_t1 = os.path.join(OUT_DIR, split, "before", os.path.basename(src_t1))
            dst_t2 = os.path.join(OUT_DIR, split, "after", os.path.basename(src_t2))
            dst_mask = os.path.join(OUT_DIR, split, "mask", os.path.basename(src_mask))
            # use copy to preserve downloads
            import shutil
            shutil.copy(src_t1, dst_t1)
            shutil.copy(src_t2, dst_t2)
            shutil.copy(src_mask, dst_mask)

    print(f"Dataset organized: train/val/test with fractions {1-VAL_FRAC-TEST_FRAC}/{VAL_FRAC}/{TEST_FRAC}")

if __name__ == "__main__":
    main()
