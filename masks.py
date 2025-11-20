import os
import numpy as np
import rasterio

# --- CONFIGURATION ---
INPUT_DIR = "downloads"
OUTPUT_DIR = "masks"

# Thresholds for Deforestation Logic
NDVI_THRESHOLD_DROP = 0.25  # Vegetation index must drop by at least this much
MIN_FOREST_NDVI = 0.50      # T1 must be at least this 'green' to count as forest (Lowered slightly to be safe)
MAX_NON_FOREST_NDVI = 0.40  # T2 must be below this to count as 'deforested'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_band(src, band_idx):
    """
    Reads a specific band index from the raster.
    Converts '0' (No Data / Cloud Mask) to NaN so it doesn't mess up the math.
    """
    data = src.read(band_idx).astype('float32')
    data[data == 0] = np.nan
    return data

def calculate_ndvi(nir, red):
    """
    Standard NDVI formula: (NIR - Red) / (NIR + Red)
    """
    # Add epsilon 1e-8 to avoid division by zero errors
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi

def process_tile(tid):
    t1_path = os.path.join(INPUT_DIR, f"{tid}_t1.tif")
    t2_path = os.path.join(INPUT_DIR, f"{tid}_t2.tif")
    out_path = os.path.join(OUTPUT_DIR, f"{tid}_mask.tif")

    # Skip if pair is incomplete
    if not (os.path.exists(t1_path) and os.path.exists(t2_path)):
        return

    with rasterio.open(t1_path) as src_t1, rasterio.open(t2_path) as src_t2:
        
        # --- STEP 1: READ BANDS ---
        # Your exported bands are: ["B2", "B3", "B4", "B8", "B11"]
        # Rasterio uses 1-based indexing:
        # 1=Blue, 2=Green, 3=Red (B4), 4=NIR (B8), 5=SWIR
        
        red_t1 = read_band(src_t1, 3)
        nir_t1 = read_band(src_t1, 4)
        
        red_t2 = read_band(src_t2, 3)
        nir_t2 = read_band(src_t2, 4)

        # --- STEP 2: CALCULATE NDVI ---
        # Numpy handles NaNs automatically (NaN + Number = NaN)
        ndvi_t1 = calculate_ndvi(nir_t1, red_t1)
        ndvi_t2 = calculate_ndvi(nir_t2, red_t2)

        # --- STEP 3: APPLY LOGIC ---
        
        # Rule A: Significant Drop
        diff = ndvi_t1 - ndvi_t2
        has_dropped = diff > NDVI_THRESHOLD_DROP
        
        # Rule B: T1 was actually Forest (filters out urban/water/bare soil in T1)
        was_forest = ndvi_t1 > MIN_FOREST_NDVI
        
        # Rule C: T2 is now Non-Forest (filters out slight seasonal browning)
        now_not_forest = ndvi_t2 < MAX_NON_FOREST_NDVI
        
        # Rule D: Data Validity (Ignore pixels that were clouds in either image)
        valid_data = (~np.isnan(ndvi_t1)) & (~np.isnan(ndvi_t2))

        # Combine Rules (All must be True)
        final_mask_bool = has_dropped & was_forest & now_not_forest & valid_data

        # Convert to clean Uint8 (0 for No Change, 1 for Change)
        final_mask_uint8 = final_mask_bool.astype('uint8')

        # --- STEP 4: SAVE MASK ---
        # Copy metadata from source but update to single-band integer
        profile = src_t1.profile
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
            compress='lzw'
        )

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(final_mask_uint8, 1)

        # --- REPORTING ---
        # Calculate percentage of pixels marked as change
        total_pixels = final_mask_uint8.size
        change_pixels = np.count_nonzero(final_mask_uint8)
        pct_change = (change_pixels / total_pixels) * 100
        
        if pct_change > 0:
            print(f"✅ {tid}: Change detected: {pct_change:.2f}%")
        else:
            print(f"🔹 {tid}: No deforestation detected.")

if __name__ == "__main__":
    # identify all unique Tile IDs based on T1 files
    files = os.listdir(INPUT_DIR)
    tile_ids = sorted(list(set([f.split("_t")[0] for f in files if "_t1.tif" in f])))
    
    print(f"Starting mask generation for {len(tile_ids)} pairs...")
    
    for tid in tile_ids:
        try:
            process_tile(tid)
        except Exception as e:
            print(f"❌ Error processing {tid}: {e}")
            
    print("\n--- Processing Complete ---")
    print(f"Masks saved to: {os.path.abspath(OUTPUT_DIR)}")