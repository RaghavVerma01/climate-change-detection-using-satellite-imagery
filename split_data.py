import os
import shutil
import random
import rasterio
import numpy as np

# --- CONFIG ---
SOURCE_IMG_DIR = "downloads"
SOURCE_MASK_DIR = "masks"
BASE_DATA_DIR = "dataset"
TRAIN_RATIO = 0.8

# Strategy: How many "boring" (no change) tiles to keep?
# 0.10 means "Keep 10% of the empty tiles, delete the rest"
EMPTY_TILE_KEEP_RATIO = 0.15 

# --- SETUP ---
if os.path.exists(BASE_DATA_DIR):
    shutil.rmtree(BASE_DATA_DIR)

for split in ['train', 'val']:
    os.makedirs(os.path.join(BASE_DATA_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DATA_DIR, split, 'masks'), exist_ok=True)

# --- FILTERING LOGIC ---
all_masks = [f for f in os.listdir(SOURCE_MASK_DIR) if f.endswith("_mask.tif")]
valid_tids = []
dropped_empty = 0
dropped_error = 0

print(f"Scanning {len(all_masks)} generated masks...")

for m in all_masks:
    tid = m.replace("_mask.tif", "")
    t1_path = os.path.join(SOURCE_IMG_DIR, f"{tid}_t1.tif")
    t2_path = os.path.join(SOURCE_IMG_DIR, f"{tid}_t2.tif")
    mask_path = os.path.join(SOURCE_MASK_DIR, m)
    
    # 1. Integrity Check
    if not (os.path.exists(t1_path) and os.path.exists(t2_path)):
        dropped_error += 1
        continue
        
    if os.path.getsize(t1_path) < 1000: # Skip 1KB error files
        dropped_error += 1
        continue

    # 2. Content Check (The Smart Filter)
    try:
        with rasterio.open(mask_path) as src:
            data = src.read(1)
            # Check if mask has any white pixels (Change)
            has_change = np.any(data > 0)
            
        if has_change:
            # ALWAYS keep interesting tiles
            valid_tids.append(tid)
        else:
            # RANDOMLY drop most empty tiles, keep a few
            if random.random() < EMPTY_TILE_KEEP_RATIO:
                valid_tids.append(tid)
            else:
                dropped_empty += 1
                
    except Exception as e:
        print(f"Error reading {mask_path}: {e}")
        dropped_error += 1

print("--- Filtering Report ---")
print(f"🚫 Dropped (Corrupt/Missing): {dropped_error}")
print(f"🧹 Dropped (Empty/No Change): {dropped_empty}")
print(f"✅ Kept (Balanced Dataset):   {len(valid_tids)}")

# --- SPLIT AND COPY ---
random.shuffle(valid_tids)
split_idx = int(len(valid_tids) * TRAIN_RATIO)
train_ids = valid_tids[:split_idx]
val_ids = valid_tids[split_idx:]

def copy_dataset(tids, split):
    for tid in tids:
        src_t1 = os.path.join(SOURCE_IMG_DIR, f"{tid}_t1.tif")
        src_t2 = os.path.join(SOURCE_IMG_DIR, f"{tid}_t2.tif")
        src_mask = os.path.join(SOURCE_MASK_DIR, f"{tid}_mask.tif")
        
        shutil.copy(src_t1, os.path.join(BASE_DATA_DIR, split, 'images', f"{tid}_t1.tif"))
        shutil.copy(src_t2, os.path.join(BASE_DATA_DIR, split, 'images', f"{tid}_t2.tif"))
        shutil.copy(src_mask, os.path.join(BASE_DATA_DIR, split, 'masks', f"{tid}_mask.tif"))

copy_dataset(train_ids, 'train')
copy_dataset(val_ids, 'val')

print(f"\nDataset ready! Train: {len(train_ids)} | Val: {len(val_ids)}")