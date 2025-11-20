import os
import shutil
import random

# --- CONFIG ---
SOURCE_IMG_DIR = "downloads"
SOURCE_MASK_DIR = "masks"
BASE_DATA_DIR = "dataset"
TRAIN_RATIO = 0.8 # 80% Train, 20% Validation

# --- SETUP ---
# Clean rebuild of dataset folder
if os.path.exists(BASE_DATA_DIR):
    shutil.rmtree(BASE_DATA_DIR)

for split in ['train', 'val']:
    os.makedirs(os.path.join(BASE_DATA_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DATA_DIR, split, 'masks'), exist_ok=True)

# --- FILTER ---
# Only pick tiles that have both an Image pair AND a Mask
all_masks = [f for f in os.listdir(SOURCE_MASK_DIR) if f.endswith("_mask.tif")]
valid_tids = []

print("Validating pairs...")
for m in all_masks:
    tid = m.replace("_mask.tif", "")
    t1 = os.path.join(SOURCE_IMG_DIR, f"{tid}_t1.tif")
    t2 = os.path.join(SOURCE_IMG_DIR, f"{tid}_t2.tif")
    
    # Check if T1/T2 exist and aren't empty error files
    if os.path.exists(t1) and os.path.exists(t2):
        if os.path.getsize(t1) > 1000 and os.path.getsize(t2) > 1000:
            valid_tids.append(tid)

print(f"Found {len(valid_tids)} complete tile pairs.")

# --- SPLIT ---
random.shuffle(valid_tids)
split_idx = int(len(valid_tids) * TRAIN_RATIO)
train_ids = valid_tids[:split_idx]
val_ids = valid_tids[split_idx:]

def copy_dataset(tids, split):
    print(f"Populating {split} set with {len(tids)} tiles...")
    for tid in tids:
        # Source
        src_t1 = os.path.join(SOURCE_IMG_DIR, f"{tid}_t1.tif")
        src_t2 = os.path.join(SOURCE_IMG_DIR, f"{tid}_t2.tif")
        src_mask = os.path.join(SOURCE_MASK_DIR, f"{tid}_mask.tif")
        
        # Dest (Renaming for clarity if needed, but keeping ID is safer)
        shutil.copy(src_t1, os.path.join(BASE_DATA_DIR, split, 'images', f"{tid}_t1.tif"))
        shutil.copy(src_t2, os.path.join(BASE_DATA_DIR, split, 'images', f"{tid}_t2.tif"))
        shutil.copy(src_mask, os.path.join(BASE_DATA_DIR, split, 'masks', f"{tid}_mask.tif"))

copy_dataset(train_ids, 'train')
copy_dataset(val_ids, 'val')

print("\n✅ Dataset Organization Complete.")
print(f"Training samples: {len(train_ids)}")
print(f"Validation samples: {len(val_ids)}")