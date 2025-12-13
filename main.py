# --- CELL 1: INSTALL LIBRARIES ---

# !pip install -q segmentation-models-pytorch rasterio

import os
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import torch.nn.functional as F

# --- CONFIGURATION ---
# Kaggle Input Paths are tricky. 
# usually: /kaggle/input/<your-dataset-name>/dataset
# We will find it dynamically below.
BATCH_SIZE = 8          
EPOCHS = 20
LEARNING_RATE = 0.0001
ENCODER = "resnet34"    # Upgraded to ResNet34 since we have better GPUs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"✅ Using device: {DEVICE}")

# --- FIND THE DATA PATH ---
# Kaggle puts datasets in /kaggle/input
# Let's find where your 'dataset' folder ended up
INPUT_ROOT = "D:/Projects/climate-change-detection-using-satellite-imagery"
DATA_DIR = ""

for root, dirs, files in os.walk(INPUT_ROOT):
    if "train" in dirs and "val" in dirs:
        DATA_DIR = root
        print(f"✅ Found dataset at: {DATA_DIR}")
        break

if not DATA_DIR:
    print("❌ Could not auto-find dataset! Please check 'Add Input' on the right.")
    # Fallback manually if needed:
    # DATA_DIR = "/kaggle/input/your-dataset-name-here/dataset"

# --- DATASET CLASS ---
class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=(256, 256)):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.files = [f for f in os.listdir(self.mask_dir) if f.endswith("_mask.tif")]
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mask_name = self.files[idx]
        tid = mask_name.replace("_mask.tif", "")
        
        t1_path = os.path.join(self.img_dir, f"{tid}_t1.tif")
        t2_path = os.path.join(self.img_dir, f"{tid}_t2.tif")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 1. Load T1 (Normalize 0-1)
        with rasterio.open(t1_path) as src:
            t1 = src.read().astype("float32") / 10000.0
            
        # 2. Load T2 (Normalize 0-1)
        with rasterio.open(t2_path) as src:
            t2 = src.read().astype("float32") / 10000.0

        # 3. Stack -> (10, H, W)
        image = np.concatenate([t1, t2], axis=0)

        # 4. Load Mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype("float32")
            mask[mask > 0] = 1.0 
        mask = np.expand_dims(mask, axis=0) # (1, H, W)

        # --- THE FIX: RESIZE TO FIXED DIMENSIONS ---
        # Convert to Tensor
        tensor_img = torch.from_numpy(image)
        tensor_mask = torch.from_numpy(mask)

        # Resize Image (Bilinear interpolation is best for continuous data)
        # interpolate expects (Batch, Channel, H, W), so we unsqueeze(0) then squeeze(0)
        tensor_img = F.interpolate(
            tensor_img.unsqueeze(0), 
            size=self.image_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)

        # Resize Mask (Nearest Exact is MANDATORY for masks to keep values 0 or 1)
        tensor_mask = F.interpolate(
            tensor_mask.unsqueeze(0), 
            size=self.image_size, 
            mode='nearest'
        ).squeeze(0)

        return tensor_img, tensor_mask

# --- TRAINING LOOP ---
def train_model():
    if not DATA_DIR: return

    train_dataset = ChangeDetectionDataset(DATA_DIR, split="train")
    val_dataset = ChangeDetectionDataset(DATA_DIR, split="val")

    # Increased workers for Kaggle
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"🚀 Starting training with {len(train_dataset)} samples...")

    # U-Net with 10 input channels
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=None, 
        in_channels=10, 
        classes=1, 
        activation=None
    )
    model.to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    
    # History for plotting
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # Save to /kaggle/working (This is the output folder)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "/kaggle/working/best_model.pth")
            print("  --> 💾 Model Saved!")

    # Plot results
    plt.figure(figsize=(10,5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title("Training Progress")
    plt.legend()
    plt.savefig("/kaggle/working/training_plot.png")
    plt.show()

if __name__ == "__main__":
    train_model()