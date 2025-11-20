import os
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# --- CONFIGURATION ---
DATA_DIR = "dataset"
BATCH_SIZE = 4        # Reduce to 2 if you run out of GPU memory
EPOCHS = 15
LEARNING_RATE = 0.0001
ENCODER = "resnet18"  # Lightweight backbone
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# --- DATASET CLASS ---
class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        
        # Find all mask files
        self.files = [f for f in os.listdir(self.mask_dir) if f.endswith("_mask.tif")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mask_name = self.files[idx]
        tid = mask_name.replace("_mask.tif", "")
        
        # Paths
        t1_path = os.path.join(self.img_dir, f"{tid}_t1.tif")
        t2_path = os.path.join(self.img_dir, f"{tid}_t2.tif")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Read T1 and T2
        # Sentinel-2 is Uint16 (0-10000). Neural Nets like 0-1.
        # We divide by 10000.0 to normalize.
        with rasterio.open(t1_path) as src:
            t1 = src.read().astype("float32") / 10000.0
            
        with rasterio.open(t2_path) as src:
            t2 = src.read().astype("float32") / 10000.0

        # Stack T1 and T2 (5 bands + 5 bands = 10 channels)
        # Shape becomes (10, Height, Width)
        image = np.concatenate([t1, t2], axis=0)

        # Read Mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype("float32")
            # Ensure mask is 0 or 1
            mask[mask > 0] = 1.0 
            
        # Add channel dimension to mask: (H, W) -> (1, H, W)
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask)

# --- TRAINING SETUP ---
def train():
    # 1. Load Data
    train_dataset = ChangeDetectionDataset(DATA_DIR, split="train")
    val_dataset = ChangeDetectionDataset(DATA_DIR, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training on {len(train_dataset)} samples. Validating on {len(val_dataset)} samples.")

    # 2. Create Model
    # We use Unet. Input channels = 10 (5 bands * 2 images). Classes = 1 (Change).
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=None, # No pre-trained weights because we have 10 channels, not 3 (RGB)
        in_channels=10, 
        classes=1, 
        activation=None # We use BCEWithLogitsLoss, so no activation at end
    )
    model.to(DEVICE)

    # 3. Loss and Optimizer
    criterion = torch.nn.BCEWithLogitsLoss() # Binary Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Step
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

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  --> Model Saved!")

if __name__ == "__main__":
    train()