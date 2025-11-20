import rasterio
import matplotlib.pyplot as plt
import numpy as np

t1 = "downloads/tile_00000_t1.tif"
t2 = "downloads/tile_00000_t2.tif"
mask = "dataset/masks/tile_00000_mask.tif"

with rasterio.open(t1) as src:
    img1 = src.read([3,2,1])  # B4,B3,B2 as RGB (1-based in rasterio -> indices given as band numbers)
    img1 = (img1 / img1.max())  # quick norm
with rasterio.open(t2) as src:
    img2 = src.read([3,2,1])  # B4,B3,B2 as RGB (1-based in rasterio -> indices given as band numbers)
    img2 = (img2 / img2.max())  # quick norm

with rasterio.open(mask) as m:
    msk = m.read(1)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(np.transpose(img1, (2,1,0))); plt.title("t1 (RGB)")
plt.subplot(1,2,1); plt.imshow(np.transpose(img2, (1,2,0))); plt.title("t2 (RGB)")
plt.subplot(1,2,2); plt.imshow(msk, cmap='gray'); plt.title("NDVI-based mask")
plt.show()
