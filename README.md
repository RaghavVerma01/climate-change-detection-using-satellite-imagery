# Satellite Climate Change Detection from Satellite Imagery

This project explores detecting environmental changes such as **deforestation** using multi-temporal **Sentinel-2 satellite imagery**.  
The repository implements an end-to-end workflow including **satellite data acquisition, preprocessing, automated label generation, and model training** for change detection.

The goal of this project is to build a reproducible pipeline for analyzing climate-related land cover changes using remote sensing and deep learning.

---

# Technologies

- **Python**
- **PyTorch** – model training and deep learning framework  
- **Google Earth Engine API** – satellite imagery retrieval  
- **Sentinel-2 Satellite Data** – multispectral Earth observation imagery  
- **NumPy & SciPy** – numerical processing  
- **Rasterio** – geospatial raster data processing  
- **Matplotlib** – visualization of outputs  
- **Git & GitHub** – version control and project management  

---

# Features

- Automated **Sentinel-2 satellite imagery retrieval** using Google Earth Engine
- **Cloud masking and temporal compositing** for cleaner satellite observations
- **Spatial tiling pipeline** for generating training-ready image tiles
- Automatic **NDVI-based change mask generation** for weakly supervised labeling
- Dataset splitting and preprocessing utilities
- **PyTorch-based change detection training pipeline**
- Visualization tools for inspecting predictions and change masks

---

# Process

The project follows a multi-stage pipeline designed to process raw satellite data into training-ready inputs.

## 1. Satellite Data Collection

Sentinel-2 imagery is fetched using the **Google Earth Engine API** for selected regions and time periods representing *before* and *after* observations.

## 2. Preprocessing

The pipeline applies:

- Cloud masking
- Temporal compositing
- Spatial tiling of large regions into smaller patches

These steps ensure the dataset is clean and suitable for machine learning workflows.

## 3. Change Mask Generation

Instead of relying on manually labeled data, this project generates **weak labels** using an NDVI-based heuristic:
NDVI = (NIR − Red) / (NIR + Red)

Pixels that show a strong NDVI decrease between time steps are considered potential deforestation events.

This provides an automatic way to produce training masks without manual annotation.

## 4. Dataset Preparation

Tiles and generated masks are organized into:

- Training set
- Validation set
- Test set

Metadata files track the geographic location and temporal windows of each tile.

## 5. Model Training

A **PyTorch segmentation model (U-Net style)** is trained to predict pixel-level change masks using paired before/after satellite tiles.

---

# What I Learned

Working on this project helped me gain practical experience with:

- Handling **real-world satellite imagery data**
- Building **data pipelines for geospatial datasets**
- Using the **Google Earth Engine API** for large-scale Earth observation analysis
- Implementing **weakly-supervised labeling strategies** when ground truth data is limited
- Designing and training **deep learning models for image segmentation**
- Debugging issues related to coordinate systems, raster formats, and large datasets

It also provided experience in building an **end-to-end machine learning pipeline**, from data collection to model evaluation.

---

# Running the Project

## 1. Clone the repository

```bash
git clone https://github.com/yourusername/climate-change-detection.git
cd climate-change-detection
```
## 2. Install dependencies
```bash
pip install -r requirements.txt
```
## 3. Authenticate Google Earth Engine
```bash
earthengine authenticate
```
## 4. Run the data pipeline
```bash
python run.py
```
This step will:
- Download satellite tiles
- generate temporal composites
- store tiles for dataset creation

## 5. Generate change masks
```bash
python mask_generator.py
```
## 6. Split the dataset
```bash
python split_data.py
```
## 7. Train the model
```bash
python train.py
```
## 8. Validate model
```bash
python validate.py
```
### Note
Raw satellite imagery and generated datasets are not included in this repository due to storage constraints.
However, the full pipeline required to reproduce the dataset is provided.
To download custom tiles from different regions, edit config/regions.json
