# import ee
# from .cloudmask import s2_mask_clouds
# from .logger import get_logger

# logger = get_logger()

# BANDS = ["B2","B3","B4","B8","B11"] #BLUE, GREEN, RED, NIR, SWIR

# def get_s2_composite(bbox,start_date,end_date,scale = 10):
#     logger.info(f"Creating Sentinel-2 composite for {start_date} -> {end_date}")

#     region = ee.geometry.Geometry.Rectangle(bbox)
#     col = (
#         ee.imagecollection.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
#         .filterBounds(region)
#         .filterDate(start_date,end_date)
#         .map(s2_mask_clouds)
#     )

#     composite = col.median().select(BANDS)
#     composite = composite.clip(region)
#     logger.info("Composite Created")

#     return composite,region
# s2_fetcher.py
import ee
from .cloudmask import s2_mask_clouds
from .logger import get_logger

logger = get_logger()

# Sentinel-2 bands are natively unsigned 16-bit integers
BANDS = ["B2","B3","B4","B8","B11"] 

def get_s2_composite(bbox, start_date, end_date):
    logger.info(f"Creating Sentinel-2 composite for {start_date} -> {end_date}")

    region = ee.Geometry.Rectangle(bbox)
    
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .map(s2_mask_clouds)
    )

    # Create median composite
    composite = col.median().select(BANDS)
    
    # CRITICAL FIX: Clip AND Cast to Uint16 to save space and fix formatting
    # median() creates Floats. We convert back to Int for standard TIFFs.
    composite = composite.clip(region).toUint16()

    logger.info("Composite Created")
    return composite, region