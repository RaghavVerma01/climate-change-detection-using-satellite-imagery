import ee
from .cloudmask import s2_mask_clouds
from .logger import get_logger

logger = get_logger()

BANDS = ["B2","B3","B4","B8","B11"] #BLUE, GREEN, RED, NIR, SWIR

def get_s2_composite(bbox,start_date,end_date,scale = 10):
    logger.info(f"Creating Sentinel-2 composite for {start_date} -> {end_date}")

    region = ee.geometry.Geometry.Rectangle(bbox)
    col = (
        ee.imagecollection.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date,end_date)
        .map(s2_mask_clouds)
    )

    composite = col.median().select(BANDS)
    composite = composite.clip(region)
    logger.info("Composite Created")

    return composite,region