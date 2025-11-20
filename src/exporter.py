# # import ee
# # from .logger import get_logger

# # logger = get_logger()

# # def export_image(image,region,out_name,bucket="S2_exports",scale = 10):
# #     logger.info(f"Exporting {out_name} to Drive...")

# #     task = ee.batch.Export.image.toCloudStorage(
# #         image = image,
# #         description=out_name,
# #         bucket=bucket,
# #         fileNamePrefix=out_name,
# #         region=region,
# #         scale=scale,
# #         maxPixels=1e13,
# #         crs="EPSG:4326"
# #     )

# #     task.start()
# #     logger.info(f"Export started: {out_name}")
# #     return task
# import ee
# import requests
# import os
# from .logger import get_logger

# logger = get_logger()

# def export_image(image, region, out_name, scale=10):
#     """
#     Downloads an EE image LOCALLY using direct download URLs.
#     Saves as: ./downloads/<out_name>.tif
#     """

#     # Ensure output directory exists
#     os.makedirs("downloads", exist_ok=True)
#     out_path = os.path.join("downloads", f"{out_name}.tif")

#     logger.info(f"Downloading locally: {out_name}")

#     # Create download request
#     request = {
#         "image": image,
#         "scale": scale,
#         "region": region.getInfo()['coordinates'],
#         "filePerBand": False
#     }

#     try:
#         # Generate download URL
#         download_id = ee.data.getDownloadId(request)
#         url = ee.data.makeDownloadUrl(download_id)

#         # Stream the download
#         response = requests.get(url, stream=True)
#         response.raise_for_status()

#         with open(out_path, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)

#         logger.info(f"Saved locally: {out_path}")
#         return out_path

#     except Exception as e:
#         logger.error(f"Local download failed for {out_name}: {e}")
#         return None
    
# exporter.py
import ee
import requests
import os
import json
from .logger import get_logger

logger = get_logger()

def export_image(image, region, out_name, scale=10):
    """
    Downloads an EE image LOCALLY using direct download URLs.
    Saves as: ./downloads/<out_name>.tif
    """
    os.makedirs("downloads", exist_ok=True)
    out_path = os.path.join("downloads", f"{out_name}.tif")

    logger.info(f"Requesting download URL: {out_name}")

    # CRITICAL FIX 1: safely convert ee.Geometry to coordinate list
    # If 'region' is an ee.Geometry, get the coordinates. 
    # If it's already a list, leave it.
    try:
        if isinstance(region, ee.Geometry):
            region_coords = region.bounds().getInfo()['coordinates']
        else:
            region_coords = region
    except Exception as e:
        logger.error(f"Region serialization failed: {e}")
        return None

    # CRITICAL FIX 2: Explicitly set format and crs
    params = {
        'name': out_name,
        'scale': scale,
        'region': region_coords,
        'crs': 'EPSG:4326', 
        'format': 'GEO_TIFF'
    }

    try:
        # Get the URL directly from the image object
        # This is often more stable than ee.data.getDownloadId for small tiles
        url = image.getDownloadURL(params)
        
        logger.info(f"Downloading content...")
        response = requests.get(url, stream=True)
        
        # Check for request errors (400, 401, 500)
        if response.status_code != 200:
            logger.error(f"Failed to download. Status: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None

        # Write file
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Check if file is actually an image (header check)
        if os.path.getsize(out_path) < 1000:
             logger.warning(f"File {out_name} is suspiciously small. Check for error text inside.")

        logger.info(f"Saved locally: {out_path}")
        return out_path

    except Exception as e:
        logger.error(f"Local download failed for {out_name}: {e}")
        return None