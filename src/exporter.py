# import ee
# from .logger import get_logger

# logger = get_logger()

# def export_image(image,region,out_name,bucket="S2_exports",scale = 10):
#     logger.info(f"Exporting {out_name} to Drive...")

#     task = ee.batch.Export.image.toCloudStorage(
#         image = image,
#         description=out_name,
#         bucket=bucket,
#         fileNamePrefix=out_name,
#         region=region,
#         scale=scale,
#         maxPixels=1e13,
#         crs="EPSG:4326"
#     )

#     task.start()
#     logger.info(f"Export started: {out_name}")
#     return task
import ee
import requests
import os
from .logger import get_logger

logger = get_logger()

def export_image(image, region, out_name, scale=10):
    """
    Downloads an EE image LOCALLY using direct download URLs.
    Saves as: ./downloads/<out_name>.tif
    """

    # Ensure output directory exists
    os.makedirs("downloads", exist_ok=True)
    out_path = os.path.join("downloads", f"{out_name}.tif")

    logger.info(f"Downloading locally: {out_name}")

    # Create download request
    request = {
        "image": image,
        "scale": scale,
        "region": region.getInfo()['coordinates'],
        "filePerBand": False
    }

    try:
        # Generate download URL
        download_id = ee.data.getDownloadId(request)
        url = ee.data.makeDownloadUrl(download_id)

        # Stream the download
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Saved locally: {out_path}")
        return out_path

    except Exception as e:
        logger.error(f"Local download failed for {out_name}: {e}")
        return None
    