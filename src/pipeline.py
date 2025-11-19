import ee
import json
from .logger import get_logger
from .tilegrid import generate_tile_grid
from .s2_fetcher import get_s2_composite
from .exporter import export_image
from .metadata import write_metadata

logger = get_logger()

def run_pipeline(config_path="config/regions.json"):
    ee.Initialize()

    logger.info("Loading configuration...")
    config = json.load(open(config_path))

    for region_name, cfg in config.items():
        logger.info(f"Processing region: {region_name}")
        bbox=cfg["bbox"]
        t1=cfg["t1"]
        t2=cfg["t2"]

        tiles =generate_tile_grid(bbox)
        metadata_rows = []
        for tile in tiles:
            tid = tile["tile_id"]
            tbbox = tile["bbox"]

            logger.info(f"Processing tile: {tid}")

            #t1 composite
            img_t1, region = get_s2_composite(tbbox,t1["start"],t1["end"])
            info = img_t1.getInfo()
            print("Here is info: ",info)
            export_image(img_t1,region,f"{tid}_t1")

            #t2 composite
            img_t2,region = get_s2_composite(tbbox,t2["start"],t2["end"])
            export_image(img_t2,region,f"{tid}_t2")

            metadata_rows.append([
                tid,region_name,
                t1["start"],t1["end"],
                t2["start"],t2["end"],
                f"{tid}_t1.tif",
                f"{tid}_t2.tif",
                tbbox
            ])

        #Save metadata
        write_metadata(region_name,metadata_rows)
    
    logger.info("Pipeline completed successfully")