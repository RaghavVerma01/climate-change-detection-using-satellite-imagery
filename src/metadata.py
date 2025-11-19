import csv
import os

def write_metadata(region_name,metadata_rows):
    os.makedirs("metadata",exist_ok=True)
    out_path = f"metadata/{region_name}_metadata.csv"

    header = [
        "tile_id","region_name",
        "t1_start","t1_end",
        "t2_start","t2_end",
        "bbox"
    ]

    with open(out_path,"w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(header)
        writer.writerows(metadata_rows)

    return out_path