import ee, json, math
ee.Initialize()  # or your project id

# Replace these two lines with the code you use to build a single composite & region for one tile
# e.g. call your get_s2_composite to produce img and region; or paste a known region list
# from your pipeline: img_t1, region = get_s2_composite(tile_bbox, t1_start, t1_end)
# For ad-hoc check paste the bbox coords here:
tile_bbox = [-63.475, -4.325, -63.45, -4.3]  # example small tile inside your bbox
region = ee.Geometry.Rectangle(tile_bbox)
col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
       .filterBounds(region)
       .filterDate('2018-01-01', '2018-12-31'))

# Use your cloudmask function here; if not available, do a no-mask run to inspect raw:
# from cloudmask import s2_mask_clouds
# col = col.map(s2_mask_clouds)

# For diagnostics, produce a median composite (no clip yet)
composite = col.median()

# 1) Band names
try:
    bands = composite.bandNames().getInfo()
except Exception as e:
    bands = f"ERROR: {e}"

# 2) Projection info & nominal scale
try:
    proj = composite.projection().getInfo()
except Exception as e:
    proj = f"ERROR: {e}"
try:
    scale = composite.projection().nominalScale().getInfo()
except Exception as e:
    scale = f"ERROR: {e}"

# 3) Region info and bounding box size (deg -> approx meters)
region_info = region.getInfo()
coords = region_info['coordinates']
min_lon = min([c[0] for c in coords[0]])
max_lon = max([c[0] for c in coords[0]])
min_lat = min([c[1] for c in coords[0]])
max_lat = max([c[1] for c in coords[0]])
width_deg = max_lon - min_lon
height_deg = max_lat - min_lat
# approximate meters: lat_m ~= deg*111320; lon_m ~= deg*111320*cos(mean_lat)
mean_lat = (min_lat + max_lat) / 2.0
lat_m = height_deg * 111320
lon_m = width_deg * 111320 * abs(math.cos(math.radians(mean_lat)))

info = {
    'bands': bands,
    'projection': proj,
    'nominal_scale': scale,
    'region_bbox_deg': tile_bbox,
    'region_bbox_m_approx': {'width_m': lon_m, 'height_m': lat_m},
}

print(json.dumps(info, indent=2))
