import math

def generate_tile_grid(bbox,tile_size_m = 2560):
    # tile_size_m = size covered in meters (256px * 10m)
    # bbox = [xmin,ymin,xmax,ymax]

    xmin,ymin,xmax,ymax = bbox
    lon_step = (xmax-xmin)/math.ceil((xmax-xmin)/0.0256)
    lat_step = (ymax-ymin)/math.ceil((ymax-ymin)/0.0256)

    tiles = []

    tid = 0
    x = xmin
    while x<xmax:
        y = ymin
        while y<ymax:
            tiles.append({
                "tile_id":f"tile_{tid:05d}",
                "bbox":[x,y,min(x+lon_step,xmax),min(y+lat_step,ymax)]
            })
            y+=lat_step
            tid +=1
        x+=lon_step
    return tiles