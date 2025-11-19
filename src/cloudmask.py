import ee

# Cloud mask based on SCL + cloud probability if available
def s2_mask_clouds(image):

    # Select the Scene Classification Layer
    scl = image.select("SCL")

    # Good SCL classes (keep):
    # 4 = vegetation
    # 5 = bare soil
    # 6 = water
    # 7 = unclassified
    # 11 = snow (optional)
    good_classes = scl.eq(4).Or(
        scl.eq(5)).Or(
        scl.eq(6)).Or(
        scl.eq(7)
    )

    # Remove clouds, shadows, cirrus, snow
    return (
        image.updateMask(good_classes)
             .copyProperties(image, image.propertyNames())
    )
