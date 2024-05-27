import numpy as np

def remove_cross(image):
    # hyperbola parameters
    height, lenght = image.shape
    center_x = lenght // 2
    center_y = height // 2
    radius = 400

    # create an array with coordinates x e y
    yy, xx = np.ogrid[:height, :lenght]
    # calculate the area of the cross
    cross_area = abs((xx - center_x) * (yy - center_y)) <= radius
    # set pixels to 0
    image[cross_area] = 0
    return image


def find_fingerprint(image,step):
    # parameters
    pixel_totali = image.size
    pixel_zero_prec = 0
    fingerprint = []
    # range in which search the fingerprint (percentage)
    range_min = 0.995
    range_max = 0.998

    for i in range(255):
        image[image < i ] = 0
        pixel_zero = np.count_nonzero(image == 0)
        if(pixel_zero>pixel_zero_prec+step):
            pixel_zero_prec = pixel_zero
            if((pixel_zero > pixel_totali*range_min) & (pixel_zero < pixel_totali*range_max)):
                fingerprint.append(image.copy())
    return fingerprint