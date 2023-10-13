import numpy as np

def log_env(debug, progress):
    if debug is True:
        debug = print
    if debug is False:
        debug = lambda *args: None

    if progress is True:
        import tqdm
        progress = tqdm.tqdm
    if progress is False:
        progress = lambda x, **kwargs: x

    return debug, progress


def standardize_format(image, expected_dims):
    """ Converts an input image into the format
    expected, this being:
    (width, height), (channels, width, height), (cycle, channels, width, height)
    depending on the expected_dims. """

    image = image.squeeze()

    if len(image.shape) == 3:
        if expected_dims < 3:
            raise ValueError("Expected image with shape: (CHANNELS, WIDTH, HEIGHT), got {}".format(image.shape))
        if image.shape[2] < 32:
            image = image.transpose((2,0,1))

    if len(image.shape) == 4:
        if expected_dims < 4:
            raise ValueError("Expected image with shape: (CYCLE, CHANNELS, WIDTH, HEIGHT), got {}".format(image.shape))
        if image.shape[2] < 32:
            image = image.transpose((0,3,1,2))
    
    return image.reshape([1] * (expected_dims - len(image.shape)) + image.shape)

def percent_normalize(image, percent=0.1):
    if len(image.shape) == 2:
        image = image.reshape([1] + image.shape)

    mins = np.percentile(image, [percent], axis=(1,2)).reshape((-1,1,1))
    maxes = np.percentile(image, [100-percent], axis=(1,2)).reshape((-1,1,1))

    image = (image - mins) / (maxes - mins)
    image[image<0] = 0
    image[image>1] = 1

    return image

def to_rgb8(image):
    if image.dtype != np.uint8:
        image = percent_normalize(image).astype(np.uint8)

    image = image.transpose((1,2,0))
    return image[:,:,:3]

def mark_dots(image, poses):
    marking_layer = np.zeros_like(image[0])
    for pos in poses:
        mark_dot(marking_layer, pos)

    return np.array([*image, marking_layer])

def mark_dot(image, dot, inner_rad=3, outer_rad=6, color=1):
    x,y = dot
    image[x-outer_rad:x-inner_rad+1, y] = color
    image[x+inner_rad:x+outer_rad+1, y] = color
    image[x, y-outer_rad:y-inner_rad+1] = color
    image[x, y+inner_rad:y+outer_rad+1] = color


