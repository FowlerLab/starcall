import numpy as np
import skimage.morphology
import skimage.filters

example_matrix = np.array([
    # G      T      A      C
    [0.890, 0.048, 0.028, 0.034], # G
    [0.163, 0.790, 0.020, 0.027], # T
    [0.051, 0.037, 0.589, 0.323], # A
    [0.092, 0.068, 0.162, 0.679], # C
])


def color_correct(image, dye_matrix=None):
    """ Takes an image taken and estimates
    the dye present by using the given dye response matrix.
    This corrects for crosstalk between different channels
    and dyes.
    
    Params:
        image : np.ndarray, shape (num_channels, width, height)
        dye_matrix : np.ndarray, shape (num_dyes, num_channels)
            if not provided this matrix is estimated by the function estimate_dye_matrix
    Returns:
        new_image : np.ndarray, shape (num_dyes, width, height)
    """

    flat_image = image.reshape(image.shape[0], -1)

    if dye_matrix is None:
        dye_matrix = estimate_dye_matrix(image)

    print (dye_matrix.shape, flat_image.shape)
    dye_levels, residuals, rank, singular_vals = np.linalg.lstsq(dye_matrix.T, flat_image, rcond=None)
    dye_levels[dye_levels<0] = 0
    return dye_levels.reshape(image.shape).astype(image.dtype)

def estimate_crosstalk(chan, chanref, percent=0.1, bins=8):
    """ Calculates the estimated crosstalk between the two channels,
    returns the crosstalk from chanref to chan, or how much chan is dependant
    on chanref.
    """
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots()

    num_bins = bins
    axis.scatter(chanref, chan)
    bins = np.linspace(chanref.min(), chanref.max(), num_bins)
    #bins = np.percentile(chanref, np.linspace(0, 100, bins))
    for val in bins:
        axis.plot([val, val], [0, chan.max()], color='red')

    mins = []
    refvals = []
    for i in range(len(bins)-1):
        print (i)
        mask = (bins[i]<=chanref) & (chanref<=bins[i+1])
        print (np.count_nonzero(mask))
        if np.any(mask):
            min_index = np.argmin(chan[mask])
            mins.append(chan[mask][min_index])
            refvals.append(chanref[mask][min_index])
    axis.plot(refvals, mins)

    fig.savefig('plots/estimate_crosstalk.png')
    ratios = np.sort(np.array(mins) / refvals)
    crosstalk = ratios[num_bins//4-1]
    #crosstalk = np.median(np.array(mins) / refvals)
    if crosstalk > 0.5:
        kslfsld
    return crosstalk

    keep = chanref > max(np.percentile(chanref, 90), 0)
    ratio = chan[keep] / chanref[keep]
    axis.hist(ratio, bins=100, alpha=0.5)
    axis.hist(chan.flatten() / chanref.flatten(), bins=100, alpha=0.5)
    fig.savefig('plots/estimate_crosstalk.png')
    print (np.percentile(ratio, percent))
    skdfjls
    return np.percentile(ratio, percent)

def estimate_dye_matrix(image):
    """ Estimates the dye response matrix given an image.
    This assumes that each dye corresponds to one channel,
    and that the majority of the response to the dye is that
    channel.
    """
    matrix = np.zeros((len(image), len(image)))
    for x in range(len(image)):
        for y in range(len(image)):
            if x == y:
                matrix[x,y] = 1
            else:
                matrix[x,y] = estimate_crosstalk(image[y], image[x])
    print (matrix)
    return matrix

def crosstalk_plot(image, corrected, dye_matrix, name="image"):
    import matplotlib.pyplot as plt

    image = image.reshape(image.shape[0], -1)
    corrected = corrected.reshape(corrected.shape[0], -1)

    xposes, yposes = np.meshgrid(range(dye_matrix.shape[0]), range(dye_matrix.shape[1]))
    xposes, yposes = xposes.reshape(-1), yposes.reshape(-1)

    indices = np.argsort(dye_matrix.reshape(-1))[::-1]
    pairs = np.stack([xposes[indices], yposes[indices]], axis=1)
    pairs = [tuple(pair) for pair in np.sort(pairs, axis=1)]
    pairs = [pairs[i] for i in range(len(pairs)) if pairs[i][0] != pairs[i][1] and pairs[i] not in pairs[:i]]

    fig, axes = plt.subplots(ncols=len(pairs), nrows=2, figsize=(5*len(pairs), 10))
    
    for i,pair in enumerate(pairs):
        axes[0,i].scatter(image[pair[0]], image[pair[1]], s=1)
        axes[0,i].set_title('Crosstalk of original image')
        axes[0,i].set_xlabel('GTAC'[pair[0]])
        axes[0,i].set_ylabel('GTAC'[pair[1]])
        axes[1,i].scatter(corrected[pair[0]], corrected[pair[1]], s=1)
        axes[1,i].set_title('Crosstalk of corrected image')
        axes[1,i].set_xlabel('GTAC'[pair[0]])
        axes[1,i].set_ylabel('GTAC'[pair[1]])
    
    fig.savefig('plots/crosstalk_{}.png'.format(name))


class ResponseCurve:
    def __init__(self, *args):
        if len(args) == 1:
            points = [[args[0]-1, 0], [args[0], 1], [args[0]+1, 0]]
        elif len(args) == 2:
            points = [[args[0]-1, 0], [args[0], 1], [args[1], 1], [args[1]+1, 0]]
        

def calculate_dye_matrix(excitation_wavelengths, dyes, filter_wavelengths):
    """ Calculates the dye response matrix given the parameters used to image.
    Parameters:
        excitation_wavelengths: array like of integers, shape (channels,)
            This specifies the wavelengths used to exite the dyes
        dyes: array like of strings, shape (channels,)
            This specifies which dyes are being used, the valid names are (case insensitive):
            DAPI, G, T, A, C, 
        dye_responses: array like of numbers, shape (channels,)
            The wavelengths 
    """
    return None


def estimate_background(images, percent=5, gaussian=50):
    #background = np.percentile(images, 5, axis=0)
    old_shape = images.shape
    #images = images.reshape(images.shape[0], -1)
    background = np.zeros(images.shape[2:], images.dtype)

    #split up work to avoid casting whole thing to float, keep memory down
    batch_size = images.shape[1] // 64
    for i in range(0, images.shape[1], batch_size):
        section = images[:,i:i+batch_size]
        section = np.percentile(section, percent, axis=(0,1))
        #section = section.mean(axis=0)
        background[i:i+batch_size] = section

    background = background / background.max()
    background = background.reshape(old_shape[1:])
    #mask = skimage.morphology.disk(8)
    for i in range(len(background)):
        #background[i] = skimage.morphology.erosion(background[i], mask)
        background[i] = skimage.filters.gaussian(background[i], gaussian, mode='reflect')
    return background

def illumination_correction(images, out=None, background=None):
    if background is None:
        background = calculate_background(images)
    if out is None:
        out = np.empty_like(images)

    for i in range(len(images)):
        newimage = images[i] / background.reshape(1, *background.shape)
        np.clip(newimage, np.iinfo(images.dtype).min, np.iinfo(images.dtype).max, out=newimage)
        out[i] = newimage.astype(images.dtype)

    return out



