import numpy as np
import skimage.io
import skimage.filters
import skimage.feature
import skimage.measure
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, minimum_filter
from scipy.special import softmax
from . import utils
from .reads import Read, make_readset
import numpy as np

def dot_filter(image, major_axis=4, minor_axis=0.5, copy=True):
    """ Filter that removes any background in the sequencing images,
    leaving only the dots from sequencing colonies. This is done using
    top hat filter, subtracting the morphological opening of the image from
    itself. This leaves only features smaller than the footprint, whose size
    is specified by the parameters major_axis and minor_axis. To further
    amplify only features that are circular, a series of filters are applied
    with ellipses at different angles.

    Args:
        major_axis (float): the major axis of the ellipse used as a footprint
        minor_axis (float): the minor axis of the ellipse used as a footprint
        copy (bool default True): Whether the input image should be copied or modified in place
    """
    if copy:
        image = image.copy()

    orig_shape = image.shape
    image = image.reshape(-1, *orig_shape[2:])

    footprint = skimage.morphology.disk(major_axis)
    footprints = np.zeros((4, footprint.shape[0], footprint.shape[1]), footprint.dtype)
    mid = major_axis
    mid2 = major_axis + 1
    #footprints[0,kernel_size+1,:] = 1
    #footprints[1,:,kernel_size+1] = 1
    #footprints[2,list(range(footprint.shape[0])),list(range(footprint.shape[0]))] = 1
    #footprints[3,list(reversed(range(footprint.shape[0]))),list(range(footprint.shape[0]))] = 1
    #footprints[0,:mid2,:mid2] = footprint[:mid2,:mid2]
    #footprints[1,:mid2,mid:] = footprint[:mid2,mid:]
    #footprints[2,mid:,:mid2] = footprint[mid:,:mid2]
    #footprints[3,mid:,mid:] = footprint[mid:,mid:]

    for i, rot in enumerate([-np.pi/4, 0, np.pi/4, np.pi/2]):
        rows, cols = skimage.draw.ellipse(major_axis, major_axis, major_axis+1, minor_axis, rotation=rot)
        footprints[i,rows,cols] = 1

    #print (footprints)
    new_background = np.empty_like(image[i])
    for i in range(image.shape[0]):
        #background[i] = skimage.morphology.opening(background[i], footprint)
        #background[i] = skimage.filters.gaussian(background[i], kernel_size)
        new_background[...] = 0
        for j in range(len(footprints)):
            new_background = np.maximum(new_background, skimage.morphology.opening(image[i], footprints[j]))
        image[i] -= new_background

    image = image.reshape(orig_shape)

    #np.clip(image, 0, None, out=image)

    return image

def dot_filter2(image, small_radius=2, large_radius=4, copy=True):
    if copy:
        image = image.copy()

    orig_shape = image.shape
    image = image.reshape(-1, *orig_shape[2:])

    #large_footprint = skimage.morphology.disk(large_radius)
    #small_footprint = skimage.morphology.disk(small_radius)
    #diff = large_radius - small_radius
    #large_footprint[diff:-diff,diff:-diff] &= ~small_footprint.astype(bool)
    footprint = skimage.morphology.disk(large_radius)
    print (footprint)

    for i in range(image.shape[0]):
        #background = skimage.morphology.dilation(image[i], footprint)
        #background[i] = skimage.filters.gaussian(background[i], kernel_size)
        #image[i] -= background
        image[i] = skimage.morphology.white_tophat(image[i], footprint)

    image = image.reshape(orig_shape)

    np.clip(image, 0, None, out=image)

    return image

def dot_filter_new(image, large_sigma=4, copy=True):
    """ Filter that removes any background in the sequencing images,
    leaving only the dots from sequencing colonies. Done using a difference
    of gaussian filter, specified by the parameter large_sigma.

    Args:
        image (np.ndarray of shape (num_cycles, num_channels, width, height)
        large_sigma (float): the gaussian sigma that should be subtracted from the sequencing
            images.
        copy (bool default True): Whether the input image should be copied or modified in place
    """
    #modifying it to ignore nan positions in the images
    if copy:
        image = image.copy()

    og_shape = image.shape
    if len(image.shape) == 3:
        image = image.reshape((1,) + image.shape)
    
    nan_filter = np.isnan(image).any(axis=(0, 1))
    #set image nan values to 0 before gaussian blur 
    #otherwise the blur will expand the NaN section
    np.nan_to_num(image, copy = False)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j] -= skimage.filters.gaussian(image[i,j], large_sigma)

    #set nan locations to NaN again (the nanfilter is going to be 2d)
    image[:, :, nan_filter] = np.nan
    image -= np.nanmean(image, axis = (2,3)).reshape(image.shape[0], image.shape[1],1,1) #image.mean(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1) # (3) compute z score
    image /= np.nanstd(image, axis = (2,3)).reshape(image.shape[0], image.shape[1],1,1) #image.std(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1) # (3) compute z score 

    return image.reshape(og_shape)

def dot_filter_old(image, large_sigma=4, copy=True):
    if copy:
        image = image.copy()

    og_shape = image.shape
    if len(image.shape) == 3:
        image = image.reshape((1,) + image.shape)

    image -= image.mean(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1)
    image /= image.std(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1)
    #image -= image.mean(axis=(0,2,3)).reshape(1,-1,1,1)
    #image /= image.std(axis=(0,2,3)).reshape(1,-1,1,1)
    np.clip(image, 0, None, out=image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j] -= skimage.filters.gaussian(image[i,j], large_sigma)
    np.clip(image, 0, None, out=image)

    return image.reshape(og_shape)

def dot_filter2_old(image, kernel_size=10):
    og_shape = image.shape
    if len(image.shape) == 3:
        image = image.reshape((1,) + image.shape)

    image -= image.mean(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1)
    image /= image.std(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1)
    #image -= image.mean(axis=(0,2,3)).reshape(1,-1,1,1)
    #image /= image.std(axis=(0,2,3)).reshape(1,-1,1,1)
    np.clip(image, 0, None, out=image)
    
    footprint = skimage.morphology.disk(kernel_size)
    for i in range(image.shape[0]):
        image[i] = skimage.morphology.white_tophat(image[i], footprint)

    np.clip(image, 0, None, out=image)

    return image.reshape(og_shape)


def detect_dots(image,
        min_sigma=1,
        max_sigma=2,
        num_sigma=7,
        return_sigmas=False,
        channels=None,
        copy=True):
    """ Takes a raw sequencing image set and identifies and extracts all sequencing reads
    from the image, filtering out cell background and debris. This is done by calling
    dot_filter to filter out any background in the image, then calling highlight_dots
    to create a single grayscale image, which is then passed to skimage.feature.blob_log.
    The values at these positions in the filtered image are extracted, and returned along
    with the positions.

    Args:
        image (ndarray of shape (n_cycles, n_channels, width, height)): The input image
        min_sigma, max_sigma, num_sigma (float): Parameters passed to skimage.feature.blob_log
        return_sigmas (bool, default False): Whether to return the sigma values returned from skimage.feature.blob_log
        channels (tuple of str): The names of the sequencing channels in the image.
            Defaults to ('G', 'T', 'A', 'C'), but if your sequencing channels are in a different
            order make sure to specify it here.
        copy (bool default True): Whether the image should be copied or modified in place.

    Returns:
        reads (DataFrame): The reads detected in the image, each with a position and read values.
            The columns in the table are:
                'position_x', 'position_y': the x and y position of the read colony
                'values_cycle{cycle}_{chan}': the values from the filtered sequencing images,
                    for each cycle and channel. The names of the channels are specified
                    in the parameter 'channels'
        if return_sigmas is specified:
        sigmas (ndarray of shape (n_dots,)): The estimated sigma of all dots detected in the image
    """

    if copy: image = image.copy()

    filtered = dot_filter_new(image, large_sigma=max_sigma, copy=False)
    
    greyimage = highlight_dots(filtered.copy())
    #tifffile.imwrite('tmp_dot_greyimage.tif', greyimage)
    new_threshold = greyimage[greyimage != -1].mean() #note that we set all the nans to -1, to differentiate them for taking a mean 
    greyimage[greyimage == -1] = 0 
    poses = skimage.feature.blob_log(greyimage,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=new_threshold,
    )
    sigmas = poses[:,2]

    footprint = skimage.morphology.disk(2)
    for i in range(filtered.shape[0]):
        for j in range(filtered.shape[1]):
            filtered[i,j] = skimage.morphology.dilation(filtered[i,j], footprint)

    intposes = poses[:,:2].astype(int)
    values = image[:,:,intposes[:,0],intposes[:,1]]
    values = values.transpose(2,0,1)

    reads = make_readset(positions=poses[:,:2], values=values, channels=channels)

    if return_sigmas:
        return reads, sigmas
    return reads


def highlight_dots(image, gaussian_blur=None):
    """ Combine an image containing multiple sequencing cycles into a single
    grayscale image, containing only the dots from sequencing colonies.
    To filter for sequencing dots, we subtract the second maximal channel, then
    take the standard deviation across cycles and sum along channels. This
    means only features that are bright in a single channel and changing frequently
    are conserved in the final image.

    Args:
        image (ndarray of shape (num_cycles, num_channels, width, height)): Input image to filter
        gaussian_blur (float, optional): if specified a gaussian blur is applied before combining
    """
    #AML - modified to ignore NaN positions in the image when finding dots 
    if len(image.shape) == 3:
        image = image.reshape((1,) + image.shape)

    nan_filter = np.isnan(image[0,0,:,:])
    sorted_image = np.sort(image, axis=1)
    image -= sorted_image[:,-2:-1]

    if gaussian_blur is not None:
        for i in range(len(image)):
            for j in range(image.shape[1]):
                image[i,j] = skimage.filters.gaussian(image[i,j], sigma=gaussian_blur)

    np.clip(image, 0, None, out=image)

    image = np.nanstd(image, axis=0 if image.shape[0] > 1 else 1) # (7) std across cycles per channel #AML- changed std to nanstd, ignore blankspace nans? 
    image = np.nansum(image, axis=0) # (8) sum across channels to get single grayscale 2d image 

    #for any spot where image was originally nan, we will set it to -1
    image[nan_filter] = -1
    
    return image


def z_score_dots_per_cycle(image,  copy=True):
    """ Filter that removes any background in the sequencing images,
    leaving only the dots from sequencing colonies. Done using a difference
    of gaussian filter, specified by the parameter large_sigma. Note that the
    image will contain nans at positions without image data

    Args:
        image (np.ndarray of shape (num_cycles, num_channels, width, height)
        large_sigma (float): the gaussian sigma that should be subtracted from the sequencing
            images.
        copy (bool default True): Whether the input image should be copied or modified in place
    """
    #copy image to prevent the original from being z-scored 
    image = image.copy()
    #control image shape
    og_shape = image.shape
    if len(image.shape) == 3:
        image = image.reshape((1,) + image.shape)
    #z-score
    image -= np.nanmean(image, axis = (2,3)).reshape(image.shape[0], image.shape[1],1,1) #image.mean(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1) # (3) compute z score
    image /= np.nanstd(image, axis = (2,3)).reshape(image.shape[0], image.shape[1],1,1) #image.std(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1) # (3) compute z score 

    return image.reshape(og_shape)


def subtract_dot_background(image, large_sigma=4):
    """ Filter that removes any background in the sequencing images,
    leaving only the dots from sequencing colonies. Done using a difference
    of gaussian filter, specified by the parameter large_sigma. Note that 
    the nans are set to 0 before the gaussian to prevent them from expanding 
    and then reset before the corrected image is returned

    Args:
        image (np.ndarray of shape (num_cycles, num_channels, width, height)
        large_sigma (float): the gaussian sigma that should be subtracted from the sequencing
            images.
    """
    #modifying it to ignore nan positions in the images

    og_shape = image.shape
    if len(image.shape) == 3:
        image = image.reshape((1,) + image.shape)
    
    nan_filter = np.isnan(image).any(axis=(0, 1))
    #set image nan values to 0 before gaussian blur 
    #otherwise the blur will expand the NaN section
    np.nan_to_num(image, copy = False)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j] -= skimage.filters.gaussian(image[i,j], large_sigma)

    #set nan locations to NaN again (the nanfilter is going to be 2d)
    image[:, :, nan_filter] = np.nan

    return image.reshape(og_shape)

#using PSF weighted sums around a dot instead of the max 
#From Claude
def psf_weighted_values(image, positions, psf_sigma, order=1, truncate=3.0):
    """PSF-weighted intensity at each dot's sub-pixel position, NaN-aware.
    For each (cycle, channel) plane, weight pixels by a Gaussian of width
    `psf_sigma` and sample at the dot's sub-pixel center. Handles NaNs so that
    they don't contribute weight to the final value

    Args:
        image (ndarray, (n_cycles, n_channels, H, W)): background-subtracted,
          May contain NaNs at no-data positions
        positions (ndarray, (n_dots, >=2)): sub-pixel (row, col) from blob_log.
        psf_sigma (float): Gaussian weighting width, shared across channels/cycles.
        order (int): sub-pixel interpolation order (1=bilinear, 3=cubic).
        truncate (float): kernel radius in units of sigma.

    Returns:
        values (ndarray, (n_dots, n_cycles, n_channels)): PSF-weighted mean
            intensity per dot. Positions whose kernel covers no valid pixels
            come back as NaN.
    """
    n_cyc, n_chan = image.shape[:2]
    coords = positions[:, :2].T  # (2, n_dots): [rows; cols]
    values = np.empty((positions.shape[0], n_cyc, n_chan), dtype=float)

    # Validity mask is 2-D: a pixel is no-data if any cycle/channel is NaN there,
    # matching how the detection path defines NaN borders.
    valid = (~np.isnan(image).any(axis=(0, 1))).astype(float)  # (H, W)
    den = gaussian_filter(valid, sigma=psf_sigma, mode="constant", cval=0.0, truncate=truncate)
    safe_den = np.where(den > 1e-6, den, np.nan)  # NaN where no valid support

    for i in range(n_cyc):
        for j in range(n_chan):
            plane = np.nan_to_num(image[i, j], nan=0.0)  # local copy, original untouched
            num = gaussian_filter(plane, sigma=psf_sigma, mode="constant",
                                  cval=0.0, truncate=truncate)
            smoothed = num / safe_den  # normalized convolution; NaN outside support
            values[:, i, j] = map_coordinates(smoothed, coords, order=order,
                                              mode="nearest", cval=np.nan)
    return values

def box_max_values(image, positions, box_radius=1):
    """Max intensity within a (2*box_radius+1)-pixel square box centered at each
    dot's position, per cycle and channel. NaN-aware: a NaN pixel never wins the
    max, but if every pixel in a dot's box is NaN, that dot's value is NaN 

    Args:
        image (ndarray, (n_cycles, n_channels, H, W)): background-subtracted,
          may contain NaNs at no-data positions.
        positions (ndarray, (n_dots, >=2)): sub-pixel (row, col) from blob_log.
        box_radius (int): half-width of the box; box_radius=1 gives a 3x3 box.

    Returns:
        values (ndarray, (n_dots, n_cycles, n_channels)): max intensity per dot
            within its box, per cycle and channel.
    """
    n_cyc, n_chan = image.shape[:2]
    H, W = image.shape[2:]
    size = 2 * box_radius + 1
    rows = np.clip(np.round(positions[:, 0]).astype(int), 0, H - 1)
    cols = np.clip(np.round(positions[:, 1]).astype(int), 0, W - 1)

    values = np.empty((positions.shape[0], n_cyc, n_chan), dtype=float)
    for i in range(n_cyc):
        for j in range(n_chan):
            plane = image[i, j]
            nanmask = np.isnan(plane)
            filled = np.where(nanmask, -np.inf, plane)
            box_max = maximum_filter(filled, size=size, mode="constant", cval=-np.inf)
            no_valid_support = minimum_filter((~nanmask).astype(np.uint8), size=size, mode="constant", cval=0) == 0
            box_max = np.where(no_valid_support, np.nan, box_max)
            values[:, i, j] = box_max[rows, cols]
    return values


def detect_dots_keep_background_corrected_intensities(image,extract_mode,
        min_sigma=1,
        max_sigma=2,
        num_sigma=7,
        return_sigmas=False,
        channels=None,
        copy=True):
    """ Takes a raw sequencing image set and identifies and extracts all sequencing reads
    from the image, filtering out cell background and debris. This is done by calling
    dot_filter to filter out any background in the image, then calling highlight_dots
    to create a single grayscale image, which is then passed to skimage.feature.blob_log.
    The values at these positions in the background corrected image are extracted, the pixels surrounding
    these positions are expanded out a distnace sigma and the sum of all intensities in 
    the expansion is passed to read set creation.

    Args:
        image (ndarray of shape (n_cycles, n_channels, width, height)): The input image
        min_sigma, max_sigma, num_sigma (float): Parameters passed to skimage.feature.blob_log
        return_sigmas (bool, default False): Whether to return the sigma values returned from skimage.feature.blob_log
        expansion_val (float): value to expand out to sum intensities for reads
        channels (tuple of str): The names of the sequencing channels in the image.
            Defaults to ('G', 'T', 'A', 'C'), but if your sequencing channels are in a different
            order make sure to specify it here.
        copy (bool default True): Whether the image should be copied or modified in place.

    Returns:
        reads (DataFrame): The reads detected in the image, each with a position and read values.
            The columns in the table are:
                'position_x', 'position_y': the x and y position of the read colony
                'values_cycle{cycle}_{chan}': the values from the filtered sequencing images,
                    for each cycle and channel. The names of the channels are specified
                    in the parameter 'channels'
        if return_sigmas is specified:
        sigmas (ndarray of shape (n_dots,)): The estimated sigma of all dots detected in the image
    """
    #
    #if copy: image = image.copy() 
    image = subtract_dot_background(image, large_sigma=max_sigma)
    filtered = z_score_dots_per_cycle(image)

    greyimage = highlight_dots(filtered.copy())

    new_threshold = greyimage[greyimage != -1].mean() #note that we set all the nans to -1 in greyimage, to differentiate them for taking a mean 
    greyimage[greyimage == -1] = 0 #reset to 0 for the acutal blob log search
    poses = skimage.feature.blob_log(greyimage,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=new_threshold,
    )
    sigmas = poses[:,2]
    intposes = poses[:,:2].astype(int)

    #PSF-weighted sum around the dots, w/ one fixed sigma shared across all channels/cycles.
    if extract_mode == 'psf':
        psf_sigma = float(np.median(sigmas)) if len(sigmas) else max_sigma
        print ('using psf weighted values with sigma: ', psf_sigma)
        values = psf_weighted_values(image, poses[:, :2], psf_sigma=psf_sigma)
    elif extract_mode == 'box':
        print('using 3x3 max box values')
        values = box_max_values(image, poses[:, :2], box_radius=1)
    else:
        print ('using default approach....(5x5 max)')
        #set nans to neg inf for max selection 
        filtered = np.nan_to_num(image, nan=-np.inf) #don't use the z scored values rn, since we'll be training a color correction method
        footprint = skimage.morphology.disk(2)
        for i in range(filtered.shape[0]):
            for j in range(filtered.shape[1]):
                filtered[i,j] = skimage.morphology.dilation(filtered[i,j], footprint)
        values = filtered[:,:,intposes[:,0],intposes[:,1]]
        values = values.transpose(2,0,1)

    reads = make_readset(positions=poses[:,:2], values=values, channels=channels)

    if return_sigmas:
        return reads, sigmas
    return reads

def detect_dots_old(image,
        min_sigma=1,
        max_sigma=2,
        num_sigma=7,
        sigma_cutoff=1,
        return_sigmas=False,
        threshold_rel=None,
        median_index=None,
        copy=True):

    if copy: image = image.copy()

    if len(image.shape) == 3:
        image = image.reshape((1,) + image.shape)

    #image = dot_filter(image)

    maximage = image.max(axis=0)

    median_index = median_index or len(maximage) // 2
    
    median = np.partition(maximage, median_index, axis=0)
    median -= median[median_index]
    np.max(median, axis=0, out=median[0])

    greyimage = median[0]
    tmp_layer = median[1]

    skimage.io.imsave('tmp_dots_greyimage.tif', greyimage)

    first_sigma = 1
    second_sigma = 2
    skimage.filters.gaussian(greyimage, second_sigma, output=tmp_layer)
    skimage.filters.gaussian(greyimage, first_sigma, output=greyimage)
    greyimage -= tmp_layer
    np.clip(greyimage, 0, None, out=greyimage)

    poses = skimage.feature.blob_log(greyimage, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=threshold_rel)
    sigmas = poses[:,2]

    intposes = poses[:,:2].astype(int)

    if return_sigmas:
        return intposes, sigmas
    return intposes

def detect_dots_debug(image,
        min_sigma=1,
        max_sigma=2,
        num_sigma=7,
        sigma_cutoff=1,
        return_sigmas=False,
        threshold_rel=None,
        median_index=None):

    orig_image = image
    #image = image / image.mean(axis=(1,2)).reshape(-1,1,1)
    image = dot_filter(image)
    skimage.io.imsave('plots/tmp_afterfilter.tif', image)
    #greyimage = image.std(axis=0)
    #greyimage = np.max(image - np.median(image, axis=0), axis=0)
    median_index = median_index or len(image) // 2
    median = np.partition(image, median_index, axis=0)[median_index]
    greyimage = np.max(image - median, axis=0)

    skimage.io.imsave('plots/tmp_outliers.tif', greyimage)
    #print (greyimage.min(), greyimage.mean(), greyimage.max())
    #skimage.io.imsave('plots/tmp_std.tif', greyimage)

    first_sigma = 1
    second_sigma = 2
    diff_of_gauss = skimage.filters.gaussian(greyimage, first_sigma) - skimage.filters.gaussian(greyimage, second_sigma)
    diff_of_gauss = np.maximum(diff_of_gauss, 0)
    skimage.io.imsave('plots/tmp_diff.tif', diff_of_gauss)

    #"""
    #greyimage = np.maximum(greyimage - greyimage.mean(), 0)
    greyimage = diff_of_gauss
    #skimage.io.imsave('plots/tmp_grey.tif', np.array([*orig_image, (greyimage / greyimage.max() * 65535).astype('uint16')]))
    poses = skimage.feature.blob_log(greyimage, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=threshold_rel)
    sigmas = poses[:,2]

    intposes = poses[:,:2].astype(int)
    values = image[:,intposes[:,0],intposes[:,1]].T

    skimage.io.imsave('plots/tmp_test_dots.tif', utils.mark_dots(diff_of_gauss[None,...], intposes))

    if return_sigmas:
        return intposes, values, sigmas
    return intposes, values
    #"""

    fig, axes = plt.subplots(nrows=2, figsize=(5,10))
    for axis in axes:
        axis.set_yscale('log')
    axes[0].hist(diff_of_gauss.flatten(), bins=100)
    axes[0].set_title('Hist of diff of gaussian')

    diff_of_gauss = np.log(diff_of_gauss+1)
    axes[1].hist(diff_of_gauss.flatten(), bins=100)

    threshold = skimage.filters.threshold_otsu(diff_of_gauss)
    labels = skimage.measure.label(diff_of_gauss > threshold)
    skimage.io.imsave('plots/tmp_labels.tif', diff_of_gauss > threshold)
    poses = [region.centroid for region in skimage.measure.regionprops(labels)]
    intposes = np.array(poses).astype(int)
    #intposes = skimage.feature.peak_local_max(diff_of_gauss, min_distance=3)

    skimage.io.imsave('plots/tmp_test_dots.tif', utils.mark_dots(diff_of_gauss[None,...], intposes))
    fig.savefig('plots/hists_diff_gauss.png')

    return intposes, None

def gaussian_kernel(radius, sigma):
    kernel = np.zeros((radius*2+1, radius*2+1))
    kernel[radius,radius] = 1
    return skimage.filters.gaussian(kernel, sigma)


def call_dots(values):
    print (values.shape)
    num_dots, num_cycles, num_base = values.shape

    tsne = sklearn.manifold.TSNE()
    tsne_poses = tsne.fit_transform(values.reshape(num_dots * num_cycles, num_base)).reshape(num_dots, num_cycles, 2)

    fig, axes = plt.subplots(nrows=num_cycles, figsize=(8, 5*num_cycles))
    for cycle in range(num_cycles):
        poses = tsne_poses[:,cycle]
        axes[cycle].scatter(poses[:,0], poses[:,1])
        axes[cycle].set_title("Cycle " + i)

    fig.savefig('plots/calling_dots.png')

    pass


