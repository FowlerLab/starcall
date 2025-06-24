import numpy as np
import skimage.io
import skimage.filters
import skimage.feature
import skimage.measure
import matplotlib.pyplot as plt
import numba
import tifffile

from . import utils
from .reads import Read, ReadSet

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
    if copy:
        image = image.copy()

    og_shape = image.shape
    if len(image.shape) == 3:
        image = image.reshape((1,) + image.shape)

    #image -= image.mean(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1)
    #image /= image.std(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1)
    #image -= image.mean(axis=(0,2,3)).reshape(1,-1,1,1)
    #image /= image.std(axis=(0,2,3)).reshape(1,-1,1,1)
    #np.clip(image, 0, None, out=image)
    
    for i in range(image.shape[0]):
        image[i] -= skimage.filters.gaussian(image[i], large_sigma, channel_axis=0)
        #for j in range(image.shape[1]):
            #image[i,j] = scipy.ndimage.gaussian_laplace(image[i,j], large_sigma)
    #np.clip(image, 0, None, out=image)

    image -= image.mean(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1)
    image /= image.std(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1)

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
        image[i] -= skimage.filters.gaussian(image[i], large_sigma, channel_axis=0)
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
    if len(image.shape) == 3:
        image = image.reshape((1,) + image.shape)

    sorted_image = np.sort(image, axis=1)
    image -= sorted_image[:,-2:-1]

    if gaussian_blur is not None:
        for i in range(len(image)):
            image[i] = skimage.filters.gaussian(image[i], sigma=gaussian_blur, channel_axis=0)

    np.clip(image, 0, None, out=image)

    image = image.std(axis=0)
    image = image.sum(axis=0)
    return image

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
        copy (bool default True): Whether the image should be copied or modified in place.

    Returns:
        reads (ReadSet): The reads detected in the image, each with a position and read values.
        if return_sigmas is specified:
        sigmas (ndarray of shape (n_dots,)): The estimated sigma of all dots detected in the image
    """

    if copy: image = image.copy()

    filtered = dot_filter_new(image, large_sigma=4, copy=False)
    #filtered = dot_filter_old(image, large_sigma=4, copy=False)
    #filtered = dot_filter(image, major_axis=4, minor_axis=0.5, copy=False)
    #tifffile.imwrite('tmp_dot_filter_filtered_12.tif', filtered)
    greyimage = highlight_dots(filtered.copy())
    #tifffile.imwrite('tmp_dot_greyimage.tif', greyimage)

    poses = skimage.feature.blob_log(greyimage,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=greyimage.mean(),
    )
    sigmas = poses[:,2]

    footprint = skimage.morphology.disk(2)
    for i in range(filtered.shape[0]):
        for j in range(filtered.shape[1]):
            filtered[i,j] = skimage.morphology.dilation(filtered[i,j], footprint)

    intposes = poses[:,:2].astype(int)
    values = image[:,:,intposes[:,0],intposes[:,1]]
    values = values.transpose(2,0,1)

    #print (values.shape)
    #print (np.percentile(values, 75, axis=0))
    #print (np.percentile(values, 99, axis=0))
    #print (np.mean(values, axis=0))
    #print (values.min(axis=0), values.max(axis=0))
    #values /= np.maximum(0.00000001, np.percentile(values, 75, axis=0)[None,:,:])
    #print (np.percentile(values, 75, axis=0))
    #print (np.percentile(values, 99, axis=0))
    #print (np.mean(values, axis=0))
    #print (values.min(axis=0), values.max(axis=0))

    reads = ReadSet(positions=poses[:,:2], values=values, channels=channels)

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


