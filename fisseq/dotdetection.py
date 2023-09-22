import numpy as np
import skimage.io
import skimage.filters
import skimage.feature
import skimage.measure
import matplotlib.pyplot as plt
import numba

from . import utils

def dot_filter(image, large_sigma=4):
    image -= image.mean(axis=(1,2)).reshape(-1,1,1)
    image /= image.std(axis=(1,2)).reshape(-1,1,1)
    np.clip(image, 0, None, out=image)
    
    image -= skimage.filters.gaussian(image, large_sigma, channel_axis=0)
    np.clip(image, 0, None, out=image)
    return image

def detect_dots(image,
        min_sigma=1,
        max_sigma=2,
        num_sigma=7,
        sigma_cutoff=1,
        return_sigmas=False,
        threshold_rel=None,
        median_index=None,
        copy=True):

    if copy: image = image.copy()
    image = dot_filter(image)
    median_index = median_index or len(image) // 2
    
    median = np.partition(image, median_index, axis=0)
    median -= median[median_index]
    np.max(median, axis=0, out=median[0])

    greyimage = median[0]
    tmp_layer = median[1]

    first_sigma = 1
    second_sigma = 2
    skimage.filters.gaussian(greyimage, second_sigma, output=tmp_layer)
    skimage.filters.gaussian(greyimage, first_sigma, output=greyimage)
    greyimage -= tmp_layer
    np.clip(greyimage, 0, None, out=greyimage)

    poses = skimage.feature.blob_log(greyimage, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_rel=threshold_rel)
    sigmas = poses[:,2]

    intposes = poses[:,:2].astype(int)
    values = image[:,intposes[:,0],intposes[:,1]].T

    if return_sigmas:
        return intposes, values, sigmas
    return intposes, values

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

