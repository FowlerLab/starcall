import numpy as np
import skimage.io
import skimage.filters
import skimage.feature
import skimage.measure
import matplotlib.pyplot as plt
import tifffile
import time
import math
import contextlib
import concurrent.futures
import os
import scipy.ndimage
import skimage.morphology
from . import utils
from .reads import Read, make_readset
import numpy as np


def _make_debug(debug):
    """ debug can be True (print), False/None (silent) or a callable (used as-is) """
    if callable(debug):
        return debug
    if debug:
        return lambda *args: print(*args)
    return lambda *args: None


class _StageTimer:
    """ Times named stages of detect_dots and reports them through debug.
    Use as `with timer('stage'): ...`. Repeated stages are accumulated.
    sync is called before stopping the clock (used to wait for the GPU).
    """
    def __init__(self, version, debug=False, sync=None, timings=None):
        self.version = version
        self.debug = _make_debug(debug)
        self.sync = sync
        self.times = timings if timings is not None else {}
        self.start = time.perf_counter()

    @contextlib.contextmanager
    def __call__(self, stage):
        t0 = time.perf_counter()
        yield
        if self.sync is not None:
            self.sync()
        dt = time.perf_counter() - t0
        self.times[stage] = self.times.get(stage, 0) + dt
        self.debug('detect_dots[{}] {}: {:.2f}s'.format(self.version, stage, dt))

    def log(self, *args):
        self.debug('detect_dots[{}]'.format(self.version), *args)

    def finish(self, image_shape, num_dots):
        total = time.perf_counter() - self.start
        self.times['total'] = total
        self.debug('detect_dots[{}] total: {:.2f}s, {} dots, image shape {}'.format(
            self.version, total, num_dots, tuple(image_shape)))


def _null_timer(stage):
    return contextlib.nullcontext()


def _make_reads(poses, values, channels):
    """ Shared tail of all detect_dots versions: values has shape (n_dots, n_cycles, n_channels) """
    return make_readset(positions=poses[:,:2], values=values, channels=channels)


def dot_filter_new(image, large_sigma=4, copy=True, timer=_null_timer):
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
    
    with timer('dog'):
        nan_filter = np.isnan(image).any(axis=(0, 1))
        #set image nan values to 0 before gaussian blur 
        #otherwise the blur will expand the NaN section
        np.nan_to_num(image, copy = False)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j] -= skimage.filters.gaussian(image[i,j], large_sigma)

    with timer('zscore'):
        #set nan locations to NaN again (the nanfilter is going to be 2d)
        image[:, :, nan_filter] = np.nan
        image -= np.nanmean(image, axis = (2,3)).reshape(image.shape[0], image.shape[1],1,1) #image.mean(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1) # (3) compute z score
        image /= np.nanstd(image, axis = (2,3)).reshape(image.shape[0], image.shape[1],1,1) #image.std(axis=(2,3)).reshape(image.shape[0], image.shape[1],1,1) # (3) compute z score 

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


def detect_dots(image,
        min_sigma=1,
        max_sigma=2,
        num_sigma=7,
        return_sigmas=False,
        channels=None,
        copy=True,
        debug=False,
        timings=None):
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
        debug (bool or callable): Report the time of each stage. A callable is used as the print function.
        timings (dict, optional): If given, filled with the seconds spent in each stage.

    See also detect_dots_cpu and detect_dots_gpu, faster versions of this function.

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

    timer = _StageTimer('original', debug, timings=timings)
    timer.log('starting on image shape {} dtype {}'.format(image.shape, image.dtype))

    if copy: image = image.copy()

    filtered = dot_filter_new(image, large_sigma=max_sigma, copy=False, timer=timer)
    
    with timer('highlight'):
        greyimage = highlight_dots(filtered.copy())
    #tifffile.imwrite('tmp_dot_greyimage.tif', greyimage)
    with timer('threshold'):
        new_threshold = greyimage[greyimage != -1].mean() #note that we set all the nans to -1, to differentiate them for taking a mean 
        greyimage[greyimage == -1] = 0 
    timer.log('blob_log threshold {}'.format(new_threshold))
    with timer('blob_log'):
        poses = skimage.feature.blob_log(greyimage,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=new_threshold,
        )
    sigmas = poses[:,2]
    timer.log('found {} dots'.format(len(poses)))

    with timer('dilate/sample'):
        footprint = skimage.morphology.disk(2)
        for i in range(filtered.shape[0]):
            for j in range(filtered.shape[1]):
                filtered[i,j] = skimage.morphology.dilation(filtered[i,j], footprint)

    with timer('extract'):
        intposes = poses[:,:2].astype(int)
        values = image[:,:,intposes[:,0],intposes[:,1]]
        values = values.transpose(2,0,1)

        reads = _make_reads(poses, values, channels)

    timer.finish(image.shape, len(reads))

    if return_sigmas:
        return reads, sigmas
    return reads



#####################################################################
# Faster versions of detect_dots: detect_dots_cpu and detect_dots_gpu
#
# Both run the same steps as detect_dots, but:
#   - the image is processed in row bands / tiles instead of all at once, so
#     work can run in parallel and the input can stay a (uint16) memmap
#   - dilation of every plane is replaced by taking the max over the footprint
#     at the detected dot positions only, which gives the same values
#   - blob_log runs on overlapping tiles, keeping dots in each tile's core
# Global statistics (per plane z-score, blob_log threshold) are computed over
# the whole image exactly as in detect_dots.
#####################################################################

def _bands(length, size, halo=0):
    """ Yields (start, end, halo_start, halo_end) ranges covering range(length) """
    size = max(int(size), 1)
    for start in range(0, length, size):
        end = min(start + size, length)
        yield start, end, max(start - halo, 0), min(end + halo, length)


def _band_rows(pixel_budget, row_pixels):
    return max(int(pixel_budget // max(row_pixels, 1)), 1)


def _gaussian_radius(sigma, truncate=4.0):
    """ Kernel radius used by scipy.ndimage gaussian filters """
    return int(truncate * float(sigma) + 0.5)


def _reflect_index(index, length):
    """ scipy.ndimage 'reflect' boundary mode (d c b a | a b c d | d c b a),
    the default mode of skimage.morphology.dilation """
    index = np.where(index < 0, -index - 1, index)
    index = np.where(index >= length, 2 * length - index - 1, index)
    return index


class _CpuBackend:
    """ xp is the array module work is done with, sxp the one the filtered and grey
    images are stored with. in_place means both are the same, so no copies are needed """
    name = 'cpu'
    on_gpu = False
    in_place = True

    def __init__(self, threads):
        self.threads = threads
        self.xp = np
        self.sxp = np
        self.ndi = scipy.ndimage
        self.blob_log = skimage.feature.blob_log
        self.sync = None

    def to_device(self, array):
        return np.asarray(array)

    def to_host(self, array):
        return array

    def store(self, dest, key, array):
        dest[key] = array

    def map(self, func, items):
        """ runs the per band/tile work """
        return cpu_map(func, items, self.threads)

    def map_loaded(self, load, func, items):
        """ func(item, load(item)) for all items """
        return cpu_map(lambda item: func(item, load(item)), items, self.threads)

    def free(self):
        pass

    def memory_info(self):
        return ''


class _GpuBackend:
    """ With resident=True the filtered and grey images are kept on the GPU,
    otherwise they are kept in host memory and bands are sent to the GPU as needed """
    name = 'gpu'
    on_gpu = True

    def __init__(self, threads, device=0, resident=False):
        try:
            import cupy
            import cupyx.scipy.ndimage
            import cucim.skimage.feature
        except ImportError as error:
            raise ImportError('detect_dots_gpu needs cupy and cucim installed, eg: '
                    'pip install cupy-cuda12x cucim-cu12') from error
        cupy.cuda.Device(device).use()
        self.cupy = cupy
        self.threads = threads
        self.xp = cupy
        self.ndi = cupyx.scipy.ndimage
        self.blob_log = cucim.skimage.feature.blob_log
        self.device = device
        self.set_resident(resident)
        self.peak_used = 0
        self.peak_reserved = 0

    def set_resident(self, resident):
        self.resident = resident
        self.in_place = resident
        self.sxp = self.cupy if resident else np

    def free_device_memory(self):
        free, total = self.cupy.cuda.Device(self.device).mem_info
        return free + self.cupy.get_default_memory_pool().free_bytes()

    def sync(self):
        self.cupy.cuda.Device(self.device).synchronize()

    def to_device(self, array):
        cupy = self.cupy
        if isinstance(array, cupy.ndarray):
            return cupy.ascontiguousarray(array)
        if array.ndim > 2 and not array.flags.c_contiguous:
            # a band across planes, copy plane by plane instead of packing it on the host first
            result = cupy.empty(array.shape, dtype=array.dtype)
            for index in np.ndindex(array.shape[:-2]):
                result[index].set(np.ascontiguousarray(array[index]))
            return result
        return cupy.asarray(np.ascontiguousarray(array))

    def to_host(self, array):
        return self.cupy.asnumpy(array)

    def store(self, dest, key, array):
        """ dest[key] = array, where dest is on the host or the device """
        if isinstance(dest, self.cupy.ndarray):
            dest[key] = array
            return
        target = dest[key]
        if target.flags.c_contiguous:
            array.get(out=target)
        else:
            for index in np.ndindex(target.shape[:-2]):
                array[index].get(out=target[index])

    def map(self, func, items):
        # one GPU, so tasks run one after another
        return [func(item) for item in items]

    def map_loaded(self, load, func, items):
        """ func(item, load(item)) for all items, running func one at a time on the gpu
        while the next items are loaded (eg read from disk) in background threads """
        items = list(items)
        results = []
        workers = max(1, min(self.threads, 8))
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            pending = [pool.submit(load, item) for item in items[:workers * 2]]
            for index, item in enumerate(items):
                data = pending[index].result()
                pending[index] = None
                if index + workers * 2 < len(items):
                    pending.append(pool.submit(load, items[index + workers * 2]))
                results.append(func(item, data))
        return results

    def free(self):
        pool = self.cupy.get_default_memory_pool()
        self.peak_used = max(self.peak_used, pool.used_bytes())
        self.peak_reserved = max(self.peak_reserved, pool.total_bytes())
        pool.free_all_blocks()

    def memory_info(self):
        pool = self.cupy.get_default_memory_pool()
        self.peak_used = max(self.peak_used, pool.used_bytes())
        self.peak_reserved = max(self.peak_reserved, pool.total_bytes())
        free, total = self.cupy.cuda.Device(self.device).mem_info
        return ('gpu memory now used {:.2f}GB, peak used {:.2f}GB, peak reserved {:.2f}GB, '
                'device free {:.2f}GB of {:.2f}GB').format(
            pool.used_bytes() / 1e9, self.peak_used / 1e9, self.peak_reserved / 1e9, free / 1e9, total / 1e9)


def cpu_map(func, items, threads):
    """ list(map(func, items)) using a thread pool """
    items = list(items)
    if threads is None or threads <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        return list(pool.map(func, items))


def _blob_log_tile(tile, offset, core, min_sigma, max_sigma, num_sigma, threshold, backend=None):
    """ Runs blob_log on one haloed tile and keeps the dots in the tile core,
    in whole image coordinates. Module level so it can be sent to a process pool. """
    if backend is None:
        blob_log, to_device, to_host = skimage.feature.blob_log, np.asarray, lambda x: x
    else:
        blob_log, to_device, to_host = backend.blob_log, backend.to_device, backend.to_host
    poses = blob_log(to_device(tile),
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
    )
    poses = np.asarray(to_host(poses), dtype=float).reshape(-1, 3)
    poses[:,0] += offset[0]
    poses[:,1] += offset[1]
    y0, y1, x0, x1 = core
    keep = (poses[:,0] >= y0) & (poses[:,0] < y1) & (poses[:,1] >= x0) & (poses[:,1] < x1)
    return poses[keep]


def _block_reduce(array, factor, xp, func='mean'):
    """ downsample a 2d array by factor, trimming the edges """
    height, width = (array.shape[0] // factor) * factor, (array.shape[1] // factor) * factor
    blocks = array[:height,:width].reshape(height // factor, factor, width // factor, factor)
    return getattr(blocks, func)(axis=(1, 3))


def _collect_diagnostics(prefix, image, mask, filtered, means, stds, greyimage, threshold,
        poses, values, reads, backend, channels, sigma_params,
        crop=None, crop_size=256, stride=8, max_dot_samples=50000, seed=0):
    """ Writes prefix.summary.csv and prefix.images.npz, see starcall.dotdetection_qc.
    filtered holds the z scored DoG image, so DoG = z * std + mean and background = raw - DoG.
    Pixel statistics use every stride'th row and column, outside the NaN mask. """
    from . import dotdetection_qc as qc
    from .qc import get_chastity_df
    sxp = backend.sxp
    num_cycles, num_channels, height, width = image.shape
    channels = list(channels) if channels is not None else list(Read.DEFAULT_CHANNELS)
    channels = [str(chan) for chan in channels] + [str(j) for j in range(len(channels), num_channels)]
    planes = [(i, j) for i in range(num_cycles) for j in range(num_channels)]
    summary = qc.SummaryWriter()

    # per plane pixel statistics
    valid = ~mask[::stride,::stride]
    z_sub = backend.to_host(filtered[:,:,::stride,::stride]) if backend.sxp is not np else filtered[:,:,::stride,::stride]
    def plane_task(plane):
        i, j = plane
        raw = np.array(image[i,j,::stride,::stride], dtype=np.float32)[valid]
        z = np.asarray(z_sub[i,j], dtype=np.float32)[valid]
        dog = z * np.float32(stds[i,j]) + np.float32(means[i,j])
        return plane, raw, dog, z
    for (i, j), raw, dog, z in cpu_map(plane_task, planes, backend.threads):
        cycle, chan = i, channels[j]
        summary.moments('raw', raw, cycle, chan)
        summary.moments('bg', raw - dog, cycle, chan)
        summary.moments('dog', dog, cycle, chan)
        summary.hist('raw_hist', qc.RAW_BINS, raw, cycle, chan)
        summary.hist('dog_hist', qc.DOG_BINS, dog, cycle, chan)
        summary.hist('z_hist', qc.Z_BINS, z, cycle, chan)
        summary.scalar('z_mean', means[i,j], cycle, chan)
        summary.scalar('z_std', stds[i,j], cycle, chan)
    del z_sub

    # dots: values are the (dilated) z scores at the dots
    intposes = poses[:,:2].astype(int)
    called = np.argmax(np.nan_to_num(values, nan=-np.inf), axis=2)
    # raw intensities (not dilated) at a random sample of the dots
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(len(poses), min(len(poses), max_dot_samples), replace=False))
    sample_raw = np.empty((len(sample), num_cycles, num_channels), dtype=np.float32)
    def raw_task(plane):
        i, j = plane
        sample_raw[:,i,j] = image[i,j][intposes[sample,0], intposes[sample,1]]
    cpu_map(raw_task, planes, backend.threads)
    chastity = get_chastity_df(reads) if len(reads) else np.zeros((0, num_cycles))
    for i in range(num_cycles):
        counts = np.bincount(called[:,i], minlength=num_channels)
        called_sample = called[sample,i]
        for j, chan in enumerate(channels):
            summary.scalar('called_count', counts[j], i, chan)
            summary.hist('dot_raw_hist', qc.RAW_BINS, sample_raw[called_sample == j, i, j], i, chan)
            summary.hist('dot_dog_hist', qc.DOG_BINS, values[:,i,j] * stds[i,j] + means[i,j], i, chan)
            summary.hist('dot_z_hist', qc.Z_BINS, values[:,i,j], i, chan)
        summary.hist('chastity_hist', qc.CHASTITY_BINS, chastity[:,i], i, '')
    min_sigma, max_sigma, num_sigma = sigma_params
    levels = np.linspace(min_sigma, max_sigma, num_sigma)
    nearest = np.abs(poses[:,2:3] - levels[None,:]).argmin(axis=1) if len(poses) else np.zeros(0, int)
    level_counts = np.bincount(nearest, minlength=len(levels))
    for level, count in zip(levels, level_counts):
        summary.counts('sigma_count', [level, level], [count])

    # grey image and tile totals
    grey_sub = backend.to_host(greyimage[::stride,::stride])
    summary.hist('grey_hist', qc.GREY_BINS, grey_sub[valid])
    summary.scalar('threshold', threshold)
    summary.scalar('n_dots', len(poses))
    summary.scalar('valid_pixels', int((~mask).sum()))

    summary_path = prefix + '.summary.csv'
    summary.table().to_csv(summary_path, index=False)

    # image crops, around the center of the imaged area by default
    if crop is None:
        rows, cols = np.nonzero(~mask[::stride,::stride])
        crop = (int(rows.mean() * stride), int(cols.mean() * stride)) if len(rows) else (height // 2, width // 2)
    size = min(crop_size, height, width)
    y0 = int(np.clip(crop[0] - size // 2, 0, height - size))
    x0 = int(np.clip(crop[1] - size // 2, 0, width - size))
    window = (slice(None), slice(None), slice(y0, y0 + size), slice(x0, x0 + size))
    crop_raw = np.array(image[window], dtype=np.float32)
    crop_z = np.asarray(backend.to_host(filtered[window]) if sxp is not np else filtered[window], dtype=np.float32)
    crop_dog = crop_z * stds[:,:,None,None].astype(np.float32) + means[:,:,None,None].astype(np.float32)
    crop_grey = np.asarray(backend.to_host(greyimage[y0:y0+size,x0:x0+size]) if sxp is not np else greyimage[y0:y0+size,x0:x0+size])
    in_crop = (poses[:,0] >= y0) & (poses[:,0] < y0 + size) & (poses[:,1] >= x0) & (poses[:,1] < x0 + size)
    crop_dots = poses[in_crop] - np.array([y0, x0, 0])
    scale = max(1, int(math.ceil(max(height, width) / 1024)))
    thumbnail = backend.to_host(_block_reduce(greyimage, scale, sxp, 'mean'))
    bins, extent = 64, [[0, height], [0, width]]
    density, _, _ = np.histogram2d(poses[:,0], poses[:,1], bins=bins, range=extent)
    # bins with no imaged pixels
    valid_rows, valid_cols = np.nonzero(valid)
    imaged, _, _ = np.histogram2d(valid_rows * stride, valid_cols * stride, bins=bins, range=extent)
    density_mask = imaged == 0
    images_path = prefix + '.images.npz'
    np.savez_compressed(images_path,
        channels=np.array(channels), num_cycles=num_cycles,
        crop_origin=np.array([y0, x0]), crop_raw=crop_raw,
        crop_dog=crop_dog.astype(np.float16), crop_z=crop_z.astype(np.float16),
        crop_grey=crop_grey.astype(np.float16), crop_dots=crop_dots.astype(np.float32),
        grey_thumbnail=thumbnail.astype(np.float16), thumbnail_scale=scale,
        mask_thumbnail=_block_reduce(mask, scale, np, 'all'),
        density_hist2d=density.astype(np.int32), density_mask=density_mask,
        sigma_params=np.array(sigma_params, dtype=float), threshold=float(threshold), stride=stride,
        image_shape=np.array(image.shape))
    return [summary_path, images_path]


def _detect_dots_fast(image, backend, timer,
        min_sigma, max_sigma, num_sigma,
        return_sigmas, channels,
        band_pixels, tile_size, blob_pool,
        diagnostics=None, diagnostics_crop=None, diagnostics_crop_size=256, diagnostics_stride=8,
        diagnostics_max_sample_dots=50000):
    xp, sxp = backend.xp, backend.sxp

    if image.ndim == 3:
        image = image.reshape((1,) + image.shape)
    num_cycles, num_channels, height, width = image.shape
    planes = [(i, j) for i in range(num_cycles) for j in range(num_channels)]
    timer.log('starting on image shape {} dtype {}, {} threads'.format(image.shape, image.dtype, backend.threads))

    # NaN positions: union over all planes, like dot_filter_new
    with timer('nanmask'):
        mask = np.zeros((height, width), dtype=bool)
        if np.issubdtype(image.dtype, np.floating):
            rows = _band_rows(band_pixels, width * len(planes))
            def mask_task(band):
                y0, y1, _, _ = band
                mask[y0:y1] = np.isnan(image[:,:,y0:y1]).any(axis=(0,1))
            cpu_map(mask_task, _bands(height, rows), backend.threads)
        any_nan = bool(mask.any())
    timer.log('{} NaN pixels'.format(int(mask.sum())))
    if backend.on_gpu:
        timer.log('filtered image kept on the {}'.format('gpu' if backend.resident else 'host'))
    # where the work is done, sent once
    work_mask = backend.to_device(mask)

    # difference of gaussian filter, per plane and row band. A halo of the
    # gaussian radius makes each band identical to filtering the whole plane
    filtered = sxp.empty(image.shape, dtype=np.float32)
    with timer('dog'):
        sigma = max_sigma
        halo = _gaussian_radius(sigma) + 1
        rows = _band_rows(band_pixels, width)
        tasks = [(i, j, band) for i, j in planes for band in _bands(height, rows, halo)]
        if backend.on_gpu:
            # read from the (memmapped) image in background threads while the gpu works
            load = lambda task: np.array(image[task[0],task[1],task[2][2]:task[2][3]])
        else:
            load = lambda task: image[task[0],task[1],task[2][2]:task[2][3]]
        def dog_task(task, data):
            i, j, (y0, y1, h0, h1) = task
            # on the gpu data was already copied by load and to_device
            src = backend.to_device(data).astype(xp.float32, copy=not backend.on_gpu)
            xp.nan_to_num(src, copy=False)
            src -= backend.ndi.gaussian_filter(src, sigma, mode='nearest', truncate=4.0)
            core = src[y0-h0:y1-h0]
            if any_nan:
                band_mask = work_mask[y0:y1]
                core[band_mask] = xp.nan
                valid = core[~band_mask].astype(xp.float64)
            else:
                valid = core.ravel().astype(xp.float64)
            # sums for the z score, accumulated in float64
            # not valid @ valid: BLAS threads spin after the call and slow down the other threads
            stats = (float(valid.sum()), float((valid * valid).sum()), valid.size)
            backend.store(filtered, (i, j, slice(y0, y1)), core)
            return i, j, stats
        sums = np.zeros((num_cycles, num_channels, 3))
        for i, j, stats in backend.map_loaded(load, dog_task, tasks):
            sums[i,j] += stats
        backend.free()
    means = sums[:,:,0] / sums[:,:,2]
    stds = np.sqrt(np.maximum(sums[:,:,1] / sums[:,:,2] - means ** 2, 0))
    timer.log('plane means {} stds {}'.format(means.round(3).tolist(), stds.round(3).tolist()))
    if backend.memory_info(): timer.log(backend.memory_info())

    # z score and combine into a single grey image, per row band of all planes
    greyimage = sxp.empty((height, width), dtype=np.float32)
    with timer('zscore+highlight'):
        dev_means = backend.to_device(means.astype(np.float32).reshape(num_cycles, num_channels, 1, 1))
        dev_stds = backend.to_device(stds.astype(np.float32).reshape(num_cycles, num_channels, 1, 1))
        rows = _band_rows(band_pixels, width * len(planes))
        def highlight_task(band):
            y0, y1, _, _ = band
            band = (slice(None), slice(None), slice(y0, y1))
            if backend.in_place:
                # a view, so filtered is z scored in place
                values = filtered[band]
            else:
                values = backend.to_device(filtered[band])
            values -= dev_means
            values /= dev_stds
            if not backend.in_place:
                backend.store(filtered, band, values)
            # same as highlight_dots: subtract second maximal channel,
            # std across cycles, sum across channels. Not in place, filtered keeps the z scores
            kth = num_channels - 2
            values = values - xp.partition(values, kth, axis=1)[:,kth:kth+1]
            xp.clip(values, 0, None, out=values)
            grey = values.std(axis=0 if num_cycles > 1 else 1).sum(axis=0)
            if any_nan:
                grey[work_mask[y0:y1]] = -1
            backend.store(greyimage, slice(y0, y1), grey)
        backend.map(highlight_task, list(_bands(height, rows)))
        backend.free()

    with timer('threshold'):
        new_threshold = np.float32(float(greyimage[greyimage != -1].mean()))
        greyimage[greyimage == -1] = 0
    timer.log('blob_log threshold {}'.format(new_threshold))

    # blob_log on overlapping tiles. The halo covers the laplacian of gaussian
    # kernel plus the largest blob overlap distance, so the dots and pruning
    # in each tile's core match running on the whole image
    with timer('blob_log'):
        halo = _gaussian_radius(max_sigma) + 2 + int(math.ceil(2 * math.sqrt(2) * max_sigma))
        tiles = []
        for y0, y1, hy0, hy1 in _bands(height, tile_size, halo):
            for x0, x1, hx0, hx1 in _bands(width, tile_size, halo):
                tiles.append(((hy0, hy1, hx0, hx1), (y0, y1, x0, x1)))
        timer.log('blob_log on {} tiles of size {} with halo {}'.format(len(tiles), tile_size, halo))
        blob_args = (min_sigma, max_sigma, num_sigma, new_threshold)
        if backend.on_gpu or blob_pool == 'thread' or backend.threads <= 1 or len(tiles) <= 1:
            def blob_task(tile):
                (hy0, hy1, hx0, hx1), core = tile
                result = _blob_log_tile(greyimage[hy0:hy1,hx0:hx1], (hy0, hx0), core, *blob_args, backend=backend)
                if backend.on_gpu:
                    backend.free()
                return result
            results = backend.map(blob_task, tiles)
        elif blob_pool == 'process':
            with concurrent.futures.ProcessPoolExecutor(backend.threads) as pool:
                futures = [pool.submit(_blob_log_tile, greyimage[hy0:hy1,hx0:hx1], (hy0, hx0), core, *blob_args)
                        for (hy0, hy1, hx0, hx1), core in tiles]
                results = [future.result() for future in futures]
        else:
            raise ValueError("blob_pool should be 'thread' or 'process', not {}".format(blob_pool))
        poses = np.concatenate(results + [np.zeros((0, 3))], axis=0)
        poses = poses[np.lexsort((poses[:,1], poses[:,0]))]
    sigmas = poses[:,2]
    timer.log('found {} dots'.format(len(poses)))
    if backend.memory_info(): timer.log(backend.memory_info())

    # detect_dots dilates every plane with disk(2) and then samples it at the dots,
    # here we take the max over the footprint only at the dots, which is the same
    with timer('dilate/sample'):
        intposes = poses[:,:2].astype(int)
        radius = 2
        offsets = np.argwhere(skimage.morphology.disk(radius)) - radius
        ys = [sxp.asarray(_reflect_index(intposes[:,0] + dy, height)) for dy, dx in offsets]
        xs = [sxp.asarray(_reflect_index(intposes[:,1] + dx, width)) for dy, dx in offsets]
        values = sxp.empty((len(poses), num_cycles, num_channels), dtype=np.float32)
        def sample_task(plane):
            i, j = plane
            image_plane = filtered[i,j]
            # grey_dilation keeps the first footprint value and replaces it by any larger
            # one, so NaN neighbours are skipped unless the first one is NaN. Same here
            plane_values = image_plane[ys[0], xs[0]]
            for y, x in zip(ys[1:], xs[1:]):
                neighbour = image_plane[y, x]
                sxp.copyto(plane_values, neighbour, where=neighbour > plane_values)
            values[:,i,j] = plane_values
        if sxp is np:
            cpu_map(sample_task, planes, backend.threads)
        else:
            for plane in planes:
                sample_task(plane)
            values = backend.to_host(values)
    if backend.memory_info(): timer.log(backend.memory_info())

    with timer('extract'):
        reads = _make_reads(poses, values, channels)

    if diagnostics is not None:
        with timer('diagnostics'):
            #note that 
            written = _collect_diagnostics(diagnostics, image=image, mask=mask, filtered=filtered,
                means=means, stds=stds, greyimage=greyimage, threshold=new_threshold,
                poses=poses, values=values, reads=reads, backend=backend, channels=channels,
                sigma_params=(min_sigma, max_sigma, num_sigma), crop=diagnostics_crop,
                crop_size=diagnostics_crop_size, stride=diagnostics_stride, max_dot_samples=diagnostics_max_sample_dots)
        timer.log('wrote diagnostics {}'.format(', '.join(written)))

    del filtered, greyimage
    backend.free()

    timer.finish(image.shape, len(reads))

    if return_sigmas:
        return reads, sigmas
    return reads


def detect_dots_cpu(image,
        min_sigma=1,
        max_sigma=2,
        num_sigma=7,
        return_sigmas=False,
        channels=None,
        copy=True,
        threads=None,
        tile_size=2048,
        band_pixels=2**24,
        blob_pool='thread',
        diagnostics=None,
        diagnostics_crop=None,
        diagnostics_crop_size=256,
        diagnostics_stride=8,
        diagnostics_max_sample_dots = 50000,
        debug=False,
        timings=None):
    """ Faster, multithreaded version of detect_dots with the same arguments and output.

    The input image is never modified, so it can be a read only memmap of the raw
    (integer) image; bands are read and converted to float32 as they are needed.
    Only one float32 copy of the image is kept in memory, instead of the several
    that detect_dots needs.

    Args:
        threads (int): number of threads, defaults to all available cores
        tile_size (int): size of the tiles blob_log is run on
        band_pixels (int): number of pixels processed per task in the filtering steps,
            larger uses more memory per thread
        blob_pool ('thread' or 'process'): run blob_log tiles in threads or processes
        diagnostics (str, optional): path prefix, if given prefix.summary.csv and prefix.images.npz
            are written with statistics of the intermediate steps, see starcall.dotdetection_qc
        diagnostics_crop ((y, x), optional): center of the image crops saved in the diagnostics,
            defaults to the center of the imaged (non NaN) area
        diagnostics_crop_size (int): size of the saved crops
        diagnostics_stride (int): pixel statistics are computed on every stride'th row and column
        debug (bool or callable): Report the time of each stage. A callable is used as the print function.
        timings (dict, optional): If given, filled with the seconds spent in each stage.
        copy: ignored, the input is never modified

    Dots are returned sorted by position. Dots and values match detect_dots, apart from
    rare differences from float rounding or blob pruning near tile edges.
    """
    if threads is None:
        threads = len(os.sched_getaffinity(0))
    backend = _CpuBackend(threads)
    timer = _StageTimer('cpu', debug, timings=timings)
    return _detect_dots_fast(image, backend, timer,
        min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
        return_sigmas=return_sigmas, channels=channels,
        band_pixels=band_pixels, tile_size=tile_size, blob_pool=blob_pool,
        diagnostics=diagnostics, diagnostics_crop=diagnostics_crop,
        diagnostics_crop_size=diagnostics_crop_size, diagnostics_stride=diagnostics_stride, diagnostics_max_sample_dots=diagnostics_max_sample_dots)


def detect_dots_gpu(image,
        min_sigma=1,
        max_sigma=2,
        num_sigma=7,
        return_sigmas=False,
        channels=None,
        copy=True,
        threads=None,
        tile_size=2048,
        band_pixels=2**28,
        device=0,
        resident=None,
        diagnostics=None,
        diagnostics_crop=None,
        diagnostics_crop_size=256,
        diagnostics_stride=8,
        diagnostics_max_sample_dots=50000,
        debug=False,
        timings=None):
    """ GPU version of detect_dots, with the same arguments and output. Needs cupy and cucim.

    The input image stays in host memory (it can be a read only memmap of the raw image)
    and is sent to the GPU in bands. If the filtered image fits in GPU memory it is kept
    there for all steps (resident), otherwise it is kept in host memory and bands are sent
    back and forth, so images larger than the GPU memory can still be processed.
    Lower band_pixels or tile_size if the GPU runs out of memory.

    Args:
        threads (int): number of CPU threads for reading the image and host side steps
        tile_size (int): size of the tiles blob_log is run on. cucim prunes overlapping
            blobs comparing every pair of dots in a tile, so large tiles with many dots are slow
        band_pixels (int): number of pixels sent to the GPU at once in the filtering steps
        device (int): cuda device to use
        resident (bool or None): keep the filtered image on the GPU. None decides
            from the image size and the free GPU memory
        diagnostics (str, optional): path prefix, if given prefix.summary.csv and prefix.images.npz
            are written with statistics of the intermediate steps, see starcall.dotdetection_qc
        diagnostics_crop ((y, x), optional): center of the image crops saved in the diagnostics,
            defaults to the center of the imaged (non NaN) area
        diagnostics_crop_size (int): size of the saved crops
        diagnostics_stride (int): pixel statistics are computed on every stride'th row and column
        debug (bool or callable): Report the time of each stage and GPU memory use.
            A callable is used as the print function.
        timings (dict, optional): If given, filled with the seconds spent in each stage.
        copy: ignored, the input is never modified

    Dots are returned sorted by position. Dots and values match detect_dots, apart from
    float rounding and rare blob pruning differences near tile edges.
    """
    if threads is None:
        threads = len(os.sched_getaffinity(0))
    backend = _GpuBackend(threads, device=device)
    timer = _StageTimer('gpu', debug, sync=backend.sync, timings=timings)
    if resident is None:
        num_pixels = int(np.prod(image.shape[-2:]))
        halo_tile = (tile_size + 64) ** 2
        needed = (
            int(np.prod(image.shape)) * 4                 # filtered
            + num_pixels * (4 + 1)                        # grey image and mask
            + max(8 * band_pixels * 4,                    # dog / highlight working space
                  4 * halo_tile * num_sigma * 4)          # blob_log scale space
        )
        free = backend.free_device_memory()
        resident = needed < 0.85 * free
        timer.log('resident needs about {:.1f}GB, gpu has {:.1f}GB free: resident={}'.format(
            needed / 1e9, free / 1e9, resident))
    backend.set_resident(resident)
    return _detect_dots_fast(image, backend, timer,
        min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
        return_sigmas=return_sigmas, channels=channels,
        band_pixels=band_pixels, tile_size=tile_size, blob_pool='thread',
        diagnostics=diagnostics, diagnostics_crop=diagnostics_crop,
        diagnostics_crop_size=diagnostics_crop_size, diagnostics_stride=diagnostics_stride,
        diagnostics_max_sample_dots=diagnostics_max_sample_dots)
