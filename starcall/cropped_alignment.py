""" FFT alignment of a tile against a window of a much larger tile, eg a sequencing tile against a
phenotype cycle imaged as a single tile at a lower magnification.

constitch.FFTAligner zero-pads the smaller image to the size of the larger one and searches the
whole phase correlation for the peak. When the larger image is several times the size of the
smaller one, most of the correlation is unrelated content and the edges of the padding, and the
peak is found at a wrong offset. Here the larger image is cropped to the expected footprint of the
smaller one plus a margin, using the offset of the constraint (from the initial positions), and the
peak is only searched in shifts within that margin. The window is clipped to the larger image, so
the score is the zncc of real pixels only.
"""
import math
import threading
import warnings
import numpy as np
import skimage.transform
import constitch
from constitch.alignment import calc_pcm, refine_peak, zncc_fast

RAMP = 50       # px of linear ramp to zero at the padded edge, as constitch.alignment.image_diff_sizes
MIN_OVERLAP = 100


def pad_to(image, shape):
    """ Pads image with a linear ramp to zero then zeros, to shape """
    if tuple(shape) == image.shape:
        return image
    ramp = [(0, min(RAMP, size - length)) for size, length in zip(shape, image.shape)]
    image = np.pad(image, ramp, mode='linear_ramp', end_values=0)
    return np.pad(image, [(0, size - length) for size, length in zip(shape, image.shape)])


def signed_shift(index, low, size):
    """ The shift in [low, low + size) that is equal to index modulo size """
    return (index - low) % size + low


class CroppedFFTAligner(constitch.Aligner):
    """ Aligns a tile against a window of a larger tile around the expected offset, see the module docstring.

    margin: px (in the scale of the composite) the window extends past the expected footprint of the
        smaller tile on each side, the largest error in the initial offset that can be corrected
    allow_edges: whether peaks on the first row or column of the phase correlation (shifts of 0 on
        one axis) are accepted, see constitch.FFTAligner. Edges of the images cause erroneous peaks there
    upscale_factor: subpixel precision of the offset, as constitch.FFTAligner
    num_peaks: number of highest peaks of the phase correlation scored with zncc
    """
    def __init__(self, margin=600, allow_edges=False, upscale_factor=1, num_peaks=2):
        self.margin, self.allow_edges = margin, allow_edges
        self.upscale_factor, self.num_peaks = upscale_factor, num_peaks
        self.resized, self.lock = {}, threading.Lock()

    def resized_image(self, constraint, index, box):
        """ The image at index resized to the size of its box, cached as the larger tile is shared by many constraints """
        original = constraint.composite.images[index]
        image = original[:,:,0] if len(original.shape) == 3 else original
        if image.shape == tuple(box.size[:2]):
            return image
        with self.lock:
            # kept with the source image, so a new image set at the same index (eg another well) is resized again
            key = (index, tuple(box.size[:2]))
            source, resized = self.resized.get(key, (None, None))
            if source is not original:
                resized = skimage.transform.resize(image.astype(np.float32), box.size[:2])
                self.resized[key] = (original, resized)
            return resized

    def align(self, constraint, precalc1=None, precalc2=None):
        image1 = self.resized_image(constraint, constraint.index1, constraint.box1)
        image2 = self.resized_image(constraint, constraint.index2, constraint.box2)
        expected = np.array([constraint.dx, constraint.dy])

        # the window is taken from the larger image, offsets are of the larger image relative to the smaller
        if np.prod(image2.shape) >= np.prod(image1.shape):
            score, offset = self.align_window(image1, image2, expected)
        else:
            score, offset = self.align_window(image2, image1, -expected)
            offset = -offset if offset is not None else None

        if offset is None:
            return constitch.Constraint(constraint, dx=constraint.dx, dy=constraint.dy, score=-math.inf, error=constraint.error)
        return constitch.Constraint(constraint, dx=offset[0], dy=offset[1], score=score, error=0)

    def align_window(self, tile, big, expected):
        """ Offset of big relative to tile, and its zncc score, searched within self.margin of expected """
        tile = tile.astype(np.float32)
        # window of big around the expected footprint of tile, clipped to big
        start = -np.round(expected).astype(int) - self.margin
        lo = np.maximum(start, 0)
        hi = np.minimum(start + np.array(tile.shape) + 2 * self.margin, big.shape)
        if np.any(hi - lo < MIN_OVERLAP):
            return -math.inf, None
        window = big[lo[0]:hi[0], lo[1]:hi[1]].astype(np.float32)

        # shifts of the window relative to the tile within margin of the expected one
        expected_shift = lo + np.round(expected).astype(int)
        low = expected_shift - self.margin
        span = 2 * self.margin + 1

        shape = np.maximum(window.shape, np.array(tile.shape) + RAMP)
        shape = np.maximum(shape, span)
        if np.any(np.array(window.shape) != shape):
            shape = np.maximum(shape, np.array(window.shape) + RAMP)
        fft_prod = calc_pcm(np.fft.fft2(pad_to(tile, shape)).reshape(-1),
                np.fft.fft2(pad_to(window, shape)).reshape(-1)).reshape(shape)
        correlation = np.fft.ifft2(fft_prod).real

        if not self.allow_edges:
            correlation[0,:] = correlation[-1,:] = correlation[:,0] = correlation[:,-1] = -np.inf
        allowed = np.zeros(correlation.shape, bool)
        allowed[np.ix_((low[0] + np.arange(span)) % shape[0], (low[1] + np.arange(span)) % shape[1])] = True
        correlation[~allowed] = -np.inf

        best_score, best_shift = -math.inf, None
        for i in range(self.num_peaks):
            index = np.unravel_index(np.argmax(correlation), correlation.shape)
            if not np.isfinite(correlation[index]):
                break
            correlation[index] = -np.inf
            shift = np.array([signed_shift(index[axis], low[axis], shape[axis]) for axis in range(2)])
            score = self.zncc(tile, window, shift)
            if score > best_score:
                best_score, best_shift = score, shift

        if best_shift is None:
            return -math.inf, None
        if self.upscale_factor > 1:
            best_shift = refine_peak(fft_prod, self.upscale_factor, best_shift.astype(float))
        # window is at best_shift relative to tile, and big at best_shift - lo
        return best_score, best_shift - lo

    @staticmethod
    def zncc(tile, window, shift):
        """ zncc of the overlap of tile and window, with window at shift relative to tile """
        a0 = np.maximum(shift, 0)
        a1 = np.minimum(tile.shape, shift + np.array(window.shape))
        if np.any(a1 - a0 < MIN_OVERLAP):
            return -math.inf
        return zncc_fast(tile[a0[0]:a1[0], a0[1]:a1[1]], window[a0[0]-shift[0]:a1[0]-shift[0], a0[1]-shift[1]:a1[1]-shift[1]])


def consistent_constraints(constraints, tolerance=20, min_constraints=4):
    """ Removes constraints between two cycles whose correction disagrees with the others.

    The correction of a constraint is its offset minus the offset of the initial positions
    (Constraint.difference). Between a sequencing cycle and a phenotype tile it is expected to be
    an affine function of the position of the sequencing tile: a constant from the stage offset
    between the cycles, plus terms linear in the position from the grid layout of the initial
    composite spacing tiles by their size instead of the stage step, and from any scale or rotation
    difference. An affine function is fit with RANSAC and constraints with a correction more than
    tolerance px from it on either axis are removed.

    With fewer than min_constraints constraints the fit is not determined, and all are kept.
    If no affine function fits at least min_constraints of them all are kept, with a warning.
    Returns the kept constraints as a ConstraintSet.
    """
    import sklearn.linear_model

    consts = list(constraints)
    if len(consts) < min_constraints:
        return constraints

    # position of the smaller tile of each pair, the larger (phenotype) tile is shared by all of them
    positions = np.array([
        (c.box1 if np.prod(c.box1.size[:2]) <= np.prod(c.box2.size[:2]) else c.box2).position[:2]
        for c in consts], dtype=float)
    corrections = np.array([c.difference for c in consts], dtype=float)

    ransac = sklearn.linear_model.RANSACRegressor(sklearn.linear_model.LinearRegression(),
            min_samples=3, residual_threshold=tolerance, max_trials=1000, random_state=0,
            loss=lambda true, pred: np.max(np.abs(true - pred), axis=1))
    try:
        ransac.fit(positions, corrections)
    except ValueError as error:
        warnings.warn('consistency check not applied, no affine fit within {} px: {}'.format(tolerance, error))
        return constraints

    if ransac.inlier_mask_.sum() < min_constraints:
        warnings.warn('consistency check not applied, only {} of {} constraints fit within {} px'.format(
                ransac.inlier_mask_.sum(), len(consts), tolerance))
        return constraints

    keep = set(c.pair for c, inlier in zip(consts, ransac.inlier_mask_) if inlier)
    return constraints.filter(lambda c: c.pair in keep)
