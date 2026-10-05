""" Detection and correction of rotation of the image content between cycles.

Stitching only translates tiles, so if the sample is rotated in one cycle relative to the
others (eg after the plate is put back on the stage) the tile centres can be registered
but the content of each tile drifts towards its edges. Here the rotation of each cycle is
measured on a sample of tiles, by matching a grid of small patches of a tile against the
same area in a reference cycle and fitting a similarity transform to the matches, and
tiles are rotated back about their centre when they are read.

Angles are in degrees, in the convention of rotate_frame (scipy.ndimage.rotate on the
image axes). The angle measured for a cycle is the rotation of its content relative to the
reference: if mov = rotate_frame(ref, a), measure_tile_rotation(ref, mov) gives a, so a cycle
measured at angle a is corrected with rotate_frame(frame, -a).
"""
import numpy as np

PATCH_SIZE = 256
SEARCH_MARGIN = 64
COARSE_FACTOR = 4
MIN_NCC = 0.3
MIN_PATCHES = 4
MIN_TILES = 4


def rotate_frame(frame, angle, order=1):
    """ Rotates a tile (H, W) or (C, H, W) by angle degrees about its centre, keeping its
    shape. The pixels uncovered at the corners are filled with the nearest edge values. """
    if angle == 0:
        return frame
    import scipy.ndimage
    frame = np.asarray(frame)
    return scipy.ndimage.rotate(frame, angle, axes=(frame.ndim - 1, frame.ndim - 2), reshape=False,
            order=order, mode='nearest')


def _peak(result):
    """ Location of the maximum of a match_template result, with subpixel refinement from
    a parabola through the neighboring values, and the value at the maximum """
    index = np.unravel_index(np.argmax(result), result.shape)
    location = np.array(index, dtype=float)
    for axis in range(2):
        if 0 < index[axis] < result.shape[axis] - 1:
            before, after = list(index), list(index)
            before[axis] -= 1
            after[axis] += 1
            low, mid, high = result[tuple(before)], result[index], result[tuple(after)]
            denom = low - 2 * mid + high
            if denom < 0:
                location[axis] += 0.5 * (low - high) / denom
    return location, float(result[index])


def measure_tile_rotation(ref, mov, mov_scale=1):
    """ Measures the rotation of the image content of mov relative to ref (see the module
    docstring for the convention), two images of
    about the same area in different cycles. mov is first rescaled by mov_scale (eg 0.5 for a
    phenotype tile imaged at twice the resolution of the base images), it can be smaller than
    ref and offset from it by up to about a quarter of the size of ref.

    Returns a dict with angle (degrees), scale (of mov after rescaling, relative to ref),
    rms_px (rms error of the fit in pixels of ref), n_patches (patches used in the fit) and
    offset (position of mov in ref), or None if not enough patches could be matched, eg when
    the tile is mostly empty.
    """
    import skimage.feature
    import skimage.measure
    import skimage.transform

    ref = np.asarray(ref, dtype=np.float32)
    mov = np.asarray(mov, dtype=np.float32)
    if mov_scale != 1:
        mov = skimage.transform.rescale(mov, mov_scale, anti_aliasing=True, preserve_range=True).astype(np.float32)

    # coarse offset of mov in ref, matching the center half of mov at a lower resolution
    ref_small = skimage.transform.downscale_local_mean(ref, COARSE_FACTOR)
    mov_small = skimage.transform.downscale_local_mean(mov, COARSE_FACTOR)
    h, w = mov_small.shape
    crop_origin = np.array([h // 4, w // 4])
    crop = mov_small[h//4:h//4 + h//2, w//4:w//4 + w//2]
    if crop.std() == 0 or ref_small.std() == 0:
        return None
    location, _ = _peak(skimage.feature.match_template(ref_small, crop))
    offset = np.round((location - crop_origin) * COARSE_FACTOR).astype(int)

    # grid of patches in the part of mov that overlaps ref
    lo = np.maximum(0, -offset) + SEARCH_MARGIN
    hi = np.minimum(mov.shape, np.array(ref.shape) - offset) - SEARCH_MARGIN - PATCH_SIZE
    if np.any(hi < lo):
        return None
    src, dst = [], []
    for py in np.linspace(lo[0], hi[0], 3).astype(int):
        for px in np.linspace(lo[1], hi[1], 3).astype(int):
            patch = mov[py:py + PATCH_SIZE, px:px + PATCH_SIZE]
            if patch.std() == 0:
                continue
            wy, wx = py + offset[0] - SEARCH_MARGIN, px + offset[1] - SEARCH_MARGIN
            window = ref[wy:wy + PATCH_SIZE + 2 * SEARCH_MARGIN, wx:wx + PATCH_SIZE + 2 * SEARCH_MARGIN]
            if window.shape != (PATCH_SIZE + 2 * SEARCH_MARGIN,) * 2:
                continue
            location, ncc = _peak(skimage.feature.match_template(window, patch))
            if ncc < MIN_NCC:
                continue
            centre = PATCH_SIZE / 2
            src.append((px + centre, py + centre))
            dst.append((wx + location[1] + centre, wy + location[0] + centre))

    if len(src) < MIN_PATCHES:
        return None
    src, dst = np.array(src), np.array(dst)
    model, inliers = skimage.measure.ransac((src, dst), skimage.transform.SimilarityTransform, min_samples=3,
            residual_threshold=1.5, max_trials=200, rng=0)
    if model is None or inliers.sum() < MIN_PATCHES:
        return None
    residuals = model.residuals(src[inliers], dst[inliers])
    # the transform maps mov onto ref, its rotation is the rotation of ref relative to mov in
    # (x, y) coordinates with y downwards, which is the rotation of mov relative to ref on screen
    return dict(angle=float(np.degrees(model.rotation)), scale=float(model.scale),
            rms_px=float(np.sqrt(np.mean(residuals ** 2))), n_patches=int(inliers.sum()),
            offset=offset)


class FrameReader:
    """ Reads single tiles of one channel from an nd2 or tif file of a cycle """

    def __init__(self, path, channel):
        self.path, self.channel = path, channel
        if path.endswith('.nd2'):
            import nd2
            self.file = nd2.ND2File(path)
            self.images = None
        else:
            import tifffile
            images = tifffile.memmap(path, mode='r')
            self.images = images.reshape(-1, *images.shape[-3:])

    def __call__(self, index):
        if self.images is None:
            return np.array(self.file.read_frame(index)[self.channel])  # copy, read_frame returns a view of the file
        return np.array(self.images[index, self.channel])

    def close(self):
        if self.images is None:
            self.file.close()


def sample_tiles(composite, layer, num_tiles):
    """ Indices of about num_tiles tiles of a layer of the composite, spread over the central
    70% of the well so that they are away from the mostly empty tiles at the edge """
    indices = np.flatnonzero(composite.boxes.positions[:,2] == layer)
    centers = composite.boxes.centers[indices,:2]
    lo, hi = centers.min(axis=0), centers.max(axis=0)
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.7
    side = int(np.ceil(np.sqrt(num_tiles)))
    grid = np.linspace(-1, 1, side)
    chosen = []
    for gy in grid:
        for gx in grid:
            point = mid + half * (gy, gx)
            order = np.argsort(np.linalg.norm(centers - point, axis=1))
            for i in order:
                if indices[i] not in chosen:
                    chosen.append(int(indices[i]))
                    break
    return chosen[:num_tiles]


def _nearest_tile(composite, layer, center):
    indices = np.flatnonzero(composite.boxes.positions[:,2] == layer)
    centers = composite.boxes.centers[indices,:2]
    return int(indices[np.argmin(np.linalg.norm(centers - center, axis=1))])


def detect_rotations(readers, cycle_names, is_phenotype, composite, phenotype_scale_factor,
        reference='median', threshold=0.05, correct=True, num_tiles=16, order=1, executor=None):
    """ Measures the rotation of every cycle and decides the correction applied to each.

    readers: one FrameReader (or callable tile index -> 2d image of the alignment channel)
        per cycle, in the order of the layers of composite
    cycle_names: name of each cycle, eg '03' or 'PT'
    is_phenotype: whether each cycle is a phenotype cycle, these are rescaled by
        phenotype_scale_factor (bases_scale / phenotype_scale) before matching
    composite: the initial composite, with the stage positions of all cycles at base scale
    reference: 'median' to use the median rotation of the sequencing cycles as zero, or the
        name of a cycle
    threshold: cycles whose rotation from the reference is smaller than this (degrees) are
        not corrected

    All cycles are measured against the first sequencing cycle. Returns (cycle_table,
    tile_table) as lists of dicts, see detect_rotation in alignment.smk for the columns.
    """
    import concurrent.futures
    executor = executor or concurrent.futures.ThreadPoolExecutor(max_workers=1)
    measure_index = [i for i, pt in enumerate(is_phenotype) if not pt][0] #by default chooses the first
    sample = sample_tiles(composite, measure_index, num_tiles)
    ref_start = np.flatnonzero(composite.boxes.positions[:,2] == measure_index)[0]
    ref_frames = {tile: readers[measure_index](tile - ref_start) for tile in sample}

    def moving_tiles(cycle):
        if is_phenotype[cycle]:
            return [_nearest_tile(composite, cycle, composite.boxes.centers[tile,:2]) for tile in sample]
        # sequencing cycles have the same tiles, in the same order
        layer_start = np.flatnonzero(composite.boxes.positions[:,2] == cycle)[0]
        return [int(tile - ref_start + layer_start) for tile in sample]

    def measure(cycle, frames, angle=0):
        scale = phenotype_scale_factor if is_phenotype[cycle] else 1
        jobs = [executor.submit(measure_tile_rotation, ref_frames[tile], rotate_frame(frame, angle, order), scale)
                for tile, frame in zip(sample, frames)]
        return [job.result() for job in jobs]

    tile_table, measured, frames_by_cycle = [], {}, {}
    for cycle, name in enumerate(cycle_names):
        tiles = moving_tiles(cycle)
        frames = [readers[cycle](tile - np.flatnonzero(composite.boxes.positions[:,2] == cycle)[0]) for tile in tiles]
        frames_by_cycle[cycle] = frames
        results = measure(cycle, frames)
        for ref_tile, tile, result in zip(sample, tiles, results):
            row = dict(cycle=name, tile=tile, reference_tile=ref_tile, angle_deg=np.nan, scale=np.nan, rms_px=np.nan, n_patches=0)
            if result is not None:
                row.update(angle_deg=result['angle'], scale=result['scale'], rms_px=result['rms_px'], n_patches=result['n_patches'])
            tile_table.append(row)
        measured[cycle] = np.array([r['angle'] for r in results if r is not None])

    seq_medians = [np.median(measured[c]) for c in range(len(cycle_names)) if not is_phenotype[c] and len(measured[c]) >= MIN_TILES]
    if reference == 'median':
        zero = float(np.median(seq_medians)) if seq_medians else 0.0
    else:
        ref_cycle = cycle_names.index(str(reference))
        zero = float(np.median(measured[ref_cycle])) if len(measured[ref_cycle]) else 0.0

    cycle_table = []
    for cycle, name in enumerate(cycle_names):
        angles = measured[cycle]
        rows = [row for row in tile_table if row['cycle'] == name and np.isfinite(row['angle_deg'])]
        row = dict(cycle=name, measured_deg=np.nan, relative_deg=np.nan, applied_deg=0.0, residual_after_deg=np.nan,
                n_tiles=len(angles), spread_deg=np.nan, scale=np.nan, fit_rms_px=np.nan, status='')
        if len(angles) < MIN_TILES:
            row['status'] = 'failed: only {} tiles could be measured'.format(len(angles))
            cycle_table.append(row)
            continue
        measured_deg = float(np.median(angles))
        relative = measured_deg - zero
        row.update(measured_deg=measured_deg, relative_deg=relative,
                spread_deg=float(np.percentile(angles, 75) - np.percentile(angles, 25)),
                scale=float(np.median([r['scale'] for r in rows])), fit_rms_px=float(np.median([r['rms_px'] for r in rows])))
        if abs(relative) < threshold:
            row.update(status='below threshold', residual_after_deg=relative)
        elif not correct:
            row.update(status='not corrected (correct: False)', residual_after_deg=relative)
        else:
            applied = -relative
            after = [r['angle'] for r in measure(cycle, frames_by_cycle[cycle], applied) if r is not None]
            row.update(applied_deg=applied, status='rotated',
                    residual_after_deg=float(np.median(after)) - zero if len(after) else np.nan)
        cycle_table.append(row)
    return cycle_table, tile_table, zero
