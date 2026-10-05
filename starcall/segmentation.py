import collections
import contextlib
import skimage.measure
import skimage.io
import numpy as np
import skimage.feature
import skimage.filters
import skimage.morphology
import skimage.segmentation
import sys

from . import utils
from . import dotdetection

def segment_nuclei(dapi, method='stardist', **kwargs):
    """ Segments nuclei from a single channel, typically stained with DAPI or another nuclear dye

    Params:
        dapi: numpy array of shape (width, height)
        method: str, default 'stardist'
            The method of nuclear segmentation that is used. Can be:
                'otsu_threshold': uses a simple otsu threshold and segments connected regions
                'stardist': uses stardist to segment nuclei
        **kwargs: arguments are passed onto the specific method selected

    Returns: numpy array of shape (width, height), dtype int, where 0 is background and nonzero integers are cell masks
    """

    if method == 'threshold_otsu':
        thresh = skimage.filters.threshold_otsu(dapi)
        mask = dapi > thresh
        #mask = skimage.morphology.closing(mask, skimage.morphology.disk(2))
        labels = skimage.measure.label(mask)
        return labels

    elif method == 'stardist':
        return segment_nuclei_stardist(dapi)

    raise ValueError('Unrecognized method for nuclear segmentation {}'.format(method))


#stardist_model = None
def segment_nuclei_stardist(dapi):
    #global stardist_model
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    #if stardist_model is None:
    stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
    labels, _ = stardist_model.predict_instances(normalize(dapi))
    return labels

def estimate_cyto(image):
    cyto = image[3]
    np.clip(cyto, 0, np.percentile(cyto, 99))
    #cyto = skimage.morphology.opening(cyto, skimage.morphology.disk(2))
    return cyto
    image = image - image.mean(axis=(1,2)).reshape(-1,1,1)
    image = image / image.std(axis=(1,2)).reshape(-1,1,1)
    dots = dotdetection.dot_filter(image)
    cyto = image - dots
    cyto = image.min(axis=0)
    np.clip(cyto, 0, None, out=cyto)
    return cyto


def segment_cells(cyto, dapi, method='cellpose', gpu=False, **kwargs):
    """ Segments cells using a cytoplasm and nuclear channel.

    Params:
        cyto: the cytoplasmic channel, numpy array of shape (width, height)
        dapi: the nuclear channel, numpy array of shape (width, height)
        method: str, default 'cellpose'
            The method to use for cell segmentation. The available methods are:
                'cellpose': Uses the 'cyto' model of cellpose to segment cells. Expects the parameter 'diameter'
                'stardist': Uses stardist with only the cyto channel to segment cells
        **kwargs: arguments are passed to the specified method

    Returns: numpy int array of shape (width, height), where 0 is background and nonzero integers are cell masks
    """

    if method == 'cellpose':
        return segment_cyto_cellpose(cyto, dapi, gpu=gpu, **kwargs)
    elif method == 'stardist':
        return segment_nuclei_stardist(cyto, **kwargs)

    raise ValueError('Unrecognized segmentation method {}'.format(method))


def segment_cyto_cellpose(cyto, dapi, diameter, use_nuclei_channel, gpu=False, 
                     cyto_model='cyto', logscale=True):
    from cellpose.models import Cellpose

    model_cyto = Cellpose(model_type=cyto_model, gpu=gpu)
    print ('diameter used: ', diameter)

    if logscale:
        cyto = image_log_scale(cyto)
    #allow the model to use just the cytoplasm info vs cytoplasm and nuclei
    with threaded_cellpose_dynamics():
        if use_nuclei_channel:
            img = np.array([dapi, cyto])
            cells, _, _, _  = model_cyto.eval(img, channels = [2,1], diameter=diameter)
        else:
            img = np.array([cyto, dapi])
            cells, _, _, _  = model_cyto.eval(img, diameter=diameter)

    return cells


""" Multithreaded versions of the slowest CPU steps of cellpose 2.2.2 mask creation, steps2D_interp (following the
flows, run by follow_flows) and masks_to_flows_cpu (the flow_threshold quality check, which cellpose runs on the CPU
for images over 1e8 pixels even with a GPU). Both work on independent pixels or masks, and do the same arithmetic in
the same order as cellpose, so the masks are identical to cellpose's; only the work is split across torch's threads.
"""

_map_coordinates_nogil = None

def _get_map_coordinates_nogil():
    """ cellpose.dynamics.map_coordinates, compiled with nogil so threads can run it at the same time """
    global _map_coordinates_nogil
    if _map_coordinates_nogil is None:
        import numba

        @numba.njit(['(int16[:,:,:], float32[:], float32[:], float32[:,:])',
                '(float32[:,:,:], float32[:], float32[:], float32[:,:])'], nogil=True)
        def map_coordinates(I, yc, xc, Y):
            C,Ly,Lx = I.shape
            yc_floor = yc.astype(np.int32)
            xc_floor = xc.astype(np.int32)
            yc = yc - yc_floor
            xc = xc - xc_floor
            for i in range(yc_floor.shape[0]):
                yf = min(Ly-1, max(0, yc_floor[i]))
                xf = min(Lx-1, max(0, xc_floor[i]))
                yf1= min(Ly-1, yf+1)
                xf1= min(Lx-1, xf+1)
                y = yc[i]
                x = xc[i]
                for c in range(C):
                    Y[c,i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
                              np.float32(I[c, yf, xf1]) * (1 - y) * x +
                              np.float32(I[c, yf1, xf]) * y * (1 - x) +
                              np.float32(I[c, yf1, xf1]) * y * x )

        _map_coordinates_nogil = map_coordinates
    return _map_coordinates_nogil

def _threaded_steps2D_interp(original, threads):
    def steps2D_interp(p, dP, niter, use_gpu=False, device=None):
        if use_gpu:
            return original(p, dP, niter, use_gpu=use_gpu, device=device)

        import concurrent.futures
        map_coordinates = _get_map_coordinates_nogil()
        shape = dP.shape[1:]
        # cellpose converts dP on every iteration, it only has to be done once
        dP = dP.astype(np.float32)

        def run_points(start, stop):
            # each point moves independently, so a contiguous block of them is stepped through all iterations
            cur_p = p[:, start:stop]
            dPt = np.zeros(cur_p.shape, np.float32)
            for t in range(niter):
                map_coordinates(dP, cur_p[0], cur_p[1], dPt)
                for k in range(len(cur_p)):
                    cur_p[k] = np.minimum(shape[k]-1, np.maximum(0, cur_p[k] + dPt[k]))

        bounds = np.linspace(0, p.shape[1], threads + 1).astype(int)
        with concurrent.futures.ThreadPoolExecutor(threads) as pool:
            for future in [pool.submit(run_points, start, stop) for start, stop in zip(bounds[:-1], bounds[1:])]:
                future.result()
        return p

    return steps2D_interp

def _threaded_masks_to_flows_cpu(threads):
    import concurrent.futures
    from scipy.ndimage import find_objects
    from cellpose import utils as cellpose_utils
    from cellpose.dynamics import _extend_centers

    def masks_to_flows_cpu(masks, device=None):
        Ly, Lx = masks.shape
        mu = np.zeros((2, Ly, Lx), np.float64)
        mu_c = np.zeros((Ly, Lx), np.float64)

        slices = find_objects(masks)
        dia = cellpose_utils.diameters(masks)[0]
        s2 = (.15 * dia)**2

        # the body of the loop over masks in cellpose, each mask only writes its own pixels of mu and mu_c
        def flows_of_mask(i, si):
            sr,sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            y,x = np.nonzero(masks[sr, sc] == (i+1))
            y = y.astype(np.int32) + 1
            x = x.astype(np.int32) + 1
            ymed = np.median(y)
            xmed = np.median(x)
            imin = np.argmin((x-xmed)**2 + (y-ymed)**2)
            xmed = x[imin]
            ymed = y[imin]

            d2 = (x-xmed)**2 + (y-ymed)**2
            mu_c[sr.start+y-1, sc.start+x-1] = np.exp(-d2/s2)

            niter = 2*np.int32(np.ptp(x) + np.ptp(y))
            T = np.zeros((ly+2)*(lx+2), np.float64)
            T = _extend_centers(T, y, x, ymed, xmed, np.int32(lx), np.int32(niter))
            T[(y+1)*lx + x+1] = np.log(1.+T[(y+1)*lx + x+1])

            dy = T[(y+1)*lx + x] - T[(y-1)*lx + x]
            dx = T[y*lx + x+1] - T[y*lx + x-1]
            mu[:, sr.start+y-1, sc.start+x-1] = np.stack((dy,dx))

        with concurrent.futures.ThreadPoolExecutor(threads) as pool:
            for future in [pool.submit(flows_of_mask, i, si) for i, si in enumerate(slices) if si is not None]:
                future.result()

        mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

        return mu, mu_c

    return masks_to_flows_cpu

@contextlib.contextmanager
def threaded_cellpose_dynamics(threads=None):
    """ Replaces cellpose's steps2D_interp and masks_to_flows_cpu with the multithreaded versions above while active.
    threads defaults to torch's thread count. Only done for cellpose 2.2.2, the version these were copied from.
    """
    import importlib.metadata
    import torch
    import cellpose.dynamics

    if importlib.metadata.version('cellpose') != '2.2.2':
        yield
        return

    threads = threads or torch.get_num_threads()
    originals = cellpose.dynamics.steps2D_interp, cellpose.dynamics.masks_to_flows_cpu
    cellpose.dynamics.steps2D_interp = _threaded_steps2D_interp(originals[0], threads)
    cellpose.dynamics.masks_to_flows_cpu = _threaded_masks_to_flows_cpu(threads)
    try:
        yield
    finally:
        cellpose.dynamics.steps2D_interp, cellpose.dynamics.masks_to_flows_cpu = originals
 
def image_log_scale(data, bottom_percentile=10, floor_threshold=50, ignore_zero=True):
    data = data.astype(float)
    if ignore_zero:
        data_perc = data[data > 0]
    else:
        data_perc = data
    bottom = np.percentile(data_perc, bottom_percentile)
    data[data < bottom] = bottom
    scaled = np.log10(data - bottom + 1)
    # cut out the noisy bits
    floor = np.log10(floor_threshold)
    scaled[scaled < floor] = floor
    return scaled - floor

def match_segmentations(cells, nuclei):
    """ Matches cellular and nuclear segmentation maps

    The provided cells and nuclei segmentation masks are matched, and any
    cell or nucleus that does not overlap with a corresponding cell/nucleus
    is removed. Matching is done by nuclei, as this is typically the more
    consistent segmentation mask. For each nucleus, all cells that are overlapping
    are scored by the percent overlap of the nucleus inside the cell, and the highest
    scoring cell is chosen as the match. Any unmatched cells or nuclei are discarded,
    and both masks are relabeled to share the same indices between them.
    """
    cell_props = skimage.measure.regionprops(cells)
    nuclei_props = skimage.measure.regionprops(nuclei)
    mapping = {}

    cell_props = {prop.label: prop for prop in cell_props}
    nuclei_props = {prop.label: prop for prop in nuclei_props}
    
    for nuclei_obj in utils.simple_progress(nuclei_props.values()):
        possible_cells = cells[nuclei_obj.coords[:,0],nuclei_obj.coords[:,1]]
        counts = collections.Counter(possible_cells[possible_cells!=0]).most_common(1)
        if len(counts) == 0:
            continue
        cell_label = counts[0][0]
        score = counts[0][1] / len(possible_cells) + counts[0][1] / cell_props[cell_label].area
        mapping.setdefault(cell_label, []).append((score, nuclei_obj.label))

    #cells[...] = 0
    #nuclei[...] = 0
    new_cells = np.zeros(cells.shape, cells.dtype)
    new_nuclei = np.zeros(nuclei.shape, nuclei.dtype)
    scores = []
    #new_image = np.zeros((2, *cells.shape), cells.dtype)
    for i, cell_label in enumerate(mapping):
        cell = cell_props[cell_label]
        nuclei_scores = sorted(mapping[cell_label])
        nuclei_label = nuclei_scores[-1][1]
        nucleus = nuclei_props[nuclei_label]
        new_cells[cell.coords[:,0],cell.coords[:,1]] = i + 1
        new_nuclei[nucleus.coords[:,0],nucleus.coords[:,1]] = i + 1
        scores.append(nuclei_scores[0][0])

    #scores = np.array(scores)
    #np.save('tmp_scores.npy', scores)

    return new_cells, new_nuclei



FILTER_TABLE_COLUMNS = ['orig_label', 'area', 'bbox_height', 'bbox_width', 'min_bbox_dim', 'max_bbox_dim',
        'on_edge', 'below_min_area', 'below_min_bbox', 'kept', 'final_label']

def filter_segmentation(masks, remove_edges=True, min_area=100, min_bbox=10, relabel=True, return_table=False):
    """ Filters a segmentation mask based on a set of thresholds typical for
    cell segmentation. 

    remove_edges: if true masks on the edge of the image are removed
        with skimage.segmentation.clear_border
    min_area: removes any cells with an area less than this value
    min_bbox: removes any cells that have a height or width less than this value
    relabel: whether to relabel the segmentation mask to be sequential, done
        with skimage segmentation.relabel_sequential
    return_table: if true, also returns a pandas DataFrame with the area and bbox
        dimensions of every mask before filtering, with columns FILTER_TABLE_COLUMNS.
        The return value is then (masks, table)
    """

    if return_table:
        orig_props = skimage.measure.regionprops(masks)

    if remove_edges:
        masks = skimage.segmentation.clear_border(masks)
        if return_table:
            # clear_border also removes masks connected to edge masks, so edge
            # status is taken from the labels that survive it
            not_edge_labels = set(np.unique(masks))

    if min_area or min_bbox:
        props = skimage.measure.regionprops(masks)
        mapping = np.arange(masks.max() + 1, dtype=masks.dtype)
        remove_masks = False

        for prop in props:
            if prop.area < min_area or prop.bbox[2] - prop.bbox[0] < min_bbox or prop.bbox[3] - prop.bbox[1] < min_bbox:
                mapping[prop.label] = 0
                remove_masks = True

        if remove_masks:
            masks = mapping[masks]

    if relabel:
        masks, mapping, rmapping = skimage.segmentation.relabel_sequential(masks)

    if not return_table:
        return masks

    import pandas

    rows = []
    for prop in orig_props:
        height, width = prop.bbox[2] - prop.bbox[0], prop.bbox[3] - prop.bbox[1]
        on_edge = remove_edges and prop.label not in not_edge_labels
        # labels are only removed or relabeled whole, so any pixel gives the final label
        final_label = int(masks[tuple(prop.coords[0])])
        rows.append([prop.label, int(prop.area), height, width, min(height, width), max(height, width),
                on_edge, bool(min_area) and prop.area < min_area, bool(min_bbox) and min(height, width) < min_bbox,
                final_label != 0, final_label])

    table = pandas.DataFrame(rows, columns=FILTER_TABLE_COLUMNS)
    return masks, table


