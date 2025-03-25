import collections
import skimage.measure
import skimage.io
import numpy as np
import skimage.feature
import skimage.filters
import skimage.morphology
import sys

from . import utils
from . import dotdetection

def segment_nuclei(dapi):
    return segment_nuclei_stardist(dapi)
    thresh = skimage.filters.threshold_otsu(dapi)
    mask = dapi > thresh
    #mask = skimage.morphology.closing(mask, skimage.morphology.disk(2))
    labels = skimage.measure.label(mask)
    return labels

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

def segment_cyto_cellpose(cyto, dapi, diameter, gpu=False, 
                     cyto_model='cyto', reconcile=True, logscale=True,
                     remove_edges=True):
    from cellpose.models import Cellpose

    if logscale:
        cyto = image_log_scale(cyto)
    img = np.array([dapi, cyto])

    model_cyto = Cellpose(model_type=cyto_model, gpu=gpu)
    
    cells, _, _, _  = model_cyto.eval(img, channels=[2, 1], diameter=diameter)

    print(f'found {cells.max()} cells', file=sys.stderr)

    return cells

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

