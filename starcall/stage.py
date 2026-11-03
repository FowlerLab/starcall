""" Placement of tiles from the stage positions recorded by the microscope.

The initial composite places the tiles of each cycle on a grid from their grid positions
(see make_initial_composite in alignment.smk), so the absolute stage position of each
cycle is lost and every cycle is placed relative to itself. This is fine when all cycles
cover the same area, but a phenotype cycle imaged as a single tile (or a few tiles) at
some point of the well can't be placed relative to the sequencing cycles that way.
Here the stage positions of a reference sequencing cycle are used to fit a map from stage
coordinates to the positions of its tiles in the composite, which is then used to place
the tiles of other cycles from their own stage positions.
"""
import numpy as np


def read_stage_positions(path):
    """ Stage positions of the tiles of an nd2 file, in the same order as the positions
    written by extract_nd2_positions. Returns an array (N, 3) of x, y (um) and the pixel size (um) """
    import nd2

    with nd2.ND2File(path) as images:
        full_meta = images.ome_metadata()

    rows = []
    for image in full_meta.images:
        pixels = image.pixels
        planes = [plane for plane in pixels.planes if plane.the_c == 0]
        for plane in planes:
            rows.append((plane.position_x, plane.position_y, pixels.physical_size_x))
    return np.array(rows, dtype=float)


def stage_to_pixels(stage, pixel_size):
    """ Stage positions (N, 2+) in um to pixel coordinates (N, 2), with the same axes
    as the positions of extract_nd2_positions: (y, -x) / pixel_size """
    return np.stack([stage[:,1], -stage[:,0]], axis=1) / pixel_size


def fit_stage_map(stage, centers, pixel_size):
    """ Fits an affine map from the stage positions of the tiles of a cycle to the centers
    of their boxes in the composite. The grid layout of the composite spaces tiles by their
    full size instead of the stage step, and may be centered or slightly rotated relative
    to the stage axes, all of which the affine map accounts for.
    Returns the (3, 2) map and the rms residual of the fit in pixels """
    design = np.concatenate([stage_to_pixels(stage, pixel_size), np.ones((len(stage), 1))], axis=1)
    stage_map, *_ = np.linalg.lstsq(design, centers, rcond=None)
    rms = np.sqrt(np.mean(np.sum((design @ stage_map - centers) ** 2, axis=1)))
    return stage_map, rms


def apply_stage_map(stage_map, stage, pixel_size):
    """ Centers in the composite of tiles at the given stage positions """
    design = np.concatenate([stage_to_pixels(stage, pixel_size), np.ones((len(stage), 1))], axis=1)
    return design @ stage_map


def layer_centers(composite, layer):
    """ Centers (N, 2) of the boxes of a layer of the composite """
    return np.array([box.position[:2] + box.size[:2] / 2 for box in composite.layer(layer).boxes])


def place_from_stage(composite, stages, reference, layers):
    """ Moves the boxes of the given layers of the composite so that their positions
    relative to the reference layer match their stage positions. The sizes of the boxes
    are kept, so phenotype boxes should already be rescaled to the scale of the composite.

    composite: constitch.CompositeImage with the reference layer already placed
    stages: list of stage position arrays from read_stage_positions, one per layer
    reference: the layer used to fit the map, normally the first sequencing cycle
    layers: the layers moved

    Returns the rms residual of the fit of the reference layer, in pixels
    """
    pixel_size = stages[reference][0,2]
    stage_map, rms = fit_stage_map(stages[reference], layer_centers(composite, reference), pixel_size)

    for layer in layers:
        boxes = composite.layer(layer).boxes
        assert len(stages[layer]) == len(boxes), 'stage positions do not match the tiles of layer {}'.format(layer)
        centers = apply_stage_map(stage_map, stages[layer], pixel_size)
        for box, center in zip(boxes, centers):
            box.position[:2] = np.round(center - box.size[:2] / 2).astype(int)

    return rms
