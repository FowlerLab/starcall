""" Classes and functions to store and process cells
generated from cell segmentations, combined with the in situ
sequencing reads that are linked to them.
"""

import warnings
import sys
import base64
import pandas
import dataclasses
from typing import Optional
import numpy as np
import collections.abc
import itertools
import sklearn.neighbors
import skimage.measure
import scipy.ndimage
import heapq
import time

from . import utils

class Cell:
    """ A cell in a segmentation mask. Can have many attributes describing the
    cells shape and position in the image. Some attributes will be inferred when
    not specified, eg if only mask16 is specified mask will be provided by scaling
    up mask16 to the size of the bounding box

    Attributes:
        index: int greater than 0
        
        # position attributes (all ndarrays of shape (2,))
        position: The centroid position of the cell
        size: The size of the bounding box of the cell
        point1: same as position
        point2: same as position + size
        center: same as (point1 + point2) / 2

        # shape attributes

        # mask attributes
        slice: a tuple of slices that can be used to extract a cells bounding box
        mask: A boolean mask of the cell inside its bbox
        global_mask: A boolean mask of the cell in the whole segmentation map
        mask<scale>: A boolean mask of the cell downscaled by scale
        segmentation: The segmentation masks cropped to the cells bbox
        global_segmentation: The entire segmentation mask

        image: phenotyping image cropped to bbox of cell. If it has multiple channels
            they will be stored in the first dimension, so shape (C, W, H)
        global_image: phenotyping image for whole field. Same as image, channels
            are the first axis
    """

    def __init__(self, index=None, bbox=None, position=None,
            global_segmentation=None, global_image=None,
            mask=None, rescaled_masks=None, attrs=None):
        self.index = index
        if mask is not None and position is not None:
            bbox = np.concatenate([position, np.asarray(position) + mask.shape])
        if mask is not None:
            rescaled_masks = {} if rescaled_masks is None else rescaled_masks
            rescaled_masks[1] = mask
        #self.position = np.asarray(position)
        #self.size = np.asarray(size)
        self.bbox = np.asarray(bbox)
        self.global_segmentation = global_segmentation
        self.global_image = global_image
        self.rescaled_masks = {} if rescaled_masks is None else rescaled_masks
        self.attrs = {} if attrs is None else attrs

    def __getitem__(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        return self.attrs[name]

    @property
    def position(self):
        return self.bbox[:2]

    @property
    def size(self):
        return self.bbox[2:] - self.bbox[:2]

    @property
    def point1(self):
        return self.bbox[:2]

    @property
    def point2(self):
        return self.bbox[2:]

    @property
    def center(self):
        return self.bbox.reshape(2,2).mean(axis=0)

    @property
    def slice(self):
        return (slice(self.point1[0], self.point2[0]), slice(self.point1[1], self.point2[1]))

    @property
    def mask(self):
        #mask = self.attrs.get('mask', None)
        mask = self.rescaled_masks.get(1, None)
        return self.segmentation == self.index if mask is None else mask

    @property
    def global_mask(self):
        return self.global_segmentation == self.index

    @property
    def segmentation(self):
        segmentation = self.attrs.get('segmentation', None)
        return self.global_segmentation[self.slice] if segmentation is None else segmentation

    @property
    def image(self):
        image = self.attrs.get('image', None)
        return self.global_image[...,self.slice] if image is None else image

    @property
    def has_mask(self):
        return self.global_segmentation is not None or len(self.rescaled_masks) != 0

    @property
    def best_mask(self):
        if 1 in self.rescaled_masks: return self.rescaled_masks[1]
        if self.global_segmentation is not None:
            return self.mask
        if len(self.rescaled_masks) != 0:
            return self.rescaled_masks[min(self.rescaled_masks)]
        raise AttributeError('best_mask')

    @property
    def best_mask_scale(self):
        if self.global_segmentation is not None:
            return 1
        if len(self.rescaled_masks) != 0:
            return min(self.rescaled_masks)
        return None

    def area(self, method='best'):
        """ Returns the area of the cell, based on the method speficied below:
            'best': the first method available in this list
            'table': the value from an attribute called area
            'mask': the sum of the mask
            'bbox': the total area of the bounding box
        """

        if np.any(self.size <= 0):
            return 0

        best = method == 'best'

        if (best and 'area' in self.attrs) or method == 'table':
            return self.attrs['area']

        if (best and self.has_mask) or method == 'mask':
            scale = self.best_mask_scale
            return self.best_mask.sum() * scale * scale

        if method[:4] == 'mask' and method[4:].isdigit():
            scale = int(method[4:])
            return self.rescaled_masks[scale].sum() * scale * scale

        if best or method == 'bbox':
            return self.size.prod()

    def intersection(self, othercell, mask=True):
        bbox = np.array([*np.maximum(self.bbox[:2], othercell.bbox[:2]),
                *np.minimum(self.bbox[2:], othercell.bbox[2:])])
        attrs = dict(othercell.attrs.items())
        attrs.update(dict(self.attrs.items()))
        newmask = None
        rescaled_masks = None

        if mask and self.has_mask and othercell.has_mask:
            assert self.best_mask_scale == othercell.best_mask_scale
            scale = self.best_mask_scale

            #offset = np.floor((othercell.bbox[:2] - self.bbox[:2]) / scale).astype(int)
            offset = othercell.bbox[:2] - self.bbox[:2]

            cur_box = bbox - [*self.bbox[:2], *self.bbox[:2]]

            #print (self.index, othercell.index, file=sys.stderr)
            combined_mask = combine_masks_round(self.best_mask, othercell.best_mask, cur_box, offset, scale)

            if scale == 1:
                newmask = combined_mask
            else:
                rescaled_masks = {scale: combined_mask}
            #if self.global_segmentation is not None and othercell.global_segmentation is not None:
                #attrs['mask'] = self.mask[bounds] & othercell.mask[otherbounds]

            #for scale in self.rescaled_masks:
                #if scale in othercell.rescaled_masks:
                    #attrs.setdefault('rescaled_masks', {})[scale] = self.rescaled_masks[scale][bounds] & othercell.rescaled_masks[scale][otherbounds]

        cell = Cell(
            index=(self.index, othercell.index),
            bbox=bbox,
            global_segmentation=self.global_segmentation,
            global_image=self.global_image,
            mask=newmask,
            rescaled_masks=rescaled_masks,
            attrs=attrs,
        )

        return cell

    def overlap_ratio(self, othercell, method='best'):
        return self.intersection(othercell, mask=method!='bbox').area / min(self.area(method=method), othercell.area(method=method))

    def decode_mask(self, encoded_str, scale):
        data = base64.a85decode(encoded_str.encode('ascii'))
        arr = np.unpackbits(np.frombuffer(data, np.uint8))
        #ratio = len(arr) / self.size.prod()
        ratio = 1 / scale
        #print (self.bbox)
        dims = np.ceil(self.size * ratio).astype(int)
        #if dims.prod() > arr.shape[0]:
            #dims = np.round(self.size * ratio).astype(int)
        #if dims.prod() > arr.shape[0]:
            #dims = np.floor(self.size * ratio).astype(int)
        #print (dims, self.size, arr.shape)
        return arr[:dims.prod()].reshape(dims)

    def encode_mask(self, mask):
        arr = np.packbits(mask.reshape(-1)).data
        data = base64.a85encode(arr).decode('ascii')
        return data

    def rescale_mask(self, scale):
        #rescaled = skimage.transform.rescale(self.mask, 1/scale, order=0, preserve_range=True)
        #newdims = np.round(self.size / scale).astype(int)
        #print ('newdims', self.size, scale, newdims)
        #rescaled = skimage.transform.resize(self.mask, newdims, order=0, preserve_range=True)
        rescaled = downscale_binary_mask(self.mask, scale)
        self.rescaled_masks[scale] = rescaled
        return rescaled

    def __getattr__(self, name):
        if name[:4] == 'mask' and name[4:].isdigit() and int(name[4:]) in self.rescaled_masks:
            return self.rescaled_masks[int(name[4:])]

        raise AttributeError(name)

    def plot(self, axes, mask=False, **kwargs):
        color = (12984923847923 * hash(self.index) % 255, 6748392493948 * hash(self.index) % 255, 2398042930222 * hash(self.index) % 255)
        if mask:
            mask = self.best_mask
            image = np.zeros((*mask.shape, 4), dtype=np.uint8)
            image[:,:,:3] = [[color]]
            image[:,:,3] = mask * 255
            axes.imshow(image, extent=(self.bbox[0], self.bbox[2], self.bbox[1], self.bbox[3]), **kwargs)
        else:
            xvals = [self.point1[0], self.point1[0], self.point2[0], self.point2[0], self.point1[0]]
            yvals = [self.point1[1], self.point2[1], self.point2[1], self.point1[1], self.point1[1]]
            axes.plot(xvals, yvals, color='#{:02x}{:02x}{:02x}'.format(*color), **kwargs)


def downscale_binary_mask(mask, scale):
    ceildims = np.ceil(np.array(mask.shape) / scale).astype(int) * scale

    if not np.all(ceildims == mask.shape):
        newmask = np.zeros(ceildims, mask.dtype)
        newmask[:mask.shape[0],:mask.shape[1]] = mask
        mask = newmask

    prevarea = mask.sum()
    sums = mask.reshape(mask.shape[0] // scale, scale, mask.shape[1] // scale, scale).sum(axis=(1,3))
    return sums != 0

def combine_masks_round(mask1, mask2, box, offset, scale, func=np.logical_and):
    box = box / scale
    offset = offset / scale
    #print ('box', box)
    #print ('offset', offset)

    result_shape = np.ceil(box[2:] - box[:2]).astype(int)
    result_shape = np.maximum(0, result_shape)
    #print ('result_shape', result_shape)

    offset[0] = np.floor(offset[0]) if offset[0] >= 0 else np.ceil(offset[0])
    offset[1] = np.floor(offset[1]) if offset[1] >= 0 else np.ceil(offset[1])
    offset = offset.astype(int)
    #print ('offset', offset)

    bounds = slice(max(0, offset[0]), max(0, offset[0]) + result_shape[0]), slice(max(0, offset[1]), max(0, offset[1]) + result_shape[1])
    otherbounds = slice(max(0, -offset[0]), max(0, -offset[0]) + result_shape[0]), slice(max(0, -offset[1]), max(0, -offset[1]) + result_shape[1])
    #print (bounds)
    #print (otherbounds)

    """
    box[:2] = np.floor(box[:2])
    box[2:] = np.floor(box[2:])
    box = box.astype(int)
    #print (offset)
    #print (box)
    #print (self.size / scale)
    #print (othercell.size / scale)
    #print (bbox / scale)

    bounds = slice(max(0, box[0]), max(0, box[2])), slice(max(0, box[1]), max(0, box[3]))
    otherbounds = slice(max(0, box[0] - offset[0]), max(0, box[2] - offset[0])), slice(max(0, box[1] - offset[1]), max(0, box[3] - offset[1]))

    cur_box = bbox / scale
    cur_box[:2] = np.floor(cur_box[:2])
    cur_box[2:] = np.ceil(cur_box[2:])
    cur_box = cur_box.astype(int)

    box1 = self.bbox / scale
    box1[:2] = np.floor(box1[:2])
    box1[2:] = np.ceil(box1[2:])
    box1 = box1.astype(int)

    box2 = othercell.bbox / scale
    box2[:2] = np.floor(box2[:2])
    box2[2:] = np.ceil(box2[2:])
    box2 = box2.astype(int)

    #bounds = (max(0, cur_box[0] - box1[0]), max(0, cur_box[2] - box1[0]),
            #max(0, cur_box[1] - box1[1]), max(0, cur_box[3] - box1[1]))
    #otherbounds = (max(0, cur_box[0] - box2[0]), max(0, cur_box[2] - box2[0]),
            #max(0, cur_box[1] - box2[1]), max(0, cur_box[3] - box2[1]))

    #bounds = slice(math.floor(bounds[0]), math.ceil(bounds[1])), slice(math.floor(bounds[2]), math.ceil(bounds[3]))
    #otherbounds = slice(math.floor(otherbounds[0]), math.ceil(otherbounds[1])), slice(math.floor(otherbounds[2]), math.ceil(otherbounds[3]))

    bounds = (slice(max(0, cur_box[0] - box1[0]), max(0, cur_box[2] - box1[0])),
            slice(max(0, cur_box[1] - box1[1]), max(0, cur_box[3] - box1[1])))
    otherbounds = (slice(max(0, cur_box[0] - box2[0]), max(0, cur_box[2] - box2[0])),
            slice(max(0, cur_box[1] - box2[1]), max(0, cur_box[3] - box2[1])))
    """
    #print (self.bbox, file=sys.stderr)
    #print (othercell.bbox, file=sys.stderr)
    #print (bounds, file=sys.stderr)
    #print (otherbounds, file=sys.stderr)
    #print (self.best_mask.shape, othercell.best_mask.shape, file=sys.stderr)
    #print ('----', file=sys.stderr)
    #print (box1, box2, cur_box, file=sys.stderr)

    return func(mask1[bounds], mask2[otherbounds])

def make_cell_table(segmentation=None, positions=None, sizes=None, image=None, properties=None):
    index = None
    cells = None
    rescaled_masks = None
    attrs = {}

    #print ('making cel table')
    #print (type(segmentation))
    if type(segmentation) == list and len(segmentation) > 0 and type(segmentation[0]) == Cell:
        cells = segmentation
        segmentation = None
        index, positions, sizes = [], [], []
        rescaled_masks = {}

        for cell in cells:
            index.append(cell.index)
            positions.append(cell.position)
            sizes.append(cell.size)

            for name, val in cell.attrs.items():
                if name in ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'):
                    continue
                attrs.setdefault(name, []).append(val)

            for scale, mask in cell.rescaled_masks.items():
                rescaled_masks.setdefault(scale, []).append(mask)
        #print ('attrs', list(attrs.keys()))

        if len(index) and type(index[0]) == tuple:
            index = pandas.MultiIndex.from_tuples(index)

        positions = np.array(positions)
        sizes = np.array(sizes)
        rescaled_masks = {scale: pandas.Series(arr, index=index) for scale, arr in rescaled_masks.items()}

    if type(segmentation) == list and len(segmentation) == 0:
        cells = []
        segmentation = None
        index = []
        positions, sizes = np.zeros((0, 2), int), np.zeros((0, 2), int)

    if segmentation is not None:
        props = skimage.measure.regionprops(segmentation)
        index = np.array([prop.label for prop in props])
        bboxes = np.array([prop.bbox for prop in props])
        positions = bboxes[:,:2]
        sizes = bboxes[:,2:] - bboxes[:,:2]
        
        if properties is not None:
            for name in properties:
                attrs[name] = np.array([prop[name] for prop in props])

    elif index is None:
        index = np.arange(1, len(positions) + 1)

    table = dict(
        bbox_x1 = positions[:,0],
        bbox_y1 = positions[:,1],
        bbox_x2 = positions[:,0] + sizes[:,0],
        bbox_y2 = positions[:,1] + sizes[:,1],
    )
    table.update(attrs)
    table = pandas.DataFrame(table, index=index)

    if segmentation is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table.segmentation = segmentation
            table.cells.segmentation = segmentation
    if image is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table.image = image
            table.cells.image = image
    if rescaled_masks is not None:
        table.cells.rescaled_masks.update(rescaled_masks)
        for scale, masks in rescaled_masks.items():
            encoded = table.cells.encode_masks(masks)
            if encoded is not None:
                table['mask' if scale == 1 else 'mask{}'.format(scale)] = encoded

    return table


@pandas.api.extensions.register_dataframe_accessor("cells")
class CellsAccessor:
    """ Accessor to provide attributes and methods to a table containing cells from
    a segmentation map
    """

    def __init__(self, table):
        self.table = table

        self.segmentation, self.image = None, None
        if hasattr(table, 'segmentation'):
            self.segmentation = table.segmentation
        if hasattr(table, 'image'):
            self.image = table.image

        self.rescaled_masks = {}
        for name in table.columns:
            if name[:4] == 'mask':
                if len(name) == 4:
                    self.rescaled_masks[1] = self.decode_masks(table[name], scale)
                if name[4:].isdigit():
                    scale = int(name[4:])
                    self.rescaled_masks[scale] = self.decode_masks(table[name], scale)

    def __getitem__(self, index):
        #row = self.table.loc[index]
        i = self.table.index.get_loc(index)
        return self.at(i)

    def at(self, i):
        index = self.table.index[i]
        #attrs = {name: self.table[name].iloc[i] for name in self.attrs}
        kwargs = {}

        if self.segmentation is not None:
            kwargs['global_segmentation'] = self.segmentation

        if self.image is not None:
            kwargs['global_image'] = self.image

        if len(self.rescaled_masks):
            kwargs['rescaled_masks'] = {scale: self.rescaled_masks[scale].iloc[i] for scale in self.rescaled_masks}

        cell = Cell(index=index, bbox=self.bboxes[i],
                attrs=self.table.iloc[i,:], **kwargs)
        return cell

    def __len__(self):
        return len(self.table.index)

    def __iter__(self):
        return iter(self[i] for i in self.table.index)

    def _consolidate(self):
        self.table._mgr._consolidate_inplace()

    @property
    def bboxes(self):
        self._consolidate()

        col1 = self.table.loc[:,'bbox_x1'].to_numpy()
        col2 = self.table.loc[:,'bbox_y1'].to_numpy()

        full = col1.base
        offset = col1.__array_interface__['data'][0] - full.__array_interface__['data'][0]
        stride = col2.__array_interface__['data'][0] - col1.__array_interface__['data'][0]

        arr = np.ndarray((4, col1.shape[0]), col1.dtype, full, offset, (stride, col1.strides[0]))
        return arr.T

    @property
    def positions(self):
        return self.bboxes[:,:2]

    @property
    def sizes(self):
        return self.bboxes[:,2:] - self.bboxes[:,:2]

    @property
    def centers(self):
        return (self.bboxes[:,2:] + self.bboxes[:,:2]) / 2

    @property
    def masks(self):
        if 1 not in self.rescaled_masks:
            masks = pandas.Series([cell.mask for cell in self], index=self.table.index)
            self.rescaled_masks[1] = masks
        return self.rescaled_masks[1]

    def decode_masks(self, column, scale):
        masks = [cell.decode_mask(column.iloc[i], scale) for i, cell in enumerate(self)]
        masks = pandas.Series(masks, index=self.table.index)
        return masks
        fullstr = ''.join(column)
        sizes = self.sizes
        bytesizes = np.ceil(self.sizes.prod(axis=1) * 5 / 32) * 8
        data = base64.a85decode(fullstr.encode('ascii'))
        arr = np.unpackbits(np.frombuffer(data))
        indices = np.zeros(len(sizes) + 1, int)
        np.cumsum(bytesizes, out=indices[1:])
        masks = [arr[begin:begin+sizes[i,0]*sizes[i,1]].reshape(*sizes[i]) for i, begin in enumerate(indices[:-1])]
        return masks

    def encode_masks(self, masks, limit=250):
        total_size = sum(masks[i].size for i in self.table.index)
        encoded_size = total_size * 5 / 32
        if encoded_size / len(self.table.index) > limit:
            return None
        strs = [self[i].encode_mask(masks[i]) for i in self.table.index]
        column = pandas.Series(strs, index=self.table.index)
        return column

    def rescale_masks(self, scale, limit=250):
        cells = list(self)
        total_size = sum(cell.mask.size for cell in cells)
        encoded_size = total_size / (scale * scale) * 5 / 32
        if encoded_size / len(cells) > limit:
            return None
        masks = [cell.rescale_mask(scale) for cell in cells]
        masks = pandas.Series(masks, index=self.table.index)
        column = self.encode_masks(masks)
        self.table['mask' if scale == 1 else 'mask{}'.format(scale)] = column
        self.rescaled_masks[scale] = masks
        return masks

    def intersecting_cells(self, othertable, method='best'):
        largest_cell = max(np.linalg.norm(self.sizes, axis=1).max(), np.linalg.norm(othertable.cells.sizes, axis=1).max())
        neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=5).fit(othertable.cells.centers)
        distances, indices = neighbors.radius_neighbors(self.centers, radius=largest_cell)

        otherareas = [othertable.cells.at(i).area(method=method) for i in range(len(othertable.index))]

        results = []
        for i in range(len(distances)):
            cellarea = self.at(i).area(method=method)
            for j in indices[i]:
                newcell = self.at(i).intersection(othertable.cells.at(j), mask=method!='bbox')
                area = newcell.area(method=method)
                if area > 0:
                    newcell.attrs['area'] = area
                    newcell.attrs['area_ratio'] = area / min(otherareas[j], cellarea)
                    results.append(newcell)

        return make_cell_table(results)

    def plot(self, axes, masks=False, **kwargs):
        for cell in self:
            cell.plot(axes, mask=masks, **kwargs)

        axes.set_xlim(self.bboxes[:,0].min(), self.bboxes[:,2].max())
        axes.set_ylim(self.bboxes[:,1].min(), self.bboxes[:,3].max())

