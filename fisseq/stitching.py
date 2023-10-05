import collections
import itertools
from PIL import Image, ImageTransform
from common import ImageLoader
import numpy as np
import glob
import os
import sys
import shutil
import pandas as pd
import tqdm

import matplotlib.pyplot as plt

import skimage.restoration


## M2STITCH functions

def pcm(image1, image2):
    """Compute peak correlation matrix for two images.

    Parameters
    ---------
    image1 : np.ndarray
        the first image (the dimension must be 2)

    image2 : np.ndarray
        the second image (the dimension must be 2)

    Returns
    -------
    PCM : np.ndarray
        the peak correlation matrix
    """
    assert image1.ndim == 2
    assert image2.ndim == 2
    assert np.array_equal(image1.shape, image2.shape)
    F1 = np.fft.fft2(image1)
    F2 = np.fft.fft2(image2)
    FC = F1 * np.conjugate(F2)
    return np.fft.ifft2(FC / np.abs(FC)).real.astype(np.float32)


def multi_peak_max(PCM):
    """Find the first to n th largest peaks in PCM.

    Parameters
    ---------
    PCM : np.ndarray
        the peak correlation matrix

    Returns
    -------
    rows : np.ndarray
        the row indices for the peaks
    cols : np.ndarray
        the column indices for the peaks
    vals : np.ndarray
        the values of the peaks
    """
    row, col = np.unravel_index(np.argsort(PCM.ravel()), PCM.shape)
    vals: FloatArray = PCM[row[::-1], col[::-1]]
    return row[::-1], col[::-1], vals


def ncc(image1, image2):
    """Compute the normalized cross correlation for two images.

    Parameters
    ---------
    image1 : np.ndarray
        the first image (the dimension must be 2)

    image2 : np.ndarray
        the second image (the dimension must be 2)

    Returns
    -------
    ncc : Float
        the normalized cross correlation
    """
    assert image1.ndim == 2
    assert image2.ndim == 2
    assert np.array_equal(image1.shape, image2.shape)
    image1 = image1.flatten()
    image2 = image2.flatten()
    n = np.dot(image1 - np.mean(image1), image2 - np.mean(image2))
    d = np.linalg.norm(image1) * np.linalg.norm(image2)
    return n / d


def extract_overlap_subregion(image, y, x):
    """Extract the overlapping subregion of the image.

    Parameters
    ---------
    image : np.ndarray
        the image (the dimension must be 2)
    y : Int
        the y (second last dim.) position
    x : Int
        the x (last dim.) position
    Returns
    -------
    subimage : np.ndarray
        the extracted subimage
    """
    sizeY = image.shape[0]
    sizeX = image.shape[1]
    assert (np.abs(y) < sizeY) and (np.abs(x) < sizeX)
    # clip x to (0, size_Y)
    xstart = int(max(0, min(y, sizeY, key=int), key=int))
    # clip x+sizeY to (0, size_Y)
    xend = int(max(0, min(y + sizeY, sizeY, key=int), key=int))
    ystart = int(max(0, min(x, sizeX, key=int), key=int))
    yend = int(max(0, min(x + sizeX, sizeX, key=int), key=int))
    return image[xstart:xend, ystart:yend]


def interpret_translation(image1, image2, yins, xins, ymin, ymax, xmin, xmax, n = 2):
    """Interpret the translation to find the translation with heighest ncc.

    Parameters
    ---------
    image1 : np.ndarray
        the first image (the dimension must be 2)
    image2 : np.ndarray
        the second image (the dimension must be 2)
    yins : IntArray
        the y positions estimated by PCM
    xins : IntArray
        the x positions estimated by PCM
    ymin : Int
        the minimum value of y (second last dim.)
    ymax : Int
        the maximum value of y (second last dim.)
    xmin : Int
        the minimum value of x (last dim.)
    xmax : Int
        the maximum value of x (last dim.)
    n : Int
        the number of peaks to check, default is 2.

    Returns
    -------
    _ncc : Float
        the highest ncc
    x : Int
        the selected x position
    y : Int
        the selected y position
    """
    assert image1.ndim == 2
    assert image2.ndim == 2
    assert np.array_equal(image1.shape, image2.shape)
    sizeY = image1.shape[0]
    sizeX = image1.shape[1]
    assert np.all(0 <= yins) and np.all(yins < sizeY)
    assert np.all(0 <= xins) and np.all(xins < sizeX)

    _ncc = -np.infty
    y = 0
    x = 0

    ymagss = [yins, sizeY - yins]
    ymagss[1][ymagss[0] == 0] = 0
    xmagss = [xins, sizeX - xins]
    xmagss[1][xmagss[0] == 0] = 0

    # concatenate all the candidates
    _poss = []
    for ymags, xmags, ysign, xsign in itertools.product(
        ymagss, xmagss, [-1, +1], [-1, +1]
    ):
        yvals = ymags * ysign
        xvals = xmags * xsign
        _poss.append([yvals, xvals])
    poss = np.array(_poss)
    valid_ind = (
        (ymin <= poss[:, 0, :])
        & (poss[:, 0, :] <= ymax)
        & (xmin <= poss[:, 1, :])
        & (poss[:, 1, :] <= xmax)
    )
    assert np.any(valid_ind)
    valid_ind = np.any(valid_ind, axis=0)
    for pos in np.moveaxis(poss[:, :, valid_ind], -1, 0)[: int(n)]:
        for yval, xval in pos:
            if (ymin <= yval) and (yval <= ymax) and (xmin <= xval) and (xval <= xmax):
                subI1 = extract_overlap_subregion(image1, yval, xval)
                subI2 = extract_overlap_subregion(image2, -yval, -xval)
                ncc_val = ncc(subI1, subI2)
                if ncc_val > _ncc:
                    _ncc = float(ncc_val)
                    y = int(yval)
                    x = int(xval)
    return _ncc, y, x

def calculate_offset(image1, image2):
    sizeY, sizeX = max(image1.shape[0], image2.shape[0]), max(image1.shape[1], image2.shape[1])
    PCM = pcm(image1, image2).real
    yins, xins, _ = multi_peak_max(PCM)
    max_peak = interpret_translation(image1, image2, yins, xins, -sizeY, sizeY, -sizeX, sizeX)
    return max_peak

####


def downscale_mean(image, factor):
    image = image[:image.shape[0]//factor*factor,:image.shape[1]//factor*factor]
    image = image.reshape(image.shape[0]//factor, factor, image.shape[1]//factor, factor, *image.shape[2:])
    return np.mean(image, axis=(1,3)).astype(image.dtype)


Constraint = collections.namedtuple('Constraint', ['image1', 'image2', 'score', 'offset'])

def stitch_cycles(images, positions, debug=True, progress=False, **kwargs):
    """ Stitches a time sequence of composite images together, aligning both between tiles as well as
        between cycles.

    """
    progress_arg = progress
    if debug: debug = print
    else: debug = lambda *args: None
    if progress:
        import tqdm
        progress = tqdm.tqdm
    else:
        progress = lambda x, **kwargs: x

    import m2stitch
    constraints = []

    for cycle in range(positions[:,0].max() + 1):
        debug ('Stitching cycle', cycle)
        mask = positions[:,0] == cycle
        indices = np.arange(len(positions))[mask]

        result, stats = m2stitch.stitch_images(images[mask], position_indices=positions[mask][:,1:], full_output=True)#, silent=not progress_arg, **kwargs)
        for index, row in result.iterrows():
            for j, direction in enumerate(['left', 'top']):
                if not pd.isna(row[direction]):
                    constraints.append(Constraint(indices[index], indices[row[direction]], row[direction+'_ncc'], (-row[direction+'_y'], -row[direction+'_x'])))
        debug('  done')


    debug('Aligning images across cycles')
    position_indices = {}
    for i in range(len(images)):
        pos = (positions[i,1], positions[i,2])
        position_indices[pos] = position_indices.get(pos, []) + [i]

    for pos, indices in progress(position_indices.items()):
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                score, dx, dy = calculate_offset(images[indices[i]], images[indices[j]])
                constraints.append(Constraint(indices[i], indices[j], score, (dx, dy)))
    debug('  done')

    solution_mat = np.zeros((len(constraints)*2+2, len(images)*2))
    solution_vals = np.zeros(len(constraints)*2+2)
    
    for index in range(len(constraints)):
        id1, id2, score, (dx, dy) = constraints[index]

        solution_mat[index*2, id1*2] = -score
        solution_mat[index*2, id2*2] = score
        solution_vals[index*2] = score * dx

        solution_mat[index*2+1, id1*2+1] = -score
        solution_mat[index*2+1, id2*2+1] = score
        solution_vals[index*2+1] = score * dy

    # anchor tile 0 to 0,0, otherwise there are inf solutions
    solution_mat[-2, 0] = 1
    solution_mat[-1, 1] = 1

    debug('Solving for final tile positions')
    solution, residuals, rank, sing = np.linalg.lstsq(solution_mat, solution_vals, rcond=None)
    debug('  done')

    poses = np.round(solution.reshape(-1,2)).astype(int)
    poses -= poses.min(axis=0).reshape(1,2)
    return poses


def make_test_image(image, grid_size, window_size=50):
    rows = []
    x_size, y_size = window_size, window_size
    for x in np.linspace(0, image.shape[0]-1, grid_size+1).astype(int):
        if x == 0 or x == image.shape[0]-1: x_size *= 2
        tiles = []
        for y in np.linspace(0, image.shape[1]-1, grid_size+1).astype(int):
            if y == 0 or y == image.shape[1]-1: y_size *= 2
            tiles.append(image[x-x_size:x+x_size+1,y-y_size:y+y_size+1])
        rows.append(np.concatenate(tiles, axis=1))
    test_image = np.concatenate(rows, axis=0)
    return test_image


class CompositeImage:
    """
    Class that can be used to combine many different images in an arbitrary coordinate system
    This is useful for stitching microscope images, where you may recieve metadata on the image
    position in some stage position unit.
    """

    def __init__(self, image=None, mask=None, x=None, y=None, width=None, height=None):
        """ Constructs the composite image. If no image is specified the composite stays
        uninitialized until the first image is added.
        """
        if image is not None:
            x = x or 0
            y = y or 0
            width = width or image.shape[0]
            height = height or image.shape[1]
            mask = mask if mask is not None else np.zeros(image.shape[:2], dtype='uint8')
        self.image, self.mask, self.x, self.y, self.width, self.height = image, mask, x, y, width, height
    
    def xres(self):
        """ Returns the resolution of the coordinate system, the width of a pixel in x coords """
        if self.image is None:
            return 1
        return self.image.shape[0]/self.width

    def yres(self):
        """ Returns the resolution of the coordinate system, the width of a pixel in y coords """
        if self.image is None:
            return 1
        return self.image.shape[1]/self.height

    def add_image(self, newimage, x, y, width=None, height=None):
        """ Adds an image to the composite at the specified coordinates. If the width and height of the image are not
        specified then they are assumed to be in line with the x and y resolution of the composite. Overlap between
        images is averaged as best as possible.
        """
        #print ('adding image', newimage.shape, x, y, width, height, self.x, self.y, self.width, self.height, self.xres(), self.yres())
        if width is None:
            width = newimage.shape[0]/self.xres()
        if height is None:
            height = newimage.shape[1]/self.yres()
        
        if (self.image is None):
            self.image = newimage
            self.mask = np.ones(newimage.shape[:2], dtype='uint8')
            self.x, self.y, self.width, self.height = x, y, width, height
        else:
            if abs(newimage.shape[0]/width - self.xres()) > 1/self.xres() or abs(newimage.shape[1]/height - self.yres()) > 1/self.yres():
                print (newimage.shape[0]/width, self.xres(), newimage.shape[1]/height, self.yres())
                newimage, width, height = self.rescale(newimage, width, height)
            pix_x, pix_y = int((x - self.x) * self.xres()), int((y - self.y) * self.yres())

            if pix_x < 0 or pix_x+newimage.shape[0] > self.image.shape[0] or pix_y < 0 or pix_y+newimage.shape[1] > self.image.shape[1]:
                newshape = (
                    max(self.image.shape[0] + max(0,-pix_x), pix_x + newimage.shape[0]),
                    max(self.image.shape[1] + max(0,-pix_y), pix_y + newimage.shape[1]),
                    *self.image.shape[2:]
                )
                print ("Reshaping", newshape)
                newrootimage = np.zeros(newshape, dtype=self.image.dtype)
                newrootmask = np.zeros(newshape[:2], dtype='uint8')
                negpix_x, negpix_y = max(0,-pix_x), max(0,-pix_y)
                newrootimage[negpix_x:negpix_x+self.image.shape[0], negpix_y:negpix_y+self.image.shape[1]] = self.image
                newrootmask[negpix_x:negpix_x+self.image.shape[0], negpix_y:negpix_y+self.image.shape[1]] = self.mask
                
                self.x = min(self.x, x)
                self.y = min(self.y, y)
                self.width = newrootimage.shape[0]/self.xres()
                self.height = newrootimage.shape[1]/self.yres()
                
                self.image = newrootimage
                self.mask = newrootmask

                pix_x, pix_y = int((x - self.x) * self.xres()), int((y - self.y) * self.yres())
            
            x0,x1,y0,y1 = pix_x,pix_x+newimage.shape[0], pix_y,pix_y+newimage.shape[1]
            #mask = self.mask[x0:x1,y0:y1]
            #self.image[x0:x1,y0:y1][mask!=0] //= 2
            #self.image[x0:x1,y0:y1][mask!=0] = (self.image[x0:x1,y0:y1][mask!=0].astype(int) * mask[mask!=0] // (mask[mask!=0]+1)).astype('uint8')
            overlap_mask = self.mask[x0:x1,y0:y1] != 0
            mask_section = self.mask[x0:x1,y0:y1][overlap_mask].reshape([-1] + [1] * (len(self.image.shape) - 2))
            self.image[x0:x1,y0:y1][overlap_mask] = (self.image[x0:x1,y0:y1][overlap_mask] * mask_section / (mask_section + 1)).astype(self.image.dtype)
            newimage[overlap_mask] = (newimage[overlap_mask] / (mask_section + 1)).astype(newimage.dtype)
            self.image[x0:x1,y0:y1] += newimage
            self.mask[x0:x1,y0:y1] += 1
            #self.image[pix_x:pix_x+newimage.shape[0], pix_y:pix_y+newimage.shape[1]] = newimage
            #self.image[pix_x:pix_x+newimage.shape[0], pix_y:pix_y+newimage.shape[1]] += newimage
            #self.mask[pix_x:pix_x+newimage.shape[0], pix_y:pix_y+newimage.shape[1]] += 1

    def full_image(self, fill_val=0):
        """ Returns a copy of the composite image, with the areas not covered by images
        filled with a specified value.
        """
        full_image = self.image.copy()
        full_image[self.mask==0] = fill_val
        return full_image
    
    def best_position(self, image, start_x, start_y, width=None, height=None, search_radius=25, grow_image=False):
        if grow_image:
            newimage = np.zeros((image.shape[0]+2, image.shape[1]+2, *image.shape[2:]), dtype=image.dtype)
            newimage[1:-1,1:-1] = image
            newimage[0,1:-1] = image[0,:]
            newimage[-1,1:-1] = image[-1,:]
            newimage[1:-1,0] = image[:,0]
            newimage[1:-1,-1] = image[:,-1]
            start_x -= 1/self.xres()
            start_y -= 1/self.yres()
            image = newimage
        
        if width is None:
            width = image.shape[0]/self.xres()
        if height is None:
            height = image.shape[1]/self.yres()
        
        start_pix_x, start_pix_y = int((start_x - self.x) * self.xres()), int((start_y - self.y) * self.yres())

        if type(search_radius) == int:
            search_radius = search_radius, search_radius
        
        best_score = 0
        best_pos = 0,0

        for xoff in range(-search_radius[0], search_radius[0]+1):
            for yoff in range(-search_radius[1], search_radius[1]+1):
                pix_x, pix_y = start_pix_x + xoff, start_pix_y + yoff
                
                img_x, img_y = max(0,-pix_x), max(0,-pix_y)
                pix_x, pix_y = max(0,pix_x), max(0,pix_y)
                width, height = max(0,image.shape[0]-img_x), max(0,image.shape[1]-img_y)
                #width = min(self.image.shape[0], pix_x+width) - self.image.shape[0] + width
                #height = min(self.image.shape[1], pix_y+height) - self.image.shape[1] + height
                
                maskslice = self.mask[pix_x:pix_x+width, pix_y:pix_y+height]
                rootslice = self.image[pix_x:pix_x+width, pix_y:pix_y+height]
                width, height = rootslice.shape[:2]
                imageslice = image[img_x:img_x+width, img_y:img_y+height]
                #print (imageslice.shape, maskslice.shape, rootslice.shape, image.shape, width, height, img_x, img_y, pix_x, pix_y)
                if np.sum(maskslice) > 0:
                    score = np.sum(rootslice[maskslice] * imageslice[maskslice]) / np.sum(maskslice)
                    if score > best_score:
                        best_score = score
                        best_pos = xoff, yoff
        
        if grow_image:
            return (start_pix_x + best_pos[0] + 1) / self.xres(), (start_pix_y + best_pos[1] + 1) / self.yres()
        return best_pos[0] / self.xres(), best_pos[1] / self.yres()





def illum_correction(images):
    images = np.array(images)
    background = np.min(images, axis=0)
    #signal = np.percentile(images, axis=0).astype(images[0].dtype) - background
    images = (images - background)
    signal_val = np.percentile(images, 99.9, axis=(0,1,2))
    images = images / signal_val
    images[images<0] = 0
    images[images>1] = 1
    return images

def stitch_grid_manual(images, param_list, grid_size, x_skew, y_skew):
    composite = CompositeImage()

    for image, params in zip(images, param_list):
        image = image[2:-2,...]
        i = params['section']-1
        x = i//grid_size
        y = i%grid_size
        x, y = x*image.shape[0] + y*x_skew, y*image.shape[1] + x*y_skew
        composite.add_image(image, x, y)

    return composite


def stitch_grid_metadata(images, param_list, imageloader, metadata_func):
    composite = CompositeImage()

    for image, params in zip(images, param_list):
        position = [*metadata_func(image, imageloader.to_path(params))]
        #if (not composite.image is None):
            #best_position = composite.best_position(image, *position, search_radius=20)
            #print (best_position, "best pos")
            #position[0] += best_position[0]
            #position[1] += best_position[1]
        composite.add_image(image, *position)

    return composite

def stitch_grid_m2stitch(images, param_list, imageloader, grid_pos_func, metadata_func = None, silent=False, **kwargs):
    if silent:
        import tqdm
        real_tqdm = tqdm.tqdm
        def replacement(iterable, *args, **kwargs):
            return iterable
        tqdm.tqdm = replacement
    import m2stitch
    composite = CompositeImage()
    
    stitch_images = np.array(images)
    if len(stitch_images.shape) > 3:
        stitch_images = np.sum(stitch_images, axis=tuple(range(3,len(stitch_images.shape))))
    
    positions = np.array([grid_pos_func(images[i], param_list[i], imageloader.to_path(param_list[i]))[:2] for i in range(len(images))])
    positions -= positions.min(axis=0)

    #if os.path.exists('result.csv'):
    #    result = pandas.from_csv('result.csv')
    #else:
    if True:
        if metadata_func:
            guesses = np.array([metadata_func(images[i], imageloader.to_path(param_list[i])) for i in range(len(images))])
            result, result_dict = m2stitch.stitch_images(stitch_images, position_indices=positions, position_initial_guesses=guesses, **kwargs)
        else:
            result, result_dict = m2stitch.stitch_images(stitch_images, position_indices=positions, **kwargs)
        #result.to_csv('result.csv')

    for i in range(len(images)):
        composite.add_image(images[i], result.x_pos[i], result.y_pos[i], images[i].shape[0], images[i].shape[1])

    if silent:
        tqdm.tqdm = real_tqdm
    
    return composite

def stitch_test():
    imageloader = ImageLoader()
    imageloader.image_path = 'tmp/'
    imageloader.file_pattern = 'slide{x:d}x{y:d}.png'
    imageloader.load()
    
    images, params = imageloader.load_params()
    
    rows = [param['x'] for param in params]
    cols = [param['y'] for param in params]
    
    positions = [(param['y'], param['x']) for param in params]
    guesses = [(param['x']*100, param['y']*100) for param in params]

    for i in range(len(rows)):
        width, height = images[i].shape[:2]
        #image = np.stack([images[i]]*8, axis=1)
        #image = np.stack([image]*8, axis=3)
        #images[i] = image.reshape(width*8, height*8, -1)
        #guesses.append((rows[i]*100, cols[i]*100))
        #fig, axis = plt.subplots()
        #axis.imshow(images[i].sum(axis=2))
        #axis.set_title("{}x{}".format(rows[i], cols[i]))
    #plt.show()
    
    import m2stitch
    stitch_images = np.array(images).sum(axis=3).astype('uint16')
    print (stitch_images.max(), stitch_images.min(), stitch_images.dtype)
    print (stitch_images.shape)
    print (positions)
    print (guesses)
    result, result_dict = m2stitch.stitch_images(stitch_images, position_indices=positions, ncc_threshold=0.1)#, position_initial_guess=guesses)

    print (result)

    composite = CompositeImage()
    for i in range(len(images)):
        composite.add_image(images[i], result.x_pos[i], result.y_pos[i], images[i].shape[0], images[i].shape[1])
    
    skimage.io.imsave('test_stitch.png', composite.full_image())


