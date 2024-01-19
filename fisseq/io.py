import numpy as np
import tifffile
import imageio.v3 as iio

from . import utils


class MemMap:
    """ A memory mapped image file that allows the memory to be freed
    when use is done
    """

    def __init__(self, path):
        self.path = path
        self.soft_load()

    def soft_load(self):
        if self.path.endswith('.tif'):
            try:
                self.arr = tifffile.memmap(self.path, mode='c')
                self.props = {}
                return
            except ValueError:
                pass
        props = iio.improps(self.path)
        nbytes = np.array(props.shape).prod() * props.dtype.itemsize
        self.props = dict(shape=props.shape, dtype=props.dtype, nbytes=nbytes)

    def hard_load(self):
        self.arr = iio.imread(self.path)
        self.props = {}

    def __getattr__(self, name):
        if name in self.props:
            return self.props[name]

        if self.arr is None:
            self.hard_load()

        return getattr(self.arr, name)

    def __array__(self):
        if self.arr is None:
            self.hard_load()
        return self.arr

    def free(self):
        del self.arr
        self.soft_load()


