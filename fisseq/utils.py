import collections
import itertools
import numpy as np
import time
import sys
import math

def memory_report():
    import psutil
    return ' '.join([key + '=' + human_readable(val) for key,val in psutil.Process().memory_info()._asdict().items()])

def human_readable(number):
    if number < 1000:
        return str(number)
    scale = min(4, int(math.log10(number)) // 3)
    return '{:.3f}'.format(number / (1000 ** scale)) + ['', 'K', 'M', 'G', 'T'][scale]

def format_time(secs):
    timestr = '{:02}:{:02}'.format(int(secs) // 60 % 60, int(secs) % 60)
    if secs > 3600:
        timestr = '{:02}:'.format(int(secs) // 3600) + timestr
    return timestr

def simple_progress(iterable, total=None):
    if total is None:
        total = len(iterable)

    print_points = [total // 20, total // 4, total // 2, total * 3 // 4]

    if 5 < print_points[0]:
        print_points.insert(0, 5)

    denom_str = str(total)

    start = time.time()
    lasttime = 0
    for i,value in enumerate(iterable):
        yield value
        dtime = time.time() - start
        index = i + 1

        if ((dtime - lasttime >= 2 and (lasttime < 2 or index in print_points))
                or (dtime >= 2 and index == total)):
            est_time = (dtime / index) * total

            padded_index = ('{:' + str(len(denom_str)) + '}').format(index)
            print ("  -- {}/{} {:3}% {} elapsed, {} left, done at {}".format(
                padded_index, denom_str, int(index / total * 100),
                format_time(dtime),
                format_time(est_time - dtime),
                time.strftime("%I:%M %p", time.localtime(start + est_time)),
            ), file=sys.stderr)
            lasttime = dtime
            

def log_env(debug, progress):
    if debug is True:
        debug = lambda *args, **kwargs: print(*args, **kwargs, file=sys.stderr)
    if debug is False:
        debug = lambda *args, **kwargs: None

    if progress is True:
        #import tqdm
        progress = simple_progress
    if progress is False:
        progress = lambda x, **kwargs: x

    return debug, progress

def to_rgb8(image, percent_norm=0.1, colormap=None):
    image = standardize_format(image, 3)
    num_channels, width, height = image.shape

    print (image.min(), image.max())
    minvals = np.percentile(image, percent_norm, axis=(1,2)).reshape(-1,1,1).astype(np.float32)
    maxvals = np.percentile(image, 100 - percent_norm, axis=(1,2)).reshape(-1,1,1).astype(np.float32)
    print (minvals, maxvals)
    image = (image - minvals) / np.maximum(maxvals - minvals, np.finfo(np.float32).tiny)
    print (image.min(), image.max())
    
    if colormap is None:
        if num_channels == 1:
            colormap = np.array([[1],[1],[1]])
        if num_channels >= 2:
            colormap = np.array([
                [1, 0],
                [0, 1],
                [0, 0]])
        if num_channels == 3:
            colormap = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
        if num_channels == 4:
            colormap = np.array([
                [0.66, 0, 0, 0.33],
                [0, 0.66, 0, 0.33],
                [0, 0, 1, 0]])
        if num_channels == 5:
            colormap = np.array([
                [0.5, 0, 0, 0.25, 0.25],
                [0, 0.5, 0, 0.25, 0],
                [0, 0, 0.5, 0, 0.25]])
        if num_channels == 6:
            colormap = np.array([
                [0.5, 0, 0, 0.25, 0.25, 0],
                [0, 0.5, 0, 0.25, 0, 0.25],
                [0, 0, 0.5, 0, 0.25, 0.25]])
        if num_channels == 7:
            colormap = np.array([
                [0.43, 0, 0, 0.21, 0.21, 0, 0.14],
                [0, 0.43, 0, 0.21, 0, 0.21, 0.14],
                [0, 0, 0.43, 0, 0.21, 0.21, 0.14]])

    print (colormap)

    flatimage = image.reshape(num_channels, -1)
    newimage = np.matmul(colormap, flatimage).T
    image = newimage.reshape(width, height, 3)

    print (image.min(), image.max())
    
    image[image<0] = 0
    image[image>1] = 1
    image = (image * 255).astype('uint8')
    
    return image


def standardize_format(image, expected_dims):
    """ Converts an input image into the format
    expected, this being:
    (width, height), (channels, width, height), (cycle, channels, width, height)
    depending on the expected_dims. """

    if len(image.shape) == 3:
        if expected_dims < 3:
            raise ValueError("Expected image with shape: (CHANNELS, WIDTH, HEIGHT), got {}".format(image.shape))
        if image.shape[2] < 32:
            image = image.transpose((2,0,1))

    if len(image.shape) == 4:
        if expected_dims < 4:
            raise ValueError("Expected image with shape: (CYCLE, CHANNELS, WIDTH, HEIGHT), got {}".format(image.shape))
        if image.shape[2] < 32:
            image = image.transpose((0,3,1,2))
    
    return image.reshape((1,) * (expected_dims - len(image.shape)) + image.shape)

def percent_normalize(image, percent=0.1):
    if len(image.shape) == 2:
        image = image.reshape([1] + image.shape)

    mins = np.percentile(image, [percent], axis=(1,2)).reshape((-1,1,1))
    maxes = np.percentile(image, [100-percent], axis=(1,2)).reshape((-1,1,1))

    image = (image - mins) / (maxes - mins)
    image[image<0] = 0
    image[image>1] = 1

    return image

"""
def to_rgb8(image):
    if image.dtype != np.uint8:
        image = percent_normalize(image).astype(np.uint8)

    image = image.transpose((1,2,0))
    return image[:,:,:3]
"""

def mark_dots(image, poses):
    marking_layer = np.zeros_like(image[0])
    for pos in poses:
        mark_dot(marking_layer, pos)

    return np.array([*image, marking_layer])

def mark_dot(image, dot, inner_rad=3, outer_rad=6, color=1):
    x,y = dot
    image[x-outer_rad:x-inner_rad+1, y] = color
    image[x+inner_rad:x+outer_rad+1, y] = color
    image[x, y-outer_rad:y-inner_rad+1] = color
    image[x, y+inner_rad:y+outer_rad+1] = color

_let_points = [
    [[1, 1/2, 0, 0, 1/2, 1, 1, 1/2], [1, 1, 2/3, 1/3, 0, 0, 1/2, 1/2]],
    [[0, 1, 1/2, 1/2], [1, 1, 1, 0]],
    [[0, 0, 1, 0, 0, 1/2, 1, 1], [0, 1/3, 1/3, 1/3, 2/3, 1, 2/3, 0]],
    [[1, 1/2, 0, 0, 1/2, 1], [1, 1, 2/3, 1/3, 0, 0]],
]

def sequence_plot(axis, sequences, separate=False):
    if separate:
        letter_sets = [[(let, 1) for let in seq] for seq in zip(*sequences)]
        print (letter_sets)
    else:
        #letter_sets = list(map(collections.Counter, zip(*sequences)))
        letter_sets = [sorted(list(collections.Counter(seq).items()), key=lambda pair: -'GTAC'.index(pair[0])) for seq in zip(*sequences)]
    axis.set_xlim(0, len(letter_sets))
    axis.set_ylim(0, 1)
    x_margin = 0.1
    y_margin = 0.02
    
    for i, lets in enumerate(letter_sets):
        total = sum(pair[1] for pair in lets)
        cur_count = 0
        for let, count in lets:
        #for let, count in sorted(list(lets.items()), key=lambda pair: -'GTAC'.index(pair[0])):
            index = 'GTAC'.index(let)
            xposes = np.array(_let_points[index][0]) * (1 - x_margin * 2) + x_margin + i
            yposes = np.array(_let_points[index][1]) * (count / total - y_margin * 2) + y_margin + cur_count / total
            axis.plot(xposes, yposes, color=['purple', 'blue', 'green', 'red'][index], linewidth=5, solid_capstyle='round')
            cur_count += count

def write_multicolumn(table, name, array, length_column=None):
    """ Splits a 2d array into multiple columns in a dataframe for saving to csv or
    similar format. Writes N columns with names name + '_0' to name + '_N' where N is array.shape[1].
    Also supports ragged arrays where not every row is the same length, in this case pass
    a list of lists. Writes the different number of items in each row to another column named
    length_column, which defaults to name + '_length'
    The complement function to read_multicolumn which can be used to extract the 2d array
    back from the table.
    """

    name_fmt = name if '{' in name else (name + '_{}')

    if type(array) == np.ndarray:
        assert array.ndim == 2
        columns = array.transpose()

    else:
        assert '{' not in name or length_column is not None, "If using a custom format string, length_column must be specifyed"
        length_column = length_column or name + '_length'

        lengths = np.fromiter(map(len, array), int, len(array))

        dtype = np.array(next(iter(itertools.chain.from_iterable(array)))).dtype
        for val in itertools.chain.from_iterable(array):
            dtype = np.result_type(dtype, np.array(val).dtype)

        columns = [np.zeros(len(array), dtype) for i in range(lengths.max())]
        for i in range(lengths.max()):
            for j in range(len(array)):
                if i < len(array[j]):
                    print ('setting', i, j, array[j][i])
                    columns[i][j] = array[j][i]

        table[length_column] = lengths

    for i in range(len(columns)):
        table[name_fmt.format(i)] = columns[i]

def read_multicolumn(table, name, length_column=None):
    """ The complement function to write_multicolumn. Finds all columns like name + '_XX' where
    XX is an integer and concatenates the columns to create a 2d array.
    If there is also a column name + '_length' it will be used to create a
    ragged array, where each row has a different length.
    """

    name_fmt = name if '{' in name else (name + '_{}')

    columns = []
    index = 0
    while name_fmt.format(index) in table:
        columns.append(table[name_fmt.format(index)])
        index += 1

    if length_column is not None or name + '_length' in table:
        length_column = length_column or name + '_length'
        lengths = table[length_column]

        array = []
        for i in range(len(lengths.index)):
            row = [columns[j].iloc[i] for j in range(lengths.iloc[i])]
            array.append(row)

        return array

    else:
        return np.stack(columns, axis=-1)

