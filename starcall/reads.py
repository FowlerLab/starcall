""" Classes and functions to store and process the
short reads that are generated from in situ sequencing.

Reads are stored as the Read dataclass and in pandas dataframes.
"""

import math
import pandas
import dataclasses
from typing import Optional
import numpy as np
import collections.abc
import itertools
import sklearn.neighbors
import skimage.measure
import scipy.ndimage
import scipy.spatial.distance
import heapq
import time

from . import utils

class Read(collections.abc.Mapping):
    """ A sequencing read found in an image. Can be constructed with
    a position and the image or with a position and the values
    from that position.

    A Read has three main components, each of which can be None, depending
    on what is known about the read and in what step of the sequencing process
    it is. These attributes are:

        position (ndarray of shape (2,)): The position of the read. This is relevant for all
        reads detected from the sequencing images, and holds the position at which they were found, in pixels.
        Reads without a position would be library barcodes that need to be compared to sequencing reads,
        or cell consensus reads that don't have a single position.

        values (ndarray of shape (n_cycles, n_channels)): The values extracted from the sequencing
        images for this read. As such, this is only present for reads that came from sequencing
        images, for example barcodes from the library don't have any raw sequencing values related to them.

        sequence (string of len n_cycles): The sequence of the read.
        The sequence of the read is always present and will not be None, however if a sequence
        is not specified when creating the Read it is inferred from the values, by taking
        the maximum channel for each cycle.

    Additional attributes can be added when creating the Read as keyword arguments, and can be accessed as if
    the read is a dictionary
    """

    #position: Optional[tuple]
    #values: Optional[np.ndarray]
    #sequence: Optional[str]
    #channels: tuple
    #attrs: Optional[dict]

    DEFAULT_CHANNELS = ('G', 'T', 'A', 'C')

    def __init__(self, position=None, values=None, sequence=None, image=None, channels=None, **kwargs):
        """ 
        """
        if isinstance(position, str):
            sequence = position
            position = None

        if position is not None:
            position = np.asarray(position)
            if values is None and image is None and len(position) != 2:
                values = position
                position = None

        if values is None and (position is None or image is None) and sequence is None:
            raise ValueError("Either values, sequence, or both position and image must be specified")

        if values is None and image is not None and position is not None:
            values = image[:,:,position[0],position[1]]

        channels = channels if channels is not None else self.DEFAULT_CHANNELS

        if values is not None:
            values = np.asarray(values)

            if sequence is None and values.dtype.kind == 'U':
                sequence = values
                values = None
            else:
                if len(values.shape) != 2 or values.shape[1] != len(channels):
                    raise ValueError("Expected an array of shape (n_cycles, n_channels) for values")

                self.n_cycles = values.shape[0]

        if sequence is not None:
            sequence = np.asarray(sequence, dtype='U').reshape(-1)
            if values is not None and len(sequence[0]) != values.shape[0]:
                raise ValueError("Expected values.shape[0] and len(sequence) to be equivalent")
            self.n_cycles = len(sequence[0])

        self.position = position
        self.values = values
        self._sequence = sequence
        self.channels = channels
        self.attrs = kwargs

    @property
    def sequence(self):
        if self._sequence is not None:
            return self._sequence[0]

        seq = np.asarray(self.channels, dtype='U1')[np.argmax(self.values, axis=1)]
        seq = np.frombuffer(seq, 'U' + str(seq.shape[0]))
        return seq[0]

    @sequence.setter
    def sequence(self, value):
        if type(value) != str or len(value) != self.n_cycles:
            raise ValueError("Expected a string of length self.n_cycles")

        if self._sequence is None:
            self._sequence = np.array([value], dtype='U' + str(self.n_cycles))
        else:
            self._sequence[0] = value

    @property
    def sequence_array(self):
        if self._sequence is not None:
            seq = np.frombuffer(self._sequence, 'U1')
            return seq

        seq = np.asarray(self.channels, dtype='U1')[np.argmax(self.values, axis=1)]
        return seq

    @sequence_array.setter
    def sequence_array(self, value):
        if self._sequence is None:
            self._sequence = np.empty(1, 'U' + str(self.n_cycles))

        seq = np.frombuffer(self._sequence, 'U1')
        seq[...] = value

    @property
    def qualities(self):
        return self.values.max(axis=1)

    def __repr__(self):
        parts = []

        if self.position is not None:
            parts.append('position={}'.format(self.position))
        if self.values is not None:
            parts.append('values=[...]')
        parts.append(str(self.sequence))

        for name, val in self.attrs.items():
            parts.append('{}={}'.format(name, val))

        return 'Read({})'.format(', '.join(parts))

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        if self.position is not None:
            yield 'position_x'
            yield 'position_y'

        if self.values is not None:
            for cycle in range(self.values.shape[0]):
                for chan in self.channels:
                    yield 'values_cycle{:02}_{}'.format(cycle, chan)

        if self.sequence is not None:
            yield 'sequence'

        for key in self.attrs:
            yield key

    def __getitem__(self, name):
        value = self.attrs.get(name, None)
        if value is not None:
            return value

        if name == 'position_x':
            return self.position[0]
        if name == 'position_y':
            return self.position[1]

        if name[:12] == 'values_cycle':
            splitindex = name.rfind('_')
            cycleno = int(name[12:splitindex])
            chan = self.channels.index(name[splitindex+1:])
            return self.values[cycleno,chan]

        if name == 'sequence':
            return self.sequence

    def __setitem__(self, name):
        if name == 'position_x':
            self.position = (value, self.position[1])
            return
        if name == 'position_y':
            self.position = (self.position[0], value)
            return

        if name[:12] == 'values_cycle':
            splitindex = name.rfind('_')
            cycleno = int(name[12:splitindex])
            chan = self.channels.index(name[splitindex+1:])
            self.values[cycleno,chan] = value
            return

        if name == 'sequence':
            self.sequence = value
            return

        self.attrs[name] = value

    def __len__(self):
        count = 0

        if self.position is not None:
            count += 2
        if self.values is not None:
            count += self.values.size
        if self.sequence is not None:
            count += 1

        return count + len(self.attrs)

    @staticmethod
    def asread(obj):
        if isinstance(obj, Read):
            return obj
        return Read(obj)

def make_readset(positions=None, values=None, sequences=None, image=None, channels=None, **kwargs):
    """ Creates a pandas DataFrame that is compatible with the ReadsAccessor provided
    by this package.

    """
    tables = []

    if positions is not None:
        if values is None and sequences is None and image is None:
            # only one argument, must be iterable of Read instances or sequences
            for read in positions:
                if type(read) == str:
                    table.setdefault('sequence', []).append(read)
                else:
                    print (read)
                    print (type(read))
                    kdsfls

            return pandas.DataFrame(table)

        positions = np.asarray(positions)
        table = pandas.DataFrame(positions, columns=['position_x', 'position_y'])
        tables.append(table)

    if values is None and (positions is None or image is None) and sequences is None:
        raise ValueError("Either values, sequence, or both position and image must be specified")

    if values is None and image is not None and position is not None:
        values = image[:,:,position[:,0],position[:,1]]

    channels = channels if channels is not None else Read.DEFAULT_CHANNELS

    if values is not None:
        values = np.asarray(values)

        columns = []
        for cycle in range(values.shape[1]):
            for chan in range(values.shape[2]):
                colname = 'values_cycle{:02}_{}'.format(cycle, channels[chan])
                columns.append(colname)

        table = pandas.DataFrame(values.reshape(values.shape[0], -1), columns=columns)
        tables.append(table)

    if sequences is not None:
        sequences = np.asarray(sequences, dtype='U')
        table = pandas.DataFrame(sequences, columns=['sequence'])
        tables.append(table)

    if len(kwargs) != 0:
        tables.append(pandas.DataFrame(kwargs))

    table = pandas.concat(tables, axis=1)
    return table


def join_contiguous_arrays(arrays):
    assert all(arr.base is arrays[0].base for arr in arrays)
    assert all(arr.shape == arrays[0].shape for arr in arrays)
    assert all(arr.strides == arrays[0].strides for arr in arrays)
    assert arrays[0].ndim == 1

    pointers = [arr.__array_interface__['data'][0] for arr in arrays]
    stride = pointers[1] - pointers[0]
    offset = pointers[0] - arrays[0].base.__array_interface__['data'][0]
    assert all(end - begin == stride for begin, end in zip(pointers, pointers[1:]))

    #print (offset, stride)

    return np.ndarray(
        shape = (len(arrays), arrays[0].shape[0]),
        dtype = arrays[0].dtype,
        buffer = arrays[0].base,
        offset = offset,
        strides = (stride, arrays[0].strides[0])
    )




@pandas.api.extensions.register_dataframe_accessor("reads")
class ReadsAccessor:
    """ Accessor object to provide attributes and functions for
    dataframes containing in situ sequencing reads

    A dataframe with the columns below can use the accessor:
        (optional) 'position_x', 'position_y': The pixel location of the sequencing read
        'values_cycle00_G', 'values_cycle00_T' ... 'values_cycle11_A', 'values_cycle11_C':
            The sequencing values of the read, from the filtered sequencing images.
            Values from every cycle and each channel are stored.
        (optional) 'sequence': The sequences of the read

    The accessor provides custom properties listed below:
        'positions': a numpy array of shape (num_reads, 2). The positions of the reads,
        'values': a numpy array of shape (num_reads, num_cycles, num_channels). 
            The sequencing values for all reads
        'sequences': a numpy array of strings of shape (num_reads,). If there is a column
            called 'sequence' in the dataframe this is just the contents of that column.
            Otherwise, the sequences are generated from the sequencing values, by selecting
            the maximum channel in each cycle to build up a sequence.

    Changes to positions and values will both propagate back to the underlying dataframe, so
    you can do something like:
        table.reads.positions *= 2
        table.reads.values /= np.linalg.norm(table.reads.values, axis=2)[:,:,None]

    Full reference documentation is available at <https://fowlerlab.github.io/starcall-docs/starcall.html>
    """
    def __init__(self, table):
        self.channels = []
        self.num_cycles = 0
        self.attrs = []

        self.has_position = False
        self.has_values = False
        self.has_sequence = False

        index = 0

        while index < len(table.columns):
            if table.columns[index] == 'position_x':
                if table.columns[index+1] != 'position_y':
                    raise AttributeError('DataFrame has column \'position_x\' but not \'position_y\'. '
                            'Both columns must be adjacent, to ensure correct order use starcall.reads.make_readset()')
                self.has_position = True
                index += 2

            elif table.columns[index][:12] == 'values_cycle':
                self.has_values = True

                if table.columns[index][12:-1] != '00_':
                            raise AttributeError('Expected value columns to be in format \'values_cycle{cycle:02}_{channel}\' '
                                    'in sequential order. To ensure correct order use starcall.reads.make_readset()')

                start_index = index
                while index < len(table.columns) and table.columns[index][:-1] == 'values_cycle00_':
                    self.channels.append(table.columns[index][-1])
                    index += 1

                self.num_cycles = 1

                while index < len(table.columns) and table.columns[index][:12] == 'values_cycle':
                    for chan in self.channels:
                        if index >= len(table.columns) or table.columns[index] != 'values_cycle{:02}_{}'.format(self.num_cycles, chan):
                            raise AttributeError('Expected value columns to be in format \'values_cycle{cycle:02}_{channel}\' '
                                    'in sequential order. To ensure correct order use starcall.reads.make_readset()')
                        index += 1
                    self.num_cycles += 1

            elif table.columns[index] == 'sequence':
                self.has_sequence = True
                index += 1

            else:
                self.attrs.append(table.columns[index])
                index += 1

        if self.has_values:
            self.channels = tuple(self.channels)
        else:
            if not self.has_sequence:
                raise AttributeError('Either values or sequences must be included in a table')

            self.channels = Read.DEFAULT_CHANNELS
            self.num_cycles = len(table['sequence'].iloc[0]) if len(table.index) > 0 else 0

        self.table = table

    def __getitem__(self, index):
        #row = self.table.loc[index]
        index = self.table.index.get_loc(index)
        position, values, sequence = None, None, None

        if self.has_position:
            position = self.positions[index,:]

        if self.has_values:
            values = self.values[index,:]

        if self.has_sequence:
            sequence = self.sequences[index]

        attrs = {name: self.table[name].iloc[index] for name in self.attrs}

        read = Read(position=position, values=values, sequence=sequence, **attrs)
        return read

        if 'position_x' in self.table.columns:
            position = row['position_x'], row['position_y']

        value_col_names = [col for col in self.table.columns if col[:12] == 'values_cycle']
        if len(value_col_names) != 0:
            first_cycle_name = value_col_names[0][:value_col_names[0].rfind('_')]
            num_channels = sum(col[:len(first_cycle_name)] == first_cycle_name for col in value_col_names)
            values = row.loc[value_col_names[0]:value_col_names[-1]].to_numpy()
            values = values.reshape(-1, num_channels)

        if 'sequence' in self.table.columns:
            sequence = row.loc['sequence':'sequence'].to_numpy()

        return Read(position=position, values=values, sequence=sequence)

    def __len__(self):
        return len(self.table.index)

    def __iter__(self):
        return iter(self[i] for i in self.table.index)

    def _consolidate(self):
        self.table._mgr._consolidate_inplace()

    @property
    def positions(self):
        self._consolidate()
        if 'position_x' in self.table.columns:
            col1 = self.table.loc[:,'position_x'].to_numpy()
            col2 = self.table.loc[:,'position_y'].to_numpy()

            full = col1.base
            offset = col1.__array_interface__['data'][0] - full.__array_interface__['data'][0]
            stride = col2.__array_interface__['data'][0] - col1.__array_interface__['data'][0]

            arr = np.ndarray((2, col1.shape[0]), col1.dtype, full, offset, (stride, col1.strides[0]))
            return arr.T

            if full.ndim == 1 or full.shape[0] < 2:
                raise AttributeError('The columns for a read set must be in a specific order, '
                        'use starcall.reads.make_readset() to reorder')
            return full.T
        return None

    @property
    def values(self):
        self._consolidate()
        if self.has_values:
            colname1 = 'values_cycle00_{}'.format(self.channels[0])
            colname2 = 'values_cycle00_{}'.format(self.channels[1])

            col2 = self.table.loc[:,colname2].to_numpy()
            col1 = self.table.loc[:,colname1].to_numpy()

            full = col1.base
            offset = col1.__array_interface__['data'][0] - full.__array_interface__['data'][0]
            stride = col2.__array_interface__['data'][0] - col1.__array_interface__['data'][0]

            arr = np.ndarray((self.num_cycles * len(self.channels), col1.shape[0]), col1.dtype, full, offset, (stride, col1.strides[0]))
            arr = arr.T
            arr.shape = (len(self.table.index), self.num_cycles, len(self.channels))
            return arr

            if full.ndim == 1 or full.shape[0] < self.num_cycles * len(self.channels):
                raise AttributeError('The columns for a read set must be in a specific order, '
                        'use starcall.reads.make_readset() to reorder')
            full = full.T
            full.shape = (full.shape[0], -1, len(self.channels))
            return full
        return None

    @property
    def sequences(self):
        if 'sequence' in self.table.columns:
            return self.table['sequence'].to_numpy().astype('U')

        values = self.values
        indices = np.argmax(values, axis=2)
        sequences = np.array(self.channels, dtype='U1')[indices]
        sequences = np.frombuffer(sequences, 'U' + str(sequences.shape[1]))
        return sequences

    @property
    def sequences_array(self):
        sequences = self.sequences
        sequences = np.frombuffer(sequences, 'U1').reshape(len(self.table.index), -1)
        return sequences

    def to_cell_table(self, cell_column='cell', include_attrs=['count'], cell_index=None):
        include_attrs = [name for name in include_attrs if name in self.table.columns]

        seq_table = {'index': self.table.index, 'sequence': self.sequences, cell_column: self.table[cell_column]}
        for name in include_attrs:
            seq_table[name] = self.table[name]

        seq_table = pandas.DataFrame(seq_table)

        groups = seq_table.groupby(cell_column)
        max_size = groups.size().max()

        #cell_index = pandas.Index(range(1, max(groups.groups.keys()) + 1))
        if cell_index is None:
            cell_index = pandas.Index(groups.groups.keys())

        num_reads = np.zeros(len(cell_index), dtype=int)
        total_count = np.zeros(len(cell_index), dtype=int)
        tables = []

        for i in range(max_size):
            new_table = groups.nth(i)
            new_table = pandas.DataFrame(new_table, index=cell_index)
            new_table['index'] = new_table['index'].fillna(-1).astype(int)
            if 'count' in include_attrs:
                new_table['count'] = new_table['count'].fillna(0).astype(int)

            num_reads += new_table['index'] != -1
            total_count += new_table['count']

            renames = {'index': 'index_{}'.format(i), 'sequence': 'read_{}'.format(i)}
            for name in include_attrs:
                renames[name] = name + '_{}'.format(i)

            new_table = new_table.rename(columns=renames)
            tables.append(new_table)

        tables.insert(0, pandas.DataFrame({'num_reads': num_reads}, index=cell_index))
        tables.append(pandas.DataFrame({'total_count': total_count}, index=cell_index))

        return pandas.concat(tables, axis=1)

    def aggfuncs(self, position=None, values=None, **kwargs):
        aggs = {}
        if position is not None:
            aggs['position_x'] = position
            aggs['position_y'] = position

        if values is not None:
            for cycle in range(self.num_cycles):
                for chan in self.channels:
                    aggs['values_cycle{:02}_{}'.format(cycle,chan)] = values

        aggs.update(kwargs)

        for name in aggs:
            if aggs[name] == 'mode':
                aggs[name] = lambda col: col.mode().max()

        aggs = {name: pandas.NamedAgg(column=name, aggfunc=val) for name, val in aggs.items()}
        return aggs

    def normalize(self, method='full'):
        """ Normalizes the values of this read set, based on the method specified
        Possible methods are:
            'full' (default): values are normalized across the channel axis, so that for each
            cycle the norm of the vector of all channels is 1
        """
        values = self.values

        if method == 'large':
            norms = np.linalg.norm(values, axis=2)
            np.maximum(1, norms, out=norms)
            values /= norms[:,:,None]
        if method == 'full':
            norms = np.linalg.norm(values, axis=2)
            np.maximum(0.0000000001,  norms, out=norms)
            values /= norms[:,:,None]
        if method == 'sub':
            sorted_values = np.sort(values, axis=2)
            values -= sorted_values[:,:,-2:-1]

    def plot_values(self, path, **kwargs):
        """ Plots the base values for all reads in this table.
        Saves an interactive 3d plot with plotly.
        """
        from . import visualization
        visualization.plot_basevalues_plotly(path, self.table, **kwargs)






def positional_distance_matrix(
            positions, positions2=None,
            cells=None,
            distance_cutoff=50,
            matrix=None,
            plot_path=None,
            debug=True, progress=True):
    """ Calculates a sparse distance matrix 
    """

    if positions2 is None:
        positions2 = positions

    debug, progress = utils.log_env(debug, progress)

    if matrix is None:
        matrix = {}

    """
    if cells is not None:
        # adding all reads in same cell as zero distance
        cell_groups = {}

        for i, (x,y) in enumerate(positions):
            cell_groups.setdefault(cells[x,y], ([], []))[0].append(i)
        for i, (x,y) in enumerate(positions2):
            cell_groups2.setdefault(cells[x,y], ([], []))[1].append(i)

        for cellid, (group1, group2) in cell_groups.items():
            for i in group1:
                for j in group2:
                    matrix[i,j] = 0
    """

    radius = distance_cutoff
    if cells is not None:
        props = skimage.measure.regionprops(cells)
        max_size = max((np.linalg.norm([prop.bbox[2] - prop.bbox[0], prop.bbox[1] - prop.bbox[3]]) for prop in props), default=0)
        radius += max_size

    cells_dists = set()
    if cells is not None:
        debug ("Calculating cell distances")

        neighbors = sklearn.neighbors.NearestNeighbors(radius=radius).fit(positions)
        neighbors2 = sklearn.neighbors.NearestNeighbors(radius=radius).fit(positions2)

        cell_poses = np.array([prop.centroid for prop in props])
        bboxes = np.array([prop.bbox for prop in props])
        dists, indices = neighbors.radius_neighbors(cell_poses, radius=radius)
        dists2, indices2 = neighbors2.radius_neighbors(cell_poses, radius=radius)

        for i, bbox in enumerate(progress(bboxes)):
            cell = props[i].label
            x1 = max(0, math.floor(bbox[0] - distance_cutoff))
            y1 = max(0, math.floor(bbox[1] - distance_cutoff))
            x2 = math.ceil(bbox[2] + distance_cutoff)
            y2 = math.ceil(bbox[3] + distance_cutoff)
            section = cells[x1:x2,y1:y2] == cell

            if distance_cutoff <= 1:
                # avoid edt calculation if we only are looking at reads inside cell
                cell_dists = ~section
            else:
                cell_dists = scipy.ndimage.distance_transform_edt(~section)
                cell_dists[section] = 0

            #print (indices[i])
            #print (indices2[i])

            cur_poses = positions[indices[i]] - [[x1, y1]]
            cur_poses = np.round(cur_poses).astype(int)
            cur_poses2 = positions2[indices2[i]] - [[x1, y1]]
            cur_poses2 = np.round(cur_poses2).astype(int)
            #print (cur_poses)
            #print (cell_dists.astype(int))

            for j in indices[i]:
                x,y = np.round(positions[j]).astype(int) - [x1, y1]
                if not (0 <= x < section.shape[0]) or not (0 <= y < section.shape[1]):
                    continue
                for k in indices2[i]:
                    #if k < j: continue

                    z,w = np.round(positions2[k]).astype(int) - [x1, y1]
                    if 0 <= z < section.shape[0] and 0 <= w < section.shape[1]:
                        #debug (positions[j])
                        #debug (x, y, z, w)
                        dist = cell_dists[x,y] + cell_dists[z,w]
                        if dist <= distance_cutoff:
                            matrix[j,k] = dist
                            cells_dists.add((j,k))

            """
            read_cell_dists = (cell_dists[cur_poses[:,0],cur_poses[:,1]].reshape(-1,1)
                             + cell_dists[cur_poses2[:,0],cur_poses2[:,1]].reshape(1,-1))
            for pair in np.argwhere(read_cell_dists > distance_cutoff):
                pair = tuple(pair) if pair[0] < pair[1] else (pair[1], pair[0])
                matrix[pair] = read_cell_dists[pair]
                cells_dists.add(pair)
            """
        debug ("  done")

    debug ("Calculating positional distances")
    if cells is None:
        neighbors = sklearn.neighbors.NearestNeighbors(radius=distance_cutoff).fit(positions)
    dists, indices = neighbors.radius_neighbors(positions2, radius=distance_cutoff)
    debug ("  done")

    for i in range(len(dists)):
        for j, dist in zip(indices[i], dists[i]):
            #pair = (i,j) if i < j else (j,i)
            pair = j,i
            if pair in cells_dists:
                matrix[pair] = min(matrix[pair], dist)
            else:
                matrix[pair] = dist

    if plot_path is not None:
        plot_positional_dist_matrix(plot_path, matrix, positions, positions2, cells)

    return matrix


def plot_positional_dist_matrix(path, matrix, positions, positions2, cells=None):
    import matplotlib.pyplot as plt
    import matplotlib.collections

    fig, axes = plt.subplots()

    if cells is not None:
        axes.imshow(cells, cmap='Greys')

    lines = []
    colors = []
    for (i,j), dist in matrix.items():
        lines.append([(positions[i][1], positions[i][0]), (positions2[j][1], positions2[j][0])])
        colors.append(dist)

    lines, colors = np.array(lines), np.array(colors)
    lines = lines[np.argsort(-colors)]
    colors = colors[np.argsort(-colors)]

    lines = matplotlib.collections.LineCollection(lines, linewidths=colors * 3 + 1, zorder=0)
    lines.set_array(colors)
    axes.add_collection(lines)

    axes.scatter(positions[:,1], positions[:,0], s=50)
    axes.scatter(positions2[:,1], positions2[:,0], s=50)

    fig.colorbar(lines)
    fig.savefig(path)


class LazyDistanceMatrix:
    def __init__(self, reads, reads2, distance_cutoff, distance_func, full_matrix_func, **args):
        self.reads, self.reads2 = reads, reads2
        self.distance_cutoff = distance_cutoff
        self.distance_func = distance_func
        self.full_matrix_func = full_matrix_func
        self.args = args

    def __contains__(self, pair):
        return self.distance_func(self.reads[pair[0]], self.reads2[pair[1]], **self.args) <= self.distance_cutoff

    def __getitem__(self, pair):
        return self.distance_func(self.reads[pair[0]], self.reads2[pair[1]], **self.args)

    def todict(self):
        return self.full_matrix_func(self.reads, self.reads2, distance_cutoff=self.distance_cutoff, lazy=False, **self.args)

    def items(self):
        return self.todict().items()


def values_distance(vals1, vals2, metric='euclidean'):
    return getattr(scipy.spatial.distance, metric)(vals1, vals2)

def value_distance_matrix(
            values, values2=None,
            distance_cutoff=50,
            metric='euclidean',
            matrix=None,
            lazy=True,
            debug=True, progress=True):

    if values2 is None:
        values2 = values

    debug, progress = utils.log_env(debug, progress)

    if lazy and matrix is None:
        return LazyDistanceMatrix(values.reshape(values.shape[0], -1), values2.reshape(values2.shape[0], -1), distance_cutoff, values_distance, value_distance_matrix, metric=metric)

    if matrix is None:
        matrix = {}

    neighbors = sklearn.neighbors.NearestNeighbors(radius=distance_cutoff, metric=metric).fit(values.reshape(values.shape[0], -1))
    dists, indices = neighbors.radius_neighbors(values2.reshape(values.shape[0], -1), radius=distance_cutoff)

    for i in range(len(dists)):
        for j, dist in zip(indices[i], dists[i]):
            #pair = (i,j) if i < j else (j,i)
            matrix[i,j] = dist

    return matrix


def sequences_to_array(barcodes):
    """ Helper function to turn an array of strings into
    a array of uint8 with an extra dimension of the characters
    of the string
    """
    barcodes = np.asarray(barcodes)

    orig_shape = barcodes.shape
    #barcodes = barcodes.reshape(-1).astype(bytes, copy=True)
    #barcodes = barcodes.reshape(-1).astype(bytes, copy=True)
    max_len = max(map(len, barcodes.flat))

    #values = np.frombuffer(barcodes, dtype=np.uint8).reshape(barcodes.shape[0], barcodes.dtype.itemsize)
    values = np.frombuffer(barcodes, dtype=np.uint8).reshape(barcodes.shape[0], barcodes.dtype.itemsize)
    return values.reshape(orig_shape + (-1,))

def sequences_to_vector(sequences, channels=('G', 'T', 'A', 'C')):
    arr = np.frombuffer(sequences, 'U1').reshape(len(sequences), -1)
    vec = np.stack([(arr == let) / 2 for let in channels], axis=-1)
    return vec

def sequence_distance(seq1, seq2):
    return sum(let1 != let2 for let1, let2 in zip(seq1, seq2)) + abs(len(seq1) - len(seq2))


def sequence_distance_matrix(
            sequences, sequences2=None,
            distance_cutoff=50,
            matrix=None, lazy=False,
            debug=True, progress=True):

    if sequences2 is None:
        sequences2 = sequences

    debug, progress = utils.log_env(debug, progress)

    if lazy and matrix is None:
        return LazyDistanceMatrix(sequences, sequences, distance_cutoff, sequence_distance, sequence_distance_matrix)

    if matrix is None:
        matrix = {}

    if distance_cutoff < 1 and False:
        # only looking for exact matches, can be much more efficient
        indices = np.argsort(sequences)
        indices2 = np.argsort(sequences2)

        index, base_index2, index2 = 0, 0, 0
        while index < len(indices):
            seq1, seq2 = sequences[indices[index]], 
            #if sequences[indices[index]]

    vecs = sequences_to_vector(sequences).reshape(len(sequences), -1)
    vecs2 = sequences_to_vector(sequences2).reshape(len(sequences), -1)
    neighbors = sklearn.neighbors.NearestNeighbors(radius=distance_cutoff, metric='cityblock').fit(vecs)
    dists, indices = neighbors.radius_neighbors(vecs2, radius=distance_cutoff)

    for i in range(len(dists)):
        for j, dist in zip(indices[i], dists[i]):
            #pair = (i,j) if i < j else (j,i)
            matrix[i,j] = dist

    return matrix


def distance_matrix(
            table, table2=None,
            cells=None,
            distance_cutoff=50,
            positional_weight=0.0,
            value_weight=0.0,
            sequence_weight=0.0,
            value_distance_metric='euclidean',
            matrix=None,
            debug=True, progress=True):
    """ Calculates a sparse distance matrix between two sets of reads.
    Can be used for multiple applications, such as clustering reads found in
    cells, matching reads to a barcode library, or clustering reads outside
    of cell boundaries.

    The distance between two reads has three components, calculated from the tree components of a read.
    Each is weighted using the parameters passed in:
        positional_weight: the euclidean distance between the two reads
        value_weight: the euclidean distance between the vector of raw read values for the two reads
        sequence_weight: the edit distance between the two read sequences

    These components are combined with their respective weights to calculate a final distance.
    If this distance is less than or equal to distance_cutoff, it is added to the matrix.
    """

    if positional_weight <= 0 and value_weight <= 0 and sequence_weight <= 0:
        raise ValueError('One of positional_weight, value_weight, sequence_weight must be nozero to calculate distances')

    if table2 is None:
        table2 = table

    debug, progress = utils.log_env(debug, progress)

    cur_matrix = None

    if table.reads.has_position and table2.reads.has_position and positional_weight > 0:
        pos_dists = positional_distance_matrix(
                        table.reads.positions, table2.reads.positions,
                        distance_cutoff=distance_cutoff / positional_weight,
                        cells=cells,
                        debug=debug, progress=progress)

        cur_matrix = {}
        for pair, dist in pos_dists.items():
            cur_matrix[pair] = dist * positional_weight

    if table.reads.has_values and table2.reads.has_values and value_weight > 0:
        value_dists = value_distance_matrix(
                        table.reads.values, table2.reads.values,
                        distance_cutoff=distance_cutoff / value_weight,
                        metric=value_distance_metric,
                        lazy=cur_matrix is not None,
                        debug=debug, progress=progress)

        if cur_matrix is None:
            cur_matrix = {}
            for pair, dist in value_dists.items():
                cur_matrix[pair] = dist * value_weight

        else:
            new_matrix = {}
            for pair, curdist in cur_matrix.items():
                if pair not in value_dists:
                    continue
                newdist = curdist + value_dists[pair] * value_weight
                if newdist > distance_cutoff:
                    continue
                new_matrix[pair] = newdist
            cur_matrix = new_matrix

    if sequence_weight > 0:
        sequence_dists = sequence_distance_matrix(
                        table.reads.sequences, table2.reads.sequences,
                        distance_cutoff=distance_cutoff / sequence_weight,
                        lazy=cur_matrix is not None,
                        debug=debug, progress=progress)

        if cur_matrix is None:
            cur_matrix = {}
            for pair, dist in sequence_dists.items():
                cur_matrix[pair] = dist * sequence_weight

        else:
            new_matrix = {}
            for pair, curdist in cur_matrix.items():
                if pair not in sequence_dists:
                    continue
                newdist = curdist + sequence_dists[pair] * sequence_weight
                if newdist > distance_cutoff:
                    continue
                new_matrix[pair] = newdist
            cur_matrix = new_matrix

    if matrix is None:
        return cur_matrix

    for pair, dist in cur_matrix:
        matrix[pair] = dist


def distance_matrix_old(
            table,
            cells=None,
            distance_cutoff=50,
            positional_weight=1.0,
            value_weight=1.0,
            sequence_weight=1.0,
            matrix=None,
            debug=True,
            progress=True):

    debug, progress = utils.log_env(debug, progress)

    if matrix is None:
        matrix = {}

    debug ("Finding neighbors")
    neighbors = sklearn.neighbors.NearestNeighbors(radius=distance_cutoff)
    neighbors = neighbors.fit(table.reads.positions)

    cell_matrix = {}

    debug ("Calculating cell distances")

    if cells is not None:
        props = skimage.measure.regionprops(cells)

        cell_poses = np.array([prop.centroid for prop in props])
        bboxes = np.array([prop.bbox for prop in props])
        max_size = np.max(bboxes[:,2:] - bboxes[:,:2])
        dists, indices = neighbors.radius_neighbors(cell_poses, radius=distance_cutoff + max_size)

        for i, bbox in enumerate(progress(bboxes)):
            cell = props[i].label
            x1 = max(0, int(bbox[0]) - distance_cutoff)
            y1 = max(0, int(bbox[1]) - distance_cutoff)
            x2 = int(bbox[2]) + distance_cutoff
            y2 = int(bbox[3]) + distance_cutoff
            section = cells[x1:x2,y1:y2] == cell

            cell_dists = scipy.ndimage.distance_transform_edt(~section)
            cell_dists[section] = 0

            for j in indices[i]:
                x, y = table.reads.positions[j]
                if x >= x1 and x < x2 and y >= y1 and y < y2:
                    dist = cell_dists[int(x-x1),int(y-y1)]
                    cell_matrix.setdefault(j, {})[cell] = dist

    #fig, axis = plt.subplots()

    debug("Calculating dot distances")
    dists, indices = neighbors.radius_neighbors(table.reads.positions)

    #full_matrix = {}
    #ofile.write('i,j,distance\n')

    #for i in progress(range(len(table.reads.positions))):
        #for j in range(i+1, len(table.reads.positions)):
            #direct_dist = np.linalg.norm(table.reads.positions[i] - table.reads.positions[j])
            #if direct_dist > distance_cutoff:
                #continue
    for i, cur_dists, cur_indices in zip(progress(range(len(dists))), dists, indices):
        for pos_dist, j in zip(cur_dists, cur_indices):
            if i >= j:
                continue

            direct_dist = pos_dist

            #begin = time.time()
            read1, read2 = table.reads[i], table.reads[j]
            seq1, seq2 = read1.sequence_array, read2.sequence_array
            #end = time.time()
            #print (end - begin)

            #times = []
            #times.append(time.time())
            seq_dist = np.sum(seq1 != seq2)
            #times.append(time.time())

            #value_dist = np.linalg.norm(table.reads.values[i]) * np.linalg.norm(table.reads.values[j]) - np.sum(table.reads.values[i] * table.reads.values[j])
            lengths = np.linalg.norm(read1.values, axis=1) * np.linalg.norm(read2.values, axis=1)
            #times.append(time.time())
            prod = np.sum(read1.values * read2.values, axis=1)
            #times.append(time.time())
            value_dist = lengths - prod
            #times.append(time.time())
            value_dist = value_dist * (seq1 != seq2)
            #times.append(time.time())
            #value_dist = value_dist * seq_dist
            #print (value_dist)
            #value_dist = np.sum(value_dist * value_dist)
            value_dist = np.sum(value_dist)
            #times.append(time.time())

            #debug ([end - start for start, end in zip(times, times[1:])])


            #debug ('value dist', value_dist, np.linalg.norm(values[i]), np.linalg.norm(values[j]))
            #debug ('   ', ''.join(np.array(list('GTAC'))[values[i].reshape(-1,4).argmax(axis=1)]))
            #debug ('   ', ''.join(np.array(list('GTAC'))[values[j].reshape(-1,4).argmax(axis=1)]))
            #debug ('   ', values[i], values[j])
            #debug ('direct dist', direct_dist)

            min_cell_dist = direct_dist
            #for cell in cells_table.index:
                #if (cell, i) not in cell_matrix or (cell, j) not in cell_matrix:
                    #continue
                #dist = cell_matrix[cell,i] + cell_matrix[cell,j]
            #if i not in cell_matrix:
                #debug ('not in cell matrix', i)
            #if j not in cell_matrix:
                #debug ('not in cell matrix', i)

            if i in cell_matrix and j in cell_matrix:
                possible_cells = set(cell_matrix[i].keys()) & set(cell_matrix[j].keys())
                #print (possible_cells)
                for cell in possible_cells:
                    dist = cell_matrix[i][cell] + cell_matrix[j][cell]
                    if dist < min_cell_dist:
                        min_cell_dist = dist
                        #cell_center = np.argwhere(cells == cell).mean(axis=0)
                        #axis.plot([table.reads.positions[i,0], cell_center[0], table.reads.positions[j,0]], [table.reads.positions[i,1], cell_center[1], table.reads.positions[j,1]], color='red')
                        #debug ('cell closer', cell, dist)

            #debug ('cell dist', min_cell_dist)
            dist = min_cell_dist * positional_weight + value_dist * value_weight + seq_dist * sequence_weight
            #dist = min_cell_dist * positional_weight / distance_cutoff + value_dist * value_weight / table.reads.n_cycles
            matrix[i,j] = dist

    #fig.savefig('tmp_dists.png')
    return matrix


class Heap:
    REMOVED = '<removed-task>' # placeholder for a removed task

    def __init__(self):
        self.pq = [] # list of entries arranged in a heap
        self.entry_finder = {} # mapping of tasks to entries
        self.counter = itertools.count() # unique sequence count

    def push(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def remove(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = Heap.REMOVED

    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not Heap.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        while self.pq and self.pq[0][2] == Heap.REMOVED:
            priority, count, task = heapq.heappop(self.pq)
        return len(self.pq) == 0


def cluster_reads(distance_matrix,
            threshold=0.2,
            linkage='mean',
            debug=True,
            num_reads=None,
            progress=False):

    if linkage == 'min':
        return _cluster_reads_linkage_min(distance_matrix, threshold, debug, progress)

    debug, progress = utils.log_env(debug, progress)
    cluster_dists = {}
    max_reads = 0

    print_matrix = False

    heap = Heap()
    clusters_added = set()

    for (i, j), distance in distance_matrix.items():
        if j < i: i, j = j, i
        if i == j or (i,j) in clusters_added:
            # remove duplicate edges in distance matrix
            continue
        cluster_dists.setdefault(i, {})[j] = distance
        cluster_dists.setdefault(j, {})[i] = distance
        max_reads = max(max_reads, i, j)
        heap.push((i,j), distance)
        clusters_added.add((i,j))

    del clusters_added

    #cluster_dists = distance_matrix.copy()
    #num_reads = max(max(pair) for pair in distance_matrix) + 1
    num_reads = num_reads or max_reads + 1

    clusters = np.arange(num_reads).reshape(-1,1).tolist()
    #debug (len(clusters))
    cluster_indices = np.arange(num_reads)

    #for pair, dist in distance_matrix.items():
        #clusters[pair[0]].extend(clusters[pair[1]])
        #clusters[pair[1]] = []

    def merge_dists_mean(cluster1, cluster2, dists1, dists2):
        weight1, weight2 = len(clusters[cluster1]), len(clusters[cluster2])
        weight1, weight2 = weight1 / (weight1 + weight2), weight2 / (weight1 + weight2)

        new_dists = {}
        for i, dist1 in dists1.items():
            if i in dists2:
                new_dists[i] = dist1 * weight1 + dists2[i] * weight2

                """
                dist_set = []
                for j in list(clusters[cluster1]) + list(clusters[cluster2]):
                    for k in clusters[i]:
                        pair = (j,k) if j < k else (k,j)
                        dist_set.append(distance_matrix[pair])
                debug (dist_set)
                debug (dist1, dists2[i], weight1, weight2)
                dist = sum(dist_set) / len(dist_set)

                debug (dist, new_dists[i])
                assert abs(dist - new_dists[i]) < 0.0001
                """

        return new_dists

    def merge_dists_max(cluster1, cluster2, dists1, dists2):
        return {pair: max(dists1[pair], dists2[pair]) for pair in set(dists1.keys()) & set(dists2.keys())}

    def merge_dists_min(cluster1, cluster2, dists1, dists2):
        return {pair: min(dists1[pair], dists2[pair]) for pair in set(dists1.keys()) & set(dists2.keys())}

    merge_dists_func = merge_dists_mean
    if linkage == 'min':
        merge_dists_func = merge_dists_min
    if linkage == 'max':
        merge_dists_func = merge_dists_max

    #next_pairs = sorted(distance_matrix.items(), key=lambda kv: kv[1])
    #next_pairs = [pair for pair, dist in next_pairs]

    #while len(next_pairs):
        #pairs = next_pairs
        #next_pairs = []
        #for pair in pairs:
    #for pair, dist in progress(sorted(distance_matrix.items(), key=lambda kv: kv[1])):
            #print (cluster_indices, pair)
    """
    while True:
            min_pair = None
            min_dist = threshold
            for cluster1 in cluster_dists.keys():
                for cluster2, dist in cluster_dists[cluster1].items():
                    if dist < min_dist:
                        min_dist = dist
                        min_pair = cluster1, cluster2

            if min_pair is None:
                break

            pair = min_pair
            if pair[0] > pair[1]:
                pair = pair[1], pair[0]

            #min_pair = min(((i, *min(cluster_dists[i].items(), key=lambda kv: kv[1])) for i in cluster_dists.keys()), key=lambda x: x[2])

            #if min_pair[2] > threshold:
                #break

            #pair = min_pair[:2]

            """
    while not heap.empty():
            """
            orig_pair = pair
            pair = cluster_indices[pair[0]], cluster_indices[pair[1]]

            assert len(clusters[pair[0]]) != 0 and len(clusters[pair[1]]) != 0

            if pair[1] not in cluster_dists[pair[0]]:
                next_pairs.append(orig_pair)
                continue

            cluster_dist = cluster_dists[pair[0]][pair[1]]
            if cluster_dist > threshold:
                continue
            #"""

            if print_matrix:
                for i in range(num_reads):
                    items = []
                    for j in range(num_reads):
                        if (i,j) not in heap.entry_finder:
                            items.append(' . ')
                        else:
                            heap_dist = heap.entry_finder[i,j][0]
                            dist1 = cluster_dists[i][j]
                            dist2 = cluster_dists[j][i]
                            assert heap_dist == dist1 == dist2
                            items.append('{:.1f}'.format(heap_dist))

                    line = ' '.join('{:3}'.format(item) for item in items)
                    debug (line)

            pair = heap.pop()
            #print (cluster_indices, pair)

            cluster_dist = cluster_dists[pair[0]][pair[1]]
            if cluster_dist > threshold:
                # this is the min pair, so no more pairs will be less
                # than the thresh
                break

            #debug ('Merging ', pair, cluster_dists[pair[0]][pair[1]])

            for other in cluster_dists[pair[0]].keys():
                del cluster_dists[other][pair[0]]
                if other != pair[1]:
                    cur_pair = (pair[0], other) if other > pair[0] else (other, pair[0])
                    #print ('removing', cur_pair)
                    heap.remove(cur_pair)
            for other in cluster_dists[pair[1]].keys():
                del cluster_dists[other][pair[1]]
                if other != pair[0]:
                    cur_pair = (pair[1], other) if other > pair[1] else (other, pair[1])
                    #print ('removing', cur_pair)
                    heap.remove(cur_pair)

            new_dists = merge_dists_func(pair[0], pair[1], cluster_dists.pop(pair[0]), cluster_dists.pop(pair[1]))

            cluster_dists[pair[0]] = new_dists
            for other, dist in new_dists.items():
                cluster_dists[other][pair[0]] = dist
                cur_pair = (pair[0], other) if other > pair[0] else (other, pair[0])
                heap.push(cur_pair, dist)

            # updating clusters, all reads in cluster pair[1] join pair[0]

            cluster_indices[clusters[pair[1]]] = pair[0]

            clusters[pair[0]].extend(clusters[pair[1]])
            clusters[pair[1]] = []



    """
    min_pair = min(cluster_dists.keys(), key=lambda pair: cluster_dists[pair])
    while cluster_dists[min_pair] < threshold:
        debug ('merging', min_pair)
        debug (len(clusters))

        clusters[min_pair[0]].extend(clusters[min_pair[1]])
        clusters[min_pair[1]] = []
        cluster_indices.remove(min_pair[1])

        dists1, dists2 = cluster_dists.pop(min_pair[0]), cluster_dists.pop(min_pair[1])


        #for other in range(len(clusters)):
        for other in cluster_indices:
            #if other == min_pair[0] or len(clusters[other]) == 0: continue

            pair = (other, min_pair[0]) if other < min_pair[0] else (min_pair[0], other)
            prev_pair = (other, min_pair[1]) if other < min_pair[1] else (min_pair[1], other)

            if prev_pair in cluster_dists:
                del cluster_dists[prev_pair]
            else:
                continue

            if pair in cluster_dists:
                del cluster_dists[pair]
            else:
                continue

            if pair[0] == pair[1]: continue

            dist_set = []
            for i in clusters[pair[0]]:
                for j in clusters[pair[1]]:
                    dist_set.append(distance_matrix.get((i,j), None))

            if any(dist is None for dist in dist_set):
                continue

            dist = sum(dist_set) / len(dist_set)
            cluster_dists[pair] = dist

        min_pair = min(cluster_dists.keys(), key=lambda pair: cluster_dists[pair])
    """

    cluster_indices = np.zeros(num_reads, dtype=int)
    for i,cluster in enumerate(filter(lambda cluster: len(cluster), clusters)):
        cluster_indices[cluster] = i

    return cluster_indices


def _cluster_reads_linkage_min(distance_matrix,
            threshold=0.2,
            debug=True,
            progress=False):
    """ Cluster reads using the min linkage. This method is separate
    because the min linkage allows for a much more efficient algorithm
    """
    debug, progress = utils.log_env(debug, progress)

    num_reads = max(max(pair) for pair in distance_matrix) + 1

    clusters = np.arange(num_reads)

    for pair, dist in progress(distance_matrix.items()):
        if dist <= threshold:
            clusters[pair[1]] = clusters[pair[0]]

    mapping = {}
    for i in range(num_reads):
        clusters[i] = mapping.setdefault(clusters[i], len(mapping))

    return clusters

