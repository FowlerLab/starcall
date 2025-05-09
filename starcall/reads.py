""" Classes that encapsulate sequencing reads detected in
images, with attributes and functions to manipulate and
process them.
"""

import re
import time
import itertools
import heapq
import numpy as np
import csv
import pandas
import scipy.ndimage
import matplotlib.pyplot as plt
import sklearn.neighbors
import skimage.measure

from . import utils

class Read:
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

    Additional attributes can be added when creating the Read as keyword arguments, and can be accessed as
    normal attributes on the Read.
    """

    DEFAULT_CHANNELS = np.array(['G', 'T', 'A', 'C'], dtype='U1')

    def __init__(self, position=None, values=None, sequence=None, image=None, channels=None, **kwargs):
        """ 
        Read(position, values, [sequence, 
        """
        if isinstance(position, str):
            sequence = position
            position = None

        if position is not None:
            position = np.asarray(position)
            if values is None and image is None and position.shape != (2,):
                values = position
                position = None

        if values is None and (position is None or image is None) and sequence is None:
            raise ValueError("Either values, sequence, or both position and image must be specified")

        if values is None and image is not None and position is not None:
            values = image[:,:,position[0],position[1]]

        channels = np.asarray(channels, dtype='U1') if channels is not None else self.DEFAULT_CHANNELS

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
        self.n_channels = len(channels)
        self._attrs = {name: np.asarray(val).reshape(-1) for name, val in kwargs.items()}

    def __repr__(self):
        parts = []

        if self.position is not None:
            parts.append('position={}'.format(self.position))
        if self.values is not None:
            parts.append('values=[...]')
        parts.append(self.sequence)

        for name, val in self._attrs.items():
            parts.append('{}={}'.format(name, val[0]))

        return 'Read({})'.format(', '.join(parts))

    @staticmethod
    def asread(obj):
        if isinstance(obj, Read):
            return obj
        return Read(obj)

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

    @property
    def attrs(self):
        return ReadAttrs(self._attrs)

    """
    def __getattr__(self, name):
        if name in self._attrs:
            return self._attrs[name]
        raise AttributeError("type object '{}' has no attribute '{}'".format(type(self).__name__, name))
    """

class ReadAttrs:
    def __init__(self, attrs):
        self.attrs = attrs

    def __getitem__(self, index):
        return self.attrs[index][0]

    def __setitem__(self, index, value):
        self.attrs[index][0] = value

    def __iter__(self):
        return iter(self.attrs)

    def keys(self):
        return self.attrs.keys()

    def values(self):
        for val in self.attrs.values():
            yield val[0]

    def items(self):
        for key, val in self.attrs.items():
            yield key, val[0]

    def __len__(self):
        return len(self.attrs)



class ReadSet:
    """ A collection of reads, all with the same number of cycles and
    channels. This can be constructed from the raw values, an image and positions,
    a list of sequences, or a combination of any. A common way to generate a read set
    is with the starcall.dotdetection.find_dots function, which detects and extracts reads
    from an image.

    ReadSet instances have similar attributes to individual Read objects, and similar to Read
    objects some can be None. When Reads are collected into a ReadSet, they are all required to
    have the same attributes.

        positions (ndarray of shape (n_reads, 2)): The positions of every read
        values (ndarray of shape (n_reads, n_cycles, n_channels): the raw values from the sequencing images for all reads
        sequences (ndarray of strings of shape (n_reads,)): The sequences of each read
        ** any extra attributes that were specified on creation

    Examples:
    
    Creating a ReadSet with just sequences:
        readset = ReadSet(['GTAC', 'GTGT', 'GGTT', 'GTAG', 'TTGT', 'ATAA'])
        readset = ReadSet(sequences=['GTAC', 'GTGT', 'GGTT', 'GTAG', 'TTGT', 'ATAA'])
        readset[0] # -> Read('GTAC')
        readset.sequences # -> np.array(['GTAC', 'GTGT', 'GGTT', 'GTAG', 'TTGT', 'ATAA'])

    Creating a ReadSet from a sequence of Reads:
        readset = ReadSet([Read('GTTT'), Read('TGTT'), Read('AAAT')])
        readset = ReadSet(reads=[Read('GTTT'), Read('TGTT'), Read('AAAT')])
        readset[0] # -> Read('GTAC')
        readset.sequences # -> np.array(['GTTT', 'TGTT', 'AAAT'])

    Creating a ReadSet with positions and values:
        readset = ReadSet(
            [(0,0), (0,5), (10, 14)],
            [
                [(1,0,0,0), (0,1,0,2), (0,3,2,1), (1,2,1,1)],
                [(1,1,0,2), (3,2,2,1), (1,0,1,2), (3,3,1,8)],
                [(1,1,0,2), (0,1,0,2), (4,0,0,2), (1,2,1,5)],
            ]
        )
        readset[0].values # -> np.array([[1,0,0,0], [0,1,0,2], [0,3,2,1], [1,2,1,1]])
        readset.sequences # -> 

    Creating a ReadSet with extra parameters:
        readset = ReadSet(sequences=['GTTT', 'AGAG', 'TGTT'], cell=[0, 1, 0], count=[1, 4, 2])
        readset.cell # -> np.array([0, 1, 0])
        readset[0].cell # -> 0

    """

    def __init__(self, positions=None, values=None, sequences=None, image=None, channels=None, reads=None, **attrs):
        positions = np.asarray(positions) if positions is not None else None

        if positions is not None and isinstance(positions[0], Read):
            # a sequence of reads is passed in
            reads = list(map(Read.asread, positions))
            positions = None
            self.n_reads = len(reads)

            if reads[0].position is not None:
                positions = np.array([read.position for read in reads])
                for i in range(self.n_reads):
                    reads[i].position = positions[i]

            if reads[0].values is not None:
                values = np.array([read.values for read in reads])
                for i in range(self.n_reads):
                    reads[i].values = values[i]

            if reads[0]._sequence is not None:
                sequences = np.array([read._sequence[0] for read in reads])
                for i in range(self.n_reads):
                    reads[i]._sequence = sequences[i:i+1]

            for name in reads[0].attrs:
                attrs[name] = np.array([read._attrs[name][0] for read in reads])
                for i in range(self.n_reads):
                    reads[i].attrs[name] = attrs[name][i:i+1]

            channels = reads[0].channels

        if positions is not None and sequences is None and positions.dtype.kind == 'U':
            sequences = positions
            positions = None

        if positions is not None and values is None and image is None and len(positions.shape) == 3:
            values = positions
            positions = None

        if values is None and (positions is None or image is None) and sequences is None:
            raise ValueError("Either values, sequences, or both positions and image must be specified")

        if values is None and image is not None and positions is not None:
            values = image[:,:,positions[:,0],positions[:,1]]
            values = values.transpose(2,0,1)

        channels = np.asarray(channels, dtype='U1') if channels is not None else Read.DEFAULT_CHANNELS

        if values is not None:
            values = np.asarray(values)

            if sequences is None and values.dtype.kind == 'U':
                sequences = values
                values = None
            else:
                if len(values.shape) != 3 or values.shape[2] != len(channels):
                    raise ValueError("Expected an array of shape (n_reads, n_cycles, n_channels) for values")
                self.n_reads = values.shape[0]
                self.n_cycles = values.shape[1]

        if sequences is not None:
            sequences = np.asarray(sequences, dtype='U')
            self.n_reads = sequences.shape[0]
            self.n_cycles = int(str(sequences.dtype).split('U')[-1])

        for name in attrs:
            attrs[name] = np.asarray(attrs[name])

        if reads is None:
            reads = []
            for i in range(self.n_reads):
                reads.append(Read(
                    position = positions[i] if positions is not None else None,
                    values = values[i] if values is not None else None,
                    sequence = sequences[i:i+1] if sequences is not None else None,
                    channels = channels,
                    **{name: vals[i:i+1] for name, vals in attrs.items()}
                ))

        self.reads = reads
        self.positions = positions
        self.values = values
        self._sequences = sequences
        self.channels = channels
        self.n_channels = len(channels)
        self.attrs = attrs

    def __repr__(self):
        if len(self) > 8:
            items = self.reads[:4] + ['\n\t...'] + self.reads[-4:]
        else:
            items = self.reads
        return 'ReadSet([{}])'.format(', '.join(str(read) for read in items))

    @staticmethod
    def asreads(obj):
        if isinstance(obj, ReadSet):
            return obj
        return ReadSet(obj)

    @property
    def sequences(self):
        if self._sequences is not None:
            return self._sequences
        seq = self.channels[np.argmax(self.values, axis=2)]
        seq = np.frombuffer(seq, 'U' + str(seq.shape[1]))
        seq.setflags(write=False)
        return seq

    @sequences.setter
    def sequences(self, value):
        if self._sequences is None:
            self._sequences = np.empty(self.n_reads, 'U' + str(self.n_cycles))
        self._sequences[...] = value

    @property
    def sequences_array(self):
        if self._sequences is not None:
            seq = np.frombuffer(self._sequences, 'U1').reshape(self.n_reads, self.n_cycles)
            return seq

        seq = self.channels[np.argmax(self.values, axis=2)]
        seq.setflags(write=False)
        return seq

    @sequences_array.setter
    def sequences_array(self, value):
        if self._sequences is None:
            self._sequences = np.empty(self.n_reads, 'U' + str(self.n_cycles))

        seq = np.frombuffer(self._sequences, 'U1').reshape(self.n_reads, self.n_cycles)
        seq[...] = value

    @property
    def qualities(self):
        return self.values.max(axis=2)

    """
    def __getattr__(self, name):
        if name in self.attrs:
            return self.attrs[name]
        raise AttributeError("type object '{}' has no attribute '{}'".format(type(self).__name__, name))
    """

    def __getitem__(self, index):
        if type(index) == slice:
            return ReadSet(
                positions = self.positions[index] if self.positions is not None else None,
                values = self.values[index] if self.values is not None else None,
                sequences = self._sequences[index] if self._sequences is not None else None,
                channels = self.channels,
                reads = self.reads[index],
                **{name: vals[index] for name, vals in self.attrs.items()}
            )
        #return self.reads[index]
        return Read(
            position = self.positions[index] if self.positions is not None else None,
            values = self.values[index] if self.values is not None else None,
            sequence = self._sequences[index:index+1] if self._sequences is not None else None,
            channels = self.channels,
            **{name: vals[index:index+1] for name, vals in self.attrs.items()}
        )

    def __iter__(self):
        for i in range(self.n_reads):
            yield self[i]

    #def __contains__(self, value):
        #return value in self.reads

    def __len__(self):
        return self.n_reads

    def copy(self, copy_arrays=True):
        positions, values, sequences = self.positions, self.values, self._sequences

        if copy_arrays and positions is not None:
            positions = positions.copy()
        if copy_arrays and values is not None:
            values = values.copy()
        if copy_arrays and sequences is not None:
            sequences = sequences.copy()

        return ReadSet(positions=positions, values=values, sequences=sequences, channels=channels)

    def normalize(self, method='full'):
        """ Normalizes the values of this read set, based on the method specified
        Possible methods are:
            'full' (default): values are normalized across the channel axis, so that for each
            cycle the norm of the vector of all channels is 1
        """
        values = self.values

        if method == 'large':
            values /= np.maximum(1, np.linalg.norm(values, axis=2)[:,:,None])
        if method == 'full':
            values /= np.maximum(0.0000000001, np.linalg.norm(values, axis=2)[:,:,None])
        if method == 'sub':
            sorted_values = np.sort(values, axis=2)
            values -= sorted_values[:,:,-2:-1]

        self.values = values

    def groupby(self, colname, sort_key=None):
        groups = {}
        for label, read in zip(self.attrs[colname], self):
            groups.setdefault(label, []).append(read)
        
        if sort_key:
            for label in groups.keys():
                groups[label] = ReadSet(sorted(groups[label], key=sort_key))
        else:
            for label in groups.keys():
                groups[label] = ReadSet(groups[label])

        return ReadSetGroups(list(groups.values()), grouped_by=colname)

    def combine(self, method=None):
        """ Combines all reads in this set into one consensus read. The different attributes
        of reads are combined in different ways:
            positions are averaged
            values are summed
            sequences, the mode is selected
            ** any custom attributes are combined as specified
            by the parameter method.

        Args:
            method (dict): Methods to use to aggregate custom attributes on reads
                Each entry in the dict specifies the aggregation method for the attribute,
                from the list, out of: 'mean', 'min', 'max', 'sum', 'mode'. A callable
                can also be passed which will take a numpy array and return a scalar.
        Returns:
            Read instance, the consensus read of this set.
        """

        position, values, sequence = None, None, None

        def mode(arr):
            vals, counts = np.unique(arr, return_counts=True)
            return vals[np.argmax(counts)]

        if self.positions is not None:
            position = self.positions.mean(axis=0)
        if self.values is not None:
            values = self.values.sum(axis=0)
        if self._sequences is not None:
            sequence = mode(self._sequences)

        def get_method(method):
            if method is None:
                method = default_method

            if callable(method):
                return method

            if method == 'mode':
                return mode
            if hasattr(np, method):
                return getattr(np, method)

            raise ValueError('Unrecognized aggregation method {}'.format(method))

        attrs = {}
        for name, attr_vals in self.attrs.items():
            default_method = 'mean' if np.issubdtype(attr_vals.dtype, np.floating) else 'mode'
            cur_method = method.get(name, default_method) if isinstance(method, dict) else method
            cur_method = get_method(cur_method)
            attrs[name] = cur_method(attr_vals)

        read = Read(position=position, values=values, sequence=sequence, channels=self.channels, **attrs)
        return read

    def to_table(self, sequences=None, qualities=False):
        """ Converts this collection of reads into a pandas DataFrame. This can be converted
        back into a ReadSet using ReadSet.from_table

            sequences (bool, default (self._sequences is None)): Whether to include sequences in the table.
                Defaults to whether sequences have been specified for this read set
            qualities (bool, default False): Whether to include quality scores for each cycle

        Returns:
            a pandas DataFrame with the following columns:
            if self.positions is not none:
            xpos, ypos: the positions of the reads
            if self.sequences is not none:
            read: the sequences of the reads
            if self.values is not None:
            value_cycle??_ch?? for all cycles and channels, containing the values of the reads
            if qualities is True:
            quality: the average quality across cycles
            quality_??: the quality score for each cycle
            chan?? for all channels, containing the names of the channels in the whole readset
        """
        table = {}
        sequences = (self._sequences is not None) if sequences is None else sequences

        if self.positions is not None:
            table['xpos'] = self.positions[:,0]
            table['ypos'] = self.positions[:,1]

        if sequences:
            table['read'] = self.sequences

        if qualities:
            qualities = self.qualities
            table['quality'] = qualities.mean(axis=1)
            for i in range(self.n_cycles):
                table['quality_{:02}'.format(i)] = qualities[:,i]

        if self.values is not None:
            for cycle in range(self.n_cycles):
                for chan in range(self.n_channels):
                    table['value_cycle{:02}_ch{:02}'.format(cycle,chan)] = self.values[:,cycle,chan]

        if not np.all(self.channels == Read.DEFAULT_CHANNELS):
            for i, chan in enumerate(self.channels):
                table['chan{:02}'.format(i)] = chan

        table.update(self.attrs)

        table = pandas.DataFrame(table)
        return table

    @staticmethod
    def from_table(table):
        """ Constructs a ReadSet from a pandas DataFrame, that was generated from ReadSet.to_table
        """

        positions, sequences = None, None

        channels = Read.DEFAULT_CHANNELS
        if 'chan00' in table.columns:
            channels = []
            i = 0
            while 'chan{:02}'.format(i) in table.columns:
                channels.append(table['chan{:02}'.format(i)].iloc[0])
                i += 1

        if 'xpos' in table.columns:
            positions = np.array([table['xpos'].to_numpy(), table['ypos'].to_numpy()]).T

        if 'read' in table.columns:
            sequences = table['read'].to_numpy()

        values = []
        i = 0
        while 'value_cycle{:02}_ch00'.format(i) in table.columns:
            values.append([table['value_cycle{:02}_ch{:02}'.format(i,j)] for j in range(len(channels))])
            i += 1

        if len(values) != 0:
            values = np.array(values).transpose(2,0,1)
        else:
            values = None

        attrs = {}
        for col in table.columns:
            if (col not in ('xpos', 'ypos', 'read', 'quality')
                    and not re.match('value_cycle\d\d_ch\d\d$', col)
                    and not re.match('chan\d\d$', col)
                    and not re.match('quality_\d+$', col)):
                attrs[col] = table[col].to_numpy()

        reads = ReadSet(positions=positions, values=values, sequences=sequences, channels=channels, **attrs)
        return reads

class ReadSetGroups:
    """ A sequence of ReadSet instances, each with a variable number of reads.
    Typically used to hold the set of reads in each cell, and is returned by
    ReadSet.groupby().
    Can be constructed directly with a sequence of ReadSet instances.
    """
    def __init__(self, groups, grouped_by=None):
        self.max_reads = max(map(len, groups))
        self.n_groups = len(groups)

        self.groups = groups
        self.grouped_by = grouped_by

        self.channels = Read.DEFAULT_CHANNELS
        if len(self.groups) != 0:
            self.channels = self.groups[0].channels

    def __repr__(self):
        if len(self) > 8:
            items = self.groups[:4] + ['\t\t\t...'] + self.groups[-4:]
        else:
            items = self.groups
        return 'ReadSetGroups([{}])'.format(',\n\t'.join(str(group) for group in items))

    def __len__(self):
        return self.n_groups

    def __getitem__(self, index):
        return self.groups[index]

    def __iter__(self):
        return iter(self.groups)

    def combine(self, method=None):
        return ReadSet([group.combine(method) for group in self.groups])
        """
        method = method or {}
        params = {}

        if self.groups[0].positions is not None:
            params['positions'] = np.array([group.positions.mean(axis=0) for group in self.groups])

        if self.groups[0].values is not None:
            params['values'] = np.array([group.values.sum(axis=0) for group in self.groups])

        if self.groups[0]._sequences is not None:
            seqs = []
            for group in self.groups:
                group_seqs, counts = np.unique(group._sequences, return_counts=True)
                seqs.append(group_seqs[np.argmax(counts)])
            params['sequences'] = np.array(seqs)

        for name in self.groups[0].attrs.keys():
            dtype = self.groups[0].attrs[name].dtype
            default_method = 'mean' if np.issubdtype(dtype, np.floating) else 'mode'
            cur_method = method.get(name, default_method)

            if callable(cur_method):
                vals = [cur_method(group.attrs[name]) for group in self.groups]
            elif cur_method == 'mean':
                vals = [group.attrs[name].mean(axis=0) for group in self.groups]
            elif cur_method == 'min':
                vals = [group.attrs[name].min(axis=0) for group in self.groups]
            elif cur_method == 'max':
                vals = [group.attrs[name].max(axis=0) for group in self.groups]
            elif cur_method == 'sum':
                vals = [group.attrs[name].sum(axis=0) for group in self.groups]
            elif cur_method == 'mode':
                vals = []
                for group in self.groups:
                    group_vals, counts = np.unique(group.attrs[name], return_counts=True)
                    vals.append(group_vals[np.argmax(counts)])

            params[name] = np.array(vals)

        readset = ReadSet(**params)
        return readset
        """

    def head(self, max_reads):
        return ReadSetGroups([readset[:max_reads] for readset in self.groups])

    def tail(self, max_reads):
        return ReadSetGroups([readset[-max_reads:] for readset in self.groups])

    def nth(self, index):
        return ReadSet([readset[index] for readset in self.groups])

    def to_table(self, columns=None, drop_columns=None, **kwargs):
        """ Creates a table by concatenating the tables of the first, second, third ... read
        in each set, each with a suffix of _0, _1, _2 respectively.
        For example, given these groups: [('GTAC', 'GGGG'), ('GTAT',), ('GTGT', 'GGTT', 'TTTT')]
        The table generated would be:
        index, num_reads, read_0, read_1, read_2
        0,     2,         'GTAC', 'GGGG', ''
        1,     1,         'GTAT', '',     ''
        2,     3,         'GTGT', 'GGTT', 'TTTT'
        
        Args:
            columns (sequence of str, optional): if specified, only these columns are included
            drop_columns (sequence of str, optional): if specified, these columns are dropped
            same as ReadSet.to_table(), arguments are forwarded to each group.
        Returns:
            pandas.DataFrame, with columns returned from ReadSet.to_table().
            There is an additional 'num_reads' column which contains the size of each group
        """
        drop_columns = drop_columns or []

        full_table = []

        for group in self.groups:
            table = group.to_table(**kwargs)
            if 'chan00' in table.columns:
                table = table.drop(columns=['chan{:02}'.format(i) for i in range(len(group.channels))])
            if columns:
                table = table[columns]
            if drop_columns:
                table = table.drop(columns=drop_columns)

            row = [pandas.Series(dict(num_reads=len(table.index)))]

            if self.grouped_by and self.grouped_by in table.columns:
                row.insert(0, pandas.Series({self.grouped_by: table[self.grouped_by].iloc[0]}))
                table = table.drop(columns=[self.grouped_by])

            for i in range(len(table.index)):
                series = table.iloc[i,:]
                series.index = [col + '_{}'.format(i) for col in series.index]
                row.append(series)

            full_table.append(pandas.concat(row))

        full_table = pandas.DataFrame(full_table)

        if not np.all(self.channels == Read.DEFAULT_CHANNELS):
            for i, chan in enumerate(self.channels):
                full_table['chan{:02}'.format(i)] = chan

        return full_table

    @staticmethod
    def from_table(table):
        """ Converts a table generated by ReadSet.to_table
        """

        channels = None
        if 'chan00' in table.columns:
            channels = []
            i = 0
            while 'chan{:02}'.format(i) in table.columns:
                channels.append(table['chan{:02}'.format(i)].iloc[0])
                i += 1

        max_reads = round(table['num_reads'].max())

        common_cols = [col for col in table.columns if col != 'num_reads' and not re.match('.*_\d+$', col)]

        #num_reads_index = table.columns.get_loc('num_reads')
        #print (num_reads_index, table.columns[num_reads_index-1])
        #if num_reads_index >= 1 and not any(table.columns[num_reads_index-1][-len('_{}'.format(i))] == '_{}'.format(i) for i in range(max_reads)):
            #common_cols.append(table.columns[num_reads_index-1])

        groups = []
        for i in range(len(table.index)):
            row = table.iloc[i,:]
            num_reads = round(row['num_reads'])

            row_table = []
            for j in range(num_reads):
                suffix = '_{}'.format(j)
                cols = [col for col in table.columns if col[-len(suffix):] == suffix]
                subrow = row[cols]
                subrow.index = [name[:-len(suffix)] for name in cols]
                row_table.append(subrow)

            row_table = pandas.DataFrame(row_table)

            if channels is not None:
                for i in range(len(channels)):
                    row_table['chan{:02}'.format(i)] = channels[i]

            for col in common_cols:
                row_table[col] = row[col]

            readset = ReadSet.from_table(row_table)
            groups.append(readset)

        return ReadSetGroups(groups, grouped_by=None if len(common_cols) == 0 else common_cols[0])

class ReadSetAttrs(dict):
    def __setitem__(self, name, value):
        super().__setitem__(name, value)
        if hasattr(self, 'callback'):
            self.callback(name, value)

class ReadTable(pandas.DataFrame):
    @property
    def _constructor(self):
        return ReadTable




def read_table(positions=None, values=None, sequences=None, channels=None, **extra_cols):
    channels = channels or ('G', 'T', 'A', 'C')
    table = {}

    if positions is not None:
        table['position','x',''] = positions[:,0]
        table['position','y',''] = positions[:,1]

    if values is not None:
        for i in range(values.shape[1]):
            for j in range(values.shape[2]):
                table['values',i,channels[j]] = values[:,i,j]

    if sequences is not None:
        table['read','',''] = sequences

    for name, value in extra_cols.items():
        table[name,'',''] = value

    return pandas.DataFrame(table)





def distance_matrix(
            reads,
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
    neighbors = neighbors.fit(reads.positions)

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
                x, y = reads.positions[j]
                if x >= x1 and x < x2 and y >= y1 and y < y2:
                    dist = cell_dists[int(x-x1),int(y-y1)]
                    cell_matrix.setdefault(j, {})[cell] = dist

    #fig, axis = plt.subplots()

    debug("Calculating dot distances")
    dists, indices = neighbors.radius_neighbors(reads.positions)

    #full_matrix = {}
    #ofile.write('i,j,distance\n')

    #for i in progress(range(len(reads.positions))):
        #for j in range(i+1, len(reads.positions)):
            #direct_dist = np.linalg.norm(reads.positions[i] - reads.positions[j])
            #if direct_dist > distance_cutoff:
                #continue
    for i, cur_dists, cur_indices in zip(progress(range(len(dists))), dists, indices):
        for pos_dist, j in zip(cur_dists, cur_indices):
            if i >= j:
                continue

            direct_dist = pos_dist

            read1, read2 = reads[i], reads[j]
            seq1, seq2 = read1.sequence_array, read2.sequence_array

            #times = []
            #times.append(time.time())
            seq_dist = np.sum(seq1 != seq2)
            #times.append(time.time())

            #value_dist = np.linalg.norm(reads.values[i]) * np.linalg.norm(reads.values[j]) - np.sum(reads.values[i] * reads.values[j])
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
                        #axis.plot([reads.positions[i,0], cell_center[0], reads.positions[j,0]], [reads.positions[i,1], cell_center[1], reads.positions[j,1]], color='red')
                        #debug ('cell closer', cell, dist)

            #debug ('cell dist', min_cell_dist)
            dist = min_cell_dist * positional_weight + value_dist * value_weight + seq_dist * sequence_weight
            #dist = min_cell_dist * positional_weight / distance_cutoff + value_dist * value_weight / reads.n_cycles
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
            progress=False):

    if linkage == 'min':
        return _cluster_reads_linkage_min(distance_matrix, threshold, debug, progress)

    debug, progress = utils.log_env(debug, progress)
    cluster_dists = {}
    num_reads = 0

    print_matrix = False

    heap = Heap()

    for (i, j), distance in distance_matrix.items():
        cluster_dists.setdefault(i, {})[j] = distance
        cluster_dists.setdefault(j, {})[i] = distance
        num_reads = max(num_reads, i, j)
        heap.push((i,j), distance)

    #cluster_dists = distance_matrix.copy()
    #num_reads = max(max(pair) for pair in distance_matrix) + 1
    num_reads += 1

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
        clusters[i] = mapping.setdefault(clusters[i], len(clusters))

    return clusters

