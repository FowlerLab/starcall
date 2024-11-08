import math
import numpy as np
import sklearn.cluster
import dataclasses
from . import utils

import time

def euclidean_distance(poses1, poses2):
    poses1, poses2 = np.asarray(poses1), np.asarray(poses2)
    assert poses1.shape[1] == poses2.shape[1]
    poses2 = np.broadcast_to(poses2.reshape((1,) + poses2.shape), (poses1.shape[0],) + poses2.shape)
    poses1 = np.broadcast_to(poses1.reshape(poses1.shape[0], 1, poses1.shape[1]), poses2.shape)
    distances = poses1 - poses2
    del poses1, poses2
    distances = np.linalg.norm(distances, axis=2)
    return distances.T

class TreeNode:
    def __init__(self, points, indices):
        self.points = points
        self.indices = indices

class NearestNeighbors:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        if metric == 'euclidean':
            metric = euclidean_distance
        self.distance_function = metric

    def fit(self, X, y=None):
        points = X
        self.distance_matrix = self.distance_function(points, points)
        batch_size = 4

        self.points = points
        self.tree = []

        all_points, all_dists, all_indices = [self.points], [self.distance_matrix], [np.arange(len(self.points))]

        self.tree = []
        while max(map(len, all_points)) / batch_size > batch_size / 2:
            new_points, new_dists, new_indices = [], [], []
            new_layer = []

            for points, dists, indices in zip(all_points, all_dists, all_indices):
                clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=batch_size, linkage='average', metric='precomputed').fit_predict(dists)

                centers = []
                indices = []
                for cluster in range(clusters.max() + 1):
                    mask = clusters == cluster
                    indices.append(len(new_points))
                    new_points.append(points[mask])
                    new_dists.append(dists[mask,mask])
                    new_indices(indices[mask])
                    center = np.argmin(new_dists[-1].sum(axis=1))
                    centers.append(np.arange(len(points))[mask][center])

                new_layer.append(TreeNode(points[centers], np.array(indices)))

            self.tree.append(new_layer)

        last_layer = []
        for points, indices in zip(new_points, new_indices):
            last_layer.append(TreeNode(points, indices))
        self.tree.append(last_layer)

    def closest(read):
        index = 0
        for layer in self.tree:
            node = layer[index]
            dists = self.distance_function(node.points, read)
            index = node.indices[np.argmin(dists.flatten())]
        
        return index


class BarcodeLibrary:
    def __init__(self, barcodes, distance_function=None, batch_size=50):
        barcodes = np.asarray(barcodes)
        if barcodes.dtype.type is np.str_:
            barcodes = barcodes_to_vector(barcodes)
        if barcodes.ndim == 3:
            barcodes = barcodes.reshape(barcodes.shape[0], -1)
        self.barcodes = barcodes

        self.neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=2, metric='manhattan')
        self.neighbors = self.neighbors.fit(self.barcodes)

    def permutations(self, max_val, num):
        perms = np.arange(max_val).reshape(-1, 1)
        for i in range(num-1):
            new_vals = np.repeat(np.arange(max_val), len(perms)).reshape(-1, 1)
            perms = np.tile(perms, (max_val, 1))
            perms = np.concatenate([new_vals, perms], axis=1)
        return perms

        """
        indices = np.zeros((math.factorial(num), num), int)
        factor = 1
        for subnum in range(2, num + 1):
            subindices = indices[:factor,num-subnum+1:]
            for i in range(1, subnum):
                indices[i*factor:(i+1)*factor,num-subnum] = i
                indices[i*factor:(i+1)*factor,num-subnum+1:] = subindices + (subindices >= i)
            subindices += 1
            factor *= subnum
        return indices
        """

    def nearest(self, reads, counts=None, match_threshold=2, second_threshold=None):
        reads = barcodes_to_vector(reads)
        counts = counts if counts is not None else np.ones(reads.shape[:-1])
        ranges = np.stack([np.arange(len(reads)), np.arange(1, len(reads) + 1)], axis=-1)

        if reads.ndim == 3:
            combined_reads, combined_counts, ranges = [], [], []

            for read_set, count_set in zip(reads, counts):
                read_set, count_set = read_set[count_set!=0], count_set[count_set!=0]
                reads_needed = self.barcodes.shape[1] / read_set.shape[1]
                assert int(reads_needed) == reads_needed
                if len(read_set) < int(reads_needed):
                    ranges.append((len(combined_reads), len(combined_reads)))
                    continue

                indices = self.permutations(len(read_set), int(reads_needed))

                new_reads = read_set[indices]
                new_reads = new_reads.reshape(new_reads.shape[0], -1)
                new_counts = count_set[indices]

                new_reads, unique_indices = np.unique(new_reads, axis=0, return_index=True)
                new_counts = new_counts[unique_indices]
                
                ranges.append((len(combined_reads), len(combined_reads) + len(new_reads)))

                combined_reads.extend(new_reads.reshape(new_reads.shape[0], -1))
                combined_counts.extend(new_counts.min(axis=1))

            reads, counts, ranges = np.array(combined_reads), np.array(combined_counts), np.array(ranges)

        dists, indices = self.neighbors.kneighbors(reads)
        matches = np.full(len(ranges), -1)
        match_dists = np.full(len(ranges), -1)

        for i, (begin, end) in enumerate(ranges):
            if begin == end: continue

            dist_section, indices_section = dists[begin:end], indices[begin:end]
            dist_section[dist_section[:,0]==dist_section[:,1]] = np.inf

            min_index = np.argmin(dist_section[:,0])
            min_val = dist_section[min_index,0]
            dist_section[min_index,0] = np.inf

            if dist_section[:,0].min() == min_val: continue

            matches[i] = indices_section[min_index,0]
            match_dists[i] = min_val / 2
        
        return match_dists, matches


def calc_barcode_distances(reads, library, counts=None, max_edit_distance=None, read_mask=None, dtype=np.float32, reads_needed=None):
    
    if type(reads) == list and type(reads[0]) == list:
        lengths = list(set(map(len, reads)))
        if len(lengths) > 1:
            target_length = max(lengths)
            mask = [[True] * len(read) + [False] * (target_length - len(read)) for read in reads]
            read_mask = read_mask & np.array(mask) if read_mask is not None else np.array(mask)
            reads = [read + ['',] * (target_length - len(read)) for read in reads]
        reads = np.asarray(reads)

    library = np.asarray(library)
    #print (reads, library)

    if counts is not None:
        counts = np.asarray(counts)
        read_mask = read_mask & (counts != 0) if read_mask is not None else counts != 0

    if reads.dtype.type is np.str_:
        reads = pack_barcodes(reads)
    if library.dtype.type is np.str_:
        library = pack_barcodes(library)

    if library.ndim == 1:
        library = library.reshape(-1, 1)
        if reads.ndim == 1:
            reads = reads.reshape(-1, 1)
    elif reads.ndim == 1:
        reads = reads.reshape(1, -1)

    if counts is not None:
        counts = counts.reshape(reads.shape)
    
    reads_needed = reads_needed or library.shape[1]

    #reads = reads[counts!=0]
    #counts = counts[counts!=0]
    #if reads.size == 0: continue

    reads = np.broadcast_to(reads.reshape(reads.shape + (1,1)), reads.shape + library.shape)
    library = np.broadcast_to(library.reshape((1,1) + library.shape), reads.shape)

    distances = np.zeros(reads.shape, dtype=dtype)
    if max_edit_distance == 0:
        np.not_equal(reads, library, out=distances)
    else:
        edit_distance(reads, library, out=distances)

    if counts is not None:
        distances += 1 / (counts.reshape(1,1,*counts.shape) + 1) # ties are broken by read count

    if read_mask is not None:
        distances[~read_mask] = np.inf

    if reads_needed == 1:
        distances = np.min(distances, axis=(1,3))
    else:
        distances = np.min(distances, axis=3)
        print (distances.shape)
        distances.partition(reads_needed - 1, axis=1)
        distances = distances[:,reads_needed-1]

    return distances

def match_barcodes(reads, counts, library, max_edit_distance=0, return_distances=False, debug=True, progress=False):
    """ Matches each set of reads given with a library barcode or set of barcodes.
    If max_edit_distance is nonzero, it will also try to match barcodes that differ
    in less that or equal to that many bases.
    """
    
    debug, progress = utils.log_env(debug, progress)
    reads, counts, library = np.asarray(reads), np.asarray(counts), np.asarray(library)
    if reads.dtype.type is np.str_:
        reads = pack_barcodes(reads)
    if library.dtype.type is np.str_:
        library = pack_barcodes(library)
    if len(reads.shape) == 1:
        reads = reads.reshape(-1, 1)
    if len(library.shape) == 1:
        library = library.reshape(-1, 1)

    matched_indices = np.full(reads.shape[0], -1, dtype=int)
    all_distances = np.full(reads.shape[0], -1, dtype=int)
    reads_needed = library.shape[1]

    for i in progress(range(reads.shape[0])):
        #"""
        read_set = reads[i]
        count_set = counts[i]
        read_set = read_set[count_set!=0]
        count_set = count_set[count_set!=0]
        if read_set.size < reads_needed: continue

        read_set = np.broadcast_to(read_set.reshape(1,1,-1), library.shape + (len(read_set),))
        assert read_set.shape[:2] == library.shape
        library_set = np.broadcast_to(library.reshape(library.shape + (1,)), read_set.shape)
        distances = np.zeros(read_set.shape, dtype=np.float32)

        if max_edit_distance == 0:
            np.not_equal(read_set, library_set, out=distances)
        else:
            edit_distance(read_set, library_set, out=distances)

        distances += 1 / (count_set.reshape(1,1,-1) + 1) # ties are broken by read count
        #distances = distances.min(axis=2).max(axis=1)
        #np.min(distances, axis=2, out=distances[:,:,0])
        #np.max(distances[:,:,0], axis=1, out=distances[:,0,0])

        distances = np.min(distances, axis=2)
        distances.partition(reads_needed - 1, axis=1)
        distances = distances[:,reads_needed-1]

        #distances = distances[:,0,0]
        #"""
        #distances = calc_barcode_distances(reads[i], counts[i], library)
        library_index = np.argmin(distances)
        min_val = distances[library_index]
        distances[library_index] = np.inf
        next_min_val = distances.min()
        del distances
        if min_val == next_min_val or min_val >= max_edit_distance + 1:
            continue
        matched_indices[i] = library_index
        all_distances[i] = int(min_val)

    if return_distances:
        return matched_indices, all_distances
    return matched_indices

def match_barcodes_bad(reads, counts, library, max_edit_distance=0, debug=True, progress=False):
    """ Matches each set of reads given with a library barcode or set of barcodes.
    If max_edit_distance is nonzero, it will also try to match barcodes that differ
    in less that or equal to that many bases.
    """
    
    debug, progress = utils.log_env(debug, progress)

    reads, counts, library = np.asarray(reads), np.asarray(counts), np.asarray(library)

    if reads.dtype.type is np.str_:
        reads = pack_barcodes(reads)
    if library.dtype.type is np.str_:
        library = pack_barcodes(library)

    if len(reads.shape) == 1:
        reads = reads.reshape(-1, 1)
    if len(library.shape) == 1:
        library = library.reshape(-1, 1)

    matched_indices = np.full(reads.shape[0], -1, dtype=int)
    for i in progress(range(reads.shape[0])):
        read_set = reads[i]
        count_set = counts[i]
        read_set = read_set[count_set!=0]
        count_set = count_set[count_set!=0]
        if read_set.size < library.shape[1]: continue

        print (read_set)
        print (read_set.shape, library.shape)
        print ('calciing distss')
        dists = calc_barcode_distances(read_set, library).reshape(-1)
        print (dists)
        indices = np.argpartition(dists, 1)
        print (indices)

        if dists[indices[0]] <= max_edit_distance and dists[indices[1]] > max_edit_distance:
            matched_indices[i] = indices[0]

    return matched_indices

def match_barcodes_nn(reads, counts, library, max_edit_distance=0):
    if len(reads.shape) == 1:
        reads = reads.reshape(-1, 1)
    if len(library.shape) == 1:
        library = library.reshape(-1, 1)

    if reads.dtype.type is np.str_:
        reads = barcodes_to_vector(reads)
    if library.dtype.type is np.str_:
        library = barcodes_to_vector(library)

    flat_library = library.reshape(-1, library.shape[-1])

    neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=2, metric='manhattan').fit(flat_library)




def pack_barcodes(barcodes, dtype=None, out=None):
    """ Helper function that converts strings of 'GTAC' into packed integers or byte strings.
    Because there are only 4 possibilities for each character, we can use two bits to represent
    them and allow for much faster comparisons and edit distance computation.
    dtype can be an unsigned integer type or bytes, if it is None it will default to uint64 unless
    the barcodes are longer than 32 characters
    """
    barcodes = np.asarray(barcodes)

    orig_shape = barcodes.shape
    barcodes = barcodes.reshape(-1).astype(bytes, copy=True)

    bytes_needed = math.ceil(barcodes.dtype.itemsize / 4)
    if bytes_needed > 8:
        raise ValueError("Unable to pack {}".format(barcodes.dtype))

    if dtype is None:
        dtype = [dt for dt in [np.uint8, np.uint16, np.uint32, np.uint64] if dt().itemsize >= bytes_needed][0]

    values = np.frombuffer(barcodes, dtype=np.uint8).reshape(barcodes.shape[0], barcodes.dtype.itemsize)

    # scale data G(71),T(84),A(65),C(67) -> 0,1,2,3
    values %= 34
    values //= 11

    if out is None:
        out = values[:,-1].astype(dtype).reshape(orig_shape)
    else:
        assert out.shape == orig_shape
        out[...] = values[:,-1].astype(dtype).reshape(orig_shape)

    for i in range(1, values.shape[1]):
        out <<= 2
        out |= values[:,values.shape[1]-i-1].reshape(orig_shape)

    return out

def unpack_barcodes(barcodes, length=None, out=None):
    """ Reverse function to utils.pack_barcodes that converts the packed integer values back to sequences.
    Importantly, this function will not give the exact same sequence given into pack_barcodes. The
    length of the barcode is not stored, so if not specified it will decode as many bases as the underlying
    dtype can hold.
    """
    length = length or barcodes.dtype.itemsize * 4
    barcodes = barcodes.copy()

    values = np.zeros((barcodes.shape[0], length), dtype=np.uint8)
    for i in range(length):
        values[:,i] = barcodes & 0b11
        barcodes >>= 2

    #more math to get them back to ascii 0,1,2,3 -> G(71),T(84),A(65),C(67)
    values += 2
    values %= 4
    values <<= values
    values += 6
    values *= 7
    values //= 9
    values += 61

    barcodes = np.frombuffer(values, dtype='S' + str(length))

    if out is None:
        return barcodes.astype(str)
    else:
        out[...] = barcodes

def barcodes_to_vector(barcodes):
    barcodes = np.asarray(barcodes)

    orig_shape = barcodes.shape
    barcodes = barcodes.reshape(-1)
    barcodes = np.frombuffer(barcodes, dtype=np.array(barcodes[0][0]).dtype).reshape(barcodes.shape[0], -1)

    vecs = np.zeros((barcodes.shape[0], barcodes.shape[1] * 2), dtype=np.float32)
    vecs[:,::2] = (barcodes == 'T').astype(np.float32) - (barcodes == 'G').astype(np.float32)
    vecs[:,1::2] = (barcodes == 'C').astype(np.float32) - (barcodes == 'A').astype(np.float32)

    return vecs.reshape(orig_shape + (-1,))

def make_edit_distance_table():
    values = np.arange(256, dtype=np.uint8)
    counts = np.zeros(256, dtype=np.uint8)

    values = (values & 0b01010101) | ((values >> 1) & 0b01010101)
    for i in range(4):
        counts += values & 0b1
        values >>= 2

    return counts

_edit_distance_table = None
def edit_distance(barcode1, barcode2, out=None):
    """ Calculates edit distance of packed barcodes. Limit specifies a limit for the distance,
    if for example you only care about distances lower than 2 this will limit the computation.
    """
    barcode1, barcode2 = np.asarray(barcode1), np.asarray(barcode2)

    global _edit_distance_table
    if _edit_distance_table is None:
        _edit_distance_table = make_edit_distance_table()

    if barcode1.dtype.type is np.str_:
        barcode1 = pack_barcodes(barcode1)
    if barcode2.dtype.type is np.str_:
        barcode2 = pack_barcodes(barcode2)

    diff = barcode1 ^ barcode2
    dists = _edit_distance_table[np.frombuffer(diff, dtype=np.uint8)].reshape(diff.shape + (diff.dtype.itemsize,))
    return dists.sum(axis=-1, out=out)



def cluster_reads(values):
    model = sklearn.cluster.AgglomerativeClustering(distance_threshold=1)
    new_values = []
    counts = []
    for i in np.unique(model.labels_):
        new_vals = values[model.labels_==i].mean(axis=0)
        new_values.append(new_vals)
        counts.append(np.sum(model.labels_==i))
    return np.array(new_values), np.array(counts)


def call_reads(cells, poses, values, num_reads=8):
    assert values.shape[2] == 4
    
    values /= values.sum(axis=2)

    cell_reads = {}

    for pos, vals in zip(poses, values):
        cell = cells[pos[0],pos[1]]
        cell_reads.setdefault(cell, []).append((pos, vals))

    merged_reads = {}
    for cell in cell_reads.keys():
        cell_vals = np.array([pair[1] for pair in cell_reads[cell]])
        cell_vals, counts = cluster_reads(cell_vals)

        sorted_indices = np.argsort(counts)[::-1]
        cell_vals, counts = cell_vals[sorted_indices], counts[sorted_indices]

        reads = np.array([''.join('GTAC'[i] for i in np.argmax(vals)) for vals in cell_vals])
        merged_reads[cell] = list(zip(reads, counts))

    return merged_reads, cell_reads
