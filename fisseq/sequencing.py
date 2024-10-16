import math
import numpy as np
import sklearn.cluster
import dataclasses
from . import utils

import time


class BarcodeLibrary:
    def __init__(self, barcodes, distance_function=None, batch_size=50):
        barcodes = np.asarray(barcodes)
        if barcodes.dtype.type is np.str_:
            barcodes = pack_barcodes(barcodes)
        if barcodes.ndim == 1:
            barcodes = barcodes.reshape(-1, 1)

        self.barcodes = barcodes
        self.distance_function = distance_function or calc_barcode_distances

        def dist_func(index1, index2):
            if index1 < 0:
                barc1 = self.current_reads[-int(index1)-1]
            else:
                barc1 = self.barcodes[int(index1)]

            if index2 < 0:
                barc2 = self.current_reads[-int(index2)-1]
            else:
                barc2 = self.barcodes[int(index2)]

            return self.distance_function(barc1, barc2)

        self.neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=2, metric=dist_func)
        self.neighbors = self.neighbors.fit(np.arange(len(self.barcodes)).reshape(-1, 1))

        """
        def split(dists):

        barcodes = self.barcodes
        dists = self.barcode_dists
        tree = []
        while len(barcodes) / batch_size > batch_size / 2:
            n_clusters = len(barcodes) // batch_size
            clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='average', metric='precomputed').fit_predict(dists)
            centers = np.empty(n_clusters, dtype=int)

            for i in range(n_clusters):
                mask = 
                indices = np.argwhere(clusters == i).flatten()
                sub_dists = dists[indices,indices]
                centers[i] = indices[np.argmin(dists.sum(axis=1))]

            new_clusters = dists[:,centers].argmin(axis=1)
            center_barcodes = barcodes[centers]

            cluster_barcodes = []
            cluster_indices = []
            for i in range(n_clusters):
                indices = np.argwhere(new_clusters == i).flatten()
                cluster_barcodes.append(barcodes[indices])
                cluster_indices.append(indices)
            tree.append((cluster_barcodes, cluster_indices))

            barcodes = center_barcodes
            dists = dists[centers,centers]

        tree.append(([barcodes], np.arange(len(barcodes))))
        self.tree = tree[::-1]
        #"""

    def closest(self, reads, match_threshold=2, second_threshold=None):
        second_threshold = second_threshold or match_threshold
        reads = np.asarray(reads)
        if reads.dtype.type is np.str_:
            reads = pack_barcodes(reads)
        if reads.ndim == 1:
            reads = reads.reshape(-1, 1)

        self.current_reads = reads
        dists, indices = self.neighbors.kneighbors(-np.arange(len(reads)).reshape(-1, 1) - 1)
        matches = dists <= match_threshold
        true_matches = matches[:,0] & ~matches[:,1]
        indices = indices[:,0]
        indices[~true_matches] = -1

        return indices


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
        distances.partition(reads_needed - 1, axis=1)
        distances = distances[:,reads_needed-1]

    return distances

def match_barcodes(reads, counts, library, max_edit_distance=0, debug=True, progress=False):
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
        #"""
        read_set = reads[i]
        count_set = counts[i]
        read_set = read_set[count_set!=0]
        count_set = count_set[count_set!=0]
        if read_set.size == 0: continue

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
        np.min(distances, axis=2, out=distances[:,:,0])
        np.max(distances[:,:,0], axis=1, out=distances[:,0,0])
        distances = distances[:,0,0]
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
    barcodes = np.frombuffer(barcodes, dtype=np.array(barcodes[0][0]).dtype).reshape(barcodes.shape[0], -1)

    vecs = np.zeros((barcodes.shape[0], barcodes.shape[1] * 2), dtype=np.float32)
    vecs[:,::2] = (barcodes == 'T').astype(np.float32) - (barcodes == 'G').astype(np.float32)
    vecs[:,1::2] = (barcodes == 'C').astype(np.float32) - (barcodes == 'A').astype(np.float32)

    return vecs

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
