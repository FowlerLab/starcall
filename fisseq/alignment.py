import numpy as np
import skimage.feature
import sklearn.neighbors
import scipy.sparse

class DistanceMatrix:
    def __init__(self, size1, size2=None, dtype=np.float32, value=None):
        self.size1, self.size2 = size1, size2 or size1
        if value is None:
            self.mat = np.zeros((self.size1, self.size2), dtype=dtype)
        else:
            self.mat = np.full((self.size1, self.size2), value, dtype=dtype)

    def __getitem__(self, indices):
        assert len(indices) == 2
        if indices[0] > indices[1]:
            indices = (indices[1], indices[0])
        return self.mat.__getitem__(indices)

    def __setitem__(self, indices, item):
        assert len(indices) == 2
        if indices[0] > indices[1]:
            indices = (indices[1], indices[0])
        return self.mat__setitem__(indices, item)

def pair_dots_func(poses1, poses2, dist_func, pass_indices=False, progress=False, debug=True):
    """ Finds pairs of dots where each one is the closest to the other, based on the given
        distance function.
            poses1, poses2: numpy arrays of positions that will be passed to the dist_func
            dist_func: a function that accepts either two poses from poses1 and poses2
                or two indices (see pass_indices) and returns a single float, the distance score
            pass_indices: if true, the indices into poses1, poses2 are passed to dist_func
            progress: if true, a progress bar is displayed
            debug: if true, debug info is printed to the console
    """
    if progress:
        import tqdm
        progress = tqdm.tqdm
    else:
        progress = lambda x: x

    if debug:
        def debug(*args):
            print (*args)
    else:
        def debug(*args):
            pass

    minvals1 = np.full(len(poses1), np.inf)
    minvals2 = np.full(len(poses2), np.inf)
    minindices1 = np.zeros(len(poses1), dtype=int)
    minindices2 = np.zeros(len(poses2), dtype=int)

    debug("Finding matches")
    for i in progress(range(len(poses1))):
        for j in range(len(poses2)):
            if pass_indices:
                score = dist_func(i, j)
            else:
                score = dist_func(poses1[i], poses2[j])
            if score < minvals1[i]:
                minvals1[i], minindices1[i] = score, j
            if score < minvals2[j]:
                minvals2[j], minindices2[j] = score, i
    debug("  done")

    indices1 = np.arange(len(poses1))
    indices2 = minindices1
    mask = indices1 == minindices2[indices2]
    indices1 = indices1[mask]
    indices2 = indices2[mask]

    matches = np.column_stack((indices1, indices2))
    return matches

def pair_dots_nn_func(poses1, poses2, dist_func, neighbors=100, pass_indices=False, progress=False, debug=True):
    """ Finds pairs of dots where each one is the closest to the other, based on the given
        distance function. Only considers matches in the n nearest neighbors of each point,
        where n is the argument neighbors.
            neighbors: number of nearest neighbors to consider when searching for pairs. This
                can also be the actual neighbors already calculated, in this case it would be
                a numpy array of shape (len(poses1), n_neighbors) of indices into poses2.
            See pair_dots_func for documentation of other args.
    """
    if progress:
        import tqdm
        progress = tqdm.tqdm
    else:
        progress = lambda x: x

    if debug:
        def debug(*args):
            print (*args)
    else:
        def debug(*args):
            pass

    if type(neighbors) == int:
        debug("Finding nearest neighbors")
        nearest_neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=neighbors).fit(poses2)
        dists, neighbors = nearest_neighbors.kneighbors(poses1)
        debug("  done")

    minvals1 = np.full(len(poses1), np.inf)
    minvals2 = np.full(len(poses2), np.inf)
    minindices1 = np.zeros(len(poses1), dtype=int)
    minindices2 = np.zeros(len(poses2), dtype=int)

    debug("Finding matches")
    for i in progress(range(len(poses1))):
        for j in neighbors[i]:
            if pass_indices:
                score = dist_func(i, j)
            else:
                score = dist_func(poses1[i], poses2[j])
            if score < minvals1[i]:
                minvals1[i], minindices1[i] = score, j
            if score < minvals2[j]:
                minvals2[j], minindices2[j] = score, i
    debug("  done")

    indices1 = np.arange(len(poses1))
    indices2 = minindices1
    mask = indices1 == minindices2[indices2]
    indices1 = indices1[mask]
    indices2 = indices2[mask]

    matches = np.column_stack((indices1, indices2))
    return matches


def pair_dots1(poses1, poses2):
    neighbors1 = sklearn.neighbors.NearestNeighbors(n_neighbors=1).fit(poses1)
    neighbors2 = sklearn.neighbors.NearestNeighbors(n_neighbors=1).fit(poses2)
    dists1, indices1 = neighbors1.kneighbors(poses2)
    dists2, indices2 = neighbors2.kneighbors(poses1)
    indices1, indices2 = indices1.flatten(), indices2.flatten()

    mutual = indices2[indices1] == indices1
    matches = np.stack([np.nonzero(mutual)[0], indices1[mutual]], axis=-1)
    return matches

import numba

def fastnorm(x):
    return np.linalg.norm(x)

@numba.extending.overload(fastnorm)
def fastnorm_overload(x):
    if x == numba.types.float64:
        def scalar_impl(x):
            return abs(x)
        return scalar_impl
    else:
        def vector_impl(x):
            return np.sqrt(np.sum(x*x))
        return vector_impl

@numba.jit(nopython=True)
def float_string_dist(vec1, vec2, penalty=10):
    last_dists = np.arange(len(vec2),-1,-1, dtype=np.float64)
    cur_dists = np.zeros(len(vec2)+1)
    for i in range(len(vec1)-1,-1,-1):
        cur_dists[-1] = len(vec2) - i + 1
        for j in range(len(vec2)-1,-1,-1):
            cur_dists[j] = min(
                last_dists[j] + penalty,
                last_dists[j+1] + fastnorm(vec1[i] - vec2[j]),
                cur_dists[j+1] + penalty,
            )
        last_dists, cur_dists = cur_dists, last_dists
    return last_dists[0]

def pair_dots2(poses1, poses2, n_neighbors=10, penalty=5, progress=False, debug=True, max_distance=None):

    neighbors1 = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors).fit(poses1)
    dists1, indices1 = neighbors1.kneighbors(poses1)
    neighbors2 = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors).fit(poses2)
    dists2, indices2 = neighbors2.kneighbors(poses2)

    #features1, features2 = dists1, dists2
    features1 = poses1[indices1] - poses1.reshape(poses1.shape[0], 1, poses1.shape[1])
    features2 = poses2[indices2] - poses2.reshape(poses2.shape[0], 1, poses2.shape[1])
    print (indices1[260])
    print (indices2[260])
    print (features1[260])
    print (features2[260])
    print (np.sqrt(np.sum((features1[260] - features2[260])**2, axis=1)))
    print (np.abs(dists1[260] - dists2[260]))
    print (np.min(dists1, axis=0))
    print (np.mean(dists1, axis=0))
    print (np.max(dists1, axis=0))

    def print_stats(arr):
        print (np.array([np.min(arr, axis=0), np.mean(arr, axis=0), np.max(arr, axis=0)]).T)

    np.set_printoptions(suppress=True)
    print_stats(np.abs(dists1 - dists2))
    print_stats(np.abs(dists1 - np.roll(dists2, 20, axis=0)))

    print_stats(np.linalg.norm(features1 - features2, axis=2))
    print_stats(np.linalg.norm(features1 - np.roll(features2, 20, axis=0), axis=2))

    def dist_func(i, j):
        #if max_distance:
            #if abs(poses1[i,0] - poses2[j,0]) + abs(poses1[i,1] - poses2[j,1]) >= max_distance:
                #return np.inf
        return float_string_dist(features1[i], features2[j], penalty=penalty)

    #matches = pair_dots_func(poses1, poses2, dist_func, pass_indices=True, progress=progress)
    matches = pair_dots_nn_func(poses1, poses2, dist_func, neighbors=100, pass_indices=True, progress=progress, debug=debug)

    return matches

def pair_dots3(poses1, poses2, n_neighbors=50):
    neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors).fit(poses1)
    dists, indices = neighbors.kneighbors(poses2)
    
    """
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots()
    #axis.bar(*np.unique(dists, return_counts=True))
    axis.hist(dists.flatten(), bins=100)
    fig.savefig('plots/dist_hist.png')
    """

def filter_pairs1(poses1, poses2, matches, n_neighbors=25):
    neighbors1 = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors).fit(poses1)
    dists1, indices1 = neighbors1.kneighbors(poses1)
    neighbors2 = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors).fit(poses2)
    dists2, indices2 = neighbors2.kneighbors(poses2)


def match_dots(position_list, min_dots=2, pair_dots_func=pair_dots2, **kwargs):
    """ Matches groups of points between the sets given in position_list
    Uses the provided pair_dots_func to match dots into pairs
    between each set, then combines these pairs into larger groups
    """
    all_matches = np.zeros((0,len(position_list)), dtype=int)

    for index1 in range(len(position_list)):
        print ('start', index1)
        current_matches = np.full((len(position_list[index1]), len(position_list)), -1)
        current_matches[:,index1] = np.arange(len(current_matches))

        #print (current_matches)
        for index2 in range(index1+1, len(position_list)):
            print ('  checking', index2)
            matches = pair_dots_func(position_list[index1], position_list[index2], **kwargs)
            current_matches[matches[:,0],index2] = matches[:,1]
            print ('   ', len(matches))

        #print (current_matches)

        for i in range(len(current_matches)):
            #mask = np.any([all_matches[:,j] == current_matches[i,j] for j in range(index1,len(position_list)) if current_matches[i,j] != -1], axis=0)
            mask = all_matches[:,index1] == i
            for j in range(index1+1, len(position_list)):
                if current_matches[i,j] != -1:
                    mask |= all_matches[:,j] == current_matches[i,j]

            #assert np.sum(mask) <= 1
            if mask.sum() != 0:
                new_index = np.nonzero(mask)[0][0]
                replacement_mask = current_matches[i,index1:] != -1
                all_matches[new_index,index1:][replacement_mask] = current_matches[i,index1:][replacement_mask]
                current_matches[i] = -1

        current_matches = current_matches[np.any(current_matches != -1, axis=1)]
        all_matches = np.concatenate((all_matches, current_matches), axis=0)
        print ('end', index1)
        print (' ', len(all_matches))
        print (' ', np.unique(np.sum(all_matches != -1, axis=1), return_counts=True)[1])

    final_matches = all_matches[np.sum(all_matches != -1, axis=1) >= min_dots]
    return final_matches


