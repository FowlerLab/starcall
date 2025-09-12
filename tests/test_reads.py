import unittest
import starcall.reads
import numpy as np
import io
import math
import matplotlib.pyplot as plt
import skimage.draw
import time

class TestReads(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.random_poses = self.rng.integers(1000, size=(100, 2))
        self.random_values = self.rng.random(size=(100, 12, 4))
        self.random_seqs = np.array(list('GTAC'), dtype='U1')[self.rng.integers(4, size=(100, 12))]
        self.random_seqs = np.frombuffer(self.random_seqs, dtype='U12')
        self.random_ints = self.rng.integers(1000, size=100)

        self.readsets = [
            starcall.reads.make_readset(values=self.random_values),
            starcall.reads.make_readset(sequences=self.random_seqs),
            starcall.reads.make_readset(positions=self.random_poses, values=self.random_values),
            starcall.reads.make_readset(positions=self.random_poses, sequences=self.random_seqs),
            starcall.reads.make_readset(positions=self.random_poses, values=self.random_values, sequences=self.random_seqs),
            starcall.reads.make_readset(values=self.random_values, cell=self.random_ints % 10, count=self.random_ints % 67),
            starcall.reads.make_readset(sequences=self.random_seqs, cell=self.random_ints % 10, count=self.random_ints % 67),
            starcall.reads.make_readset(positions=self.random_poses, values=self.random_values, cell=self.random_ints % 10, count=self.random_ints % 67),
            starcall.reads.make_readset(positions=self.random_poses, sequences=self.random_seqs, cell=self.random_ints % 10, count=self.random_ints % 67),
            starcall.reads.make_readset(positions=self.random_poses, values=self.random_values, sequences=self.random_seqs, cell=self.random_ints % 10, count=self.random_ints % 67),
        ]

    def assert_equal_numpy(self, arr1, arr2):
        self.assertEqual(type(arr1), type(arr2))
        if arr1 is not None:
            #self.assertEqual(arr1.dtype, arr2.dtype)
            if np.issubdtype(arr1.dtype, np.floating) or np.issubdtype(arr2.dtype, np.floating):
                self.assertTrue(np.all(np.abs(arr1 - arr2) < 0.000001))
            else:
                self.assertTrue(np.all(arr1 == arr2))

    def test_read(self):
        values = self.rng.random(size=(12, 4))
        read1 = starcall.reads.Read(values)
        read2 = starcall.reads.Read(values)
        self.assertEqual(len(read1.sequence), 12)
        self.assertEqual(read1.sequence_array.shape, (12,))

        read1.sequence = 'GTACGTACGTAC'
        self.assertEqual(read1.sequence, 'GTACGTACGTAC')
        self.assertEqual(read1.sequence_array.tolist(), list('GTACGTACGTAC'))

        #read2.sequence_array = 'G'
        #self.assertEqual(read2.sequence, 'GGGGGGGGGGGG')
        #read2.sequence_array[::4] = 'C'
        #self.assertEqual(read2.sequence, 'CGGGCGGGCGGG')

    def test_readset(self):
        values = self.rng.random(size=(100, 12, 4))

        readset = starcall.reads.make_readset(values=values)
        self.assertEqual(readset.reads.sequence.shape, (100,))
        self.assertEqual(str(readset.reads.sequence.dtype)[1:], 'U12')
        self.assertEqual(readset.reads.sequence_array.shape, (100, 12))
        self.assertEqual(str(readset.reads.sequence_array.dtype)[1:], 'U1')

        #readset.reads.sequence_array = 'G'
        #self.assertEqual(readset.reads.sequence[51], 'GGGGGGGGGGGG')
        #readset.reads.sequence_array[:,::2] = 'T'
        #self.assertEqual(readset.reads.sequence[51], 'TGTGTGTGTGTG')
        #readset.reads.sequence[51] = 'GTACGTACGTAC'
        #self.assertEqual(readset.reads.sequence[51], 'GTACGTACGTAC')

    def test_attrs(self):
        reads = starcall.reads.make_readset(values=self.random_values, count=self.rng.integers(0, 5, size=100), cell=np.arange(100))
        
        self.assertEqual(reads.reads[0]['count'], reads['count'][0])
        self.assertEqual(reads.reads[0]['cell'], 0)
        self.assertEqual(reads.reads[50]['count'], reads['count'][50])
        self.assertEqual(reads.reads[50]['cell'], 50)

    def test_clustering_line(self):
        poses = np.array([np.arange(20), np.zeros(20, dtype=int)]).T
        values = np.zeros((20, 12, 4))
        reads = starcall.reads.make_readset(poses, values)

        dists = starcall.reads.distance_matrix(reads, distance_cutoff=10.5, positional_weight=1, value_weight=1, debug=False)

        for i in range(len(reads)):
            for j in range(i+1, len(reads)):
                dist = j - i
                if dist < 1:
                    self.assertIn((i,j), dists)
                    #print (dists[i,j], dist)
                    self.assertTrue(abs(dists[i,j] - dist) < 0.00001)

        clusters = starcall.reads.cluster_reads(dists, threshold=2.5, linkage='min', debug=False)
        #print (clusters)
        #print (dists[0,1], dists[1,2], dists[2,3], dists[3,4])
        vals, counts = np.unique(clusters, return_counts=True)
        #print (vals, counts)
        #self.assertEqual(np.max(counts), 8)
        # Because of distance cutoff, otherwise it would be all one cluster
        # is 8 because ties between clusters to be merged is broken by insertion order,
        # so clusters build up by powers of 2, and so the max size is 8

        clusters = starcall.reads.cluster_reads(dists, threshold=2.5, linkage='max', debug=False)
        vals, counts = np.unique(clusters, return_counts=True)
        #print (vals, counts)
        self.assertEqual(np.max(counts), 2)

        clusters = starcall.reads.cluster_reads(dists, threshold=2.5, linkage='mean', debug=False)
        vals, counts = np.unique(clusters, return_counts=True)
        #print (vals, counts)
        self.assertEqual(np.max(counts), 4)

    def test_heap(self):
        numbers = self.rng.permutation(100)
        indices = np.arange(100)

        heap = starcall.reads.Heap()
        for i, num in zip(indices, numbers):
            heap.push(i, num)

        #sorted_indices = [heap.pop() for i in range(100)]
        sorted_indices = []
        for i in range(50):
            sorted_indices.append(heap.pop())

        for i in range(50, 100, 3):
            if i not in sorted_indices:
                numbers[i] *= 2
                heap.remove(i)
                heap.push(i, numbers[i])

        for i in range(50, 100):
            sorted_indices.append(heap.pop())

        self.assertEqual(list(sorted_indices), list(np.argsort(numbers)))

    def test_clustering_random(self):
        return
        poses = self.rng.integers(200, size=(1000, 2))
        values = np.zeros((1000, 12, 4))
        reads = starcall.reads.make_readset(poses, values)

        cells = np.zeros((200, 200), dtype=int)

        xposes, yposes = skimage.draw.ellipse(50, 50, 5, 10)
        cells[xposes,yposes] = 1

        xposes, yposes = skimage.draw.ellipse(100, 50, 8, 20)
        cells[xposes,yposes] = 2

        xposes, yposes = skimage.draw.ellipse(70, 150, 10, 4)
        cells[xposes,yposes] = 3

        dists = starcall.reads.distance_matrix(reads, distance_cutoff=50)
        #dists = starcall.reads.distance_matrix(reads, cells=cells, distance_cutoff=50)
        clusters = starcall.reads.cluster_reads(dists, threshold=15, linkage='mean')

        return
        fig, axis = plt.subplots()
        axis.imshow(cells.T)
        axis.scatter(poses[:,0], poses[:,1], c=clusters)

        for i in range(clusters.max() + 1):
            cur_poses = poses[clusters==i]
            centroid = cur_poses.mean(axis=0)
            for j in range(cur_poses.shape[0]):
                axis.plot([centroid[0], cur_poses[j,0]], [centroid[1], cur_poses[j,1]], c='C{}'.format(i))

        fig.savefig('tmp_plot_clusters.png')

    def test_clustering_reads(self):
        return
        poses = np.zeros((25, 2), int)
        rng = np.random.default_rng(12345)
        values = rng.random(size=(25,6,4))
        reads = starcall.reads.make_readset(poses, values)

        reads.reads.normalize()
        dists = starcall.reads.distance_matrix(reads, value_weight=0, sequence_weight=1, debug=False, progress=False)

        #for (i, j), dist in dists.items():
            #print (reads.reads[i].sequence, reads.reads[j].sequence, dist)

        #for threshold in [0.05, 0.1, 0.2, 0.3]:
        #for threshold in [0.25, 0.5, 1, 2]:
        for threshold in [0, 1, 2, 3]:
            reads['cluster'] = starcall.reads.cluster_reads(dists, threshold=threshold)
            groups = reads.groupby('cluster')
            #for i in range(groups.size().max()):
                #print (' '.join((group.reads.sequence[i] if len(group.index) > i else ' ' * reads.reads.num_cycles) for cluster, group in groups))
            #for i in range(groups.max_reads):
                #print (' '.join((group[i].sequence if len(group) > i else ' ' * reads.n_cycles) for group in groups))
            #print()

    def test_clustering_reads_seq(self):
        return
        poses = np.zeros((20, 2), int)
        values = self.rng.random(size=(20,6,4))
        reads = starcall.reads.make_readset(poses, values)

        reads.reads.normalize()
        dists = starcall.reads.distance_matrix(reads, value_weight=0, sequence_weight=1, debug=False, progress=False)

        #for (i, j), dist in dists.items():
            #print (reads.reads[i].sequence, reads.reads[j].sequence, dist)
        
        #for threshold in [0.05, 0.1, 0.2, 0.3]:
        for threshold in [1, 2, 3, 4]:
            reads['cluster'] = starcall.reads.cluster_reads(dists, threshold=threshold)
            groups = reads.groupby('cluster')
            for i in range(groups.size().max()):
                print (' '.join((group.reads.sequence[i] if len(group.index) > i else ' ' * reads.reads.num_cycles) for cluster, group in groups))
            print()

    def test_times(self):
        table = self.readsets[0]

        table.reads
        begin = time.time()
        #for i in range(len(table.index)):
        for i in range(5):
            table.reads[i]
        print (time.time() - begin)


if __name__ == '__main__':
    unittest.main()

