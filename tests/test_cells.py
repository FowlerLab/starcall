import unittest
import starcall.cells
import pandas
import numpy as np
import io
import math
import matplotlib.pyplot as plt
import skimage.draw
import time

class TestCells(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.random_poses = self.rng.integers(1000, size=(100, 2))
        self.random_sizes = self.rng.integers(100, size=(100, 2))

        self.tables = [
            starcall.cells.make_cell_table(positions=self.random_poses, sizes=self.random_sizes),
        ]

    def assertEqualNumpy(self, arr1, arr2):
        self.assertEqual(type(arr1), type(arr2))
        if arr1 is not None:
            #self.assertEqual(arr1.dtype, arr2.dtype)
            if np.issubdtype(arr1.dtype, np.floating) or np.issubdtype(arr2.dtype, np.floating):
                self.assertTrue(np.all(np.abs(arr1 - arr2) < 0.000001))
            else:
                self.assertTrue(np.all(arr1 == arr2))

    def test_attrs(self):
        cells = starcall.cells.make_cell_table(positions=self.random_poses, sizes=self.random_sizes)
        cells['count'] = np.arange(len(cells.index)) % 17
        cells['cell'] = np.arange(len(cells.index)) % 9

        self.assertEqual(cells.cells[1]['count'], cells['count'][1])
        self.assertEqual(cells.cells[1]['cell'], 0)
        self.assertEqual(cells.cells[51]['count'], cells['count'][51])
        self.assertEqual(cells.cells[51]['cell'], 50 % 9)

    def test_consolidation(self):
        table = pandas.DataFrame({'bbox_x1': np.arange(10), 'bbox_y1': np.arange(10),
                'bbox_x2': np.arange(10) + 5, 'bbox_y2': np.arange(10) + 10})
        self.assertEqual(table.cells.positions.shape, (10, 2))
        table.cells.positions[0,0] = 100
        self.assertEqual(table.cells.positions[0,0], 100)

        table = pandas.DataFrame()
        table['bbox_x1'] = np.arange(10)
        table['bbox_y1'] = np.arange(10)
        table['bbox_x2'] = np.arange(10)
        table['bbox_y2'] = np.arange(10)
        self.assertEqual(table.cells.positions.shape, (10, 2))
        table.cells.positions[0,0] = 100
        self.assertEqual(table.cells.positions[0,0], 100)

    def to_segmentation(self, segmentation):
        return np.array([list(map(int, line.strip())) for line in segmentation.split()])

    def to_mask(self, mask):
        return self.to_segmentation(mask).astype(bool)

    def test_segmentation(self):
        segmentation = self.to_segmentation("""
        00100000022200000400
        01110000005200000400
        00100003005200000400
        00000333005550000400
        00000300000000000400
        """)
        table = starcall.cells.make_cell_table(segmentation)

        self.assertEqualNumpy(table.cells.bboxes, np.array([[0,1,3,4], [0,9,3,12], [2,5,5,8], [0,17,5,18], [1,10,4,13]]))
        self.assertEqualNumpy(table.cells[1].segmentation, self.to_segmentation('010 111 010'))
        self.assertEqualNumpy(table.cells[2].segmentation, self.to_segmentation('222 052 052'))
        self.assertEqualNumpy(table.cells[3].segmentation, self.to_segmentation('003 333 300'))
        self.assertEqualNumpy(table.cells[4].segmentation, self.to_segmentation('4 4 4 4 4'))
        self.assertEqualNumpy(table.cells[5].segmentation, self.to_segmentation('520 520 555'))
        self.assertEqualNumpy(table.cells[1].mask, self.to_segmentation('010 111 010').astype(bool))
        self.assertEqualNumpy(table.cells[2].mask, self.to_segmentation('111 001 001').astype(bool))
        self.assertEqualNumpy(table.cells[3].mask, self.to_segmentation('001 111 100').astype(bool))
        self.assertEqualNumpy(table.cells[4].mask, self.to_segmentation('1 1 1 1 1').astype(bool))
        self.assertEqualNumpy(table.cells[5].mask, self.to_segmentation('100 100 111').astype(bool))

    def test_rescaling(self):
        segmentation = self.to_segmentation("""
        001100002200003344
        001100002200223333
        111100002222220033
        111100002200220033
        001111002222003333
        001111002255222200
        001100002222220000
        001100000000000000
        """)
        table = starcall.cells.make_cell_table(segmentation)

        table.cells.rescale_masks(2)
        self.assertEqualNumpy(table.cells.rescaled_masks[2][1], self.to_mask('010 110 011 010'))
        self.assertEqualNumpy(table.cells[1].mask2, self.to_mask('010 110 011 010'))

        file = io.StringIO()
        table.to_csv(file)
        file.seek(0)
        table2 = pandas.read_csv(file, index_col=0)
        self.assertEqualNumpy(table2.cells[1].mask2, self.to_mask('010 110 011 010'))
        for cell, cell2 in zip(table.cells, table2.cells):
            self.assertEqualNumpy(cell.mask2, cell2.mask2)

    def test_intersection(self):
        mask1 = self.to_mask('001100 011110 110011 111111 001100')
        mask2 = self.to_mask('110000 111111 111111 110011 110000')
        cell1, cell2 = starcall.cells.Cell(position=(0,0), mask=mask1), starcall.cells.Cell(position=(0,0), mask=mask2)
        self.assertEqualNumpy(cell1.intersection(cell2).mask, mask1 & mask2)
        self.assertEqualNumpy(cell1.intersection(starcall.cells.Cell(position=(1,1), mask=mask2)).mask, mask1[1:,1:] & mask2[:-1,:-1])
        #print (cell1.intersection(cell2).bbox)
        #print (cell1.intersection(cell2).point1)
        #print (cell1.intersection(cell2).point2)
        #print (cell1.intersection(cell2).mask)
        #print (cell1.intersection(starcall.cells.Cell(position=(1,1), mask=mask2)).mask)
        #print (cell1.intersection(cell2).area())

    def test_list_cells(self):
        segmentation = self.to_segmentation("""
        001100002200003344
        001100002200223333
        111100002222220033
        111100002200220033
        001111002222003333
        001111002255222200
        001100002222220000
        001100000000000000
        """)
        table = starcall.cells.make_cell_table(segmentation)
        table.cells.rescale_masks(2)
        table['cell'] = np.arange(len(table.index))
        table2 = starcall.cells.make_cell_table(list(table.cells))
        self.assertEqualNumpy(table.cells.bboxes, table2.cells.bboxes)
        for i in table.index:
            self.assertEqualNumpy(table.cells[i].mask2, table2.cells[i].mask2)
        #print (table2)

    def test_intersecting_cells(self):
        segmentation = self.to_segmentation("""
        001100002200003344
        001100002200223333
        111100002222220033
        111100002200220033
        001111002222003333
        001111002255222200
        001100002222220000
        001100000000000000
        """)

        segmentation2 = self.to_segmentation("""
        220000001100003344
        220000001100003333
        222200001100003300
        222200001100003300
        002222001100000000
        002222001111000000
        002200001100000000
        002200000000000000
        """)

        table = starcall.cells.make_cell_table(segmentation)
        table['cell'] = np.zeros(len(table.index))
        table2 = starcall.cells.make_cell_table(segmentation2)
        table['count'] = np.arange(len(table.index))
        matches = table.cells.intersecting_cells(table2)
        #print (matches)
        #print (matches.cells[1,2].mask)
        #print (matches.cells[1,2].rescaled_masks)

    def test_intersecting_cells_downscaled(self):
        segmentation = self.to_segmentation("""
        1100000000
        1100000000
        1111000000
        1111000000
        0000000000
        """)

        segmentation2 = self.to_segmentation("""
        0011110000
        0011110000
        0001110000
        0000110000
        0000000000
        """)

        table = starcall.cells.make_cell_table(segmentation)
        table2 = starcall.cells.make_cell_table(segmentation2)
        table.cells.rescale_masks(2)
        table2.cells.rescale_masks(2)

        file = io.StringIO()
        table.to_csv(file)
        file.seek(0)
        table = pandas.read_csv(file, index_col=0)

        file = io.StringIO()
        table2.to_csv(file)
        file.seek(0)
        table2 = pandas.read_csv(file, index_col=0)

        #print (table)
        #print (table2)
        #print (table.cells[1].mask2)
        #print (table2.cells[1].mask2)

        matches = table.cells.intersecting_cells(table2)
        #print (matches)
        #print (matches.cells[1,1].mask2)
        #print (matches.cells[1,2].mask)
        #print (matches.cells[1,2].rescaled_masks)

    def test_list_cells(self):
        segmentation = self.to_segmentation("""
        001100002200003344
        001100002200223333
        111100002222220033
        111100002200220033
        001111002222003333
        001111002255222200
        001100002222220000
        001100000000000000
        """)
        table = starcall.cells.make_cell_table(segmentation)
        table.cells.rescale_masks(2)

        fig, axes = plt.subplots()
        table.cells.plot(axes, masks=True)
        #fig.savefig('tmp_cells_plot.png')

        file = io.StringIO()
        table.to_csv(file)
        file.seek(0)
        table = pandas.read_csv(file, index_col=0)

        fig, axes = plt.subplots()
        table.cells.plot(axes, masks=True)
        #fig.savefig('tmp_cells_plot2.png')

    def test_rescale_from_rescaled(self):
        segmentation = self.to_segmentation("""
        001100002200003344
        001100002200223333
        111100002222220033
        111100002200220033
        001111002222003333
        001111002255222200
        001100002222220000
        001100000000000000
        """)
        table = starcall.cells.make_cell_table(segmentation)
        table.cells.rescale_masks(2)

        file = io.StringIO()
        table.to_csv(file)
        file.seek(0)
        table2 = pandas.read_csv(file, index_col=0)

        table.cells.rescale_masks(4)
        table2.cells.rescale_masks(4)

        for i in table.index:
            #print (table.cells.rescaled_masks[4][i], table2.cells.rescaled_masks[4][i])
            self.assertEqualNumpy(table.cells.rescaled_masks[4][i], table2.cells.rescaled_masks[4][i])



if __name__ == '__main__':
    unittest.main()

