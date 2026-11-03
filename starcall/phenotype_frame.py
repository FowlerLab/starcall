""" Correction of the rotation and scale of phenotype cycles in coordinates instead of pixels.

detect_rotation measures the rotation and scale of every cycle (see starcall.rotation). With
stitching.rotation.phenotype_correction: 'coordinates', phenotype tiles are still rotated and
rescaled when they are read for alignment, so the positions solved in composite.json are those of
corrected tiles, but the phenotype images that are stitched, segmented and measured are the raw
frames, only translated. The rotation and scale are kept as a transform between the phenotype frame
(the coordinates of the stitched phenotype image) and the sequencing images, which is used to find the
phenotype pixel of each sequencing dot.

Coordinates are (row, col) pixel indices, with pixel centres at integers:
    base     global coordinates of the composite, in sequencing pixels
    nominal  base scaled by the nominal ratio r = phenotype_scale / bases_scale, n = (b + 0.5) * r - 0.5
    frame    the phenotype frame, n = g + M @ (f - g)
where M = content_matrix(applied_deg, applied_scale) of the reference phenotype cycle (the first one)
and g, the pivot, is the centre of the well, so the frame stays close to the nominal coordinates.

The content at raw pixel u of a tile of phenotype cycle k, whose corrected tile is at nominal position p
(composite.json scaled by r), is at n = p + c + M_k @ (u - c), c the centre of the tile. In the frame
    f = g + M^-1 @ (p + c - g) + M^-1 M_k @ (u - c)
so the tile is placed at q = g + M^-1 @ (p + c - g) - c and its pixels are moved by M^-1 M_k about its
centre: the identity for the reference cycle, which keeps its raw pixels. Other phenotype cycles are
resampled by the difference between their correction and the reference's, if there is one.

The positions q only come from the solve in base pixels, so they are good to about a phenotype pixel.
Multi tile phenotype cycles are refined with the offsets between overlapping phenotype tiles measured
at full resolution on the frames, softly anchored to q (refine_positions), which keeps the seams between
phenotype tiles aligned without moving the mosaic away from the sequencing cycles.
"""
import json

import numpy as np

from starcall.rotation import content_matrix


class PhenotypeFrame:
    """ The transform between the base coordinates of the composite and the phenotype frame, see the
    module docstring.

        ratio: phenotype_scale / bases_scale
        pivot: g, in nominal coordinates
        angle, scale: correction of the reference phenotype cycle, M = content_matrix(angle, scale)
        cycles: phenotype cycle name -> dict(angle, scale) of its correction (applied_deg, applied_scale)
        reference: name of the reference phenotype cycle
    """

    def __init__(self, ratio, pivot, cycles, reference=None):
        self.ratio = float(ratio)
        self.pivot = np.asarray(pivot, dtype=float)
        self.cycles = {str(name): dict(angle=float(c['angle']), scale=float(c['scale'])) for name, c in cycles.items()}
        self.reference = str(reference if reference is not None else next(iter(self.cycles)))
        self.angle = self.cycles[self.reference]['angle']
        self.scale = self.cycles[self.reference]['scale']
        self.matrix = content_matrix(self.angle, self.scale)
        self.inverse = np.linalg.inv(self.matrix)

    @property
    def is_identity(self):
        return self.angle == 0 and self.scale == 1

    def relative(self, cycle):
        """ (angle, scale) by which the raw frames of a phenotype cycle are resampled to place them in the
        frame, M^-1 M_k, (0, 1) for the reference cycle """
        c = self.cycles[str(cycle)]
        return c['angle'] - self.angle, c['scale'] / self.scale

    def base_to_frame(self, points):
        """ Phenotype frame coordinates of points (N, 2) in base coordinates """
        nominal = (np.asarray(points, dtype=float) + 0.5) * self.ratio - 0.5
        return self.pivot + (nominal - self.pivot) @ self.inverse.T

    def frame_to_base(self, points):
        """ Base coordinates of points (N, 2) in phenotype frame coordinates """
        nominal = self.pivot + (np.asarray(points, dtype=float) - self.pivot) @ self.matrix.T
        return (nominal + 0.5) / self.ratio - 0.5

    def affine(self):
        """ (A, t) with base = A @ frame + t, the whole transform as one affine map """
        A = self.matrix / self.ratio
        t = (self.pivot - self.matrix @ self.pivot + 0.5) / self.ratio - 0.5
        return A, t

    def anchor_positions(self, cycle, positions, sizes):
        """ Unrounded frame positions q of the tiles of a phenotype cycle, from the positions (N, 2) of its
        corrected tiles in composite.json (base coordinates) and their sizes (N, 2) in phenotype pixels """
        c = (np.asarray(sizes, dtype=float) - 1) / 2
        nominal = np.asarray(positions, dtype=float) * self.ratio
        return self.pivot + (nominal + c - self.pivot) @ self.inverse.T - c

    def to_dict(self):
        A, t = self.affine()
        return dict(ratio=self.ratio, pivot=self.pivot.tolist(), reference=self.reference, cycles=self.cycles,
                frame_to_base=dict(A=A.tolist(), t=t.tolist()),
                note='(row, col) pixel indices; base = A @ frame + t, see starcall.phenotype_frame')

    @classmethod
    def from_dict(cls, obj):
        return cls(obj['ratio'], obj['pivot'], obj['cycles'], obj['reference'])


def make_frame(cycle_table, phenotype_cycles, ratio, mins, maxes):
    """ The PhenotypeFrame of a well, from the rotation.csv table of detect_rotation (a pandas DataFrame) and
    the extent (mins, maxes) of the composite in base coordinates """
    table = cycle_table.set_index(cycle_table['cycle'].astype(str))
    cycles = {}
    for cycle in phenotype_cycles:
        row = table.loc[str(cycle)]
        scale = row['applied_scale'] if 'applied_scale' in row and np.isfinite(row['applied_scale']) else 1.0
        cycles[str(cycle)] = dict(angle=float(row['applied_deg']), scale=float(scale))
    pivot = (np.asarray(mins, dtype=float) + np.asarray(maxes, dtype=float)) / 2 * ratio
    return PhenotypeFrame(ratio, pivot, cycles, reference=str(phenotype_cycles[0]))


def save(path, frame, positions, extra=None):
    """ Writes phenotype_frame.json: the frame and the integer frame positions of the tiles of each phenotype
    cycle (dict of cycle name to (N, 2) positions, in the order of the tiles in composite.json) """
    obj = frame.to_dict()
    obj['positions'] = {str(cycle): np.asarray(p).astype(int).tolist() for cycle, p in positions.items()}
    obj.update(extra or {})
    with open(path, 'w') as ofile:
        json.dump(obj, ofile, indent=1)


def load(path):
    """ (PhenotypeFrame, dict of cycle name to (N, 2) integer frame positions) from phenotype_frame.json """
    with open(path) as ifile:
        obj = json.load(ifile)
    return PhenotypeFrame.from_dict(obj), {cycle: np.array(p, dtype=int).reshape(-1, 2) for cycle, p in obj['positions'].items()}


def _solve_axis(n, i1, i2, d, w, anchors, lam):
    """ L1 solve of one axis as a linear program: variables x (n), constraint slacks (m), anchor slacks (n) """
    import scipy.optimize
    import scipy.sparse
    m = len(d)
    rows = np.arange(m)
    # x_b - x_a - t <= d  and  -(x_b - x_a) - t <= -d
    D = scipy.sparse.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[rows, rows], np.r_[i2, i1])), shape=(m, n))
    T = scipy.sparse.identity(m, format='csr')
    I = scipy.sparse.identity(n, format='csr')
    Zmn, Znm = scipy.sparse.csr_matrix((m, n)), scipy.sparse.csr_matrix((n, m))
    A = scipy.sparse.vstack([
        scipy.sparse.hstack([D, -T, Zmn]),
        scipy.sparse.hstack([-D, -T, Zmn]),
        scipy.sparse.hstack([I, Znm, -I]),
        scipy.sparse.hstack([-I, Znm, -I]),
    ], format='csr')
    b = np.r_[d, -d, anchors, -anchors]
    c = np.r_[np.zeros(n), w, np.full(n, lam)]
    bounds = [(None, None)] * n + [(0, None)] * (m + n)
    result = scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    assert result.success, result.message
    return result.x[:n]


def refine_positions(anchors, i1, i2, offsets, weights, lam=1.0, outlier_px=None):
    """ Frame positions of the tiles of a phenotype cycle that fit the offsets between overlapping tiles,
    softly anchored to the positions from the joint solve:

        minimise   sum over pairs  w * | x_b - x_a - d_ab |   +   lam * sum over tiles | x_i - a_i |

    independently for rows and columns (L1, like the mae solver).

        anchors: (N, 2) unrounded frame positions, PhenotypeFrame.anchor_positions
        i1, i2, offsets, weights: tile indices (into anchors), offset (M, 2) from tile i1 to tile i2 measured on
            the frames, and weight of each pair
        lam: how strongly tiles are held to their anchors, relative to a pair of weight 1
        outlier_px: pairs whose offset disagrees with the offset between the anchors by more than this are
            dropped, eg wrong alignments of nearly empty tiles at the edge of the well

    Returns (positions (N, 2), number of pairs used). Tiles without pairs stay at their anchors.
    """
    anchors = np.asarray(anchors, dtype=float)
    i1, i2 = np.asarray(i1, dtype=int), np.asarray(i2, dtype=int)
    offsets, weights = np.asarray(offsets, dtype=float).reshape(-1, 2), np.asarray(weights, dtype=float)
    if outlier_px is not None and len(i1):
        ok = np.linalg.norm(anchors[i2] - anchors[i1] - offsets, axis=1) <= outlier_px
        i1, i2, offsets, weights = i1[ok], i2[ok], offsets[ok], weights[ok]
    if len(i1) == 0:
        return anchors.copy(), 0
    n = len(anchors)
    solved = np.stack([_solve_axis(n, i1, i2, offsets[:, k], weights, anchors[:, k], lam) for k in range(2)], axis=1)
    return solved, len(i1)


def describe(frame, tile_size=None, translation=True):
    """ Lines of text describing what the transform of a frame does, for the QC plots and phenotype_frame.txt.
    tile_size: (rows, cols) of a phenotype tile, to say what the correction means at its corners """
    lines = []
    ratio = frame.ratio
    for cycle, c in frame.cycles.items():
        rel_angle, rel_scale = frame.relative(cycle)
        lines.append('cycle{}: rotated {:+.4f} deg, scaled x{:.5f}{}'.format(cycle, c['angle'], c['scale'],
                ' (reference phenotype cycle, raw pixels)' if cycle == frame.reference else
                ', resampled by {:+.4f} deg, x{:.5f} relative to cycle{}'.format(rel_angle, rel_scale, frame.reference)
                if (rel_angle, rel_scale) != (0, 1) else ', same as cycle{}, raw pixels'.format(frame.reference)))
    lines.append('1 phenotype px = {:.5f} sequencing px (nominal {:.5f} = bases_scale / phenotype_scale)'.format(
            frame.scale / ratio, 1 / ratio))
    if tile_size is not None:
        corner = (np.asarray(tile_size, dtype=float) - 1) / 2
        moved = np.linalg.norm((frame.matrix - np.eye(2)) @ corner)
        lines.append('correction at the corner of a {}x{} phenotype tile: {:.2f} phenotype px = {:.2f} sequencing px'.format(
                int(tile_size[0]), int(tile_size[1]), moved, moved / ratio))
    if translation:
        A, t = frame.affine()
        lines.append('sequencing (row, col) = A @ phenotype (row, col) + t,  A = [[{:+.6f}, {:+.6f}], [{:+.6f}, {:+.6f}]],  t = ({:+.2f}, {:+.2f})'.format(
                A[0, 0], A[0, 1], A[1, 0], A[1, 1], t[0], t[1]))
        lines.append('  (global composite coordinates; pivot {:.1f}, {:.1f} phenotype px)'.format(*frame.pivot))
    return lines
