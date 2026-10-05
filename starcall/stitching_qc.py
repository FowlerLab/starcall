""" QC plots for stitching: tile layouts, the constraints calculated and filtered between
pairs of cycles, and the positions and residuals after solving.

All plots are drawn in stage coordinates at the scale of the base (sequencing) images,
with x the image column and y the image row (increasing downwards), and cycles are
the z planes of the composite. Scores are the alignment score of each constraint (the
zero normalized cross correlation of the overlapping region, from -1 to 1), residuals
are the distance in pixels between the offset of a constraint and the offset between
the solved positions of its two tiles.
"""
import numpy as np

SCORE_LABEL = 'alignment score (ZNCC of overlap)'
RESIDUAL_LABEL = 'residual after solving (px)'
MAX_RESIDUAL = 5
MAX_LABELLED_TILES = 500
DPI = 150


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save(plt, fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def _labels(composite, cycle_labels):
    layers = sorted(set(int(z) for z in composite.boxes.positions[:,2]))
    if cycle_labels is None or len(cycle_labels) < len(layers):
        return {layer: 'z={}'.format(layer) for layer in layers}
    return {layer: str(cycle_labels[layer]) for layer in layers}


def _layers(composite):
    return composite.boxes.positions[:,2].astype(int)


def _key(const):
    return const.index1, const.index2


def is_modeled(const):
    """ Modeled constraints (estimated from the stage model instead of aligned) are saved
    as normal constraints, but unlike aligned constraints they have a nonzero error. """
    return const.modeled or (const.error is not None and const.error > 0)


def _tile_outlines(ax, composite, indices, color='0.8', linewidth=0.3):
    import matplotlib.collections
    import matplotlib.patches
    points1, sizes = composite.boxes.points1, composite.boxes.sizes
    patches = [matplotlib.patches.Rectangle((points1[i,1], points1[i,0]), sizes[i,1], sizes[i,0]) for i in indices]
    ax.add_collection(matplotlib.collections.PatchCollection(patches, facecolor='none', edgecolor=color, linewidth=linewidth))


def _setup_map(ax, composite):
    points1, points2 = composite.boxes.points1, composite.boxes.points2
    pad = 0.02 * (points2[:,:2].max() - points1[:,:2].min())
    ax.set_xlim(points1[:,1].min() - pad, points2[:,1].max() + pad)
    ax.set_ylim(points2[:,0].max() + pad, points1[:,0].min() - pad)
    ax.set_aspect('equal')
    ax.set_xlabel('stage x (px)')
    ax.set_ylabel('stage y (px)')
    ax.tick_params(labelsize=7)


def _marker_size(num):
    return float(np.clip(20000 / max(num, 1), 4, 60))


def _midpoints(consts):
    return np.array([(c.box1.center[:2] + c.box2.center[:2]) / 2 for c in consts]).reshape(-1, 2)


def _differences(consts):
    return np.array([c.difference for c in consts], dtype=float).reshape(-1, 2)


def relative_corrections(consts, reference=None):
    """ The correction of each constraint (see _differences) relative to the median
    correction of the constraints with the same stage offset direction (its neighbor
    direction, eg left or below), so that the spread of corrections is visible even when
    the stage positions are off by a constant amount per direction. The medians are taken
    from the constraints where reference is True (all by default). Returns the relative
    corrections and a dict of direction name to median correction. """
    diffs = _differences(consts)
    if reference is None:
        reference = np.ones(len(consts), dtype=bool)
    groups = []
    for c in consts:
        offset = c.box2.position[:2] - c.box1.position[:2]
        # in units of the smaller tile, so each phenotype tile inside a base tile is its own group
        size = np.maximum(np.minimum(c.box1.size[:2], c.box2.size[:2]), 1)
        groups.append(tuple(np.round(offset / size).astype(int)))
    medians = {}
    relative = diffs.copy()
    for group in set(groups):
        mask = np.array([g == group for g in groups], dtype=bool)
        ref = mask & reference
        median = np.median(diffs[ref if ref.any() else mask], axis=0)
        relative[mask] -= median
        medians[group] = median
    return relative, medians


def _medians_text(medians):
    names = {(0,0): 'same position', (0,1): 'right', (0,-1): 'left', (1,0): 'below', (-1,0): 'above'}
    parts = ['{}: x {:.0f}, y {:.0f}'.format(names.get(group, 'offset x {} y {} tiles'.format(group[1], group[0])), median[1], median[0])
            for group, median in sorted(medians.items())]
    lines = ['median correction by neighbor direction (px) - ']
    for part in parts:
        if len(lines[-1]) > 110:
            lines.append('')
        lines[-1] += ('; ' if lines[-1] and not lines[-1].endswith('- ') else '') + part
    return '\n'.join(lines)


def _scores(consts):
    return np.array([np.nan if c.score is None else c.score for c in consts], dtype=float)


def _grid(num):
    ncols = int(np.ceil(np.sqrt(num)))
    return int(np.ceil(num / ncols)), ncols


def _pair_title(composite, consts, cycle_labels):
    labels = _labels(composite, cycle_labels)
    for c in consts:
        z1, z2 = int(c.box1.position[2]), int(c.box2.position[2])
        return labels[z1] if z1 == z2 else '{} → {}'.format(labels[z1], labels[z2])
    return ''


def _score_histogram(ax, scores, background, threshold):
    bins = np.linspace(-0.2, 1, 61)
    scores, background = scores[np.isfinite(scores)], background[np.isfinite(background)]
    ax.hist(np.clip(scores, bins[0], bins[-1]), bins=bins, color='tab:blue', alpha=0.7,
            label='overlapping pairs (n={})'.format(len(scores)))
    ax.hist(np.clip(background, bins[0], bins[-1]), bins=bins, color='tab:orange', alpha=0.7,
            label='non-overlapping pairs (n={})'.format(len(background)))
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1,
                label='filter threshold {:.2f}\n(95th pct of non-overlapping)'.format(threshold))
    ax.set_yscale('log')
    ax.set_xlabel(SCORE_LABEL + ', clipped to [-0.2, 1]')
    ax.set_ylabel('number of pairs')
    ax.set_title('Score distribution')
    ax.legend(fontsize=7, loc='upper center')


def _correction_axes(ax, diffs, title):
    """ Limits from the 1st-99th percentile of diffs, notes how many points are outside """
    ax.axhline(0, color='0.6', linewidth=0.5)
    ax.axvline(0, color='0.6', linewidth=0.5)
    ax.set_xlabel('correction x relative to median (px)')
    ax.set_ylabel('correction y relative to median (px)')
    if len(diffs) == 0:
        ax.set_title(title)
        return
    lo, hi = np.percentile(diffs, 1, axis=0), np.percentile(diffs, 99, axis=0)
    pad = np.maximum((hi - lo) * 0.15, 5)
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[1], hi[1])
    ax.set_ylim(hi[0], lo[0])
    return lo, hi


def _outside(diffs, limits):
    if limits is None or len(diffs) == 0:
        return 0
    lo, hi = limits
    return int(np.sum(np.any((diffs < lo) | (diffs > hi), axis=1)))


def plot_tile_layout(composite, path, cycle_labels=None, title=None):
    """ One panel per cycle with the outline of every tile, labelled with its tile index
    in the cycle when there are few enough tiles to be legible. """
    plt = _plt()
    labels, layers = _labels(composite, cycle_labels), _layers(composite)
    nrows, ncols = _grid(len(labels))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows), squeeze=False, sharex=True, sharey=True,
            layout='constrained')
    centers = composite.boxes.centers
    for ax, (layer, label) in zip(axes.flat, labels.items()):
        indices = np.flatnonzero(layers == layer)
        _tile_outlines(ax, composite, indices, color='0.4', linewidth=0.4)
        if len(indices) <= MAX_LABELLED_TILES:
            for i, index in enumerate(indices):
                ax.text(centers[index,1], centers[index,0], str(i), ha='center', va='center', fontsize=4)
        _setup_map(ax, composite)
        ax.set_title('{} ({} tiles{})'.format(label, len(indices),
                '' if len(indices) <= MAX_LABELLED_TILES else ', too many to label'))
    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)
    fig.suptitle(title or 'Tile positions, labelled with the tile index within each cycle', fontsize=14)
    _save(plt, fig, path)


def plot_pair_constraints(composite, overlapping, calculated, background, path, cycle_labels=None):
    """ Constraints calculated between two cycles: a map of their scores, the score
    distribution against non-overlapping pairs, and the correction from the stage offset. """
    plt = _plt()
    calculated = list(calculated)
    calculated_keys = set(_key(c) for c in calculated)
    missing = [c for c in overlapping if _key(c) not in calculated_keys]
    scores, background = _scores(calculated), _scores(background)
    threshold = np.percentile(background[np.isfinite(background)], 95) if np.isfinite(background).any() else None
    name = _pair_title(composite, list(overlapping) or calculated, cycle_labels)

    fig, (ax_map, ax_hist, ax_diff) = plt.subplots(1, 3, figsize=(19, 6), width_ratios=[1.2, 1, 1.1], layout='constrained')
    if calculated:
        _tile_outlines(ax_map, composite, np.flatnonzero(_layers(composite) == int(calculated[0].box1.position[2])))
        mid = _midpoints(calculated)
        points = ax_map.scatter(mid[:,1], mid[:,0], c=scores, cmap='viridis', vmin=0, vmax=1, s=_marker_size(len(overlapping)))
        fig.colorbar(points, ax=ax_map, label=SCORE_LABEL, extend='min', shrink=0.8)
    if missing:
        mid = _midpoints(missing)
        ax_map.scatter(mid[:,1], mid[:,0], marker='x', color='0.5', s=_marker_size(len(overlapping)),
                label='not calculated (n={})'.format(len(missing)))
        ax_map.legend(fontsize=7, loc='upper right')
    _setup_map(ax_map, composite)
    ax_map.set_title('Score of each overlapping pair (n={})'.format(len(calculated)))

    _score_histogram(ax_hist, scores, background, threshold)

    diffs, medians = relative_corrections(calculated)
    if len(diffs):
        points = ax_diff.scatter(diffs[:,1], diffs[:,0], c=scores, cmap='viridis', vmin=0, vmax=1, s=8)
        fig.colorbar(points, ax=ax_diff, label=SCORE_LABEL, extend='min', shrink=0.8)
    limits = _correction_axes(ax_diff, diffs, '')
    outside = _outside(diffs, limits)
    ax_diff.set_title('Correction from stage offset' + (' ({} pairs outside view)'.format(outside) if outside else ''))

    fig.suptitle('Calculated constraints ' + name + '\n' + _medians_text(medians), fontsize=12)
    _save(plt, fig, path)


def plot_filtered_constraints(composite, calculated, background, threshold, above_threshold, inliers, modeled, path, cycle_labels=None):
    """ The outcome of filter_constraints for each calculated constraint: kept, removed
    for a score below the threshold, or removed as an outlier of the stage model. """
    plt = _plt()
    calculated = list(calculated)
    above_keys = set(_key(c) for c in above_threshold)
    inlier_keys = set(_key(c) for c in inliers)
    keys = [_key(c) for c in calculated]
    kept = np.array([k in inlier_keys for k in keys], dtype=bool)
    categories = [
        ('removed: score below threshold', np.array([k not in above_keys for k in keys], dtype=bool), dict(marker='x', color='0.5')),
        ('removed: stage model outlier', np.array([k in above_keys and k not in inlier_keys for k in keys], dtype=bool), dict(marker='x', color='tab:red')),
    ]
    scores, mid = _scores(calculated), _midpoints(calculated)
    diffs, medians = relative_corrections(calculated, reference=kept)
    name = _pair_title(composite, calculated, cycle_labels)
    size = _marker_size(len(calculated))

    fig, (ax_map, ax_hist, ax_diff) = plt.subplots(1, 3, figsize=(19, 6), width_ratios=[1.2, 1, 1.1], layout='constrained')
    if calculated:
        _tile_outlines(ax_map, composite, np.flatnonzero(_layers(composite) == int(calculated[0].box1.position[2])))
    for ax, xy, s in ((ax_map, mid, size), (ax_diff, diffs, 8)):
        points = ax.scatter(xy[kept,1], xy[kept,0], c=scores[kept], cmap='viridis', vmin=0, vmax=1, s=s,
                label='kept (n={})'.format(kept.sum()))
        for label, mask, style in categories:
            ax.scatter(xy[mask,1], xy[mask,0], s=s, label='{} (n={})'.format(label, mask.sum()), **style)
    fig.colorbar(points, ax=ax_diff, label=SCORE_LABEL + ', kept pairs', extend='min', shrink=0.8)
    _setup_map(ax_map, composite)
    removed = len(calculated) - kept.sum()
    ax_map.set_title('Filtering outcome' + ('\nthe stage model estimate is used for the {} removed pairs'.format(removed)
            if len(modeled) and removed else ''))
    ax_map.legend(fontsize=7, loc='upper right')

    _score_histogram(ax_hist, scores, _scores(background), threshold)

    # limits from the kept pairs, removed pairs can have corrections far larger than the tile size
    limits = _correction_axes(ax_diff, diffs[kept], '')
    outside = _outside(diffs, limits)
    ax_diff.set_title('Correction from stage offset' + (' ({} pairs outside view)'.format(outside) if outside else ''))
    ax_diff.legend(fontsize=7, loc='upper right')

    fig.suptitle('Filtered constraints ' + name + '\n' + _medians_text(medians) + ', from kept pairs', fontsize=12)
    _save(plt, fig, path)


def _pair_matrix(composite, consts, value_func, reduce=np.median):
    layer_index = {layer: i for i, layer in enumerate(_labels(composite, None))}
    groups = {}
    for c in consts:
        key = tuple(sorted((layer_index[int(c.box1.position[2])], layer_index[int(c.box2.position[2])])))
        value = value_func(c)
        if value is not None and np.isfinite(value):
            groups.setdefault(key, []).append(value)
    grid = np.full((len(layer_index), len(layer_index)), np.nan)
    for (i, j), values in groups.items():
        grid[i, j] = reduce(values)
    return grid


def _matrix_heatmap(fig, ax, grid, cycle_labels, title, label, fmt, **imshow_kwargs):
    image = ax.imshow(grid, cmap=imshow_kwargs.pop('cmap', 'viridis'), **imshow_kwargs)
    ax.set_xticks(range(len(cycle_labels)))
    ax.set_xticklabels(cycle_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cycle_labels)))
    ax.set_yticklabels(cycle_labels, fontsize=8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                dark = image.norm(grid[i, j]) < 0.6
                ax.text(j, i, fmt.format(grid[i, j]), ha='center', va='center', fontsize=6, color='white' if dark else 'black')
    ax.set_xlabel('cycle')
    ax.set_ylabel('cycle')
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=label, shrink=0.8)


def _cycle_names(composite, cycle_labels):
    return list(_labels(composite, cycle_labels).values())


def plot_presolve(composite, constraints, path, cycle_labels=None):
    """ Number of constraints and median score between each pair of cycles, for all
    constraints used when solving. """
    plt = _plt()
    constraints = list(constraints)
    names = _cycle_names(composite, cycle_labels)
    measured = [c for c in constraints if not is_modeled(c)]
    counts = _pair_matrix(composite, constraints, lambda c: 1, reduce=len)
    modeled_counts = _pair_matrix(composite, [c for c in constraints if is_modeled(c)], lambda c: 1, reduce=len)
    scores = _pair_matrix(composite, measured, lambda c: c.score)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6.5), layout='constrained')
    _matrix_heatmap(fig, ax1, counts, names, 'All constraints used for solving (n={})'.format(len(constraints)),
            'number of constraints', '{:.0f}')
    _matrix_heatmap(fig, ax2, modeled_counts, names, 'Of which modeled from the stage model (n={})'.format(len(constraints) - len(measured)),
            'number of modeled constraints', '{:.0f}')
    _matrix_heatmap(fig, ax3, scores, names, 'Median score of aligned constraints', SCORE_LABEL, '{:.2f}', vmin=0, vmax=1)
    fig.suptitle('Constraints before solving', fontsize=14)
    _save(plt, fig, path)


def _tile_maps(fig, axes, composite, values, titles, label, vmax, cmap):
    layers = _layers(composite)
    centers = composite.boxes.centers
    for ax, layer, title in zip(axes.flat, sorted(set(layers)), titles):
        indices = np.flatnonzero(layers == layer)
        _tile_outlines(ax, composite, indices)
        points = ax.scatter(centers[indices,1], centers[indices,0], c=values[indices], s=_marker_size(len(indices)),
                cmap=cmap, vmin=0, vmax=vmax)
        _setup_map(ax, composite)
        ax.set_title(title, fontsize=10)
    for ax in axes.flat[len(titles):]:
        ax.set_visible(False)
    fig.colorbar(points, ax=axes, label=label, shrink=0.6, extend='max')


def fit_affine(initial, solved):
    """ Least squares affine transform from initial to solved positions (n, 2), returns
    the residual shift (n, 2), the scale (sqrt of the determinant) and rotation in degrees """
    design = np.concatenate([initial, np.ones((len(initial), 1))], axis=1)
    transform, *_ = np.linalg.lstsq(design, solved, rcond=None)
    matrix = transform[:2].T
    scale = np.sqrt(abs(np.linalg.det(matrix)))
    rotation = np.degrees(np.arctan2(matrix[1,0], matrix[0,0]))
    return solved - design @ transform, scale, rotation


def plot_solved_positions(composite, initial_positions, path, cycle_labels=None):
    """ How far each tile moved from its stage position when solving, after removing the
    best fitting affine transform of each cycle (the offset, scale and rotation of the
    stage coordinates of the cycle), so only the corrections of individual tiles remain. """
    plt = _plt()
    names = _cycle_names(composite, cycle_labels)
    layers = _layers(composite)
    initial = np.asarray(initial_positions, dtype=float)[:,:2]
    solved = composite.boxes.positions[:,:2].astype(float)
    distance = np.zeros(len(solved))
    titles = []
    for layer, name in zip(sorted(set(layers)), names):
        mask = layers == layer
        residual, scale, rotation = fit_affine(initial[mask], solved[mask])
        distance[mask] = np.linalg.norm(residual, axis=1)
        titles.append('{}: scale {:.4f}, rotation {:.2f}°\nmedian {:.1f} px, max {:.1f} px'.format(
                name, scale, rotation, np.median(distance[mask]), distance[mask].max()))
    vmax = max(float(np.percentile(distance, 99)), 1)
    nrows, ncols = _grid(len(names))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols + 1.5, 4.8*nrows), squeeze=False, sharex=True, sharey=True,
            layout='constrained')
    _tile_maps(fig, axes, composite, distance, titles, 'shift from stage position after removing the affine fit (px)', vmax, 'viridis')
    fig.suptitle('Solved tile positions: shift of each tile from its stage position, after removing the best fitting\n'
            'affine transform of each cycle (scale and rotation of the stage coordinates are given per cycle)', fontsize=13)
    _save(plt, fig, path)


def plot_solved_accuracy(composite, constraints, path, cycle_labels=None):
    """ Residuals of all constraints after solving: median per pair of cycles, their
    distribution, and the largest residual of the constraints of each tile. """
    plt = _plt()
    constraints = list(constraints)
    names = _cycle_names(composite, cycle_labels)
    residuals = np.linalg.norm(_differences(constraints), axis=1)
    modeled = np.array([is_modeled(c) for c in constraints], dtype=bool)

    tile_max = np.zeros(len(composite.boxes))
    for c, residual in zip(constraints, residuals):
        tile_max[c.index1] = max(tile_max[c.index1], residual)
        tile_max[c.index2] = max(tile_max[c.index2], residual)

    nrows, ncols = _grid(len(names))
    fig = plt.figure(figsize=(4.5*ncols + 1.5, 4.5*nrows + 6.5), layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=[6.5, 4.5*nrows])
    ax_matrix, ax_hist = top.subplots(1, 2)
    grid = _pair_matrix(composite, constraints, lambda c: np.linalg.norm(c.difference))
    _matrix_heatmap(top, ax_matrix, grid, names, 'Median residual per cycle pair', RESIDUAL_LABEL, '{:.2f}',
            cmap='magma', vmin=0, vmax=MAX_RESIDUAL)

    bins = np.logspace(-2, np.log10(max(residuals.max() if len(residuals) else 1, 10)), 80)
    for mask, label, color in ((~modeled, 'aligned', 'tab:blue'), (modeled, 'modeled from stage model', 'tab:orange')):
        if mask.any():
            ax_hist.hist(np.clip(residuals[mask], bins[0], None), bins=bins, alpha=0.7, color=color,
                    label='{} (n={}, median {:.2f} px, {:.1%} > 2 px)'.format(label, mask.sum(), np.median(residuals[mask]), np.mean(residuals[mask] > 2)))
    ax_hist.axvline(2, color='0.4', linestyle=':', linewidth=1)
    ax_hist.set_xscale('log')
    ax_hist.set_yscale('log')
    ax_hist.set_xlabel(RESIDUAL_LABEL + ', values below 0.01 in the first bin')
    ax_hist.set_ylabel('number of constraints')
    ax_hist.set_title('Residual distribution')
    ax_hist.legend(fontsize=8)

    axes = bottom.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True)
    layers = _layers(composite)
    titles = ['{} (median {:.2f} px)'.format(name, np.median(tile_max[layers == layer]))
            for layer, name in zip(sorted(set(layers)), names)]
    _tile_maps(bottom, axes, composite, tile_max, titles, 'largest residual of the tile\'s constraints (px)', MAX_RESIDUAL, 'magma')
    fig.suptitle('Residuals after solving', fontsize=14)
    _save(plt, fig, path)


def plot_rotation(cycle_table, tile_table, path, threshold, reference='median', tile_size=2304, phenotype_cycles=()):
    """ Rotation of the image content of each cycle, measured on sample tiles before
    alignment (see starcall.rotation): the rotation of each tile relative to the reference,
    what it means in pixels at the tile edge, and the scale of phenotype cycles.
    cycle_table and tile_table are pandas DataFrames as written by detect_rotation. """
    plt = _plt()
    cycles = list(cycle_table['cycle'].astype(str))
    names = ['cycle' + c for c in cycles]
    x = np.arange(len(cycles))
    measured = cycle_table.set_index(cycle_table['cycle'].astype(str))
    zero = float(np.nanmedian(measured['measured_deg'] - measured['relative_deg']))
    rotated = measured['status'] == 'rotated'
    phenotype = [c for c in cycles if c in set(map(str, phenotype_cycles))]

    ncols = 3 if phenotype else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5.5), layout='constrained',
            width_ratios=[1.4, 1.4, 0.6][:ncols])
    ax = axes[0]
    ax.axhspan(-threshold, threshold, color='0.9', label='±{} deg threshold (not corrected)'.format(threshold))
    for i, cycle in enumerate(cycles):
        angles = tile_table.loc[tile_table['cycle'].astype(str) == cycle, 'angle_deg'].to_numpy(dtype=float) - zero
        angles = angles[np.isfinite(angles)]
        jitter = (np.arange(len(angles)) % 7 - 3) * 0.04
        color = 'tab:red' if rotated[cycle] else 'tab:blue'
        ax.scatter(i + jitter, angles, s=10, color=color, alpha=0.6)
        ax.plot([i - 0.3, i + 0.3], [measured.loc[cycle, 'relative_deg']] * 2, color='black', linewidth=1.5)
        if rotated[cycle]:
            ax.scatter(i, measured.loc[cycle, 'residual_after_deg'], marker='D', s=30, color='tab:green', zorder=3)
    ax.scatter([], [], s=10, color='tab:blue', label='tile, not corrected')
    ax.scatter([], [], s=10, color='tab:red', label='tile, cycle rotated back')
    ax.plot([], [], color='black', label='median of the cycle')
    ax.scatter([], [], marker='D', s=30, color='tab:green', label='median after correction')
    ax.axhline(0, color='0.5', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylabel('rotation relative to reference (deg)')
    ax.set_title('Rotation of each sampled tile')
    ax.legend(fontsize=7, loc='best')

    # rotation by theta moves a pixel at distance d from the tile centre by d * theta
    edge = tile_size / 2 * np.sqrt(2) * np.radians(1)
    before = np.abs(measured['relative_deg'].to_numpy(dtype=float)) * edge
    after = np.abs(measured['residual_after_deg'].to_numpy(dtype=float)) * edge
    ax = axes[1]
    ax.bar(x - 0.2, before, width=0.4, color=['tab:red' if r else 'tab:blue' for r in rotated], label='before correction')
    ax.bar(x + 0.2, after, width=0.4, color='tab:green', label='after correction')
    ax.axhline(threshold * edge, color='black', linestyle='--', linewidth=1, label='threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylabel('displacement at the tile corner (px of the cycle\'s own images)')
    ax.set_title('What the rotation means at the tile corners ({:.0f} px from the centre)'.format(tile_size / 2 * np.sqrt(2)), fontsize=10)
    ax.legend(fontsize=7)

    if phenotype:
        ax = axes[2]
        scales = [measured.loc[c, 'scale'] for c in phenotype]
        ax.bar(range(len(phenotype)), [(s - 1) * 100 for s in scales], color='tab:purple')
        ax.axhspan(-0.5, 0.5, color='0.9', zorder=0, label='±0.5%')
        ax.axhline(0, color='0.5', linewidth=0.5)
        ax.set_xticks(range(len(phenotype)))
        ax.set_xticklabels(['cycle' + c for c in phenotype])
        ax.set_ylabel('scale relative to nominal magnification (%)')
        ax.set_title('Phenotype scale check\n(not corrected)', fontsize=10)
        ax.legend(fontsize=7)

    corrected = [n for n, r in zip(names, rotated) if r]
    fig.suptitle('Rotation of the image content of each cycle (reference: {}, {:.3f} deg from the first cycle); corrected: {}'.format(
            'median of sequencing cycles' if reference == 'median' else 'cycle' + str(reference), zero,
            ', '.join(corrected) if corrected else 'none'), fontsize=12)
    _save(plt, fig, path)
