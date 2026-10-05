""" Diagnostics for dot detection: the summary format written by detect_dots_cpu / detect_dots_gpu
with diagnostics=prefix, and QC plots made from it.

The summary (prefix.summary.csv) is a long table with columns
    stat, cycle, channel, bin_lo, bin_hi, value
that only holds quantities that can be added up across tiles: counts on fixed
histogram bins and n / sum / sum of squares. Summing the summaries of all tiles of a
well (merge_summary) gives the summary of the well, so the same plots can be made for a
tile or a whole well. Per tile values (z score means and stds, the blob_log threshold)
are kept per tile.

prefix.images.npz holds crops of the intermediate images of one tile, used for the
image panels of the tile plots.
"""
import os
import re
import numpy as np
import pandas


class Bins:
    """ Fixed histogram bins, uniform after a transform (eg log10), so histograms from
    different tiles and runs can be added. Values outside the range go in the end bins. """

    def __init__(self, name, lo, hi, num, forward=None, inverse=None):
        self.name = name
        self.lo, self.hi, self.num = lo, hi, num
        self.forward = forward or (lambda x: x)
        self.inverse = inverse or (lambda x: x)

    def edges(self):
        return self.inverse(np.linspace(self.lo, self.hi, self.num + 1))

    def hist(self, values):
        values = np.asarray(values, dtype=np.float64).ravel()
        values = values[np.isfinite(values)]
        index = np.floor((self.forward(values) - self.lo) * (self.num / (self.hi - self.lo)))
        index = np.clip(index, 0, self.num - 1).astype(np.int64)
        return np.bincount(index, minlength=self.num)


def _symlog(x):
    return np.sign(x) * np.log10(1 + np.abs(x))

def _symexp(x):
    return np.sign(x) * (10 ** np.abs(x) - 1)

RAW_BINS = Bins('raw', 0, np.log10(65536), 200,
        lambda x: np.log10(np.clip(x, 1, None)), lambda x: 10 ** x)
DOG_BINS = Bins('dog', -4.5, 4.5, 180, _symlog, _symexp)
Z_BINS = Bins('z', -5, 20, 250)
GREY_BINS = Bins('grey', -4, 3, 140,
        lambda x: np.log10(np.clip(x, 1e-4, None)), lambda x: 10 ** x)
CHASTITY_BINS = Bins('chastity', 0.5, 1.0, 50)

# summary stats that are summed when merging tiles, the rest are kept per tile
ADDITIVE_SUFFIXES = ('_n', '_sum', '_sumsq', '_hist', '_count')
ADDITIVE_STATS = ('n_dots', 'valid_pixels')
SUMMARY_COLUMNS = ['stat', 'cycle', 'channel', 'bin_lo', 'bin_hi', 'value']


def is_additive(stat):
    return stat in ADDITIVE_STATS or stat.endswith(ADDITIVE_SUFFIXES)


#####################################################################
# Writing (used by dotdetection._collect_diagnostics)
#####################################################################

class SummaryWriter:
    def __init__(self):
        self.parts = []

    def scalar(self, stat, value, cycle=-1, channel=''):
        self.parts.append(pandas.DataFrame({'stat': [stat], 'cycle': [cycle], 'channel': [channel],
                'bin_lo': [np.nan], 'bin_hi': [np.nan], 'value': [float(value)]}))

    def moments(self, prefix, values, cycle=-1, channel=''):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        self.scalar(prefix + '_n', values.size, cycle, channel)
        self.scalar(prefix + '_sum', values.sum(), cycle, channel)
        self.scalar(prefix + '_sumsq', (values * values).sum(), cycle, channel)

    def hist(self, stat, bins, values, cycle=-1, channel=''):
        self.counts(stat, bins.edges(), bins.hist(values), cycle, channel)

    def counts(self, stat, edges, counts, cycle=-1, channel=''):
        edges = np.asarray(edges, dtype=float)
        self.parts.append(pandas.DataFrame({'stat': stat, 'cycle': cycle, 'channel': channel,
                'bin_lo': edges[:-1], 'bin_hi': edges[1:], 'value': np.asarray(counts, dtype=float)}))

    def table(self):
        return pandas.concat(self.parts, ignore_index=True)[SUMMARY_COLUMNS]


#####################################################################
# Reading and merging
#####################################################################

def tile_name(path):
    """ a short name for the tile a summary is from, eg well1_grid5/tile01x04y """
    parts = os.path.normpath(path).split(os.sep)
    return '/'.join(parts[-3:-1]) if len(parts) >= 3 else parts[0]


def load_summary(paths, names=None):
    """ Concatenates tile summaries, adding a tile column """
    if isinstance(paths, str):
        paths = [paths]
    names = names or [tile_name(path) for path in paths]
    tables = []
    for path, name in zip(paths, names):
        table = pandas.read_csv(path, keep_default_na=False, na_values=[''])
        if len(table) == 0:
            # tiles with an empty image
            continue
        table['channel'] = table['channel'].fillna('').astype(str)
        table['tile'] = name
        tables.append(table)
    if not tables:
        return pandas.DataFrame(columns=SUMMARY_COLUMNS + ['tile'])
    return pandas.concat(tables, ignore_index=True)


def merge_summary(table):
    """ Sums the additive stats over tiles (tile='all'). Per tile stats, and the per tile
    n_dots and valid_pixels totals, are kept with their tile names """
    additive = table['stat'].map(is_additive)
    keys = ['stat', 'cycle', 'channel', 'bin_lo', 'bin_hi']
    merged = table[additive].fillna({'bin_lo': -np.inf, 'bin_hi': -np.inf}).groupby(keys, sort=False, as_index=False)['value'].sum()
    merged[['bin_lo', 'bin_hi']] = merged[['bin_lo', 'bin_hi']].replace(-np.inf, np.nan)
    merged['tile'] = 'all'
    # per tile totals are also kept, for plots per tile
    per_tile = table[table['stat'].isin(ADDITIVE_STATS)]
    return pandas.concat([merged, table[~additive], per_tile], ignore_index=True)


def ensure_merged(table):
    if 'tile' not in table.columns:
        table = table.assign(tile='all')
    if (table['tile'] == 'all').any():
        return table
    return merge_summary(table)


def num_tiles(table):
    return len(set(table['tile']) - {'all'})


def _select(table, stat, tile='all'):
    rows = table[table['stat'] == stat]
    if tile is not None and is_additive(stat):
        rows = rows[rows['tile'] == tile]
    return rows


def moments(table, prefix):
    """ DataFrame of cycle, channel, n, mean, std from the prefix_n/_sum/_sumsq stats """
    table = ensure_merged(table)
    parts = {}
    for suffix in ('n', 'sum', 'sumsq'):
        rows = _select(table, prefix + '_' + suffix)
        parts[suffix] = rows.set_index(['cycle', 'channel'])['value']
    result = pandas.DataFrame(parts)
    result['mean'] = result['sum'] / result['n']
    result['std'] = np.sqrt(np.maximum(result['sumsq'] / result['n'] - result['mean'] ** 2, 0))
    return result.reset_index()


def histograms(table, stat):
    """ {(cycle, channel): (bin_lo, bin_hi, counts)} """
    table = ensure_merged(table)
    result = {}
    for (cycle, channel), rows in _select(table, stat).groupby(['cycle', 'channel'], sort=True):
        rows = rows.sort_values('bin_lo')
        result[cycle, channel] = (rows['bin_lo'].values, rows['bin_hi'].values, rows['value'].values)
    return result


def sum_histograms(hists, keys):
    keys = [key for key in keys if key in hists]
    if not keys:
        return None
    lo, hi, _ = hists[keys[0]]
    return lo, hi, np.sum([hists[key][2] for key in keys], axis=0)


def hist_quantiles(hist, quantiles):
    """ Quantiles from a histogram, interpolating linearly inside bins """
    lo, hi, counts = hist
    total = counts.sum()
    if total <= 0:
        return np.full(len(quantiles), np.nan)
    cumulative = np.concatenate([[0], np.cumsum(counts)]) / total
    edges = np.concatenate([lo, hi[-1:]])
    return np.interp(quantiles, cumulative, edges)


BOX_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

def hist_box(hist, label):
    """ matplotlib bxp stats from a histogram: whiskers at 5 and 95 percent """
    q05, q25, q50, q75, q95 = hist_quantiles(hist, BOX_QUANTILES)
    return dict(label=label, whislo=q05, q1=q25, med=q50, q3=q75, whishi=q95, fliers=[])


def per_tile_values(table, stat):
    """ DataFrame tile, cycle, channel, value for a per tile stat """
    rows = table[(table['stat'] == stat) & (table['tile'] != 'all')]
    return rows[['tile', 'cycle', 'channel', 'value']]


def load_images(path):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as data:
        images = {key: data[key] for key in data.files}
    return images or None


#####################################################################
# Plots. Each returns (figure, DataFrame of the values drawn)
#####################################################################

def _channels(table):
    chans = [chan for chan in dict.fromkeys(table.loc[table['channel'] != '', 'channel'])]
    return chans


def _cycles(table):
    return sorted(int(cycle) for cycle in table.loc[table['cycle'] >= 0, 'cycle'].unique())


def _cycle_labels(cycles, labels):
    if labels is not None and len(labels) >= len(cycles):
        return [str(labels[cycle]) for cycle in cycles]
    return [str(cycle) for cycle in cycles]


def _channel_colors(channels):
    default = {'G': 'tab:green', 'T': 'tab:red', 'A': 'tab:blue', 'C': 'tab:orange'}
    fallback = ['tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    return {chan: default.get(chan, fallback[i % len(fallback)]) for i, chan in enumerate(channels)}


def _data(rows):
    columns = ['panel', 'cycle', 'channel', 'quantity', 'value']
    return pandas.DataFrame(rows, columns=columns)


def _title(table, name):
    tiles = num_tiles(table)
    return '{} ({})'.format(name, 'well, {} tiles'.format(tiles) if tiles > 1 else 'tile')


def _heatmap(ax, grid, cycle_labels, channels, title, fmt='{:.2f}', cmap='viridis'):
    import matplotlib.pyplot as plt
    image = ax.imshow(grid, aspect='auto', cmap=cmap)
    ax.set_xticks(range(len(cycle_labels)))
    ax.set_xticklabels(cycle_labels, rotation=90)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, fmt.format(grid[i, j]), ha='center', va='center', color='white', size=6)
    ax.set_xlabel('cycle')
    ax.set_title(title)
    plt.colorbar(image, ax=ax)


def plot_intensity_trends(table, images=None, cycle_labels=None):
    """ raw intensity percentiles per cycle and channel, raw intensity at dots, base call fractions """
    import matplotlib.pyplot as plt
    table = ensure_merged(table)
    channels, cycles = _channels(table), _cycles(table)
    labels = _cycle_labels(cycles, cycle_labels)
    colors = _channel_colors(channels)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    raw = histograms(table, 'raw_hist')
    for chan in channels:
        for quantile, style in ((0.5, '-'), (0.99, '--'), (0.999, ':')):
            values = [hist_quantiles(raw[cycle, chan], [quantile])[0] if (cycle, chan) in raw else np.nan for cycle in cycles]
            axes[0].plot(labels, values, style, color=colors[chan],
                    label='{} p{:g}'.format(chan, quantile * 100))
            rows += [('raw_percentiles', cycle, chan, 'p{:g}'.format(quantile * 100), value) for cycle, value in zip(cycles, values)]
    axes[0].set_yscale('log')
    axes[0].set_title('raw pixel intensity percentiles')
    axes[0].set_xlabel('cycle')
    axes[0].legend(fontsize=6, ncol=len(channels), loc='center left')

    dot_raw = histograms(table, 'dot_raw_hist')
    for chan in channels:
        values = [hist_quantiles(dot_raw[cycle, chan], [0.5])[0] if (cycle, chan) in dot_raw else np.nan for cycle in cycles]
        axes[1].plot(labels, values, '-o', color=colors[chan], label=chan, markersize=3)
        rows += [('dot_raw_median', cycle, chan, 'median', value) for cycle, value in zip(cycles, values)]
    axes[1].set_yscale('log')
    axes[1].set_title('median raw intensity of the called channel at dots')
    axes[1].set_xlabel('cycle')
    axes[1].legend(fontsize=8)

    called = _select(table, 'called_count').pivot_table(index='cycle', columns='channel', values='value', aggfunc='sum')
    called = called.reindex(index=cycles, columns=channels).fillna(0)
    fractions = called.div(called.sum(axis=1).replace(0, np.nan), axis=0)
    bottom = np.zeros(len(cycles))
    for chan in channels:
        axes[2].bar(labels, fractions[chan].values, bottom=bottom, color=colors[chan], label=chan)
        bottom += fractions[chan].values
        rows += [('called_fraction', cycle, chan, 'fraction', value) for cycle, value in zip(cycles, fractions[chan].values)]
    axes[2].axhline(0.25, color='k', lw=0.5, ls='--')
    axes[2].set_title('fraction of dots called as each base')
    axes[2].set_xlabel('cycle')
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.tick_params(axis='x', rotation=90)
    fig.suptitle(_title(table, 'Intensity trends'))
    fig.tight_layout()
    return fig, _data(rows)


def plot_dog_effect(table, images=None, cycle_labels=None):
    """ background removed by the difference of gaussians filter """
    import matplotlib.pyplot as plt
    table = ensure_merged(table)
    channels, cycles = _channels(table), _cycles(table)
    labels = _cycle_labels(cycles, cycle_labels)
    colors = _channel_colors(channels)
    rows = []
    nrows = 3 if images is not None else 1
    fig = plt.figure(figsize=(20, 5 * nrows))
    grid = fig.add_gridspec(nrows, 4)
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]

    background = moments(table, 'bg')
    for chan in channels:
        values = background[background['channel'] == chan].set_index('cycle').reindex(cycles)['mean'].values
        axes[0].plot(labels, values, '-o', color=colors[chan], label=chan, markersize=3)
        rows += [('background_mean', cycle, chan, 'mean', value) for cycle, value in zip(cycles, values)]
    axes[0].set_title('mean background removed (raw - DoG)')
    axes[0].set_xlabel('cycle')
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].legend(fontsize=8)

    for ax, stat, name, xscale in ((axes[1], 'raw_hist', 'raw', 'log'), (axes[2], 'dog_hist', 'DoG', 'symlog')):
        hists = histograms(table, stat)
        for chan in channels:
            hist = sum_histograms(hists, [(cycle, chan) for cycle in cycles])
            if hist is None:
                continue
            lo, hi, counts = hist
            centers = (lo + hi) / 2
            density = counts / counts.sum() / (hi - lo)
            ax.plot(centers, density, color=colors[chan], label=chan)
            rows += [(name + '_density', -1, chan, center, value) for center, value in zip(centers, density)]
        ax.set_xscale(xscale)
        ax.set_yscale('log')
        ax.set_title('{} pixel values, all cycles'.format(name))
        ax.legend(fontsize=8)

    raw, dog = moments(table, 'raw'), moments(table, 'dog')
    merged = raw.merge(dog, on=['cycle', 'channel'], suffixes=('_raw', '_dog'))
    merged['var_removed'] = 1 - merged['std_dog'] ** 2 / merged['std_raw'] ** 2
    heat = merged.pivot(index='channel', columns='cycle', values='var_removed').reindex(index=channels, columns=cycles)
    _heatmap(axes[3], heat.values, labels, channels, 'fraction of variance removed by DoG')
    rows += [('var_removed', row.cycle, row.channel, 'fraction', row.var_removed) for row in merged.itertuples()]

    if images is not None:
        crop_raw, crop_dog = images['crop_raw'].astype(np.float32), images['crop_dog'].astype(np.float32)
        crop_bg = np.nan_to_num(crop_raw) - crop_dog
        raw_means = raw.groupby('channel')['mean'].mean()
        chan = raw_means.idxmax()
        chan_index = list(images['channels']).index(chan)
        show_cycles = sorted(set([cycles[0], cycles[len(cycles) // 2], cycles[-1]]))
        sub = grid[1:, :].subgridspec(len(show_cycles), 3)
        for row_index, cycle in enumerate(show_cycles):
            for col, (name, crop) in enumerate((('raw', crop_raw), ('background', crop_bg), ('DoG', crop_dog))):
                ax = fig.add_subplot(sub[row_index, col])
                plane = crop[cycle, chan_index]
                vmin, vmax = np.nanpercentile(plane, [1, 99.5])
                image = ax.imshow(plane, cmap='gray', vmin=vmin, vmax=vmax)
                ax.set_title('cycle {} {} {}'.format(labels[cycles.index(cycle)], chan, name), fontsize=9)
                ax.axis('off')
                plt.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(_title(table, 'Difference of gaussian background correction'))
    fig.tight_layout()
    return fig, _data(rows)


def plot_zscore(table, images=None, cycle_labels=None, cycle=None):
    """ the per plane z score parameters and their effect """
    import matplotlib.pyplot as plt
    table = ensure_merged(table)
    channels, cycles = _channels(table), _cycles(table)
    labels = _cycle_labels(cycles, cycle_labels)
    colors = _channel_colors(channels)
    cycle = cycles[0] if cycle is None else cycle
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    many = num_tiles(table) > 1

    for ax, stat, name in ((axes[0,0], 'z_mean', 'z score mean (DoG)'), (axes[0,1], 'z_std', 'z score std (DoG)')):
        values = per_tile_values(table, stat)
        grid = values.pivot_table(index='channel', columns='cycle', values='value', aggfunc='mean').reindex(index=channels, columns=cycles)
        _heatmap(ax, grid.values, labels, channels, name + (', mean over tiles' if many else ''),
                fmt='{:.1f}' if stat == 'z_std' else '{:.2g}')
        rows += [(stat, row.cycle, row.channel, row.tile, row.value) for row in values.itertuples()]

    values = per_tile_values(table, 'z_std')
    for chan in channels:
        chan_values = values[values['channel'] == chan]
        if many:
            for tile, tile_values in chan_values.groupby('tile'):
                series = tile_values.set_index('cycle').reindex(cycles)['value'].values
                axes[0,2].plot(labels, series, color=colors[chan], alpha=0.2, lw=0.7)
        series = chan_values.groupby('cycle')['value'].mean().reindex(cycles).values
        axes[0,2].plot(labels, series, '-o', color=colors[chan], label=chan, lw=2, markersize=3)
    axes[0,2].set_title('z score std per cycle' + (' (thin lines: tiles)' if many else ''))
    axes[0,2].set_xlabel('cycle')
    axes[0,2].tick_params(axis='x', rotation=90)
    axes[0,2].legend(fontsize=8)

    label = labels[cycles.index(cycle)]
    for ax, stat, name, xscale in ((axes[1,0], 'dog_hist', 'DoG before z score', 'symlog'), (axes[1,1], 'z_hist', 'after z score', 'linear')):
        hists = histograms(table, stat)
        for chan in channels:
            if (cycle, chan) not in hists:
                continue
            lo, hi, counts = hists[cycle, chan]
            centers = (lo + hi) / 2
            density = counts / counts.sum() / (hi - lo)
            ax.plot(centers, density, color=colors[chan], label=chan)
            rows += [(stat + '_density', cycle, chan, center, value) for center, value in zip(centers, density)]
        ax.set_xscale(xscale)
        ax.set_yscale('log')
        ax.set_title('pixel values {}, cycle {}'.format(name, label))
        ax.legend(fontsize=8)

    # dot values per channel before and after, from the histograms over all cycles
    ax = axes[1,2]
    ax2 = ax.twinx()
    boxes_before, boxes_after = [], []
    for stat, boxes in (('dot_dog_hist', boxes_before), ('dot_z_hist', boxes_after)):
        hists = histograms(table, stat)
        for chan in channels:
            hist = sum_histograms(hists, [(cyc, chan) for cyc in cycles])
            if hist is None:
                continue
            box = hist_box(hist, chan)
            boxes.append(box)
            rows += [(stat + '_box', -1, chan, key, box[key]) for key in ('whislo', 'q1', 'med', 'q3', 'whishi')]
    positions = np.arange(len(boxes_before))
    if boxes_before:
        ax.bxp(boxes_before, positions=positions - 0.2, widths=0.3, showfliers=False,
                boxprops=dict(color='tab:gray'), medianprops=dict(color='k'))
    if boxes_after:
        ax2.bxp(boxes_after, positions=positions + 0.2, widths=0.3, showfliers=False,
                boxprops=dict(color='tab:purple'), medianprops=dict(color='tab:purple'))
    ax.set_xticks(positions)
    ax.set_xticklabels([box['label'] for box in boxes_before])
    ax.set_ylabel('DoG value at dots (gray)')
    ax2.set_ylabel('z score at dots (purple)')
    ax.set_title('values at dots per channel, all cycles')

    fig.suptitle(_title(table, 'Z score normalization'))
    fig.tight_layout()
    return fig, _data(rows)


def _tile_position(name):
    match = re.search(r'tile(\d+)x(\d+)y', name)
    return (int(match.group(2)), int(match.group(1))) if match else None


def plot_dots(table, images=None, cycle_labels=None):
    """ dot sizes, chastity, the blob_log threshold and where dots are """
    import matplotlib.pyplot as plt
    import matplotlib.colors
    table = ensure_merged(table)
    channels, cycles = _channels(table), _cycles(table)
    labels = _cycle_labels(cycles, cycle_labels)
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    many = num_tiles(table) > 1

    sigmas = _select(table, 'sigma_count').groupby('bin_lo')['value'].sum()
    radius = sigmas.index.values * np.sqrt(2)
    axes[0,0].bar(['{:.2f}\n(r {:.1f})'.format(sigma, r) for sigma, r in zip(sigmas.index, radius)], sigmas.values)
    axes[0,0].set_title('dots per blob_log sigma (radius = sigma * sqrt 2)')
    axes[0,0].set_ylabel('dots')
    rows += [('sigma_count', -1, '', sigma, value) for sigma, value in sigmas.items()]

    chastity = histograms(table, 'chastity_hist')
    boxes = [hist_box(chastity[cycle, ''], label) for cycle, label in zip(cycles, labels) if (cycle, '') in chastity]
    if boxes:
        axes[0,1].bxp(boxes, showfliers=False)
    axes[0,1].axhline(0.6, color='r', lw=0.8, ls='--')
    axes[0,1].set_title('chastity per cycle (top / (top + second)), 0.6 threshold')
    axes[0,1].set_xlabel('cycle')
    axes[0,1].tick_params(axis='x', rotation=90)
    for cycle, box in zip(cycles, boxes):
        rows += [('chastity_box', cycle, '', key, box[key]) for key in ('whislo', 'q1', 'med', 'q3', 'whishi')]

    grey = histograms(table, 'grey_hist')
    if (-1, '') in grey:
        lo, hi, counts = grey[-1, '']
        centers = (lo + hi) / 2
        axes[0,2].plot(centers, counts, color='k')
        rows += [('grey_hist', -1, '', center, value) for center, value in zip(centers, counts)]
    thresholds = per_tile_values(table, 'threshold')['value'].values
    for threshold in thresholds:
        axes[0,2].axvline(threshold, color='r', lw=0.8, alpha=0.3 if many else 1)
    rows += [('threshold', -1, '', tile, value) for tile, value in per_tile_values(table, 'threshold')[['tile', 'value']].values]
    axes[0,2].set_xscale('log')
    axes[0,2].set_yscale('log')
    axes[0,2].set_title('grey image values and blob_log threshold' + (' per tile' if many else ''))

    counts = table[(table['stat'].isin(['n_dots', 'valid_pixels'])) & (table['tile'] != 'all')]
    per_tile_counts = counts.pivot_table(index='tile', columns='stat', values='value', aggfunc='sum')
    per_tile_counts['dots_per_mpx'] = per_tile_counts['n_dots'] / per_tile_counts['valid_pixels'] * 1e6
    rows += [('dots_per_megapixel', -1, '', tile, value) for tile, value in per_tile_counts['dots_per_mpx'].items()]
    positions = {tile: _tile_position(tile) for tile in per_tile_counts.index}
    if many and all(position is not None for position in positions.values()):
        size = max(max(position) for position in positions.values()) + 1
        grid = np.full((size, size), np.nan)
        for tile, (y, x) in positions.items():
            grid[y, x] = per_tile_counts.loc[tile, 'dots_per_mpx']
        image = axes[1,0].imshow(grid, cmap='viridis')
        for (y, x), value in np.ndenumerate(grid):
            if np.isfinite(value):
                axes[1,0].text(x, y, '{:.0f}'.format(value), ha='center', va='center', color='white', size=7)
        axes[1,0].set_xticks(range(size))
        axes[1,0].set_yticks(range(size))
        axes[1,0].set_xlabel('tile x')
        axes[1,0].set_ylabel('tile y')
        axes[1,0].set_title('dots per megapixel of imaged area, per tile')
        plt.colorbar(image, ax=axes[1,0])
    elif images is not None and 'density_hist2d' in images:
        density = images['density_hist2d'].astype(float)
        density[images['density_mask']] = np.nan
        image = axes[1,0].imshow(density, cmap='viridis')
        axes[1,0].set_title('dots per bin ({} x {} bins over the tile)'.format(*density.shape))
        plt.colorbar(image, ax=axes[1,0])
    else:
        axes[1,0].bar(range(len(per_tile_counts)), per_tile_counts['dots_per_mpx'].values)
        axes[1,0].set_title('dots per megapixel of imaged area')

    if images is not None:
        crop = images['crop_grey'].astype(np.float32)
        axes[1,1].imshow(crop, cmap='gray', vmax=np.nanpercentile(crop, 99.5))
        dots = images['crop_dots']
        if len(dots):
            norm = matplotlib.colors.Normalize(*images['sigma_params'][:2])
            axes[1,1].scatter(dots[:,1], dots[:,0], s=(dots[:,2] * np.sqrt(2) * 3) ** 2,
                    facecolors='none', edgecolors=plt.cm.plasma(norm(dots[:,2])), linewidths=0.8)
            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=axes[1,1], label='sigma')
        axes[1,1].set_title('grey image crop with dots (circle size ~ radius)')
        axes[1,1].axis('off')

        thumb = images['grey_thumbnail'].astype(np.float32)
        axes[1,2].imshow(thumb, cmap='gray', vmax=np.nanpercentile(thumb, 99.5))
        y0, x0 = images['crop_origin']
        scale = images['thumbnail_scale']
        size = images['crop_raw'].shape[-1]
        axes[1,2].add_patch(plt.Rectangle((x0 / scale, y0 / scale), size / scale, size / scale, fill=False, color='r'))
        axes[1,2].set_title('grey image (crop in red)')
        axes[1,2].axis('off')
    else:
        for ax in (axes[1,1], axes[1,2]):
            ax.axis('off')
        total = _select(table, 'n_dots')['value'].sum()
        axes[1,1].text(0.1, 0.5, '{:,.0f} dots in {} tiles'.format(total, max(num_tiles(table), 1)), fontsize=14)

    fig.suptitle(_title(table, 'Dots'))
    fig.tight_layout()
    return fig, _data(rows)


PLOTS = {
    'intensity_trends': plot_intensity_trends,
    'dog_effect': plot_dog_effect,
    'zscore': plot_zscore,
    'dots': plot_dots,
}


def plot_all(summary_paths, out_dir, images_path=None, prefix='', cycle_labels=None, names=None):
    """ Writes the 4 plots (svg) and the values drawn in each (_data.csv) to out_dir.
    With several summaries (tiles of a well) they are merged, and the merged and
    concatenated summaries are saved too. Returns the list of files written. """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    table = load_summary(summary_paths, names=names)
    written = []
    if len(table) == 0:
        # only empty tiles, write placeholders so the outputs exist
        for name in PLOTS:
            fig = plt.figure(figsize=(4, 2))
            fig.text(0.5, 0.5, 'no dots: empty image', ha='center')
            for path in (os.path.join(out_dir, prefix + name + '.svg'), os.path.join(out_dir, prefix + name + '_data.csv')):
                fig.savefig(path) if path.endswith('.svg') else _data([]).to_csv(path, index=False)
                written.append(path)
            plt.close(fig)
        return written
    merged = merge_summary(table)
    images = load_images(images_path)
    if num_tiles(table) > 1:
        for name, data in (('tile_summary', table), ('well_summary', merged)):
            path = os.path.join(out_dir, name + '.csv')
            data.to_csv(path, index=False)
            written.append(path)
    for name, func in PLOTS.items():
        fig, data = func(merged, images=images, cycle_labels=cycle_labels)
        for path, save in ((os.path.join(out_dir, prefix + name + '.svg'), lambda path: fig.savefig(path)),
                           (os.path.join(out_dir, prefix + name + '_data.csv'), lambda path: data.to_csv(path, index=False))):
            save(path)
            written.append(path)
        plt.close(fig)
    return written
