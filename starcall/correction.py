import sys
import numpy as np
import skimage.morphology
import skimage.filters
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
from scipy.optimize import nnls
from .reads import ReadsAccessor


#AML - replacing unused correction scripts with those needed to make a color crosstalk correction matrix and apply it

def get_cycle_str(i):
    if i >= 10:
        return str(i)
    else:
        return "0" + str(i)


def stack_cycles_of_dots_table(dot_table, num_cycles, sequencing_channels_order, cell_filter = True):
    """
    Turns tables with per cycle phred scores and initial sequences (determined by the max
    intensity value per dot per cycle) into a cycle-wise table for selecting cycles to use 
    in crosstalk matrix construction.

    Args:
        dot_table (pd.DataFrame): table of per dot intensities, quality information, and cell IDs (should be the output
        of rule attach_cell_ids in the starcall-workflow sequencing.smk file)
        cell_filter (bool default True): if only dots occuring in cells should be used to make the per-cycle table
        num_cycles (int): Number of cycles to be expected when table splitting
    Returns: 
        stacked_df (pd.DataFrame): Cycle level table from the input (each dot is split into cycle rows) where each
        row contains position info, original table index of the dot, intensity values, max called base, and phred score
    """

    #split the max seq into indiv letters and attach to the original table 
    df_letters = dot_table['max_seq'].str.split('', expand=True).iloc[:, 1:-1]
    col_names = ['base_cycle' + get_cycle_str(i) for i in range(num_cycles)]
    df_letters.columns = col_names
    dot_table = pd.concat([dot_table,df_letters], axis = 1)
    dot_table['cells_quality_table_index'] = dot_table.index

    #assign if in cell or not, filter out those not in cell  
    if cell_filter:
        dot_table['in_cell'] = dot_table['cell'] !=0
        dot_table = dot_table[dot_table.in_cell].copy()

    #make stacked version of table 
    common_cols = ['cells_quality_table_index','position_x', 'position_y', 'cell']
    to_stack = []
    for i in range(0, num_cycles):
        cycle_cols = ['values_cycle' + get_cycle_str(i) + f'_{nuc}' for nuc in sequencing_channels_order]
        cycle_cols = cycle_cols + ['phred_cycle' + get_cycle_str(i), 'base_cycle' + get_cycle_str(i)]
        slice = dot_table[common_cols + cycle_cols]
        #rename things? 
        rename_dict = {'values_cycle' + get_cycle_str(i) + f'_{nuc}': f'values_{nuc}' for nuc in sequencing_channels_order}
        rename_dict['phred_cycle' + get_cycle_str(i)] = 'phred'
        rename_dict['base_cycle' + get_cycle_str(i)]  = 'base'
        slice.rename(columns = rename_dict, inplace = True)
        slice['cycle'] = i
        to_stack.append(slice)

    stacked_df = pd.concat(to_stack, axis = 0)
    return stacked_df


def binned_cycle_sampler(cycle_table, num_samples, sequencing_channels_order,  phred_min = 60, num_bins = 10):
    """
    Use pandas qcut to bin the table and sample equal numbers of samples per base 
    from the bins for high quality cycles.  A percentile cutoff removes extra high 
    intensity values before samples are taken, to avoid highly saturating dots being 
    used for crosstalk correction.
    Args: 
        cycle_table (pd.DataFrame): cycle-level table of intensities per dot (should be the output of stack_cycles_of_dots_table)
        num_samples (int): how many samples to take for each base 
        percentile_cutoff (float, default 0.99): the percentile to use threshold up to before sampling
        phred_min (int, default 90): the minimum phred value to filter the cycle-level table with before sampling
        Note that 90 is the max value for phred with threshold set to 1e-9
    Returns: 
        samples (pd.DataFrame): samples from the table to use in crosstalk calculations 

        REMOVING PERCENTILE CUFOFF? 
    """

    samples = []
    for base in sequencing_channels_order:
        #look at percentiles in the values.... exclude the top 1% of values pulls off what I want? 
        #percentiles = cycle_table[cycle_table.base == base][f'values_{base}'].quantile([percentile_cutoff])
        #candidates = cycle_table[(cycle_table.base == base) & (cycle_table.phred >= phred_min) & (cycle_table[f'values_{base}'] <= percentiles[0.99])]
        candidates = cycle_table[(cycle_table.base == base) & (cycle_table.phred >= phred_min)]
        #do bin based sampling if there are more than num_samples
        if candidates.shape[0] <= num_samples:
            samples.append(candidates)
        else:
            num_samples = min(candidates.shape[0], num_samples)
            candidates['bins'] = pd.qcut(candidates[f'values_{base}'], q=num_bins, labels=['bin' + str(i) for i in list(range(0,num_bins))])
            #drawing from the bins
            counts = candidates['bins'].value_counts(sort=False)
            proportions = counts / counts.sum()
            allocations = (proportions * num_samples).round().astype(int)
            difference = num_samples - allocations.sum()
            allocations[allocations.idxmax()] += difference
            exact_sample = candidates.groupby('bins', observed=False).apply(lambda x: x.sample(n=min(allocations[x.name], len(x)), random_state=42)).reset_index(drop=True)
            samples.append(exact_sample)

    samples = pd.concat(samples, axis = 0, ignore_index=True)
    return samples


def quality_filtered_sampler(cycle_table, num_samples, sequencing_channels_order, phred_min = 60):
    """
    Sample equal numbers of samples per base over high quality cycles.  A percentile cutoff removes
    extra high intensity values before samples are taken, to avoid highly saturating dots being 
    used for crosstalk correction. No binning used since these samples are for the GMM model (not NNLS).

    Args:
        cycle_table (pd.DataFrame): cycle-level table of intensities per dot (should be the output
            of stack_cycles_of_dots_table)
        num_samples (int): how many samples to take for each base
        percentile_cutoff (float, default 0.99): the percentile to threshold up to before sampling
        phred_min (int, default 90): the minimum phred value to filter the cycle-level table with
            before sampling
    Returns:
        samples (pd.DataFrame): samples from the table to use in crosstalk calculations

    REMOVING PERCENTILE CUTOFF
    """

    samples = []
    for base in sequencing_channels_order:
        base_table = cycle_table[cycle_table.base == base]
        #cutoff = base_table[f'values_{base}'].quantile(percentile_cutoff)
        #candidates = base_table[(base_table.phred >= phred_min) & (base_table[f'values_{base}'] <= cutoff)]
        candidates = base_table[(base_table.phred >= phred_min)]
        n = min(num_samples, len(candidates))
        samples.append(candidates.sample(n=n, random_state=42))

    samples = pd.concat(samples, axis = 0, ignore_index=True)
    return samples

#crosstalk matrix creation methods 

def calculate_crosstalk_median_ratio(a: np.ndarray) -> np.ndarray:
    """Compute the median ratio of the input array to quantify crosstalk between channels.

    :param a: Input 2D array where each row represents data for a specific observation, and each
        column corresponds to a channel.
    :return: A normalized 2D array of median ratios for each channel.
    """
    max_indices = a.argmax(axis=1)  # Indices of maximum values per row
    median_array = np.array(
        [np.median(a[max_indices == i], axis=0) for i in range(a.shape[1])]
    ).T
    totals = median_array.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        median_array = median_array / totals

    median_array[np.isnan(median_array)] = 1  # Replace NaNs with 1
    return np.linalg.inv(median_array)

def fit_crosstalk_nnls(a: np.ndarray) -> np.ndarray:
    """Estimate a crosstalk correction matrix by regressing each channel's intensity against
    the called channel's intensity with non-negative least squares, instead of collapsing each
    class down to a single median vector like calculate_crosstalk_median_ratio does.

    For reads called as channel i (a.argmax(axis=1) == i), the called channel's own intensity is
    used as a stand-in for the read's unknown true brightness, and every other channel j is fit
    against it through the origin: values_j ~= k_ji * values_i, with k_ji constrained >= 0 since
    a dye can only add signal into another channel, not subtract from it. This is meant to be fit
    on samples from binned_cycle_sampler, which deliberately spreads samples across the intensity
    range per called base -- exactly the spread a regression needs to pin down a slope reliably,
    and which a single median statistic ignores.

    :param a: Input 2D array of sampled read intensities, shape (reads, channels). Should span a
        wide intensity range per called base (as binned_cycle_sampler's output does) -- a
        narrow-range sample gives NNLS little leverage to estimate a slope.
    :return: A normalized, inverted crosstalk correction matrix in the same convention as
        calculate_crosstalk_median_ratio and fit_crosstalk_em (a drop-in for
        apply_channel_crosstalk_matrix).
    """
    nchannels = a.shape[1]
    max_indices = a.argmax(axis=1)
    median_array = np.ones((nchannels, nchannels))  # column i defaults to all-1s if base i is unseen

    for i in range(nchannels):
        called = a[max_indices == i]
        if len(called) == 0:
            continue
        reference = called[:, i]
        for j in range(nchannels):
            if j == i:
                continue
            slope, _residual = nnls(reference[:, None], called[:, j])
            median_array[j, i] = slope[0]

    totals = median_array.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        median_array = median_array / totals

    median_array[np.isnan(median_array)] = 1
    return np.linalg.inv(median_array)


def normalize_reads(a: np.ndarray) -> np.ndarray:
    """Project each read's channel intensities onto the unit simplex (sum to 1), removing the
    effect of per-dot brightness so only the relative color/crosstalk signature remains.

    :param a: Input 2D array of shape (reads, channels).
    :return: Array of shape (kept_reads, channels), non-negative, each row summing to 1. Rows
        that sum to ~0 (no signal in any channel) are dropped.
    """
    a = np.clip(a, 0, None)
    totals = a.sum(axis=1)
    keep = totals > 1e-6
    return a[keep] / totals[keep, None]


def _crosstalk_em_init_means(a: np.ndarray) -> np.ndarray:
    """Warm-start point for fit_crosstalk_em: the per-channel argmax means, i.e. the same
    classification calculate_crosstalk_median_ratio relies on entirely. Used only to seed the EM
    fit so its components stay aligned to channel/base identity instead of being free to permute;
    the fit itself is free to move reads away from this initial hard assignment.
    """
    max_indices = a.argmax(axis=1)
    return np.array([
        a[max_indices == i].mean(axis=0) if np.any(max_indices == i) else a.mean(axis=0)
        for i in range(a.shape[1])
    ])


def fit_crosstalk_em(a: np.ndarray, covariance_type: str = 'full', max_iter: int = 100, tol: float = 1e-4, random_state: int = 42):
    """Estimate a crosstalk correction matrix by fitting a Gaussian mixture model with
    expectation-maximization, instead of hard-classifying each read by its argmax channel
    before averaging (as calculate_crosstalk_median_ratio does).

    Reads are normalized onto the unit simplex first (see normalize_reads) so that per-dot
    brightness -- which varies hugely between amplicons and carries no crosstalk information --
    doesn't affect the fit. Each read is then treated as a soft mixture over the `channels`
    possible true bases, and the base assignment and the crosstalk matrix are refined together
    until convergence. This avoids the circularity of committing to an argmax call on
    uncorrected data before the correction is known, which matters most for reads near a cluster
    boundary -- exactly where crosstalk does the most damage.

    :param a: Input 2D array of raw sampled read intensities, shape (reads, channels). Use
        quality_filtered_sampler rather than binned_cycle_sampler to build this -- the binning
        that sampler does exists to balance representation across the intensity range for a
        median statistic, which this fit has no need for since brightness is normalized away.
    :param covariance_type: sklearn GaussianMixture covariance_type. 'full' captures correlated
        crosstalk noise between channels and is the most expressive option; fall back to 'diag'
        if the fit is unstable with small sample counts.
    :param max_iter: Maximum number of EM iterations.
    :param tol: Convergence tolerance on the per-sample average log-likelihood gain.
    :param random_state: Random seed for sklearn; the fit is otherwise warm-started from the
        argmax means (see _crosstalk_em_init_means) so results should be stable regardless.
    :return: (correction_matrix, model) -- the normalized, inverted crosstalk matrix (same
        convention as calculate_crosstalk_median_ratio, so it's a drop-in replacement for
        apply_channel_crosstalk_matrix) and the fitted sklearn GaussianMixture, for inspecting
        convergence or per-read posterior probabilities.
    """

    normalized = normalize_reads(a)
    means_init = normalize_reads(_crosstalk_em_init_means(a))

    model = GaussianMixture(
        n_components = a.shape[1],
        covariance_type = covariance_type,
        means_init = means_init,
        max_iter = max_iter,
        tol = tol,
        random_state = random_state,
    )
    model.fit(normalized)

    if not model.converged_:
        print('warning: crosstalk EM fit did not converge in', max_iter, 'iterations', file=sys.stderr)

    median_array = model.means_.T
    totals = median_array.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        median_array = median_array / totals
    median_array[np.isnan(median_array)] = 1

    return np.linalg.inv(median_array), model


def apply_channel_crosstalk_matrix(df, sequencing_channels_order, w: np.ndarray, dtype=np.float32):
    """Apply the linear transformation w * x = y to a dots dataframe.

    :param df: dots dataframe with 'values_cycleXX_[GTAC]' columns.
    :param w: Crosstalk compensation matrix, (c, c) or (t, c, c).
    :param dtype: Corrected data type.
    :return: A copy of `df` with the intensity columns replaced by corrected values.
    """
    X = df.reads.values        # (read, t, c)
    num_cycles = X.shape[1]
    nchannels = X.shape[2]

    if w.ndim == 2:
        Y = w.dot(X.reshape(-1, nchannels).T).T.reshape(X.shape)
    else:
        Y = np.stack([
            w[t].dot(X[:, t, :].T).T     # (read, c) per cycle
            for t in range(len(w))
        ], axis=1)

    out = df.copy()
    #build cols 
    cols = []
    for i in range(0, num_cycles):
        for base in sequencing_channels_order: 
            cols.append('values_cycle'+get_cycle_str(i) + f'_{base}')
    out[cols] = Y.astype(dtype, copy=False).reshape(len(df), -1)
    return out