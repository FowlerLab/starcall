import numpy as np
import skimage.io
import pandas
from scipy.special import softmax
import numpy as np
from . import reads #to use the reads accessor for table values
from . import utils
import numpy as np


def calculate_peaks(df):
    """
    Calculate the peak score (std across cycles, mean across channels)
    equivalent to the local maxima detection image value in SCALLOPS.
    Note that the expected values for this are intensity, not anything normalized.
    
    For each spot:
        1. compute std across cycles for each channel
        2. take mean across channels
    
    Returns an array of shape (N,) where N is the number of dots.
    """
    X = df.reads.values   # (read, cycle, channel)
    std_across_cycles = X.std(axis=1)   # (N, 4) - std over cycles per channel
    peak = std_across_cycles.mean(axis=1)  # (N,) - mean over channels
    
    return peak

def calculate_snr(pixels, background_percentile=50, signal_percentile=99.9):
    """
    Percentile-based, dot-agnostic SNR for a raw pixel intensity distribution.

    background/signal are given by percentiles of the raw pixel values, and noise
    is the median absolute deviation scaled to a std-equivalent (1.4826 * MAD),
    which is robust to the small fraction of bright dot pixels skewing a plain std().

    :param pixels: 1D array of raw pixel intensities for one (well, cycle, channel).
    :return: (background, signal, noise, snr)
    """
    background = np.percentile(pixels, background_percentile)
    signal = np.percentile(pixels, signal_percentile)
    noise = 1.4826 * np.median(np.abs(pixels - np.median(pixels)))
    snr = (signal - background) / noise if noise > 0 else np.nan
    return background, signal, noise, snr

def get_softmax_df(df, min_error: float = 1e-9, use_min: bool = True) -> np.ndarray:
    """Compute phred quality scores directly from a normalized spot-intensity dataframe.

    Columns matching 'values_cycleXX_[GTAC]' are extracted, softmaxed over the
    channel axis, and converted to phred scores (higher is better).

    :param df: DataFrame with one read per row.
    :param min_error: Minimum p-value error.
    :return: np.ndarray of shape (n_reads, n_cycles) — quality score per read per cycle.
    """

    X = df.reads.values   # (read, cycle, channel)
    p = np.max(softmax(X, axis=2), axis=2) # (read, cycle)
    p_error = 1 - p
    if use_min:
        p_error = np.maximum(p_error, min_error)
    return -10 * np.log10(p_error)


#new metrics 

def get_dominance_signed_df(df, min_norm: float = 1e-9) -> np.ndarray:
    """Scale-invariant separability using the raw (unclipped) z-scores: the top channel's
    value as a share of the whole vector's L2 norm, top / ||x||_2. Unlike get_dominance_df,
    negative channels aren't zeroed out first -- they still shape the norm, so a channel
    that's strongly negative (vs. just below zero) pulls this score down.

    :param df: DataFrame with one read per row.
    :param min_norm: floor for ||x||_2 to avoid dividing by ~0 on an all-zero cycle.
    :return: np.ndarray of shape (n_reads, n_cycles).
    """
    X = df.reads.values                     # (read, cycle, channel)
    norm = np.linalg.norm(X, axis=2)
    degenerate = np.isnan(norm) | (norm < min_norm)
    norm = np.where(degenerate, 1.0, norm)
    dominance = np.nan_to_num(X, nan=0.0).max(axis=2) / norm
    return np.where(degenerate, 0.0, dominance)


def get_signed_margin_df(df, min_spread: float = 1e-9) -> np.ndarray:
    """How many standard deviations (of the non-winning channels) the top channel clears
    the rest by: (top - mean(rest)) / std(rest). Exactly scale-invariant under x -> c*x
    for c > 0 (numerator and denominator scale together), and uses the signed z-scores
    directly -- no clipping, no total/sum needed at all.

    :param df: DataFrame with one read per row.
    :param min_spread: floor for std(rest) to avoid dividing by ~0 when the losing
        channels are (almost) identical.
    :return: np.ndarray of shape (n_reads, n_cycles).
    """
    X = df.reads.values                     # (read, cycle, channel)
    # some (read, cycle) slices are entirely NaN -- e.g. spots sitting on a masked-out
    # region of the source image, which dot_filter_new NaNs out across every channel and
    # cycle at once. Plain argmax/nanmean choke on those (argmax silently "picks" channel 0
    # since NaN > x is always False; nanmean/nanstd warn "Mean of empty slice" when every
    # remaining value is also NaN). Substitute a safe placeholder before reducing, and mark
    # those slices degenerate explicitly instead.
    all_nan = np.all(np.isnan(X), axis=2)           # (read, cycle)
    X_safe = np.where(all_nan[..., None], 0.0, X)

    top_idx = np.argmax(X_safe, axis=2)
    top = np.take_along_axis(X_safe, top_idx[..., None], axis=2)[..., 0]
    mask = np.ones_like(X_safe, dtype=bool)
    np.put_along_axis(mask, top_idx[..., None], False, axis=2)
    rest = np.where(mask, X_safe, np.nan)
    rest_mean = np.nanmean(rest, axis=2)
    rest_std = np.nanstd(rest, axis=2)
    degenerate = all_nan | (rest_std < min_spread)
    rest_std = np.where(degenerate, 1.0, rest_std)
    margin = (top - rest_mean) / rest_std
    return np.where(degenerate, 0.0, margin)

def get_chastity_df(df, min_total: float = 1e-9) -> np.ndarray:
    """Illumina-style chastity: the top channel's share of the top two channels only,
    I_top / (I_top + I_second). Like get_dominance_df this is scale-invariant, but it
    ignores the bottom two channels entirely. Classic Illumina QC threshold is chastity < 0.6.

    :param df: DataFrame with one read per row.
    :param min_total: floor for (I_top + I_second) to avoid dividing by ~0.
    :return: np.ndarray of shape (n_reads, n_cycles), each value in [0.5, 1].
    """
    X = df.reads.values                      # (read, cycle, channel)
    Xc = np.clip(X, 0, None)
    top2 = np.sort(Xc, axis=2)[:, :, -2:]     # (read, cycle, 2), ascending -> [second, top]
    total = top2.sum(axis=2)
    degenerate = total < min_total
    total = np.where(degenerate, 1.0, total)
    chastity = top2[:, :, 1] / total
    return np.where(degenerate, 0.5, chastity)


def get_log_margin_df(df, eps: float = 1e-3) -> np.ndarray:
    """Log-ratio between the top and second-highest clipped channel values per cycle:
    log((top + eps) / (second + eps)). Approximately scale-invariant like get_dominance_df,
    but unbounded -- it can tell apart "clearly winning by a lot" reads that dominance's
    bounded [1/n_channels, 1] range compresses together near 1.

    :param df: DataFrame with one read per row.
    :param eps: additive floor so the ratio/log stay finite when a channel is ~0 or negative.
    :return: np.ndarray of shape (n_reads, n_cycles).
    """
    X = df.reads.values
    Xc = np.clip(X, 0, None)
    top2 = np.sort(Xc, axis=2)[:, :, -2:]     # ascending: [second, top]
    return np.log((top2[:, :, 1] + eps) / (top2[:, :, 0] + eps))


def get_purity_df(df, min_total: float = 1e-9) -> np.ndarray:
    """1 - normalized Shannon entropy of the clipped per-channel shares (the same shares
    get_dominance_df computes, but using the whole distribution instead of just the max).
    Distinguishes "one plausible second-place channel" from "uniform noise across all four" --
    two cycles that get_dominance_df alone can't tell apart if they have the same top share.

    :param df: DataFrame with one read per row.
    :param min_total: floor for the clipped channel total, as in get_dominance_df.
    :return: np.ndarray of shape (n_reads, n_cycles), each value in [0, 1]
        (1 = one channel has all the signal, 0 = perfectly uniform across channels).
    """
    X = df.reads.values
    Xc = np.clip(X, 0, None)
    n_channels = X.shape[2]
    totals = Xc.sum(axis=2, keepdims=True)
    degenerate = totals[..., 0] < min_total
    totals = np.where(totals < min_total, 1.0, totals)
    shares = Xc / totals
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(shares > 0, shares * np.log(shares), 0.0)
    entropy = -term.sum(axis=2)
    purity = 1.0 - entropy / np.log(n_channels)
    return np.where(degenerate, 0.0, purity)


def get_decay_slope(per_cycle_scores: np.ndarray) -> np.ndarray:
    """Per-read least-squares slope of a per-cycle score against cycle index. Negative slope
    means signal fading over the run -- a read-level temporal-quality axis that's orthogonal
    to any single cycle's separability/magnitude.

    :param per_cycle_scores: array of shape (n_reads, n_cycles), e.g. the output of
        get_zscore_magnitude_df or get_dominance_df.
    :return: np.ndarray of shape (n_reads,) -- the slope per read.
    """
    n_cycles = per_cycle_scores.shape[1]
    cycles_centered = np.arange(n_cycles) - (n_cycles - 1) / 2
    denom = (cycles_centered ** 2).sum()
    scores_centered = per_cycle_scores - per_cycle_scores.mean(axis=1, keepdims=True)
    return (scores_centered * cycles_centered).sum(axis=1) / denom


def get_adjacent_cycle_similarity(df) -> np.ndarray:
    """Per-read mean cosine similarity between each cycle's 4-channel vector and the next
    cycle's. High similarity suggests phasing/prephasing-style bleed between adjacent
    cycles, rather than an in-cycle channel-separability problem.

    :param df: DataFrame with one read per row.
    :return: np.ndarray of shape (n_reads,).
    """
    X = df.reads.values                 # (read, cycle, channel)
    a, b = X[:, :-1, :], X[:, 1:, :]
    dot = (a * b).sum(axis=2)
    denom = np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)
    cos_sim = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 1e-9)
    return cos_sim.mean(axis=1)

def get_dominance_df(df, min_total: float = 1e-9) -> np.ndarray:
    """Compute a scale-invariant separability score: the top channel's share of the
    total (clipped, non-negative) per-cycle signal.

    Unlike get_softmax_df, this is invariant to uniform rescaling of a cycle's channel
    values (e.g. [10,10,10,20] and [20,20,20,40] both score 0.4) — it measures how
    cleanly one channel dominates, independent of overall signal amplitude.

    :param df: DataFrame with one read per row.
    :param min_total: floor for the clipped channel total, to avoid dividing by ~0 when
        every channel is at or below the per-cycle mean (degenerate/no-signal cycle).
    :return: np.ndarray of shape (n_reads, n_cycles), each value in [1/n_channels, 1].
    """
    X = df.reads.values                     # (read, cycle, channel)
    Xc = np.clip(X, 0, None)
    totals = Xc.sum(axis=2)                 # (read, cycle)
    n_channels = X.shape[2]
    degenerate = totals < min_total
    totals = np.where(degenerate, 1.0, totals) #if the total is too small, divide by one instead of the total
    dominance = Xc.max(axis=2) / totals
    dominance = np.where(degenerate, 1.0 / n_channels, dominance)
    return dominance

def get_zscore_magnitude_df(df) -> np.ndarray:
    """Magnitude of the called channel's z-score per cycle: max(x) across channels.

    Deliberately scale-sensitive (unlike get_dominance_df) — this is the confidence
    signal that dominance throws away. Expects z-scored values_cycleXX_* input.

    :param df: DataFrame with one read per row.
    :return: np.ndarray of shape (n_reads, n_cycles).
    """
    X = df.reads.values   # (read, cycle, channel)
    return X.max(axis=2)

def get_fac_delta_of_top_pos(df) -> np.ndarray:
    """Delta (in absolute value) between the top channel and the second-highest channel
    per cycle, i.e. how far the winner clears the runner-up (unlike get_top_range below,
    this ignores the two lowest channels entirely).

    :param df: DataFrame with one read per row.
    :return: np.ndarray of shape (n_reads, n_cycles).
    """
    X = df.reads.values                   # (read, cycle, channel)
    top2 = np.sort(X, axis=2)[:, :, -2:]   # (read, cycle, 2), ascending -> [second, top]
    return (np.abs(top2[:, :, 1] - top2[:, :, 0]))/(np.abs(top2[:, :, 1]))
