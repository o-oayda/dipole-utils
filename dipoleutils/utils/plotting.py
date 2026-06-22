from typing import Sequence
import numpy as np
from numpy.typing import NDArray 
from matplotlib.patches import Patch
import warnings
import matplotlib.pyplot as plt
import matplotlib
import healpy as hp
from dipoleutils.utils.physics import omega_to_theta


DEFAULT_BOOTSTRAP_RESAMPLES = 1000


def plot_log_log_histogram(
        data: Sequence[float] | NDArray[np.floating],
        bins: int | Sequence[float] = 10,
        color: str = 'cornflowerblue',
        **hist_kwargs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], list[Patch]]:
    '''
    Plot a histogram with logarithmic scales on both axes, using bins that are
    uniformly spaced in log space.

    :param data: Input array-like of values; non-positive entries are dropped.
    :param bins: Either the number of bins (int) or a sequence of log-uniform
        bin edges to use directly.
    :param color: Color applied to both the filled bars and their outlines.
    :param hist_kwargs: Extra keyword arguments forwarded to ``plt.hist``.
    :return: The ``(counts, bin_edges, patches)`` tuple returned by
        ``plt.hist``.
    '''
    if 'bins' in hist_kwargs:
        raise TypeError('Pass bin specification via the explicit `bins` argument.')
    if 'color' in hist_kwargs:
        raise TypeError('Pass bar color via the explicit `color` argument.')

    values = np.asarray(data, dtype=np.float64)
    positive_mask = values > 0
    if not np.all(positive_mask):
        removed = int(values.size - positive_mask.sum())
        warnings.warn(
            f'Removed {removed} non-positive entries before plotting on log-log axes.',
            RuntimeWarning,
            stacklevel=2
        )
        values = values[positive_mask]

    if values.size == 0:
        raise ValueError('Log-log histogram requires at least one positive value.')

    if isinstance(bins, (int, np.integer)):
        if bins < 1:
            raise ValueError('Number of bins must be a positive integer.')
        edges = np.logspace(
            np.log10(values.min()),
            np.log10(values.max()),
            int(bins) + 1
        )
    else:
        edges = np.asarray(bins, dtype=np.float64)
        if np.any(edges <= 0):
            raise ValueError('Bin edges must be positive for log spacing.')
        log_widths = np.diff(np.log10(edges))
        if not np.allclose(log_widths, log_widths[0]):
            raise ValueError('Provided bin edges are not uniformly spaced in log space.')

    # note: we make two plt.hist calls to get the 'solid edge with alpha' style
    # the first call needs stepfilled with alpha, the second just an edge
    fill_kwargs = dict(hist_kwargs)
    fill_kwargs.setdefault('histtype', 'stepfilled')
    fill_kwargs.setdefault('alpha', 0.3)
    fill_kwargs['color'] = color
    counts, bin_edges, patches = plt.hist(
        values, bins=edges, **fill_kwargs
    )

    edge_kwargs = dict(hist_kwargs)
    edge_kwargs.setdefault('histtype', 'step')
    edge_kwargs['color'] = color
    edge_kwargs['lw'] = 1.5
    plt.hist(
        values, bins=edges, **edge_kwargs
    )

    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log', nonpositive='clip')

    return counts, bin_edges, patches


def _compute_binned_mean_statistics(
        x: Sequence[float] | NDArray[np.floating],
        y: Sequence[float] | NDArray[np.floating],
        bins: int | Sequence[float] = 10,
        x_range: tuple[float, float] | None = None,
        log_bins: bool = False,
        bootstrap_resamples: int | None = DEFAULT_BOOTSTRAP_RESAMPLES,
        bootstrap_seed: int = 0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x_values = np.ravel(np.asarray(x, dtype=np.float64))
    y_values = np.ravel(np.asarray(y, dtype=np.float64))

    if x_values.size != y_values.size:
        raise ValueError('`x` and `y` must contain the same number of elements.')

    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]

    if x_range is not None:
        x_min, x_max = map(float, x_range)
        if x_min >= x_max:
            raise ValueError('`x_range` must satisfy xmin < xmax.')
        in_range = (x_values >= x_min) & (x_values <= x_max)
        x_values = x_values[in_range]
        y_values = y_values[in_range]
    else:
        x_min = float(np.min(x_values)) if x_values.size else np.nan
        x_max = float(np.max(x_values)) if x_values.size else np.nan

    if x_values.size == 0:
        raise ValueError('No finite samples remain after applying filters.')

    if isinstance(bins, (int, np.integer)):
        if int(bins) < 1:
            raise ValueError('Number of bins must be a positive integer.')
        if x_range is None:
            x_min = float(np.min(x_values))
            x_max = float(np.max(x_values))
        if log_bins:
            if x_min <= 0 or np.any(x_values <= 0):
                raise ValueError('Log-spaced bins require strictly positive `x` values.')
            bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), int(bins) + 1)
        else:
            bin_edges = np.linspace(x_min, x_max, int(bins) + 1)
    else:
        bin_edges = np.asarray(bins, dtype=np.float64)
        if bin_edges.ndim != 1 or bin_edges.size < 2:
            raise ValueError('Bin edges must be a one-dimensional array of length >= 2.')
        if not np.all(np.isfinite(bin_edges)):
            raise ValueError('Bin edges must be finite.')
        if np.any(np.diff(bin_edges) <= 0):
            raise ValueError('Bin edges must be strictly increasing.')
        if log_bins and np.any(bin_edges <= 0):
            raise ValueError('Log-spaced bins require strictly positive bin edges.')

    rng = np.random.default_rng(bootstrap_seed)
    bin_centres: list[float] = []
    mean_y: list[float] = []
    mean_y_err: list[float] = []

    for bin_index in range(bin_edges.size - 1):
        left_edge = bin_edges[bin_index]
        right_edge = bin_edges[bin_index + 1]
        in_bin = (x_values >= left_edge) & (
            x_values < right_edge
            if bin_index < bin_edges.size - 2
            else x_values <= right_edge
        )
        y_in_bin = y_values[in_bin]
        if y_in_bin.size == 0:
            continue

        if bootstrap_resamples is None or int(bootstrap_resamples) < 2 or y_in_bin.size <= 1:
            bootstrap_error = 0.0
        else:
            bootstrap_means = np.empty(int(bootstrap_resamples), dtype=np.float64)
            for bootstrap_index in range(int(bootstrap_resamples)):
                sampled_y = rng.choice(y_in_bin, size=y_in_bin.size, replace=True)
                bootstrap_means[bootstrap_index] = np.mean(sampled_y)
            bootstrap_error = float(np.std(bootstrap_means, ddof=1))

        if log_bins:
            bin_centre = float(np.sqrt(left_edge * right_edge))
        else:
            bin_centre = float(0.5 * (left_edge + right_edge))

        bin_centres.append(bin_centre)
        mean_y.append(float(np.mean(y_in_bin)))
        mean_y_err.append(bootstrap_error)

    return (
        np.asarray(bin_centres, dtype=np.float64),
        np.asarray(mean_y, dtype=np.float64),
        np.asarray(mean_y_err, dtype=np.float64),
    )


def plot_binned_mean(
        x: Sequence[float] | NDArray[np.floating],
        y: Sequence[float] | NDArray[np.floating],
        bins: int | Sequence[float] = 10,
        x_range: tuple[float, float] | None = None,
        log_bins: bool = False,
        bootstrap_resamples: int | None = DEFAULT_BOOTSTRAP_RESAMPLES,
        bootstrap_seed: int = 0,
        ax=None,
        **errorbar_kwargs
    ):
    '''
    Plot the mean of ``y`` in bins of ``x``, with optional bootstrap errors.

    :param x: Input x-values used for bin assignment.
    :param y: Input y-values averaged within each x-bin.
    :param bins: Either the number of bins (int) or an explicit sequence of bin
        edges.
    :param x_range: Inclusive ``(xmin, xmax)`` domain restriction applied
        before binning. When ``bins`` is an int, this range is also used to
        construct the bin edges.
    :param log_bins: If True, construct log-spaced bins and use geometric bin
        centres. Requires strictly positive x-values in the active domain.
    :param bootstrap_resamples: Number of within-bin bootstrap resamples used to
        estimate one-sigma uncertainty on the mean. Defaults to
        ``DEFAULT_BOOTSTRAP_RESAMPLES``. If None or less than 2, zero errors are
        returned.
    :param bootstrap_seed: Random seed for bootstrap resampling.
    :param ax: Optional matplotlib axes to plot on. Defaults to ``plt.gca()``.
    :param errorbar_kwargs: Extra keyword arguments forwarded to
        ``Axes.errorbar``.
    :return: Tuple of ``(bin_centres, mean_y, mean_y_err, errorbar_container)``.
    '''
    if 'yerr' in errorbar_kwargs:
        raise TypeError('Pass bootstrap uncertainty via `bootstrap_resamples`, not `yerr`.')

    bin_centres, mean_y, mean_y_err = _compute_binned_mean_statistics(
        x,
        y,
        bins=bins,
        x_range=x_range,
        log_bins=log_bins,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    axis = plt.gca() if ax is None else ax
    errorbar_kwargs.setdefault('fmt', 'o')
    errorbar_kwargs.setdefault('capsize', 3)
    errorbar_kwargs.setdefault('markersize', 6)
    errorbar_kwargs.setdefault('markeredgewidth', 0.0)
    errorbar_container = axis.errorbar(
        bin_centres,
        mean_y,
        yerr=mean_y_err,
        **errorbar_kwargs
    )
    return bin_centres, mean_y, mean_y_err, errorbar_container

def plot_binned_quantile(
    x,
    y,
    bins=20,
    quantile=0.5,
    min_count=1,
    n_bootstrap=0,
    ax=None,
    plot=True,
    **kwargs,
):
    """Plot an arbitrary quantile of y in bins of x.

    Returns
    -------
    bin_centres, y_quantile, y_quantile_err, counts
        ``y_quantile_err`` is NaN unless ``n_bootstrap > 0``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1].")

    bin_edges = np.histogram_bin_edges(x, bins=bins)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    y_quantile = np.full(bin_centres.shape, np.nan, dtype=float)
    y_quantile_err = np.full(bin_centres.shape, np.nan, dtype=float)
    counts = np.zeros(bin_centres.shape, dtype=int)

    bin_index = np.digitize(x, bin_edges) - 1
    bin_index[x == bin_edges[-1]] = bin_centres.size - 1

    rng = np.random.default_rng()
    for i in range(bin_centres.size):
        values = y[bin_index == i]
        counts[i] = values.size
        if values.size < min_count:
            continue

        y_quantile[i] = np.nanquantile(values, quantile)
        if n_bootstrap > 0 and values.size > 1:
            draws = rng.choice(values, size=(n_bootstrap, values.size), replace=True)
            y_quantile_err[i] = np.nanstd(
                np.nanquantile(draws, quantile, axis=1),
                ddof=1,
            )

    if plot:
        if ax is None:
            ax = plt.gca()
        valid = np.isfinite(y_quantile)
        if np.any(np.isfinite(y_quantile_err)):
            ax.errorbar(
                bin_centres[valid],
                y_quantile[valid],
                yerr=y_quantile_err[valid],
                marker="o",
                linestyle="-",
                **kwargs,
            )
        else:
            ax.scatter(bin_centres[valid], y_quantile[valid], **kwargs)

    return bin_centres, y_quantile, y_quantile_err, counts

def smooth_map(
        healpy_map: NDArray,
        weights: NDArray | None = None,
        angle_scale: float = 1.,
        only_return_data: bool = False,
        fig: matplotlib.figure.Figure | None = None,
        map_is_nested: bool = False,
        **kwargs
    ) -> NDArray | None:
    smoothed_map_to_plot = average_smooth_map(
        healpy_map,
        weights=weights,
        angle_scale=angle_scale
    )

    if only_return_data:
        return smoothed_map_to_plot

    hp.projview(
        smoothed_map_to_plot,
        nest=map_is_nested,
        fig=fig.number if fig is not None else None,
        **{
            'cb_orientation': 'vertical',
            **kwargs
        }
    )
    return None

def average_smooth_map(
        healpy_map: NDArray[np.floating],
        weights: NDArray[np.floating] | None = None, 
        angle_scale: float = 1.,
        map_is_nested: bool = False
    ) -> NDArray:
    '''
    Smooth a healpy map using a moving average.
    '''
    included_pixels = np.where(~np.isnan(healpy_map))[0]
    smoothed_map = np.nan * np.empty_like(healpy_map)
    nside = hp.get_nside(healpy_map)
    
    if weights is None:
        weights = np.ones_like(healpy_map)

    smoothing_radius = omega_to_theta(angle_scale)
    for p_index in included_pixels:
        vec = hp.pix2vec(nside, p_index, nest=map_is_nested)
        disc = hp.query_disc(nside, vec, smoothing_radius, nest=map_is_nested)
        smoothed_map[p_index] = np.nanmean(healpy_map[disc] * weights[disc])

    return smoothed_map

def density_map(
        healpy_map: NDArray,
        **projview_kwargs
) -> None:
    n_sources = int(np.nansum(healpy_map))
    hp.projview(
        healpy_map,
        **{
            'cbar': True,
            'cb_orientation': 'vertical',
            'unit': 'Source count per pixel',
            'title': rf'Sources: {n_sources:,}',
            **projview_kwargs
        }
    )
