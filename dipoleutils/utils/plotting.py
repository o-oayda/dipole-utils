from typing import Sequence
import numpy as np
from numpy.typing import NDArray 
from matplotlib.patches import Patch
import warnings
import matplotlib.pyplot as plt
import matplotlib
import healpy as hp
from dipoleutils.utils.physics import omega_to_theta


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
        **kwargs
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
