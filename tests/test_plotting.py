import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dipoleutils.utils.plotting import (
    _compute_binned_mean_statistics,
    plot_binned_mean,
)


def test_compute_binned_mean_statistics_linear_bins():
    x = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    y = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=float)

    centres, mean_y, mean_y_err = _compute_binned_mean_statistics(
        x,
        y,
        bins=2,
        bootstrap_resamples=None,
    )

    assert np.allclose(centres, [0.75, 2.25])
    assert np.allclose(mean_y, [15.0, 35.0])
    assert np.allclose(mean_y_err, [0.0, 0.0])


def test_compute_binned_mean_statistics_respects_explicit_edges_and_x_range():
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    y = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=float)

    centres, mean_y, mean_y_err = _compute_binned_mean_statistics(
        x,
        y,
        bins=[0.0, 2.0, 4.0, 6.0],
        x_range=(1.0, 3.0),
        bootstrap_resamples=None,
    )

    assert np.allclose(centres, [1.0, 3.0])
    assert np.allclose(mean_y, [10.0, 25.0])
    assert np.allclose(mean_y_err, [0.0, 0.0])


def test_compute_binned_mean_statistics_log_bins_use_geometric_centres():
    x = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float)
    y = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float)

    centres, mean_y, _ = _compute_binned_mean_statistics(
        x,
        y,
        bins=[1.0, 4.0, 16.0],
        log_bins=True,
        bootstrap_resamples=None,
    )

    assert np.allclose(centres, [2.0, 8.0])
    assert np.allclose(mean_y, [1.5, 6.0])


def test_compute_binned_mean_statistics_skips_empty_bins():
    x = np.asarray([0.0, 0.5, 9.0], dtype=float)
    y = np.asarray([1.0, 3.0, 5.0], dtype=float)

    centres, mean_y, _ = _compute_binned_mean_statistics(
        x,
        y,
        bins=[0.0, 1.0, 2.0, 10.0],
        bootstrap_resamples=None,
    )

    assert np.allclose(centres, [0.5, 6.0])
    assert np.allclose(mean_y, [2.0, 5.0])


def test_compute_binned_mean_statistics_drops_non_finite_pairs():
    x = np.asarray([0.0, 1.0, np.nan, 3.0], dtype=float)
    y = np.asarray([10.0, np.nan, 30.0, 40.0], dtype=float)

    centres, mean_y, mean_y_err = _compute_binned_mean_statistics(
        x,
        y,
        bins=[0.0, 2.0, 4.0],
        bootstrap_resamples=None,
    )

    assert np.allclose(centres, [1.0, 3.0])
    assert np.allclose(mean_y, [10.0, 40.0])
    assert np.allclose(mean_y_err, [0.0, 0.0])


def test_compute_binned_mean_statistics_bootstrap_is_reproducible():
    x = np.asarray([1.0, 1.2, 1.4, 2.0, 2.2, 2.4], dtype=float)
    y = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=float)

    first = _compute_binned_mean_statistics(
        x,
        y,
        bins=[1.0, 1.5, 2.5],
        bootstrap_resamples=64,
        bootstrap_seed=7,
    )
    second = _compute_binned_mean_statistics(
        x,
        y,
        bins=[1.0, 1.5, 2.5],
        bootstrap_resamples=64,
        bootstrap_seed=7,
    )

    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])
    assert np.allclose(first[2], second[2])
    assert np.all(first[2] > 0.0)


def test_compute_binned_mean_statistics_bootstraps_by_default():
    _, _, mean_y_err = _compute_binned_mean_statistics(
        [1.0, 1.2, 1.4, 2.0, 2.2, 2.4],
        [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        bins=[1.0, 1.5, 2.5],
        bootstrap_seed=7,
    )

    assert np.all(mean_y_err > 0.0)


def test_compute_binned_mean_statistics_rejects_invalid_ranges_and_log_inputs():
    with pytest.raises(ValueError, match="xmin < xmax"):
        _compute_binned_mean_statistics([1.0, 2.0], [3.0, 4.0], x_range=(2.0, 2.0))

    with pytest.raises(ValueError, match="strictly positive"):
        _compute_binned_mean_statistics(
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            bins=2,
            log_bins=True,
            bootstrap_resamples=None,
        )


def test_plot_binned_mean_returns_arrays_and_artist():
    fig, ax = plt.subplots()
    try:
        centres, mean_y, mean_y_err, artist = plot_binned_mean(
            [0.0, 1.0, 1.2, 2.0, 2.2, 3.0],
            [1.0, 3.0, 4.0, 5.0, 7.0, 9.0],
            bins=2,
            ax=ax,
            bootstrap_seed=3,
        )
    finally:
        plt.close(fig)

    assert np.allclose(centres, [0.75, 2.25])
    assert np.allclose(mean_y, [8.0 / 3.0, 7.0])
    assert np.all(mean_y_err > 0.0)
    assert artist.lines[0].axes is ax
    assert artist.lines[0].get_linestyle() == 'None'
    assert artist.lines[0].get_marker() == 'o'
    assert artist.lines[0].get_markeredgewidth() == 0.0
    assert artist.lines[0].get_markersize() == 6.0
    assert len(artist.lines[1]) == 2
