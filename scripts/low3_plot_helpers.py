import matplotlib.pyplot as plt
import numpy as np


def _get_sorted_scatter(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    sort_idx = np.argsort(x_values)
    return x_values[sort_idx], y_values[sort_idx]


def _compute_moving_average(x_values, y_values, half_window, trim_edges=False):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if trim_edges:
        valid = (
            (x_values >= x_values.min() + half_window)
            & (x_values <= x_values.max() - half_window)
        )
        x_values = x_values[valid]
        y_values = y_values[valid]

    return x_values, np.asarray([
        np.mean(
            y_values[
                (x_values >= x_value - half_window)
                & (x_values <= x_value + half_window)
            ]
        )
        for x_value in x_values
    ])


def _compute_binned_mean(x_values, y_values, bin_width, x_min=0.0, x_max=24.0):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    bin_edges = np.arange(x_min, x_max + bin_width, bin_width)
    bin_centres = bin_edges[:-1] + bin_width / 2
    binned_mean = np.asarray([
        np.mean(
            y_values[
                (x_values >= left_edge)
                & (x_values < right_edge)
            ]
        )
        for left_edge, right_edge in zip(bin_edges[:-1], bin_edges[1:], strict=True)
    ])
    return bin_centres, binned_mean


def estimate_density_temperature_trend(
    density_map,
    temperature_map,
    temperature_window_c=1.0,
):
    valid_temperature_pixels = np.isfinite(density_map) & np.isfinite(temperature_map)
    sorted_temperature_x, sorted_temperature_y = _get_sorted_scatter(
        temperature_map[valid_temperature_pixels],
        density_map[valid_temperature_pixels],
    )
    trend_temperature, temperature_moving_average = _compute_moving_average(
        sorted_temperature_x,
        sorted_temperature_y,
        temperature_window_c / 2,
        trim_edges=True,
    )
    return trend_temperature, temperature_moving_average


def _interp_with_linear_edges(x_eval, x_data, y_data):
    x_eval = np.asarray(x_eval, dtype=float)
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    interpolated = np.interp(x_eval, x_data, y_data)

    if x_data.size < 2:
        return interpolated

    left_mask = x_eval < x_data[0]
    right_mask = x_eval > x_data[-1]

    left_slope = (y_data[1] - y_data[0]) / (x_data[1] - x_data[0])
    right_slope = (y_data[-1] - y_data[-2]) / (x_data[-1] - x_data[-2])

    interpolated[left_mask] = y_data[0] + left_slope * (x_eval[left_mask] - x_data[0])
    interpolated[right_mask] = y_data[-1] + right_slope * (x_eval[right_mask] - x_data[-1])
    return interpolated


def apply_temperature_density_correction(
    density_map,
    temperature_map,
    temperature_window_c=1.0,
    reference_density=None,
):
    density_map = np.asarray(density_map, dtype=float)
    temperature_map = np.asarray(temperature_map, dtype=float)
    trend_temperature, trend_density = estimate_density_temperature_trend(
        density_map,
        temperature_map,
        temperature_window_c=temperature_window_c,
    )

    valid_temperature_pixels = np.isfinite(density_map) & np.isfinite(temperature_map)
    if reference_density is None:
        reference_density = float(np.nanmean(density_map[valid_temperature_pixels]))

    corrected_density_map = density_map.copy()
    trend_at_pixel_temperature = _interp_with_linear_edges(
        temperature_map[valid_temperature_pixels],
        trend_temperature,
        trend_density,
    )
    corrected_density_map[valid_temperature_pixels] = (
        density_map[valid_temperature_pixels]
        * reference_density
        / trend_at_pixel_temperature
    )

    return corrected_density_map, trend_temperature, trend_density, reference_density


def plot_density_relationships(
    density_map,
    start_time_map,
    temperature_map,
    time_hours,
    source_temperatures,
    time_window_hours=1.0,
    temperature_window_c=1.0,
    temperature_time_bin_hours=0.25,
    title_prefix='',
    density_label='Sources Per Pixel',
    correction_temperature=None,
    correction_factor=None,
):
    valid_time_pixels = np.isfinite(density_map) & np.isfinite(start_time_map)
    sorted_time_x, sorted_time_y = _get_sorted_scatter(
        start_time_map[valid_time_pixels],
        density_map[valid_time_pixels],
    )

    sorted_time_x, time_moving_average = _compute_moving_average(
        sorted_time_x,
        sorted_time_y,
        time_window_hours / 2,
    )

    valid_temperature_observations = (
        np.isfinite(time_hours) & np.isfinite(source_temperatures)
    )
    time_bin_centres, temperature_vs_time_average = _compute_binned_mean(
        np.asarray(time_hours[valid_temperature_observations], dtype=float),
        np.asarray(source_temperatures[valid_temperature_observations], dtype=float),
        temperature_time_bin_hours,
    )

    sorted_temperature_x, sorted_temperature_y = _get_sorted_scatter(
        temperature_map[np.isfinite(density_map) & np.isfinite(temperature_map)],
        density_map[np.isfinite(density_map) & np.isfinite(temperature_map)],
    )
    trend_temperature_x, temperature_moving_average = _compute_moving_average(
        sorted_temperature_x,
        sorted_temperature_y,
        temperature_window_c / 2,
        trim_edges=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(sorted_time_x, sorted_time_y, s=8, alpha=0.5)
    axes[0].plot(sorted_time_x, time_moving_average, color='tab:red', linewidth=2)
    axes[0].set_xlim(0, 24)
    axes[0].set_xticks(np.arange(24))
    axes[0].set_xticklabels([f'{hour:02d}:00' for hour in range(24)], rotation=45)
    axes[0].set_xlabel('Mean Scan Start Time Per Pixel (24 hr)')
    axes[0].set_ylabel(density_label)
    axes[0].set_title(f'{title_prefix}Density vs Mean Scan Start Time')

    temperature_axis = axes[0].twinx()
    temperature_axis.plot(
        time_bin_centres,
        temperature_vs_time_average,
        color='tab:orange',
        linewidth=2,
    )
    temperature_axis.set_ylabel('Average Temperature (C)', color='tab:orange')
    temperature_axis.tick_params(axis='y', colors='tab:orange')

    axes[1].scatter(sorted_temperature_x, sorted_temperature_y, s=8, alpha=0.5)
    axes[1].plot(trend_temperature_x, temperature_moving_average, color='tab:red', linewidth=2)
    axes[1].set_xlabel('Mean Temperature Per Pixel (C)')
    axes[1].set_ylabel(density_label)
    axes[1].set_title(f'{title_prefix}Density vs Mean Temperature')

    if correction_temperature is not None and correction_factor is not None:
        correction_axis = axes[1].twinx()
        correction_axis.plot(
            np.asarray(correction_temperature, dtype=float),
            np.asarray(correction_factor, dtype=float),
            color='black',
            linestyle='--',
            linewidth=2,
        )
        correction_axis.set_ylabel('Correction Factor', color='black')
        correction_axis.tick_params(axis='y', colors='black')

    plt.tight_layout()
    return fig, axes
