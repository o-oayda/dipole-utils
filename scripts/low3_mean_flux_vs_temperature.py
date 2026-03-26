from dipoleutils.utils.data_loader import DataLoader
from dipoleutils.utils.weather import get_temperatures_for_mjd
import matplotlib.pyplot as plt
import numpy as np
import warnings


FLUX_MIN_MJY = 15
FLUX_MAX_MJY = 1000
TEMPERATURE_WINDOW_C = 6.0
FLUX_AGGREGATION_FUNCTION = np.mean
FLUX_AGGREGATION_LABEL = 'Mean'


def compute_moving_average(x_values, y_values, half_window_c):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid = (
        (x_values >= x_values.min() + half_window_c)
        & (x_values <= x_values.max() - half_window_c)
    )
    x_values = x_values[valid]
    y_values = y_values[valid]
    return x_values, np.asarray([
        np.mean(
            y_values[
                (x_values >= x_value - half_window_c)
                & (x_values <= x_value + half_window_c)
            ]
        )
        for x_value in x_values
    ])


data = DataLoader('racs', 'low3').load()
flux_values = np.asarray(data['Total_flux'], dtype=float)
valid_flux = (flux_values >= FLUX_MIN_MJY) & (flux_values <= FLUX_MAX_MJY)
catalogue = data[valid_flux]

try:
    catalogue['Temperature_C'] = get_temperatures_for_mjd(catalogue['Scan_start_MJD'])
except Exception as exc:
    warnings.warn(f'Unable to fetch weather data: {exc}')
    catalogue['Temperature_C'] = np.full(len(catalogue), np.nan)

sbid = np.asarray(catalogue['SBID'], dtype=np.int64)
total_flux = np.asarray(catalogue['Total_flux'], dtype=float)
temperature_c = np.asarray(catalogue['Temperature_C'], dtype=float)

unique_sbid, inverse_indices = np.unique(sbid, return_inverse=True)
tile_flux_statistic = np.asarray([
    FLUX_AGGREGATION_FUNCTION(total_flux[inverse_indices == tile_index])
    for tile_index in range(unique_sbid.size)
])
tile_temperature = np.empty(unique_sbid.size, dtype=float)
for tile_index in range(unique_sbid.size):
    tile_temperature[tile_index] = temperature_c[inverse_indices == tile_index][0]

valid_tiles = np.isfinite(tile_flux_statistic) & np.isfinite(tile_temperature)
tile_flux_statistic = tile_flux_statistic[valid_tiles]
tile_temperature = tile_temperature[valid_tiles]
unique_sbid = unique_sbid[valid_tiles]

sort_idx = np.argsort(tile_temperature)
sorted_temperature = tile_temperature[sort_idx]
sorted_flux_statistic = tile_flux_statistic[sort_idx]
sorted_sbid = unique_sbid[sort_idx]

trend_temperature, trend_flux_statistic = compute_moving_average(
    sorted_temperature,
    sorted_flux_statistic,
    TEMPERATURE_WINDOW_C / 2,
)

print(f'Number of LOW3 tiles (SBIDs): {sorted_sbid.size}')

plt.figure(figsize=(7, 5))
plt.scatter(sorted_temperature, sorted_flux_statistic, s=12, alpha=0.7)
plt.plot(trend_temperature, trend_flux_statistic, color='tab:red', linewidth=2)
plt.xlabel('Mean Temperature Per SBID Tile (C)')
plt.ylabel(f'{FLUX_AGGREGATION_LABEL} Total Flux Per SBID Tile (mJy)')
plt.title(f'LOW3 {FLUX_AGGREGATION_LABEL} Flux vs SBID Tile Temperature')
plt.tight_layout()
plt.yscale('log')
# plt.ylim(0, 15)
plt.show()
