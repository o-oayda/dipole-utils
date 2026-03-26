from dipoleutils.utils.data_loader import DataLoader
from dipoleutils.utils.plotting import smooth_map
from dipoleutils.utils.samples import CatalogueToMap
from dipoleutils.utils.mask import Masker
from dipoleutils.utils.weather import get_temperatures_for_mjd
from scripts.low3_plot_helpers import (
    apply_temperature_density_correction,
    plot_density_relationships,
)
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import warnings
from dipoleska.models.multipole import Multipole


NSIDE = 64
ASKAP_UTC_OFFSET_HOURS = 8.0
FLUX_MIN_MJY = 15

data = DataLoader('racs', 'low3').load()
low3 = CatalogueToMap(data)
data = DataLoader('racs', 'low3-scaled').load()
low3_scaled = CatalogueToMap(data)

low3.make_cut('Total_flux', minimum=FLUX_MIN_MJY, maximum=1000)
low3_scaled.make_cut('Total_flux', minimum=FLUX_MIN_MJY, maximum=1000)
low3.catalogue['Start_time_hours'] = np.mod(
    np.asarray(low3.catalogue['Scan_start_MJD'], dtype=float) % 1.0 * 24.0
    + ASKAP_UTC_OFFSET_HOURS,
    24.0,
)
try:
    low3.catalogue['Temperature_C'] = get_temperatures_for_mjd(
        low3.catalogue['Scan_start_MJD']
    )
except Exception as exc:
    warnings.warn(f'Unable to fetch weather data: {exc}')
    low3.catalogue['Temperature_C'] = np.full(len(low3.catalogue), np.nan)
dmap = low3.make_density_map('equatorial', nside=NSIDE)
dmap_scaled = low3_scaled.make_density_map('equatorial', nside=NSIDE)
low3.catalogue['perc_err'] = (
        low3.catalogue['E_Total_flux'] / low3.catalogue['Total_flux']
) * 100
fmap = low3.make_parameter_map(
    column_name='Total_flux', coordinate_system='equatorial', operation='mean',
    nside=NSIDE
)
start_time_map = low3.make_parameter_map(
    column_name='Start_time_hours',
    coordinate_system='equatorial',
    operation='mean',
    nside=NSIDE
)
temperature_map = low3.make_parameter_map(
    column_name='Temperature_C',
    coordinate_system='equatorial',
    operation='mean',
    nside=NSIDE
)
sbidmap = low3.make_parameter_map(
    column_name='SBID',
    coordinate_system='equatorial',
    operation='mean',
    nside=NSIDE
)
rmsmap = low3.make_parameter_map(
    column_name='perc_err',
    coordinate_system='equatorial',
    operation='median',
    nside=NSIDE
)

mask = Masker([dmap, fmap, start_time_map, temperature_map, rmsmap], 'equatorial')
mask.mask_galactic_plane(5)
mask.mask_a_team_sources(radius_deg=3, source_names=['Cygnus A'])
mask.mask_a_team_sources(radius_deg=2)
mask.mask_equatorial_poles(north_radius=42)
# mask.mask_around_bright_sources(
#     5000, 0.1,
#     low3.catalogue['Total_flux'],
#     low3.catalogue['RA'],
#     low3.catalogue['Dec']
# )
dmap, fmap, start_time_map, temperature_map, rmsmap = mask.get_masked_density_map()

cat = low3.get_catalogue()
time_hours = np.asarray(cat['Start_time_hours'], dtype=float)
source_temperatures = np.asarray(cat['Temperature_C'], dtype=float)

window_minutes = 120
temperature_time_bin_minutes = 15
temperature_window_c = 3.0
temperature_time_bin_hours = temperature_time_bin_minutes / 60
dmap_temperature_corrected, trend_temperature, trend_density, reference_density = apply_temperature_density_correction(
    dmap,
    temperature_map,
    temperature_window_c=temperature_window_c,
    reference_density=float(np.nanmean(dmap[np.isfinite(dmap)])),
)
correction_factor = reference_density / trend_density
plot_density_relationships(
    dmap,
    start_time_map,
    temperature_map,
    time_hours,
    source_temperatures,
    time_window_hours=window_minutes / 60,
    temperature_window_c=temperature_window_c,
    temperature_time_bin_hours=temperature_time_bin_hours,
    correction_temperature=trend_temperature,
    correction_factor=correction_factor,
)
print(
    'Applied temperature-density correction with reference density:',
    f'{reference_density:.3f} sources per pixel.'
)
plot_density_relationships(
    dmap_temperature_corrected,
    start_time_map,
    temperature_map,
    time_hours,
    source_temperatures,
    time_window_hours=window_minutes / 60,
    temperature_window_c=temperature_window_c,
    temperature_time_bin_hours=temperature_time_bin_hours,
    title_prefix='Corrected ',
    correction_temperature=trend_temperature,
    correction_factor=correction_factor,
)

smooth_map(dmap_temperature_corrected)
# hp.projview(start_time_map, title='Mean LOW3 Scan Start Time', unit='hours')
# plt.close()
plt.show()

model = Multipole(dmap_temperature_corrected, ells=[1,2], likelihood='point')
model.run_nested_sampling(step=True)
model.corner_plot()
model.sky_direction_posterior()
plt.show()
