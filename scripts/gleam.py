'''
clearly some negative correlation between rms noise (lrmswide) and source count
'''
from dipoleutils.utils.data_loader import DataLoader
from dipoleutils.utils.mask import Masker
from astropy.io import fits
import numpy as np
from astropy.table import Table
from dipoleutils.utils.plotting import plot_log_log_histogram
from dipoleutils.utils.samples import CatalogueToMap
import healpy as hp
import matplotlib.pyplot as plt
from dipoleska.models.dipole import Dipole


data = DataLoader('gleam', 'x-dr2')
gleam = data.load(
    columns=[
        'RAdeg', 'DEdeg', 'Fintwide', 'lrmswide', 'e_Fintwide', 'eFitFlux',
        'awide'
    ]
)

processor = CatalogueToMap(gleam) # pyright: ignore
emap = processor.make_parameter_map(
    'lrmswide',
    'equatorial',
    # no_source_val=0.015, 
    operation='median'
)
processor.make_cut(column_name='Fintwide', minimum=0.015, maximum=None)
dmap = processor.make_density_map(coordinate_system='equatorial')

masker = Masker(dmap, coordinate_system='equatorial')
masker.mask_equatorial_poles(north_radius=61)
masker.mask_equatorial_longitude(ra_min_deg=99, ra_max_deg=311)
dmap = masker.get_masked_density_map()


masker = Masker(emap, coordinate_system='equatorial')
masker.mask_equatorial_poles(north_radius=61)
masker.mask_equatorial_longitude(ra_min_deg=99, ra_max_deg=311)
emap = masker.get_masked_density_map()

hp.projview(dmap, coord=['C', 'G'], sub=211)
hp.projview(emap, coord=['C', 'G'], sub=212)
plt.figure()
plot_log_log_histogram(processor.catalogue['Fintwide'], bins=100)
plt.figure()
plt.scatter(emap, dmap, s=1)
plt.show()

# model = Dipole(dmap, likelihood='general_poisson')
# model.run_nested_sampling(step=True)
# model.corner_plot(coordinates=['equatorial', 'galactic'])
# model.sky_direction_posterior(coordinates=['equatorial', 'galactic'])
# plt.show()

# with fits.open('~/catalogue_data/gleam/x-dr2/VIII_113_catalog2.dat.gz.fits', memmap=True) as hdul:
#   raw = hdul[1].data.view(np.ndarray)  # keep ASCII field bytes/strings
#   t = Table(raw)
