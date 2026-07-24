from typing import Literal
from dipoleska.models.dipole import Dipole
from dipoleska.models.multipole import Multipole
from dipoleska.models.priors import Prior
from dipoleutils.utils.crossmatch import CrossMatch
from dipoleutils.utils.data_loader import DataLoader
from dipoleutils.utils.mask import Masker
from dipoleutils.utils.samples import CatalogueToMap
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from dipoleutils.utils.plotting import smooth_map


def run_dipole(dmap, large_D: bool = False):
    prior = Prior(choose_prior='dipole')
    if large_D:
        prior.change_prior(2, new_prior=['Uniform', 0., 0.3])
    model = Dipole(dmap, likelihood='general_poisson', prior=prior)
    model.run_nested_sampling()
    model.corner_plot(coordinates=['equatorial', 'galactic'])
    model.sky_direction_posterior(coordinates=['equatorial', 'galactic'])
    plt.show()

def run_dipole_quadrupole(dmap):
    model = Multipole(dmap, likelihood='general_poisson', ells=[0,1,2])
    model.run_nested_sampling(step=True)
    model.corner_plot(coordinates=['equatorial', 'galactic'])
    model.sky_direction_posterior(coordinates=['equatorial', 'galactic'])
    plt.show()

def racs_low1_mask(maps):
    masker = Masker(maps, coordinate_system='equatorial')
    masker.mask_equatorial_poles(north_radius=61, south_radius=13)
    masker.mask_galactic_plane(5)
    masker.mask_slice(0.65, 18.4, 4)
    masker.mask_slice(333.4, 19.1, 4)
    masker.mask_slice(256.2, 5.1, 4)
    masker.mask_slice(188.2, 11.2, 4)
    masker.mask_slice(84.0, 22.0, 4)
    masker.mask_slice(80.0, -70.0, 4)
    maps = masker.get_masked_density_map()
    return maps

RLOW_FREQ = 887.5e6
RMID_FREQ = 1367.5e6
FLUX_CUT_MJY = 15
FLUX_MAX_MJY = 1000
DATASET: Literal['mid1', 'mid1-25as'] = 'mid1-25as'

mapping = {
    'mid1': {
        'flux': 'Total_flux',
        'name': 'name'
    },
    'mid1-25as': {
        'flux': 'Total_flux',
        'name': 'Name'
    } 
}

data = DataLoader('racs', DATASET).load()
rmid = CatalogueToMap(data)
data = DataLoader('racs', 'low1').load()
rlow = CatalogueToMap(data)

rmid.make_cut(mapping[DATASET]['flux'], FLUX_CUT_MJY, FLUX_MAX_MJY)
rlow.make_cut('total_flux_source', FLUX_CUT_MJY, FLUX_MAX_MJY)

dmap = rmid.make_density_map('equatorial')
dmap = racs_low1_mask(dmap)
smooth_map(dmap)
plt.show()

# match mid (A) to low (B)
xmatch = CrossMatch(
    rmid.catalogue,
    rlow.catalogue,
    coordinate_system='equatorial'
)
xmatch.cross_match(radius=5, source_name_A_column=mapping[DATASET]['name'])
matches = xmatch.get_crossmatch_table()
where_match_exists = matches['source_idx_B'] != -1
matched = matches[where_match_exists]

# Use row indices from crossmatch output so non-unique source_name values
# are handled correctly (each matched row maps to exactly one source)
# there is a duplicate name in racs low but they are distinct astrophysical sources
rlow_flux = rlow.catalogue['total_flux_source'][matched['source_idx_B']]
rmid_flux = rmid.catalogue[mapping[DATASET]['flux']][matched['source_idx_A']]

# S_nu ~ nu ** (-alpha)
alpha = (
    ( np.log(rmid_flux) - np.log(rlow_flux) )
  / ( np.log(RLOW_FREQ) - np.log(RMID_FREQ) )
)

# scale rlow flow -> rmid flux given median alpha used in duchesne+23
rmid_flux_scaled = rlow_flux * (RMID_FREQ / RLOW_FREQ) ** -0.88

# Subset of rmid containing only sources with successful rlow matches,
# augmented with spectral index for each matched source.
rmid_xmatched = rmid.catalogue[matched['source_idx_A']].copy()
rmid_xmatched['alpha'] = alpha
rmid_xmatched['total_flux_scaled'] = rmid_flux_scaled
ALPHA_CUT = 0.3
mean_alpha = 0.9 # np.mean(rmid_xmatched['alpha'])
std_alpha = np.std(rmid_xmatched['alpha'])
to_include = alpha > ALPHA_CUT
# to_include = (alpha > (mean_alpha - std_alpha)) & (alpha < (mean_alpha + std_alpha))
rmid_xmatched_cut = rmid_xmatched[to_include]
rmid_xmatched_notcut = rmid_xmatched[~to_include]

plt.hist(rmid_xmatched_cut['alpha'], bins=200)
plt.show()

plt.hist(
    rmid_xmatched[mapping[DATASET]['flux']] / rmid_xmatched['total_flux_scaled'],
    bins=200,
    range=[0.5, 2.5],
    alpha=0.3
)
plt.hist(
    rmid_xmatched_cut[mapping[DATASET]['flux']] / rmid_xmatched_cut['total_flux_scaled'],
    bins=200,
    range=[0.5, 2.5],
    alpha=0.3
)
plt.show()

NSIDE = 64
rmid_match = CatalogueToMap(rmid_xmatched_cut)
dmap = rmid_match.make_density_map(coordinate_system='equatorial', nside=NSIDE)
amap = rmid_match.make_parameter_map(
    column_name='alpha', 
    coordinate_system='equatorial',
    nside=NSIDE
)
fmap = rmid_match.make_parameter_map(
    column_name=[mapping[DATASET]['flux'], 'total_flux_scaled'],
    coordinate_system='equatorial',
    operation='/',
    nside=NSIDE
)
fmap = np.asarray([np.nanmedian(i) for i in fmap])

dmap, amap, fmap = racs_low1_mask([dmap, amap, fmap])

# hp.projview(dmap)
# hp.projview(amap)
hp.projview(fmap, min=0.8, max=1.2, cmap='coolwarm_r')
smooth_map(dmap)
plt.show()

# run_dipole_quadrupole(dmap)
run_dipole(dmap)

# rmid_match = CatalogueToMap(rmid_xmatched_notcut)
# cut_dmap = rmid_match.make_density_map(coordinate_system='equatorial')
# cut_amap = rmid_match.make_parameter_map(column_name='alpha', coordinate_system='equatorial')
#
# cut_dmap, cut_amap = racs_low1_mask([cut_dmap, cut_amap])
#
# hp.projview(cut_dmap)
# hp.projview(cut_amap)
# smooth_map(cut_dmap)
# plt.show()
#
# run_dipole(cut_dmap, large_D=True)
