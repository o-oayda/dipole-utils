# %%
from dipoleutils.utils import DataLoader, CrossMatch, CatalogueToMap, Masker
import matplotlib.pyplot as plt
from astropy.table import Table
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
# %%
loader = DataLoader('racs', 'low1')
racs = loader.load()
loader = DataLoader('nvss')
nvss = loader.load()

assert type(racs) is Table
assert type(nvss) is Table

racs = CatalogueToMap(racs)
nvss = CatalogueToMap(nvss)
nvss.add_coordinate_system(starting_frame='equatorial', target_frame='galactic')
racs.make_cut('total_flux_source', 15, 1000)
nvss.make_cut('integrated_flux', 15, 1000)

# %%
racs_unique = racs
racs_overlap = racs.copy_independent()
nvss_unique = nvss
nvss_overlap = nvss.copy_independent()

# %%
# build the masks
COORD_SYSTEM = 'galactic'
custom_mask_racs = [
    (0.65, 18.4, 3.44),
    (333.4, 19.1, 3.44),
    (256.2, 5.1, 3.44),
    (188.2, 11.2, 3.44),
    (84.0, 22.0, 3.44),
    (80.0, -70.0, 3.44)
]
custom_mask_nvss = [
    (50.8, -37.1, 1.15),
    (84.0, -4.0, 2.86),
    (117, -3, 2.86),
    (188, 11, 2.86),
    (360-99.8, -0.5, 1.15),
    (360-172.2, 2.2, 1.15),
    (7.9, 43.5, 1.15),
    (97.2, 59.0, 1.15)
]

def to_galactic(coord_list):
  galactic_coords = []
  for ra, dec, radius in coord_list:
    sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs') # type: ignore
    gal = sc.galactic
    galactic_coords.append((gal.l.degree, gal.b.degree, radius)) # type: ignore
  return galactic_coords

custom_mask_racs = to_galactic(custom_mask_racs)
custom_mask_nvss = to_galactic(custom_mask_nvss)
# RACS RANGE: -84.56 <--> 29.13
# NVSS RANGE: -40.37 <--> 89.72
# OVERLAP   : -40.37 <--> 29.13

# (1) MAKE RACS TOTAL SURVEY MASK
blank_map = np.ones(hp.nside2npix(64), dtype=np.int64)
racs_mask = Masker(blank_map, coordinate_system=COORD_SYSTEM)
racs_mask.mask_equatorial_poles(north_radius=61, south_radius=13)
racs_mask.mask_galactic_plane(latitude_cut=5)
for cmask in custom_mask_racs: racs_mask.mask_slice(*cmask)
racs_mmap = racs_mask.get_mask_map()
racs_pixels = racs_mask.get_unmasked_pixels()

# (2) MAKE NVSS TOTAL SURVEY MASK
nvss_mask = Masker(blank_map, coordinate_system=COORD_SYSTEM)
nvss_mask.mask_equatorial_poles(south_radius=50)
for cmask in custom_mask_nvss: nvss_mask.mask_slice(*cmask)
nvss_mask.mask_galactic_plane(latitude_cut=10)
nvss_mmap = nvss_mask.get_mask_map()
nvss_pixels = nvss_mask.get_unmasked_pixels()

# (3) & (4) define overlap with set intersections of pixels
overlap_mmap = racs_mmap & nvss_mmap
unique_racs_mmap = racs_mmap & ~overlap_mmap
unique_nvss_mmap = nvss_mmap & ~overlap_mmap

hp.projview(racs_mmap, graticule=True, sub=511, title='RACS survey mask')
hp.projview(nvss_mmap, graticule=True, sub=512, title='NVSS survey mask')
hp.projview(overlap_mmap, graticule=True, sub=513, title='Common pixels')
hp.projview(unique_racs_mmap, graticule=True, sub=514, title='Unique RACS pixels')
hp.projview(unique_nvss_mmap, graticule=True, sub=515, title='Unique NVSS pixels')

# %%
# exclude all sources which lie inside the pixels-space masks
racs.mask_in_pixel_space(
    pixels_to_mask=np.where(unique_racs_mmap == 0)[0],
    coordinate_system=COORD_SYSTEM,
    nside=64
)
racs_dmap = racs.make_density_map(coordinate_system=COORD_SYSTEM)

nvss.mask_in_pixel_space(
    pixels_to_mask=np.where(unique_nvss_mmap == 0)[0],
    coordinate_system=COORD_SYSTEM,
    nside=64
)
nvss_dmap = nvss.make_density_map(coordinate_system=COORD_SYSTEM)

# overlap regions
racs_overlap.mask_in_pixel_space(
    pixels_to_mask=np.where(overlap_mmap == 0)[0],
    coordinate_system=COORD_SYSTEM,
    nside=64
)
nvss_overlap.mask_in_pixel_space(
    pixels_to_mask=np.where(overlap_mmap == 0)[0],
    coordinate_system=COORD_SYSTEM,
    nside=64
)
xmatch_overlap = CrossMatch(
    racs_overlap.get_catalogue(),
    nvss_overlap.get_catalogue(),
    coordinate_system=COORD_SYSTEM
)
xmatch_overlap.cross_match(radius=5)
unique_racs_table = xmatch_overlap.get_unique_A_sources()
unique_nvss_table = xmatch_overlap.get_unique_B_sources()
common_sources_table = xmatch_overlap.get_common_sources()

assert (
      2 * len(common_sources_table)
    + len(unique_racs_table)
    + len(unique_nvss_table)
    ==
      racs_overlap.get_source_count()
    + nvss_overlap.get_source_count()
)

nu_overlap = CatalogueToMap(unique_nvss_table)
c_overlap = CatalogueToMap(common_sources_table)
ru_overlap = CatalogueToMap(unique_racs_table)
ru_overlap_dmap = ru_overlap.make_density_map(coordinate_system=COORD_SYSTEM)
nu_overlap_dmap = nu_overlap.make_density_map(coordinate_system=COORD_SYSTEM)
c_overlap_dmap = c_overlap.make_density_map(coordinate_system=COORD_SYSTEM)

assert (
      2 * np.sum(c_overlap_dmap)
    + np.sum(ru_overlap_dmap)
    + np.sum(nu_overlap_dmap)
    ==
      racs_overlap.get_source_count()
    + nvss_overlap.get_source_count()
)

print('% of RACS sources with counterpart: '
  f'{xmatch_overlap.get_number_of_matches()*100 / racs_overlap.get_source_count():.1f}'
)
print('% of NVSS sources with counterpart: '
  f'{xmatch_overlap.get_number_of_matches()*100 / nvss_overlap.get_source_count():.1f}'
)

# %%
hp.projview(racs_dmap, sub=511, cmap='plasma', title='Unique RACS density map')
hp.projview(nvss_dmap, sub=512, cmap='plasma', title='Unique NVSS density map')
hp.projview(ru_overlap_dmap, sub=513, cmap='plasma', title='Unique RACS sources (overlap)')
hp.projview(nu_overlap_dmap, sub=514, cmap='plasma', title='Unique NVSS sources (overlap)')
hp.projview(c_overlap_dmap, sub=515, cmap='plasma', title='Common sources (overlap)')
# %%
## now apply masks
racs_dmap = racs_dmap.astype(np.float64)
nvss_dmap = nvss_dmap.astype(np.float64)
ru_overlap_dmap = ru_overlap_dmap.astype(np.float64)
nu_overlap_dmap = nu_overlap_dmap.astype(np.float64)
c_overlap_dmap = c_overlap_dmap.astype(np.float64)

racs_dmap[~unique_racs_mmap.astype(bool)] = np.nan
nvss_dmap[~unique_nvss_mmap.astype(bool)] = np.nan
ru_overlap_dmap[~overlap_mmap.astype(bool)] = np.nan
nu_overlap_dmap[~overlap_mmap.astype(bool)] = np.nan
c_overlap_dmap[~overlap_mmap.astype(bool)] = np.nan

hp.projview(racs_dmap, sub=511, cmap='plasma', title='Unique RACS density map (masked)')
hp.projview(nvss_dmap, sub=512, cmap='plasma', title='Unique NVSS density map (masked)')
hp.projview(ru_overlap_dmap, sub=513, cmap='plasma', title='Unique RACS sources (overlap, masked)')
hp.projview(nu_overlap_dmap, sub=514, cmap='plasma', title='Unique NVSS sources (overlap, masked)')
hp.projview(c_overlap_dmap, sub=515, cmap='plasma', title='Common sources (overlap, masked)')
# %%
np.save('data/ru_dmap.npy', racs_dmap)
np.save('data/nu_dmap.npy', nvss_dmap)
np.save('data/ru_overlap_dmap.npy', ru_overlap_dmap)
np.save('data/nu_overlap_dmap.npy', nu_overlap_dmap)
np.save('data/c_overlap_dmap.npy', c_overlap_dmap)
# %%
racs_dmap = np.load('data/ru_dmap.npy')
nvss_dmap = np.load('data/nu_dmap.npy')
ru_overlap_dmap = np.load('data/ru_overlap_dmap.npy')
nu_overlap_dmap = np.load('data/nu_overlap_dmap.npy')
c_overlap_dmap = np.load('data/c_overlap_dmap.npy')

hp.projview(racs_dmap, sub=511, cmap='plasma', title='Unique RACS density map (masked)')
hp.projview(nvss_dmap, sub=512, cmap='plasma', title='Unique NVSS density map (masked)')
hp.projview(ru_overlap_dmap, sub=513, cmap='plasma', title='Unique RACS sources (overlap, masked)')
hp.projview(nu_overlap_dmap, sub=514, cmap='plasma', title='Unique NVSS sources (overlap, masked)')
hp.projview(c_overlap_dmap, sub=515, cmap='plasma', title='Common sources (overlap, masked)')
# %%
