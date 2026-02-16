from corner import corner
from dipoleutils.utils.data_loader import DataLoader
from dipoleutils.utils.plotting import smooth_map
from dipoleutils.utils.samples import CatalogueToMap
import healpy as hp
import matplotlib.pyplot as plt
from dipoleska.utils.map_process import MapProcessor
import numpy as np
from dipoleska.models.dipole import Dipole
'''
'id','catalogue_id','name','source_id','field_id','ra',
'dec','dec_corr','e_ra','e_dec','e_dec_corr','total_flux','e_total_flux_pybdsf','e_t
otal_flux','peak_flux','e_peak_flux_pybdsf','e_peak_flux','maj_axis','min_axis','pa'
,'e_maj_axis','e_min_axis','e_pa','dc_maj_axis','dc_min_axis','dc_pa','e_dc_maj_axis
','e_dc_min_axis','e_dc_pa','noise','tile_l','tile_m','tile_sep','gal_lon','gal_lat'
,'psf_maj','psf_min','psf_pa','s_code','n_gaussians','flag','scan_start_mjd','scan_l
ength','sbid','e_flux_scale'
'''

data = DataLoader('racs', 'mid1').load()
rmid = CatalogueToMap(data)
rmid.catalogue['perc_err'] = rmid.catalogue['e_total_flux'] / rmid.catalogue['total_flux']
psf_maj = rmid.make_parameter_map('psf_maj', coordinate_system='equatorial')
psf_min = rmid.make_parameter_map('psf_min', coordinate_system='equatorial')
psf_area = np.pi * psf_maj * psf_min
noise = rmid.make_parameter_map('noise', coordinate_system='equatorial')
era = rmid.make_parameter_map('e_ra', coordinate_system='equatorial')
edec = rmid.make_parameter_map('e_dec', coordinate_system='equatorial')
flux = rmid.make_parameter_map('perc_err', coordinate_system='equatorial')
rmid.make_cut(column_name='total_flux', minimum=10, maximum=None)
dmap = rmid.make_density_map('equatorial')

processor = MapProcessor([psf_area, noise / psf_area, dmap, dmap])
processor.mask(
    classification=['galactic_plane', 'north_equatorial', 'south_equatorial'], 
    output_frame='C', 
    radius=[8, 70, 35]
)
maps = processor.density_maps

hp.projview(maps[-1])
plt.show()

plt.figure(figsize=(5,7))
for i, map in enumerate(maps):
    if i == len(maps) - 1:
        smooth_map(maps[i], sub=int(f'{len(maps)}1{i+1}'))
    else:
        hp.projview(
            maps[i], 
            sub=int(f'{len(maps)}1{i+1}'), 
            norm='log' if i in [1, 2] else None
        )
plt.show()

# samples = np.vstack(maps).T
# mask = (~np.isnan(samples)).all(axis=1)
# corner(samples[mask])
# plt.show()

model = Dipole(maps[-1], likelihood='general_poisson')
model.run_nested_sampling()
model.corner_plot(coordinates=['equatorial', 'galactic'])
model.sky_direction_posterior(coordinates=['equatorial', 'galactic'])
plt.show()

# ## permute source positions and observe effects
# cat = rmid.get_catalogue()
# ra, dec = cat['ra'], cat['dec']
# sigra, sigdec = cat['e_ra'], cat['e_dec']
#
# N_STRAP = 10
# for i in range(N_STRAP):
#     new_ra = ra + np.random.normal(scale=10*sigra)
#     new_dec = dec + np.random.normal(scale=10*sigdec)
#
# new_ra = cat['ra']
# new_dec = cat['dec_corr']
# ipix = hp.ang2pix(64, new_ra, new_dec, lonlat=True)
# new_dmap = np.bincount(ipix, minlength=49152)
# processor = MapProcessor(new_dmap)
# processor.mask(
#     classification=['galactic_plane', 'north_equatorial'], 
#     output_frame='C', 
#     radius=[8, 42]
# )
# new_dmap = processor.density_map
#
# smooth_map(new_dmap)
# plt.show()
