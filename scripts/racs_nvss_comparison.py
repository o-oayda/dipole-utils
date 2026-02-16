from dipoleska.models.dipole import Dipole
from dipoleska.utils.posterior import Posterior
from dipoleutils.utils import DataLoader, CatalogueToMap
import matplotlib.pyplot as plt
from astropy.table import Table
import healpy as hp
import numpy as np
import argparse
from dipoleutils.utils.mask import Masker
import matplotlib as mpl


argparser = argparse.ArgumentParser()
argparser.add_argument(
    '--load-previous',
    action='store_true'
)
args = argparser.parse_args()

LOAD_PREVIOUS = args.load_previous

if not LOAD_PREVIOUS:
    loader = DataLoader('racs', 'low1')
    racs = loader.load()
    loader = DataLoader('nvss')
    nvss = loader.load()

    assert type(racs) is Table
    assert type(nvss) is Table

    racs = CatalogueToMap(racs)
    nvss = CatalogueToMap(nvss)
    racs.add_coordinate_system(starting_frame='equatorial', target_frame='galactic')
    nvss.add_coordinate_system(starting_frame='equatorial', target_frame='galactic')

    racs.make_cut('total_flux_source', 15, 1000)
    nvss.make_cut('integrated_flux', 15, 1000)

    racs_dmap = racs.make_density_map()
    nvss_dmap = nvss.make_density_map()

    racs_masks = [
        (0.65, 18.4, 3), (333.4, 19.1, 3), (256.2, 5.1, 3),
        (188.2, 11.2, 3), (84.0, 22.0, 3), (80.0, -70.0, 3)
    ]
    masker = Masker(racs_dmap, coordinate_system='equatorial')
    masker.mask_galactic_plane(latitude_cut=5)
    masker.mask_equatorial_poles(north_radius=61, south_radius=13)
    for ra, dec, rad in racs_masks:
        masker.mask_slice(0.65, 18.4, 3)
        masker.mask_slice(333.4, 19.1, 3)
        masker.mask_slice(256.2, 5.1, 3)
        masker.mask_slice(188.2, 11.2, 3)
        masker.mask_slice(84.0, 22.0, 3)
        masker.mask_slice(80.0, -70.0, 3)
    racs_dmap = masker.get_masked_density_map()

    nvss_masks = [
        (50.8, -37.1, 0.02), (84.0, -4.0, 0.05), (117, -3, 0.05), (188, 11, 0.05),
        (360-99.8, -0.5, 0.02), (360-172.2, 2.2, 0.02), (7.9, 43.5, 0.02), 
        (97.2, 59.0, 0.02)
    ]
    masker = Masker(nvss_dmap, coordinate_system='equatorial')
    masker.mask_galactic_plane(latitude_cut=10)
    masker.mask_equatorial_poles(south_radius=50)
    for ra, dec, radius_rad in nvss_masks:
        masker.mask_slice(ra, dec, np.rad2deg(radius_rad))
    nvss_dmap = masker.get_masked_density_map()

    hp.projview(
        racs_dmap, title='RACS Density Map', sub=211, graticule=True,
        graticule_labels=True, coord=['C', 'G']
    )
    hp.projview(
        nvss_dmap, title='NVSS Density Map', sub=212, graticule=True,
        graticule_labels=True, coord=['C', 'G']
    )

    plt.tight_layout()
    plt.show()

    LIKELIHOOD = 'poisson'
    RUN_OUT_DIR = 'racs_nvss_out'
    samples = ['racs', 'nvss']
    for i, dmap in enumerate([racs_dmap, nvss_dmap]):
        model = Dipole(dmap, likelihood=LIKELIHOOD)
        if 'rms' in LIKELIHOOD:
            model.prior.change_prior(1, new_prior=['Uniform', -1., 1.])
        model.run_nested_sampling(
            output_dir=f'{RUN_OUT_DIR}/{samples[i]}',
            reactive_sampler_kwargs={'resume': 'overwrite'}
        )
        model.corner_plot(
            coordinates=['equatorial', 'galactic'],
            save_path=model.ultranest_sampler.logs['run_dir']
        )
        model.sky_direction_posterior(
            coordinates=['equatorial', 'galactic'],
            save_path=model.ultranest_sampler.logs['run_dir'],
            contour_levels=[1., 2.]
        )
        plt.close('all')

CMB_L = 264.021
CMB_B = 48.253
mpl.rc('text', usetex=True)
model_racs = Posterior('racs_nvss_out/racs')
model_nvss = Posterior('racs_nvss_out/nvss')
model_nvss.add_comparison_run(model_racs)
model_nvss.sky_direction_posterior(
    contour_levels=[1., 2.], 
    coordinates=['equatorial', 'galactic'],
    label='nice',
    smooth=0.08
)
ax = plt.gca()
ax.get_legend().remove()
ax.scatter(np.deg2rad(360-CMB_L), np.deg2rad(CMB_B), marker='*', s=100, color='black')
plt.savefig('racs_nvss_out/combined_sky.png', dpi=300, bbox_inches='tight')
plt.show()
