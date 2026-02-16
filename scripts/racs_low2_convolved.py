from dipoleutils.utils import DataLoader, CatalogueToMap, Masker
import matplotlib.pyplot as plt
import numpy as np
from dipoleska.models.dipole import Dipole
from scripts.racs_low2 import (
    smooth_wrapper, density_wrapper, make_integer_cmap, coolwarm_upper_half
)


# VARIANTS = ['low3', 'low3-scaled']
VARIANTS = ['mid1-25as']
target_to_coord = {
    'equatorial': ['C'],
    'galactic': ['C', 'G']
}
LIKELIHOOD = 'general_poisson_rms'
colnames = {
    **dict.fromkeys(
        ['low3', 'low3-scaled', 'low2-25as-path', 'low2-45as-patch', 'low2-patch', 'mid1-25as'],
        ('Total_flux', 'Noise')
    ),
    **dict.fromkeys(
        ['low1'],
        ('total_flux_source', 'noise')
    ),
    **dict.fromkeys(
        ['mid1'],
        ('total_flux', 'noise')
    )
}


if __name__ == '__main__':
    dmaps = []

    for variant in VARIANTS:
        RUN_OUT_DIR = f'tmp/{variant}'
        loader = DataLoader('racs', variant)
        rl2 = loader.load()

        processor = CatalogueToMap(rl2) # type: ignore
        print(processor.get_column_names())
        fluxkey, noisekey = colnames[variant]
        emap = processor.make_parameter_map(noisekey, 'equatorial', operation='mean')
        processor.make_cut(column_name=fluxkey, minimum=40, maximum=None)
        dmap = processor.make_density_map(coordinate_system='equatorial')
        masker = Masker(dmap, coordinate_system='equatorial')
        if variant == 'mid1':
            masker.mask_galactic_plane(latitude_cut=5)
        else:
            masker.mask_galactic_plane(latitude_cut=5)

        if 'low2-45as' in variant:
            masker.mask_equatorial_poles(north_radius=55)
            masker.mask_a_team_sources(radius_deg=3)
        elif variant == 'low2-25as-patch':
            # north radius = 65
            # 67 thresh
            masker.mask_equatorial_poles(north_radius=68, south_radius=17)
            masker.mask_slice(23, 22, 8) # square patch at northern limits
            masker.mask_slice(260, -50, 3) # no a team sources hit this
            masker.mask_slice(95, -5, 3) # area of heightened density
            masker.mask_slice(126, -65, 2) # drop in diffmap
            masker.mask_slice(350, -74, 4) # source eq pole survey limit
            masker.mask_a_team_sources(radius_deg=3)
        elif variant == 'low1':
            masker.mask_equatorial_poles(north_radius=61, south_radius=13)
            masker.mask_slice(0.65, 18.4, 3)
            masker.mask_slice(333.4, 19.1, 3)
            masker.mask_slice(256.2, 5.1, 3)
            masker.mask_slice(188.2, 11.2, 3)
            masker.mask_slice(84.0, 22.0, 3)
            masker.mask_slice(80.0, -70.0, 3)
        elif variant == 'low2-patch':
            masker.mask_equatorial_poles(north_radius=43)
            masker.mask_a_team_sources(radius_deg=3)
        elif variant == 'low3':
            masker.mask_equatorial_poles(north_radius=43)
            masker.mask_a_team_sources(radius_deg=3)
        elif variant == 'low3-scaled':
            masker.mask_equatorial_poles(north_radius=43)
            masker.mask_a_team_sources(radius_deg=3)
        elif variant == 'mid1':
            masker.mask_equatorial_poles(north_radius=41)
            masker.mask_slice(300, 40, 3)
        elif variant == 'mid1-25as':
            masker.mask_equatorial_poles(north_radius=61)
            masker.mask_slice(300, 40, 3)
        else:
            raise Exception

        # masker.mask_a_team_sources(radius_deg=6, source_names=['Cygnus A'])
        dmap = masker.get_masked_density_map()
        binary_mask = np.asarray(masker.get_mask_map(), dtype=np.bool_)
        emap[~binary_mask] = np.nan

        # major hack for now just to see what's happening in 25"
        # if variant == 'low2-25"-patch':
            # mask = masker.get_mask_map()
            # zero_pixels = np.where(dmap <= 1)[0]
            # dmap[zero_pixels] = np.nan

        print(f'Number of sources: {int( np.nansum(dmap) )}')

        smooth_wrapper(dmap, survey=variant, min=13.84, max=14.94)
        density_wrapper(dmap, survey=variant)
        density_wrapper(emap, survey=f'{variant}_error', norm='log')

        plt.close('all')
        dmaps.append(dmap)

        model = Dipole(dmap, likelihood=LIKELIHOOD, rms_map=emap)
        if 'rms' in LIKELIHOOD:
            model.prior.change_prior(1, new_prior=['Uniform', -1., 1.])
        model.run_nested_sampling(
            output_dir=f'{RUN_OUT_DIR}/{LIKELIHOOD}',
            reactive_sampler_kwargs={'resume': 'overwrite'}
        )
        model.corner_plot(
            coordinates=['equatorial', 'galactic'],
            save_path=model.ultranest_sampler.logs['run_dir']
        )
        model.sky_direction_posterior(
            coordinates=['equatorial', 'galactic'],
            save_path=model.ultranest_sampler.logs['run_dir']
        )
        plt.close('all')

    if len(dmaps) > 1:
        diffmap = dmaps[1] - dmaps[0]
        discrete_kwargs = make_integer_cmap(diffmap, cmap='plasma')
        
        diff_survey = rf'25" $-$ 45"'
        density_wrapper(
            diffmap, 
            survey=diff_survey,
            # survey=rf'Original $-$ patch (45")', 
            title=rf'$\Delta = {np.nansum(diffmap)}$',
            unit='Source count difference',
            **discrete_kwargs
        )
        smooth_wrapper(
            diffmap,
            survey=diff_survey,
            cmap='plasma'
        )

