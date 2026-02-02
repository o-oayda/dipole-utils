from dipoleutils.utils import DataLoader, CatalogueToMap, Masker
import matplotlib.pyplot as plt
import numpy as np
from dipoleska.models.dipole import Dipole
from scripts.racs_low2 import (
    smooth_wrapper, density_wrapper, make_integer_cmap, coolwarm_upper_half
)


VARIANTS = ['low2-45as-patch', 'low2-25as-patch']
# VARIANTS = ['low2-25as-patch']
target_to_coord = {
    'equatorial': ['C'],
    'galactic': ['C', 'G']
}


if __name__ == '__main__':
    dmaps = []

    for variant in VARIANTS:
        loader = DataLoader('racs', variant)
        rl2 = loader.load()

        processor = CatalogueToMap(rl2) # type: ignore
        print(processor.get_column_names())
        processor.make_cut(column_name='Total_flux', minimum=15, maximum=None)
        dmap = processor.make_density_map(coordinate_system='equatorial')
        emap = processor.make_parameter_map('Noise', 'equatorial')
        masker = Masker(dmap, coordinate_system='equatorial')
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

    # model = Dipole(dmap, likelihood='general_poisson')
    # model.run_nested_sampling()
    # model.corner_plot(coordinates=['equatorial', 'galactic'])
    # model.sky_direction_posterior(coordinates=['equatorial', 'galactic'])
    # plt.show()
