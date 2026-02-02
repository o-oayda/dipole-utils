from typing import Any
from numpy.typing import NDArray
from dipoleutils.utils import DataLoader, CatalogueToMap, Masker
from dipoleutils.utils.plotting import smooth_map, density_map
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import os


target_to_coord = {
    'equatorial': ['C'],
    'galactic': ['C', 'G']
}


def coolwarm_upper_half(name: str = 'coolwarm_upper') -> colors.LinearSegmentedColormap:
    """Return a coolwarm colormap truncated to its warm half."""
    base_cmap = plt.get_cmap('coolwarm')
    samples = base_cmap(np.linspace(0.5, 1.0, 256))
    return colors.LinearSegmentedColormap.from_list(name, samples)


def smooth_wrapper(dmap, survey: str, **kwargs):
    for target, coord in target_to_coord.items():
        smooth_map(
            dmap, 
            coord=coord,
            rlabel=survey,
            graticule=True,
            graticule_labels=True,
            unit='Smoothed count per pixel',
            **kwargs
        )
        os.makedirs(f'tmp/{target}', exist_ok=True)
        plt.savefig(f'tmp/{target}/{survey}_smooth.png', dpi=300, bbox_inches='tight')


def density_wrapper(dmap, survey: str, **kwargs):
    for target, coord in target_to_coord.items():
        density_map(
            dmap, 
            graticule=True, 
            graticule_labels=True, 
            rlabel=survey,
            coord=coord,
            **kwargs,
        )
        os.makedirs(f'tmp/{target}', exist_ok=True)
        plt.savefig(f'tmp/{target}/{survey}_counts.png', dpi=300, bbox_inches='tight')


def make_integer_cmap(valid_pixels: NDArray, cmap: str = 'coolwarm') -> dict[str, Any]:
    # Flatten and drop NaNs so we can inspect actual integer levels.
    finite_pixels = np.asarray(valid_pixels).ravel()
    finite_pixels = finite_pixels[np.isfinite(finite_pixels)]
    if finite_pixels.size == 0:
        return {}

    # Discretise coolwarm so each integer difference gets a single color.
    unique_levels = np.unique(np.rint(finite_pixels).astype(int))
    if 0 not in unique_levels:
        zero_idx = np.searchsorted(unique_levels, 0)
        unique_levels = np.insert(unique_levels, zero_idx, 0)

    base_cmap = plt.get_cmap(cmap)

    # Force the diverging normalisation to be centred on zero.
    neg_span = -min(unique_levels[0], 0)
    pos_span = max(unique_levels[-1], 0)

    def _position(level: int) -> float:
        if level < 0 and neg_span:
            return 0.5 - 0.5 * (abs(level) / neg_span)
        if level > 0 and pos_span:
            return 0.5 + 0.5 * (level / pos_span)
        return 0.5

    color_positions = np.array([_position(level) for level in unique_levels])
    discrete_cmap = colors.ListedColormap(base_cmap(color_positions))

    bounds = np.concatenate((unique_levels - 0.5, [unique_levels[-1] + 0.5]))
    if unique_levels[0] >= 0:
        bounds[0] = 0.0

    norm = colors.BoundaryNorm(bounds, discrete_cmap.N, clip=True)
    discrete_kwargs = {'cmap': discrete_cmap, 'norm': norm}

    return discrete_kwargs


if __name__ == '__main__':
    ## Racs-low2
    loader = DataLoader('racs', 'low2')
    rl2 = loader.load()

    processor = CatalogueToMap(rl2) # type: ignore
    og_rl2 = processor.make_density_map(coordinate_system='equatorial')
    processor.make_cut(column_name='Total_flux', minimum=15, maximum=None)
    rl2 = processor.get_catalogue()

    # plot_log_log_histogram(rl2['Total_flux'], bins=100)

    dmap = processor.make_density_map(coordinate_system='equatorial')
    masker = Masker(dmap, coordinate_system='equatorial')
    masker.mask_equatorial_poles(north_radius=45)
    masker.mask_galactic_plane(latitude_cut=5)
    masker.mask_a_team_sources(radius_deg=6, source_names=['Cygnus A'])
    dmap2 = masker.get_masked_density_map()
    emap = processor.make_parameter_map('Noise', 'equatorial')

    binary_mask = np.asarray(masker.get_mask_map(), dtype=np.bool_)
    emap[~binary_mask] = np.nan

    print(f'Number of sources: {int( np.nansum(dmap2) )}')

    smooth_wrapper(dmap2, survey='RACS-low2')
    density_wrapper(dmap2, survey='RACS-low2')
    density_wrapper(emap, survey=f'RACS-low2_error', norm='log')

    del rl2

    ## RACS-low2-patch
    loader = DataLoader('racs', 'low2-patch')
    rl2p = loader.load()

    processor = CatalogueToMap(rl2p) # type: ignore
    og_rl2p = processor.make_density_map(coordinate_system='equatorial')
    processor.make_cut(column_name='Total_flux', minimum=15, maximum=None)
    rl2p = processor.get_catalogue()

    # plt.figure()
    # plot_log_log_histogram(rl2p['Total_flux'], bins=100)

    dmap = processor.make_density_map(coordinate_system='equatorial')
    masker = Masker(dmap, coordinate_system='equatorial')
    masker.mask_equatorial_poles(north_radius=45)
    masker.mask_galactic_plane(latitude_cut=5)
    masker.mask_a_team_sources(radius_deg=6, source_names=['Cygnus A'])
    dmap2p = masker.get_masked_density_map()

    print(f'Number of sources: {int( np.nansum(dmap2p) )}')

    smooth_wrapper(dmap2p, survey='RACS-low2-patch')
    density_wrapper(dmap2p, survey='RACS-low2-patch')

    dmap_sub = np.subtract(dmap2, dmap2p)

    valid_pixels = dmap_sub[~np.isnan(dmap_sub)]
    discrete_kwargs = make_integer_cmap(valid_pixels)

    density_wrapper(
        dmap_sub, 
        survey=rf'Original $-$ patch', 
        title=rf'$\Delta = {np.nansum(dmap_sub)}$',
        unit='Source count difference',
        **discrete_kwargs
    )

    ### check OG sample
    density_wrapper(og_rl2, survey='RACS-low2 (full)')
    density_wrapper(og_rl2p, survey='RACS-low2-patch (full)')
    plt.close('all')
