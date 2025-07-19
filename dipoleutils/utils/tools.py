import numpy as np
from numpy.typing import NDArray
import healpy as hp

def angles_to_density_map(
        azimuthal_angles: NDArray[np.float_],
        polar_angles: NDArray[np.float_],
        lonlat: bool = True,
        nest: bool = False,
        nside: int = 64
    ) -> NDArray[np.int_]:
    '''Turn vectors of angles to a healpy density map.'''
    if lonlat:
        pixels = hp.ang2pix(
            nside,
            azimuthal_angles, # lon (deg)
            polar_angles,     # lat (deg)
            lonlat=lonlat,
            nest=nest
        )
    else:
        pixels = hp.ang2pix(
            nside,
            polar_angles,     # colat (rad)
            azimuthal_angles, # lon (rad)
            lonlat=lonlat,
            nest=nest
        )
    density_map = np.bincount(pixels, minlength=hp.nside2npix(nside))
    return density_map