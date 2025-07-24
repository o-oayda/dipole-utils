import numpy as np
from numpy.typing import NDArray
import healpy as hp
from dipoleutils.utils.physics import change_source_coordinates

class Masker:
    def __init__(self,
            density_map: NDArray[np.int_],
            coordinate_system: str
    ) -> None:
        self.density_map = density_map
        self.coordinate_system = coordinate_system
        self.mask_map = np.ones_like(density_map, dtype=bool)
        self.masked_pixel_indices = set()
        self.nside = hp.get_nside(density_map)
        self.npix = hp.nside2npix(self.nside)
        self.all_indices = set(np.arange(self.npix))

    def _get_pole_vecs_in_native_coords(self, 
            pole_lon_deg: NDArray[np.float64],
            pole_lat_deg: NDArray[np.float64],
            source_coordinate_system: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Convert pole coordinates to native coordinate system and return as unit
        vectors.
        
        :param pole_lon_deg: Pole longitudes in degrees
        :param pole_lat_deg: Pole latitudes in degrees  
        :param source_coordinate_system: The coordinate system the poles are
            defined in
        :return: Tuple of (north_pole_vector, south_pole_vector) in native
            coordinates
        """
        # Convert to native coordinates if needed
        if self.coordinate_system != source_coordinate_system:
            pole_lon_native_deg, pole_lat_native_deg = change_source_coordinates(
                pole_lon_deg,
                pole_lat_deg,
                native_coordinates=source_coordinate_system,
                target_coordinates=self.coordinate_system
            )
        else:
            pole_lon_native_deg = pole_lon_deg
            pole_lat_native_deg = pole_lat_deg
        
        # Convert to unit vectors (healpy expects colatitude, longitude)
        north_pole_vector = hp.ang2vec(
            np.deg2rad(90.0 - pole_lat_native_deg[0]),  # to colatitude
            np.deg2rad(pole_lon_native_deg[0])
        )
        south_pole_vector = hp.ang2vec(
            np.deg2rad(90.0 - pole_lat_native_deg[1]),  # to colatitude
            np.deg2rad(pole_lon_native_deg[1])
        )
        
        return north_pole_vector, south_pole_vector

    def _mask_around_poles(self, 
            north_pole_vector: NDArray[np.float64],
            south_pole_vector: NDArray[np.float64], 
            north_radius: float,
            south_radius: float,
            inverse_mask: bool = False
    ) -> None:
        """
        Mask pixels around both poles using disc queries.
        
        :param north_pole_vector: Unit vector pointing to north pole
        :param south_pole_vector: Unit vector pointing to south pole
        :param north_latitude_cut: Latitude threshold in degrees for north pole
        :param south_latitude_cut: Latitude threshold in degrees for south pole.
            If None, uses the same value as north_latitude_cut.
        """
        # Query discs around both poles
        north_indices = hp.query_disc(
            self.nside,
            north_pole_vector,
            radius=np.deg2rad(north_radius)
        )
        south_indices = hp.query_disc(
            self.nside,
            south_pole_vector,
            radius=np.deg2rad(south_radius)
        )
        
        # Combine and apply mask
        mask_indices = np.concatenate([north_indices, south_indices])
        if inverse_mask:
            mask_indices = np.asarray(
                list(self.all_indices - set(mask_indices)),
                dtype=np.int64
            )
        self._update_mask(mask_indices)
    
    def _update_mask(self, mask_indices: NDArray[np.int64]) -> None:
        self.mask_map[mask_indices] = False
        self.masked_pixel_indices.update(mask_indices)

    def mask_galactic_plane(self, 
            latitude_cut: float = 10.0
    ) -> None:
        '''
        Mask the galactic plane by masking pixels within ±latitude_cut degrees
        of the galactic equator (b=0).
        
        :param latitude_cut: Default latitude range in degrees to mask around b=0.
        '''
        # Use default value if specific pole cuts not provided        
        # Galactic pole coordinates (l=0, b=±90)
        pole_lon_deg = np.asarray([0., 0.])
        pole_lat_deg = np.asarray([90., -90.])
        
        north_pole_vector, south_pole_vector = self._get_pole_vecs_in_native_coords(
            pole_lon_deg, pole_lat_deg, 'galactic'
        )
        
        self._mask_around_poles(
            north_pole_vector, south_pole_vector, 
            90-latitude_cut, 90-latitude_cut,
            inverse_mask=True
        )

    def mask_equatorial_poles(self, 
            north_radius: float = 0.,
            south_radius: float = 0.
    ) -> None:
        """
        Mask the equatorial poles by masking pixels with |dec| > latitude_cut.
        
        :param latitude_cut: Default declination threshold in degrees above which to
            mask (used if north_latitude_cut and south_latitude_cut are not specified)
        :param north_latitude_cut: Declination threshold for north pole. If None,
            uses latitude_cut.
        :param south_latitude_cut: Declination threshold for south pole. If None,
            uses north_latitude_cut (or latitude_cut if north_latitude_cut is also None).
        """
        # Equatorial pole coordinates (RA=0, Dec=±90)
        pole_lon_deg = np.asarray([0., 0.])
        pole_lat_deg = np.asarray([90., -90.])
        
        north_pole_vector, south_pole_vector = self._get_pole_vecs_in_native_coords(
            pole_lon_deg, pole_lat_deg, 'equatorial'
        )
        
        self._mask_around_poles(
            north_pole_vector, south_pole_vector,
            north_radius, south_radius
        )

    def mask_ecliptic_poles(self, 
            north_latitude_cut: float = 90.,
            south_latitude_cut: float = 90.
    ) -> None:
        """
        Mask the ecliptic poles by masking pixels with
        |ecliptic_lat| > latitude_cut.
        
        :param latitude_cut: Default ecliptic latitude threshold in degrees above
            which to mask (used if north_latitude_cut and south_latitude_cut are not specified)
        :param north_latitude_cut: Ecliptic latitude threshold for north pole. If None,
            uses latitude_cut.
        :param south_latitude_cut: Ecliptic latitude threshold for south pole. If None,
            uses north_latitude_cut (or latitude_cut if north_latitude_cut is also None).
        """
        # Ecliptic pole coordinates (longitude=0, latitude=±90)
        pole_lon_deg = np.asarray([0., 0.])
        pole_lat_deg = np.asarray([90., -90.])
        
        north_pole_vector, south_pole_vector = self._get_pole_vecs_in_native_coords(
            pole_lon_deg, pole_lat_deg, 'ecliptic'
        )
        
        self._mask_around_poles(
            north_pole_vector, south_pole_vector,
            north_latitude_cut, south_latitude_cut
        )

    def mask_slice(self,
            slice_longitude: float,
            slice_latitude: float,
            radius: float
    ) -> None:
        """
        Mask a circular region around a given coordinate.
        
        :param slice_longitude: Longitude of center in degrees (in native
            coordinate system).
        :param slice_latitude: Latitude of center in degrees (in native
            coordinate system).
        :param radius: Radius of circular mask in degrees.
        """        
        # Convert center coordinates to colatitude/longitude for healpy
        center_theta = np.deg2rad(90.0 - slice_latitude)  # latitude to colat
        center_phi = np.deg2rad(slice_longitude)
        
        # Get center pixel vector
        center_vec = hp.ang2vec(center_theta, center_phi)

        # Mask pixels within disc
        mask_indices = hp.query_disc(
            self.nside, center_vec, radius=np.deg2rad(radius)
        )
        self._update_mask(mask_indices)

    def get_masked_density_map(self) -> NDArray[np.float64]:
        """
        Return the density map with masked pixels set to NaN.
        
        :return: Density map with masked pixels as NaN
        """
        masked_map = self.density_map.astype(np.float64)
        masked_map[~self.mask_map] = np.nan
        return masked_map

    def get_masked_pixels(self) -> NDArray[np.int64]:
        return np.asarray(list(self.masked_pixel_indices), dtype=np.int64)
    
    def get_unmasked_pixels(self) -> NDArray[np.int64]:
        unmasked_pixels = self.all_indices - self.masked_pixel_indices
        return np.asarray(list(unmasked_pixels), dtype=np.int64)
    
    def get_mask_map(self) -> NDArray[np.int64]:
        '''
        Ones for unmasked area; zeros masked areas.
        '''
        map_to_return = self.mask_map
        return map_to_return.astype(np.int64)

    def reset_mask(self) -> None:
        """
        Reset the mask to include all pixels (i.e., remove all masking).
        """
        self.mask_map = np.ones_like(self.density_map, dtype=bool)
        self.masked_pixel_indices = set()