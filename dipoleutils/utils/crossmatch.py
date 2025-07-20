from astropy.table import Table
from astropy.coordinates import match_coordinates_sky, SkyCoord, Angle
from.coordinate_parser import CoordinateSystemParser
import astropy.units as u
import numpy as np
from numpy.typing import NDArray

class CrossMatch:
    def __init__(self,
            catalogueA: Table,
            catalogueB: Table,
            coordinate_system: str
    ) -> None:
        '''
        Notation: matching A to B; i.e., for each A source, find the counterpart
        B source.
        '''
        self.catalogueA = catalogueA
        self.catalogueB = catalogueB
        self.crossmatch_catalogue: Table | None = None

        parser = CoordinateSystemParser()
        self.coords_A = parser.parse_coordinate_systems(catalogueA)
        self.coords_B = parser.parse_coordinate_systems(catalogueB)

        self.coordinate_system = coordinate_system
        assert coordinate_system in self.coords_A.keys(), (
            f'Coordinate system {coordinate_system} not found in catalogue A.'
        )
        assert coordinate_system in self.coords_B.keys(), (
            f'Coordinate system {coordinate_system} not found in catalogue B.'
        )

        self.lonA_column, self.latA_column = (
            self.coords_A[coordinate_system]['azimuthal'],
            self.coords_A[coordinate_system]['polar']
        )
        self.lonB_column, self.latB_column = (
            self.coords_B[coordinate_system]['azimuthal'],
            self.coords_B[coordinate_system]['polar']
        )
    
    def cross_match(self, radius: float) -> Table:
        '''
        Cross-match catalogues A and B within the specified radius.
        Creates and stores a crossmatch_catalogue Table with matched sources.
        
        :param radius: Angular distance in arcseconds.
        :return: Table containing A source ID, counterpart B source ID, and
            angular distance.
        '''
        skycoordA = SkyCoord(
            self.catalogueA[self.lonA_column],
            self.catalogueA[self.latA_column],
            unit='deg'
        )
        skycoordB = SkyCoord(
            self.catalogueB[self.lonB_column],
            self.catalogueB[self.latB_column],
            unit='deg'
        )
        
        output = match_coordinates_sky(skycoordA, skycoordB)

        # Indices in catalogue B for each source in A
        matched_indices: NDArray[np.int64] = output[0]
        distances: Angle = output[1]  # Angular distances
        
        # Convert distances to arcseconds
        distances_arcsec = distances.to(u.arcsec).value # type: ignore
        
        # Create arrays for the crossmatch table
        source_idx_A = np.arange(len(self.catalogueA))
        source_idx_B = matched_indices.copy().astype(np.float64)
        angular_distances = distances_arcsec.copy()
        
        # Set values outside radius to NaN (no match)
        outside_radius = distances_arcsec >= radius
        source_idx_B[outside_radius] = np.nan
        angular_distances[outside_radius] = np.nan
        
        # Create the crossmatch catalogue
        self.crossmatch_catalogue = Table(
            {
                'source_idx_A': source_idx_A,
                'source_idx_B': source_idx_B,
                'angular_distance_arcsec': angular_distances
            }
        )
        
        return self.crossmatch_catalogue
    
    