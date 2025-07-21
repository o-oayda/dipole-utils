from astropy.table import Table
from astropy.coordinates import match_coordinates_sky, SkyCoord, Angle
from.coordinate_parser import CoordinateSystemParser
import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from typing import Optional

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
        
        # Source name columns - will be determined later
        self.source_name_A_column: str | None = None
        self.source_name_B_column: str | None = None

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
    
    def cross_match(self,
            radius: float,
            source_name_A_column: Optional[str] = None,
            source_name_B_column: Optional[str] = None
        ) -> Table:
        '''
        Cross-match catalogues A and B within the specified radius.
        Creates and stores a crossmatch_catalogue Table with matched sources.
        
        :param radius: Angular distance in arcseconds.
        :param source_name_A_column: Column name for source names in catalogue A. 
            If None, will try 'source_name' or raise an error if not found.
        :param source_name_B_column: Column name for source names in catalogue B.
            If None, will try 'source_name' or raise an error if not found.
        :return: Table containing A source ID, counterpart B source ID, angular distance,
            and source names from both catalogues.
        '''
        # Determine source name columns
        self._determine_source_name_columns(source_name_A_column, source_name_B_column)
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
        source_idx_B = matched_indices.copy()
        angular_distances = distances_arcsec.copy()
        
        # Get source names
        source_names_A = self.catalogueA[self.source_name_A_column]
        source_names_B = np.full(len(source_idx_A), None, dtype=object)
        
        # Set values outside radius to -1 (no match)
        outside_radius = distances_arcsec >= radius
        source_idx_B[outside_radius] = -1
        angular_distances[outside_radius] = np.nan
        
        # Fill in source names for B (only for matches)
        valid_matches = ~outside_radius
        if np.any(valid_matches):
            source_names_B[valid_matches] = self.catalogueB[self.source_name_B_column][matched_indices[valid_matches]]
        
        # Create the crossmatch catalogue
        self.crossmatch_catalogue = Table(
            {
                'source_idx_A': source_idx_A,
                'source_idx_B': source_idx_B,
                'source_name_A': source_names_A,
                'source_name_B': source_names_B,
                'angular_distance_arcsec': angular_distances
            }
        )
        
        return self.crossmatch_catalogue

    def _determine_source_name_columns(self,
            source_name_A_column: Optional[str],
            source_name_B_column: Optional[str]
        ) -> None:
        '''
        Determine and store the source name columns for both catalogues.
        '''
        # Handle catalogue A source name column
        if source_name_A_column is not None:
            if source_name_A_column not in self.catalogueA.colnames:
                raise ValueError(
                    f"Column '{source_name_A_column}' not found in catalogue A. "
                    f"Available columns: {self.catalogueA.colnames}"
                )
            self.source_name_A_column = source_name_A_column
        elif self.source_name_A_column is None:  # First time determining
            if 'source_name' in self.catalogueA.colnames:
                self.source_name_A_column = 'source_name'
            else:
                raise ValueError(
                    f"No 'source_name' column found in catalogue A. "
                    f"Available columns: {self.catalogueA.colnames}. "
                    f"Please specify source_name_A_column parameter."
                )
        
        # Handle catalogue B source name column
        if source_name_B_column is not None:
            if source_name_B_column not in self.catalogueB.colnames:
                raise ValueError(
                    f"Column '{source_name_B_column}' not found in catalogue B. "
                    f"Available columns: {self.catalogueB.colnames}"
                )
            self.source_name_B_column = source_name_B_column
        elif self.source_name_B_column is None:  # First time determining
            if 'source_name' in self.catalogueB.colnames:
                self.source_name_B_column = 'source_name'
            else:
                raise ValueError(
                    f"No 'source_name' column found in catalogue B. "
                    f"Available columns: {self.catalogueB.colnames}. "
                    f"Please specify source_name_B_column parameter."
                )