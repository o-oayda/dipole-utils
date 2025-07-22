from astropy.table import Table
from numpy.typing import NDArray
import numpy as np
from typing import Dict, List, Optional, Tuple, cast
from .tools import angles_to_density_map
from .coordinate_parser import CoordinateSystemParser


class CatalogueToMap:
    def __init__(self, catalogue: Table) -> None:
        '''
        Initialize the CatalogueToMap object with an astropy Table catalogue.
        Parses the catalogue to identify available coordinate systems and their
        columns.
        
        :param catalogue: Astropy Table containing the catalogue data.
        '''
        self.catalogue = catalogue
        self.coordinate_systems = {}
        self.coordinate_columns = {}
        self.map_coordinate_system = None
        self.parser = CoordinateSystemParser()
        self._parse_angular_coordinates()
    
    def _parse_angular_coordinates(self):
        '''
        Parse the columns of the catalogue and check if any are angular
        coordinates. Identifies all coordinate systems present and their
        corresponding azimuthal/polar angle columns. Populates
        self.coordinate_systems and self.coordinate_columns.
        '''
        # Use the coordinate parser to find coordinate systems
        self.coordinate_systems = self.parser.parse_coordinate_systems(
            self.catalogue
        )
        
        # Update coordinate_columns for backward compatibility and summary
        self.coordinate_columns = {
            'systems': self.coordinate_systems,
            'count': len(self.coordinate_systems)
        }
    
    def get_coordinate_info(self) -> Dict:
        '''
        Return information about all identified coordinate systems in the
        catalogue.
        
        :return: Dictionary containing coordinate systems, their names, count,
            and all coordinate info.
        '''
        return {
            'coordinate_systems': self.coordinate_systems,
            'systems_found': list(self.coordinate_systems.keys()),
            'total_systems': len(self.coordinate_systems),
            'all_coordinate_info': self.coordinate_columns
        }
    
    def get_coordinates(self, system: str) -> Optional[Tuple[str, str]]:
        '''
        Get the azimuthal and polar column names for a specific coordinate
        system.
        
        :param system: Name of the coordinate system (e.g., 'equatorial',
            'galactic').
        :return: Tuple of (azimuthal_column, polar_column) if found, else None.
        '''
        if system in self.coordinate_systems:
            coords = self.coordinate_systems[system]
            return coords['azimuthal'], coords['polar']
        return None
    
    def add_coordinate_system(self, starting_frame: str, target_frame: str) -> None:
        '''
        Add a new coordinate system to the catalogue by transforming coordinates
        from an existing coordinate system. New columns are added to the catalogue
        with the transformed coordinates.
        
        :param starting_frame: The coordinate system to transform from 
            (e.g., 'equatorial', 'galactic', 'ecliptic').
        :param target_frame: The coordinate system to transform to
            (e.g., 'equatorial', 'galactic', 'ecliptic').
        :raises ValueError: If the starting frame is not available in the catalogue
            or if the coordinate transformation fails.
        '''
        # Import the coordinate transformation function
        from .physics import change_source_coordinates
        
        # Check if starting frame exists in the catalogue
        if not self.has_coordinate_system(starting_frame):
            available = ', '.join(self.get_available_systems())
            raise ValueError(
                f"Starting coordinate system '{starting_frame}' not available. "
                f"Available systems: {available}"
            )
        
        # Check if target frame is already present
        if self.has_coordinate_system(target_frame):
            print(f"Target coordinate system '{target_frame}' already exists in catalogue.")
            return
        
        # Get the source coordinates
        source_coords = self.get_coordinates(starting_frame)
        if source_coords is None:
            raise ValueError(
                f"Could not retrieve coordinates for system '{starting_frame}'."
            )
        
        source_azimuthal_col, source_polar_col = source_coords
        source_azimuthal = np.asarray(self.catalogue[source_azimuthal_col], dtype=np.float64)
        source_polar = np.asarray(self.catalogue[source_polar_col], dtype=np.float64)
        
        # Transform coordinates
        try:
            transformed_azimuthal, transformed_polar = change_source_coordinates(
                source_azimuthal, source_polar, starting_frame, target_frame
            )
        except (AssertionError, ValueError) as e:
            raise ValueError(f"Coordinate transformation failed: {e}")
        
        # Get the standard column names for the target coordinate system
        # Use the first pattern from the coordinate parser patterns
        target_patterns = self.parser.coordinate_patterns.get(target_frame)
        if target_patterns is None:
            raise ValueError(f"Unsupported target coordinate system: {target_frame}")
        
        # Extract the first (canonical) name from each pattern list
        azimuthal_pattern = target_patterns['azimuthal'][0]
        polar_pattern = target_patterns['polar'][0]
        
        # Clean up regex patterns to get simple column names
        def clean_pattern(pattern: str) -> str:
            # Remove regex markers and get the base name
            cleaned = pattern.replace(r'\b', '').replace('.*', '')
            cleaned = cleaned.replace(r'[\s_-]*', '').replace('\\', '')
            return cleaned
        
        target_azimuthal_col = clean_pattern(azimuthal_pattern)
        target_polar_col = clean_pattern(polar_pattern)
        
        # Add new columns to the catalogue
        self.catalogue[target_azimuthal_col] = transformed_azimuthal
        self.catalogue[target_polar_col] = transformed_polar
        
        print(f"Added {target_frame} coordinates: {target_azimuthal_col}, {target_polar_col}")
        
        # Re-parse the catalogue to update coordinate systems
        self._parse_angular_coordinates()

    def has_coordinate_system(self, system: str) -> bool:
        '''
        Check if a specific coordinate system is available in the catalogue.
        
        :param system: Name of the coordinate system to check.
        :return: True if the system is available, False otherwise.
        '''
        return system in self.coordinate_systems
    
    def has_valid_coordinates(self) -> bool:
        '''
        Check if any valid coordinate systems were identified in the catalogue.
        
        :return: True if at least one coordinate system is found, False
            otherwise.
        '''
        return len(self.coordinate_systems) > 0
    
    def get_available_systems(self) -> List[str]:
        '''
        Return a list of all available coordinate systems identified in the
        catalogue.
        
        :return: List of coordinate system names.
        '''
        return list(self.coordinate_systems.keys())

    def get_column_names(self) -> list[str]:
        return self.catalogue.colnames

    def get_source_count(self) -> int:
        return len(self.catalogue)

    def get_catalogue(self) -> Table:
        return self.catalogue

    def make_cut(self,
        column_name: str,
        minimum: Optional[float],
        maximum: Optional[float],
        cut_outside: bool = True
    ) -> None:
        '''
        Apply a cut/filter to the catalogue based on a column's value range.
        Only rows with column values within [minimum, maximum] are kept.
        
        :param column_name: Name of the column to apply the cut on.
        :param minimum: Minimum value (inclusive) for the cut. If None, no lower
            bound.
        :param maximum: Maximum value (inclusive) for the cut. If None, no upper
            bound.
        :param cut_outside: If True, cut all values which fall outside the
            specified range (default behaviour). If False, cut all values which
            fall inside the range.
        '''
        cut = np.ones(len(self.catalogue), dtype=bool)
        column_data = np.asarray(self.catalogue[column_name])
        
        if minimum is not None:
            cut &= column_data >= minimum
        if maximum is not None:
            cut &= column_data <= maximum

        if not cut_outside:
            cut = ~cut

        # Explicitly cast to Table for type checker
        self.catalogue = cast(Table, self.catalogue[cut])

    def make_density_map(
            self,
            coordinate_system: Optional[str] = None,
            nside: int = 64,
            nest: bool = False
        ) -> NDArray[np.int_]:
        '''
        Create a density map from the catalogue coordinates using the specified
        coordinate system. Bins the azimuthal and polar angles into a healpy map
        grid.
        
        :param coordinate_system: Which coordinate system to use ('equatorial',
            'galactic', 'ecliptic') when binning the angles into a healpy map.
            If None, uses the first available system.
        :param nside: The nside parameter for the healpy map (must be a power of
            2, e.g., 64, 128).
        :param nest: If True, use NESTED pixel ordering. If False, use RING
            pixel ordering.
        :return: A numpy array representing the density map (number of objects
            per healpy pixel).
        :raises ValueError: If no valid coordinate systems are found or if the
            requested system is unavailable.
        '''
        if not self.has_valid_coordinates():
            raise ValueError(
                "No valid coordinate systems identified. Cannot create density map."
            )
        
        if coordinate_system is None:
            system_to_use = self.get_available_systems()[0]
        
        elif coordinate_system not in self.coordinate_systems:
            available = ', '.join(self.get_available_systems())
            raise ValueError(
                f"Coordinate system '{coordinate_system}' not available. "
                f"Available systems: {available}"
            )
        else:
            system_to_use = coordinate_system
        
        # Get coordinate columns
        self.map_coordinate_system = system_to_use
        coords = self.get_coordinates(system_to_use)
        if coords is None:
            raise ValueError(
                f"Could not retrieve coordinates for system '{system_to_use}'."
            )
        azimuthal_col, polar_col = coords
        
        azimuthal_angles = np.asarray(self.catalogue[azimuthal_col], dtype=float)
        polar_angles = np.asarray(self.catalogue[polar_col], dtype=float)

        print(
            f"Binning density map in {system_to_use} coordinates: "
            f"{azimuthal_col}, {polar_col}"
        )
        self.density_map = angles_to_density_map(
            azimuthal_angles, polar_angles,
            lonlat=True, nest=nest, nside=nside
        )
        return self.density_map
    
    def get_map_coordinate_system(self) -> str | None:
        """
        Returns the map coordinate system associated with the generated
        density map.
        :return: The map coordinate system as a string, or None if no map has
        been generated.
        """
        return self.map_coordinate_system