from astropy.table import Table
from numpy.typing import NDArray
import numpy as np
from typing import Dict, List, Optional, Tuple, cast, Literal, Callable, Union
from .tools import angles_to_density_map
from .coordinate_parser import CoordinateSystemParser
import copy
import healpy as hp
from .constants import CMB_L, CMB_B
from .math import (
    compute_dipole_signal,
    multipole_pixel_product_vectorised,
    multipole_tensor_vectorised,
    vectorised_quadrupole_tensor,
    vectorised_spherical_to_cartesian
)
from scipy.stats import poisson


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
    
    def make_parameter_map(self,
            column_name: str | list[str],
            coordinate_system: str,
            no_source_val: float = np.nan,
            nside: int = 64,
            nest: bool = False,
            operation: Literal[
                'median', 'mean', 'sum', 'std', 'dist', '-', '+', '/', '*'
            ] = 'median'
        ) -> NDArray[np.float64] | list:
        '''
        Creates a healpy map of parameter values aggregated within each pixel.
        
        This function uses optimized vectorized operations for performance:
        - scipy.ndimage functions for grouped aggregation (mean, sum, std)
        - Sort-based grouping for median and distribution operations
        - Only processes pixels that contain sources (sparse optimization)
        
        Performance scales with number of sources rather than number of pixels,
        making it efficient for high-resolution maps and sparse data.
        
        :param column_name: Name of the parameter column to be mapped. Must be a 
            valid column name in the catalogue. If a list of two column names is 
            provided, binary operations (+, -, /, *) can be applied between them.
        :param coordinate_system: Coordinate system to use for pixel assignment 
            ('equatorial', 'galactic', 'ecliptic'). Must match a system available 
            in the catalogue.
        :param no_source_val: Value assigned to pixels containing no sources. 
            Only applies to aggregation operations (median, mean, sum, std). 
            Defaults to np.nan.
        :param nside: HEALPix nside parameter (must be power of 2, e.g., 64, 128, 256). 
            Higher values give finer angular resolution but more pixels.
        :param nest: If True, use NESTED pixel ordering. If False (default), 
            use RING pixel ordering.
        :param operation: Statistical operation to perform on parameter values 
            within each pixel:
            - 'median': Median value (default)
            - 'mean': Arithmetic mean  
            - 'sum': Sum of all values
            - 'std': Standard deviation
            - 'dist': Return list of all values (distribution)
            - '+', '-', '/', '*': Binary operations (requires list of 2 column names)
        :return: For aggregation operations (median, mean, sum, std): numpy array 
            of shape (npix,) with parameter values. For 'dist' and binary operations: 
            list of length npix where each element contains the values/results for 
            that pixel.
        :raises ValueError: If coordinate_system is not available, column_name is 
            invalid, or operation is incompatible with column_name type.
        '''
        operation_dict = {
            'median': np.median,
            'mean': np.mean,
            'sum': np.sum,
            'std': np.std,
            'dist': list,
            '+': lambda x, y: list(x + y),
            '-': lambda x, y: list(x - y),
            '/': lambda x, y: list(x / y),
            '*': lambda x, y: list(x * y)
        }
        assert operation in operation_dict.keys(), (
            f'Invalid operation. Available options: {list(operation_dict.keys())}.'
        )
        npix = hp.nside2npix(nside)

        if 'pixel_indices' not in self.get_column_names():
            self.compute_pixel_indices(coordinate_system, nside, nest)
        
        operation_function: Callable = operation_dict[operation]
        
        # Initialize the appropriate container based on operation
        result_map: list | NDArray[np.float64]
        if operation in ['dist', '+', '-', '/', '*']:
            result_map = [[] for _ in range(npix)]  # List of lists for distributions
        else:
            result_map = np.full(npix, no_source_val, dtype=np.float64)  # Float array
        
        # Extract data as numpy arrays for faster processing
        pixel_indices = np.asarray(self.catalogue['pixel_indices'])
        
        # Use numpy groupby-like functionality for much faster processing
        if type(column_name) is list:
            if operation not in ['+', '-', '/', '*']:
                raise ValueError(
                    f"Operation '{operation}' not supported with multiple columns. "
                    f"Use one of: +, -, /, *"
                )
            col1_data = np.asarray(self.catalogue[column_name[0]])
            col2_data = np.asarray(self.catalogue[column_name[1]])
            
            # Sort by pixel indices for efficient grouping
            sort_idx = np.argsort(pixel_indices)
            sorted_pixels = pixel_indices[sort_idx]
            sorted_col1 = col1_data[sort_idx]
            sorted_col2 = col2_data[sort_idx]
            
            # Find unique pixels and their boundaries
            unique_pixels, start_indices = np.unique(
                sorted_pixels,
                return_index=True
            )
            end_indices = np.append(start_indices[1:], len(sorted_pixels))
            
            # Process each unique pixel
            for i, pix_ind in enumerate(unique_pixels):
                start, end = start_indices[i], end_indices[i]
                col1_vals = sorted_col1[start:end]
                col2_vals = sorted_col2[start:end]
                sources_statistic = operation_function(col1_vals, col2_vals)
                result_map[pix_ind] = sources_statistic
        else:
            col_data = np.asarray(self.catalogue[column_name])
            
            if operation == 'dist':
                # For distribution, we need to handle differently
                sort_idx = np.argsort(pixel_indices)
                sorted_pixels = pixel_indices[sort_idx]
                sorted_col = col_data[sort_idx]
                
                unique_pixels, start_indices = np.unique(
                    sorted_pixels,
                    return_index=True
                )
                end_indices = np.append(start_indices[1:], len(sorted_pixels))
                
                for i, pix_ind in enumerate(unique_pixels):
                    start, end = start_indices[i], end_indices[i]
                    col_vals = sorted_col[start:end]
                    result_map[pix_ind] = col_vals.tolist()
            else:
                # For aggregation operations, use even faster vectorized approach
                from scipy import ndimage
                
                # Use scipy's labeled statistics for ultra-fast aggregation
                unique_pixels = np.unique(pixel_indices)
                
                if operation == 'mean':
                    pixel_means = ndimage.mean(
                        col_data,
                        labels=pixel_indices,
                        index=unique_pixels
                    )
                    result_map[unique_pixels] = pixel_means
                elif operation == 'sum':
                    pixel_sums = ndimage.sum(
                        col_data,
                        labels=pixel_indices,
                        index=unique_pixels
                    )
                    result_map[unique_pixels] = pixel_sums
                elif operation == 'std':
                    pixel_stds = ndimage.standard_deviation(
                        col_data,
                        labels=pixel_indices,
                        index=unique_pixels
                    )
                    result_map[unique_pixels] = pixel_stds
                elif operation == 'median':
                    # Median doesn't have a direct ndimage function, fall back
                    # to groupby approach
                    sort_idx = np.argsort(pixel_indices)
                    sorted_pixels = pixel_indices[sort_idx]
                    sorted_col = col_data[sort_idx]
                    
                    unique_pixels, start_indices = np.unique(
                        sorted_pixels,
                        return_index=True
                    )
                    end_indices = np.append(start_indices[1:], len(sorted_pixels))
                    
                    for i, pix_ind in enumerate(unique_pixels):
                        start, end = start_indices[i], end_indices[i]
                        col_vals = sorted_col[start:end]
                        result_map[pix_ind] = np.median(col_vals)
        
        return result_map

    def get_map_coordinate_system(self) -> str | None:
        """
        Returns the map coordinate system associated with the generated
        density map.
        :return: The map coordinate system as a string, or None if no map has
        been generated.
        """
        return self.map_coordinate_system
    
    def copy_independent(self) -> "CatalogueToMap":
        '''
        Returns a totally independent (deep) copy of this CrossMatch object.
        All attributes are recursively copied, so changes to the copy do not
        affect the original.
        '''
        return copy.deepcopy(self)
    
    def mask_in_pixel_space(self,
            pixels_to_mask: NDArray[np.int64],
            coordinate_system: str,
            nside: int,
            nest: bool = False
    ) -> None:
        if 'pixel_indices' not in self.get_column_names():
            self.compute_pixel_indices(coordinate_system, nside, nest)
        
        mask = ~np.isin(self.catalogue['pixel_indices'], pixels_to_mask) # type: ignore
        self.catalogue = cast(Table, self.catalogue[mask])
            
    def compute_pixel_indices(self,
            coordinate_system: str,
            nside: int,
            nest: bool = False
    ) -> None:
        coords = self.get_coordinates(coordinate_system)
        if coords is None:
            raise ValueError(
                f"Could not retrieve coordinates for system "
                f"{coordinate_system}'."
            )
        azimuthal_col, polar_col = coords
        azimuthal_angles = np.asarray(self.catalogue[azimuthal_col], dtype=float)
        polar_angles = np.asarray(self.catalogue[polar_col], dtype=float)
        pixel_indices = hp.ang2pix(
            nside,
            azimuthal_angles, # lon (deg)
            polar_angles,     # lat (deg)
            lonlat=True,
            nest=nest
        )
        self.catalogue.add_column(pixel_indices, name='pixel_indices')

class SimulatedDipoleMap:
    def __init__(self, nside: int = 64) -> None:
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        pixel_vectors_tuple = hp.pix2vec(nside=self.nside, ipix=np.arange(self.npix))
        self.pixel_vectors = np.vstack(pixel_vectors_tuple).T
        self.mean_density = None
        self.dipole_amplitude = None
        self.dipole_longitude = None
        self.dipole_latitude = None

    def make_map(
        self,
        mean_density: float = 50.,
        dipole_amplitude: float = 0.007,
        dipole_longitude: float = CMB_L,
        dipole_latitude: float = CMB_B
    ) -> NDArray[np.float64]:
        self.mean_density = mean_density
        self.dipole_amplitude = dipole_amplitude
        self.dipole_longitude = np.deg2rad(dipole_longitude)
        self.dipole_latitude = np.pi / 2 - np.deg2rad(dipole_latitude)
        self._compute_map_signal()
        return self._signal_to_poisson()

    def _compute_map_signal(self):
        dipole_signal = compute_dipole_signal(
            np.asarray(self.dipole_amplitude).reshape((1,)),
            np.asarray(self.dipole_longitude).reshape((1,)),
            np.asarray(self.dipole_latitude).reshape((1,)),
            np.asarray(self.pixel_vectors)
        ).squeeze()
        assert self.mean_density is not None, 'Dipole parameters not set.'
        self.dipole_and_monopole_signal = self.mean_density * (1 + dipole_signal)
        
    def _signal_to_poisson(self) -> NDArray[np.float64]:
        assert hasattr(self, 'dipole_and_monopole_signal'), 'Compute the map signal first.'
        return poisson.rvs(self.dipole_and_monopole_signal) # type: ignore


class SimulatedMultipoleMap:
    """
    Build simulated count maps that contain arbitrary multipole structure.

    Parameters for the multipoles must follow the naming convention used by
    :class:`dipoleutils.models.multipole.Multipole`, e.g.
    ``{'M0': 50., 'M1': 0.01, 'phi_l1_0': ..., 'theta_l1_0': ...}``.
    Each ell >= 1 requires an amplitude ``M{ell}`` and `ell` azimuthal/polar
    pairs ``phi_l{ell}_{i}`` / ``theta_l{ell}_{i}``.  Angles are interpreted as
    radians unless ``angles_in_degrees`` is set at initialisation.
    """

    def __init__(
        self,
        nside: int = 64,
        ells: Optional[List[int]] = None,
        angles_in_degrees: bool = False
    ) -> None:
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        pixel_vectors = hp.pix2vec(nside=self.nside, ipix=np.arange(self.npix))
        self.pixel_vectors = np.vstack(pixel_vectors).T
        self.pixel_vectors_xyz = [pixel_vectors[0], pixel_vectors[1], pixel_vectors[2]]
        self.ells = self._validate_ells(ells) if ells is not None else None
        self.mean_density: float | None = None
        self.angles_in_degrees = angles_in_degrees

    def make_map(
        self,
        parameters: Dict[Union[str, int], Union[float, NDArray[np.float64]]],
        ells: Optional[List[int]] = None,
        poisson_seed: Union[int, np.random.Generator, None] = None
    ) -> NDArray[np.float64]:
        """
        Generate a Poisson realisation of a map that contains the requested
        multipole content.

        :param parameters: Dictionary of monopole/multipole parameters.
            The monopole is expected under ``'M0'`` (or synonyms listed
            in :meth:`_extract_mean_density`), amplitudes under ``'M{ell}'``,
            and angles under ``phi_l{ell}_{i}`` / ``theta_l{ell}_{i}``.
        :param ells: Optional override of the multipole orders to include.
            When omitted, the orders provided at instantiation are used.
        :raises ValueError: If the inputs are inconsistent or incomplete.
        :param poisson_seed: Optional seed (or Generator instance) passed to the
            Poisson sampler to obtain deterministic count maps.
        :return: Poisson deviates for each HEALPix pixel.
        """
        active_ells = self._get_active_ells(ells)
        self.mean_density = self._extract_mean_density(parameters)
        multipole_signal = self._assemble_signal(
            parameters=parameters,
            ells=active_ells
        )
        rate_map = self.mean_density * multipole_signal
        if np.any(rate_map < 0):
            raise ValueError(
                'Computed rate map contains negative entries. '
                'Check that the requested amplitudes keep the signal positive.'
            )
        return self._poisson_sample(rate_map, poisson_seed)

    def _poisson_sample(
        self,
        rate_map: NDArray[np.float64],
        poisson_seed: Union[int, np.random.Generator, None]
    ) -> NDArray[np.float64]:
        if poisson_seed is None:
            return poisson.rvs(rate_map) # type: ignore[arg-type]
        rng = poisson_seed if isinstance(poisson_seed, np.random.Generator) else np.random.default_rng(poisson_seed)
        return rng.poisson(rate_map)

    def _assemble_signal(
        self,
        parameters: Dict[Union[str, int], Union[float, NDArray[np.float64]]],
        ells: List[int]
    ) -> NDArray[np.float64]:
        signal = np.ones(self.npix, dtype=np.float64)
        for ell in ells:
            if ell == 0:
                continue
            ell_signal = self._order_signal(
                ell=ell,
                parameters=parameters,
                angles_in_degrees=self.angles_in_degrees
            )
            signal += ell_signal
        return signal

    def _order_signal(
        self,
        ell: int,
        parameters: Dict[Union[str, int], Union[float, NDArray[np.float64]]],
        angles_in_degrees: bool
    ) -> NDArray[np.float64]:
        amplitude = self._extract_amplitude(parameters, ell)
        if np.isclose(amplitude, 0.0):
            return np.zeros(self.npix, dtype=np.float64)
        phi_angles = self._extract_angles(
            parameters=parameters,
            ell=ell,
            angle_type='phi',
            amplitude=amplitude,
            angles_in_degrees=angles_in_degrees
        )
        theta_angles = self._extract_angles(
            parameters=parameters,
            ell=ell,
            angle_type='theta',
            amplitude=amplitude,
            angles_in_degrees=angles_in_degrees
        )
        amplitude_like = np.asarray([amplitude], dtype=np.float64)
        phi_like = phi_angles[None, :]
        theta_like = theta_angles[None, :]

        if ell == 1:
            signal = compute_dipole_signal(
                dipole_amplitude=amplitude_like,
                dipole_longitude=phi_like.squeeze(),
                dipole_colatitude=theta_like.squeeze(),
                pixel_vectors=self.pixel_vectors
            )
        elif ell == 2:
            cartesian_vectors = vectorised_spherical_to_cartesian(
                phi_like=phi_like,
                theta_like=theta_like
            )
            tensors = vectorised_quadrupole_tensor(
                amplitude_like=amplitude_like,
                cartesian_quadrupole_vectors=cartesian_vectors
            )
            signal = multipole_pixel_product_vectorised(
                multipole_tensors=tensors,
                pixel_vectors=self.pixel_vectors_xyz,
                ell=ell
            )
        else:
            cartesian_vectors = vectorised_spherical_to_cartesian(
                phi_like=phi_like,
                theta_like=theta_like
            )
            tensors = multipole_tensor_vectorised(
                amplitude_like=amplitude_like,
                cartesian_multipole_vectors=cartesian_vectors
            )
            signal = multipole_pixel_product_vectorised(
                multipole_tensors=tensors,
                pixel_vectors=self.pixel_vectors_xyz,
                ell=ell
            )
        return signal.squeeze()

    def _extract_angles(
        self,
        parameters: Dict[Union[str, int], Union[float, NDArray[np.float64]]],
        ell: int,
        angle_type: Literal['phi', 'theta'],
        amplitude: float,
        angles_in_degrees: bool
    ) -> NDArray[np.float64]:
        prefix = f'{angle_type}_l{ell}_'
        matches: Dict[int, float] = {}
        for key, value in parameters.items():
            if not isinstance(key, str):
                continue
            if not key.startswith(prefix):
                continue
            try:
                idx = int(key.split('_')[-1])
            except ValueError:
                continue
            if idx in matches:
                raise ValueError(f'Duplicate {angle_type} index {idx} for ell={ell}.')
            matches[idx] = self._to_float(value)
        if not matches:
            if np.isclose(amplitude, 0.0):
                return np.zeros(ell, dtype=np.float64)
            raise ValueError(
                f'Missing {angle_type} angles for ell={ell}. '
                f'Expected keys {prefix}0..{ell-1}.'
            )
        expected = set(range(ell))
        if set(matches) != expected:
            raise ValueError(
                f'Incomplete {angle_type} specification for ell={ell}. '
                f'Found indices {sorted(matches)}, expected {sorted(expected)}.'
            )
        ordered = np.array([matches[i] for i in range(ell)], dtype=np.float64)
        if angles_in_degrees:
            ordered = np.deg2rad(ordered)
        return ordered

    def _extract_amplitude(
        self,
        parameters: Dict[Union[str, int], Union[float, NDArray[np.float64]]],
        ell: int
    ) -> float:
        keys_to_try: List[Union[str, int]] = [f'M{ell}', str(ell), ell]
        return self._get_scalar_parameter(parameters, keys_to_try, missing_msg=f'Missing amplitude for ell={ell}.')

    def _extract_mean_density(
        self,
        parameters: Dict[Union[str, int], Union[float, NDArray[np.float64]]]
    ) -> float:
        keys_to_try: List[Union[str, int]] = ['M0', 'mean_density', 'density', 'monopole', 0, '0']
        return self._get_scalar_parameter(parameters, keys_to_try, missing_msg='Missing monopole / mean density (key "M0").')

    def _get_scalar_parameter(
        self,
        parameters: Dict[Union[str, int], Union[float, NDArray[np.float64]]],
        candidate_keys: List[Union[str, int]],
        missing_msg: str
    ) -> float:
        for key in candidate_keys:
            if key in parameters:
                return self._to_float(parameters[key])
        raise ValueError(missing_msg)

    def _to_float(self, value: Union[float, NDArray[np.float64]]) -> float:
        arr = np.asarray(value, dtype=np.float64)
        if arr.size != 1:
            raise ValueError('Expected scalar parameter value.')
        return float(arr.reshape(-1)[0])

    def _get_active_ells(self, ells: Optional[List[int]]) -> List[int]:
        if ells is not None:
            validated = self._validate_ells(ells)
            self.ells = validated
        if self.ells is None:
            raise ValueError('No multipole orders (ells) specified.')
        return self.ells

    def _validate_ells(self, ells: Optional[List[int]]) -> List[int]:
        if ells is None:
            return []
        unique: List[int] = []
        seen = set()
        for ell in ells:
            ell_int = int(ell)
            if ell_int < 0:
                raise ValueError('Multipole orders must be non-negative.')
            if ell_int in seen:
                continue
            seen.add(ell_int)
            unique.append(ell_int)
        return unique
