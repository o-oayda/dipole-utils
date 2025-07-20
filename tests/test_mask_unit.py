import pytest
import numpy as np
import healpy as hp
from unittest.mock import Mock, patch, MagicMock

from dipoleutils.utils.mask import Masker


class TestMaskerUnit:
    """
    Unit tests for the Masker class.
    
    These tests focus on individual methods and components in isolation,
    testing the masking logic without external dependencies.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.nside = 32
        self.npix = hp.nside2npix(self.nside)
        self.mock_density_map = np.ones(self.npix, dtype=np.int_) * 10
        
    def test_masker_initialization(self):
        """Test that Masker initializes correctly."""
        masker = Masker(self.mock_density_map, 'equatorial')
        
        # Check basic attributes
        assert masker.coordinate_system == 'equatorial'
        assert np.array_equal(masker.density_map, self.mock_density_map)
        assert masker.nside == self.nside
        assert masker.npix == self.npix
        assert np.all(masker.mask_map == True)  # Initially all unmasked
        assert len(masker.masked_pixel_indices) == 0
    
    def test_masker_initialization_different_coordinate_systems(self):
        """Test initialization with different coordinate systems."""
        coord_systems = ['equatorial', 'galactic', 'ecliptic']
        
        for coord_sys in coord_systems:
            masker = Masker(self.mock_density_map.copy(), coord_sys)
            assert masker.coordinate_system == coord_sys
            assert np.all(masker.mask_map == True)
    
    def test_update_mask_method(self):
        """Test the _update_mask helper method."""
        masker = Masker(self.mock_density_map, 'equatorial')
        initial_masked_count = len(masker.masked_pixel_indices)
        
        # Mock some indices to mask
        test_indices = np.array([0, 1, 100, 200])
        masker._update_mask(test_indices)
        
        # Check that mask_map is updated correctly
        assert not masker.mask_map[0]
        assert not masker.mask_map[1] 
        assert not masker.mask_map[100]
        assert not masker.mask_map[200]
        
        # Check that masked_pixel_indices is updated
        assert len(masker.masked_pixel_indices) == initial_masked_count + len(test_indices)
        for idx in test_indices:
            assert idx in masker.masked_pixel_indices
    
    @patch('dipoleutils.utils.mask.change_source_coordinates')
    def test_get_pole_vecs_in_native_coords_no_conversion(self, mock_change_coords):
        """Test pole vector calculation when no coordinate conversion needed."""
        masker = Masker(self.mock_density_map, 'galactic')
        
        pole_lon = np.array([0., 0.])
        pole_lat = np.array([90., -90.])
        
        north_vec, south_vec = masker._get_pole_vecs_in_native_coords(
            pole_lon, pole_lat, 'galactic'
        )
        
        # Should not call coordinate conversion
        mock_change_coords.assert_not_called()
        
        # Check vector shapes
        assert north_vec.shape == (3,)
        assert south_vec.shape == (3,)
        
        # Check that vectors are unit vectors (approximately)
        assert np.isclose(np.linalg.norm(north_vec), 1.0)
        assert np.isclose(np.linalg.norm(south_vec), 1.0)
    
    @patch('dipoleutils.utils.mask.change_source_coordinates')
    def test_get_pole_vecs_in_native_coords_with_conversion(self, mock_change_coords):
        """Test pole vector calculation when coordinate conversion is needed."""
        masker = Masker(self.mock_density_map, 'equatorial')
        
        # Mock the coordinate transformation
        mock_change_coords.return_value = (np.array([45., 45.]), np.array([60., -60.]))
        
        pole_lon = np.array([0., 0.])
        pole_lat = np.array([90., -90.])
        
        north_vec, south_vec = masker._get_pole_vecs_in_native_coords(
            pole_lon, pole_lat, 'galactic'
        )
        
        # Should call coordinate conversion
        mock_change_coords.assert_called_once_with(
            pole_lon, pole_lat,
            native_coordinates='galactic',
            target_coordinates='equatorial'
        )
        
        # Check vector properties
        assert north_vec.shape == (3,)
        assert south_vec.shape == (3,)
        assert np.isclose(np.linalg.norm(north_vec), 1.0)
        assert np.isclose(np.linalg.norm(south_vec), 1.0)
    
    @patch('healpy.query_disc')
    def test_mask_around_poles_symmetric(self, mock_query_disc):
        """Test _mask_around_poles with symmetric latitude cuts."""
        masker = Masker(self.mock_density_map, 'equatorial')
        
        # Mock query_disc to return some test indices
        mock_query_disc.side_effect = [
            np.array([0, 1, 2]),  # North pole indices
            np.array([100, 101, 102])  # South pole indices
        ]
        
        north_vec = np.array([0., 0., 1.])
        south_vec = np.array([0., 0., -1.])
        
        masker._mask_around_poles(north_vec, south_vec, 70.0)
        
        # Check that query_disc was called correctly
        assert mock_query_disc.call_count == 2
        
        # Check that the correct indices are masked
        expected_masked = {0, 1, 2, 100, 101, 102}
        assert masker.masked_pixel_indices == expected_masked
        
        for idx in expected_masked:
            assert not masker.mask_map[idx]
    
    @patch('healpy.query_disc')
    def test_mask_around_poles_asymmetric(self, mock_query_disc):
        """Test _mask_around_poles with asymmetric latitude cuts."""
        masker = Masker(self.mock_density_map, 'equatorial')
        
        # Mock different return values for different radii
        mock_query_disc.side_effect = [
            np.array([0, 1]),  # North pole with smaller radius
            np.array([100, 101, 102, 103])  # South pole with larger radius
        ]
        
        north_vec = np.array([0., 0., 1.])
        south_vec = np.array([0., 0., -1.])
        
        masker._mask_around_poles(north_vec, south_vec, 80.0, 60.0)  # asymmetric
        
        # Verify calls with different radii
        expected_north_radius = np.deg2rad(90.0 - 80.0)
        expected_south_radius = np.deg2rad(90.0 - 60.0)
        
        calls = mock_query_disc.call_args_list
        assert len(calls) == 2
        
        # Check the radius arguments (third argument to query_disc)
        assert np.isclose(calls[0][1]['radius'], expected_north_radius)
        assert np.isclose(calls[1][1]['radius'], expected_south_radius)
    
    def test_get_masked_density_map(self):
        """Test get_masked_density_map method."""
        masker = Masker(self.mock_density_map, 'equatorial')
        
        # Mask some pixels manually
        test_indices = np.array([0, 10, 100])
        masker._update_mask(test_indices)
        
        masked_map = masker.get_masked_density_map()
        
        # Check that masked pixels are NaN
        for idx in test_indices:
            assert np.isnan(masked_map[idx])
        
        # Check that unmasked pixels retain original values
        unmasked_indices = np.where(masker.mask_map)[0]
        for idx in unmasked_indices:
            assert masked_map[idx] == self.mock_density_map[idx]
        
        # Check return type
        assert masked_map.dtype == np.float64
    
    def test_reset_mask(self):
        """Test reset_mask method."""
        masker = Masker(self.mock_density_map, 'equatorial')
        
        # Apply some masking
        test_indices = np.array([5, 15, 25])
        masker._update_mask(test_indices)
        
        # Verify masking was applied
        assert len(masker.masked_pixel_indices) > 0
        assert not np.all(masker.mask_map)
        
        # Reset mask
        masker.reset_mask()
        
        # Verify reset worked
        assert len(masker.masked_pixel_indices) == 0
        assert np.all(masker.mask_map == True)
    
    @patch('healpy.query_disc')
    def test_mask_slice(self, mock_query_disc):
        """Test mask_slice method."""
        masker = Masker(self.mock_density_map, 'equatorial')
        
        # Mock query_disc return
        mock_query_disc.return_value = np.array([50, 51, 52])
        
        masker.mask_slice(slice_longitude=45.0, slice_latitude=30.0, radius=10.0)
        
        # Verify query_disc was called with correct parameters
        mock_query_disc.assert_called_once()
        args, kwargs = mock_query_disc.call_args
        
        assert args[0] == masker.nside  # nside
        assert args[1].shape == (3,)  # center vector
        assert np.isclose(kwargs['radius'], np.deg2rad(10.0))
        
        # Verify masking was applied
        expected_masked = {50, 51, 52}
        assert masker.masked_pixel_indices == expected_masked
    
    def test_mask_galactic_plane_default(self):
        """Test mask_galactic_plane with default parameters."""
        masker = Masker(self.mock_density_map, 'galactic')
        
        initial_unmasked = np.sum(masker.mask_map)
        masker.mask_galactic_plane()
        final_unmasked = np.sum(masker.mask_map)
        
        # Should mask some pixels
        assert final_unmasked < initial_unmasked
        assert len(masker.masked_pixel_indices) > 0
    
    def test_mask_equatorial_poles_default(self):
        """Test mask_equatorial_poles with default parameters."""
        masker = Masker(self.mock_density_map, 'equatorial')
        
        initial_unmasked = np.sum(masker.mask_map)
        masker.mask_equatorial_poles()
        final_unmasked = np.sum(masker.mask_map)
        
        # Should mask some pixels
        assert final_unmasked < initial_unmasked
        assert len(masker.masked_pixel_indices) > 0
    
    def test_mask_equatorial_poles_asymmetric_parameters(self):
        """Test mask_equatorial_poles with asymmetric parameters."""
        masker = Masker(self.mock_density_map, 'equatorial')
        
        # Test with explicit asymmetric parameters
        masker.mask_equatorial_poles(north_latitude_cut=70.0, south_latitude_cut=85.0)
        asymmetric_unmasked = np.sum(masker.mask_map)
        
        # Reset and test symmetric
        masker.reset_mask()
        masker.mask_equatorial_poles(latitude_cut=77.5)  # average of above
        symmetric_unmasked = np.sum(masker.mask_map)
        
        # Asymmetric should generally mask different number of pixels
        # (exact comparison depends on sky map structure)
        assert asymmetric_unmasked != symmetric_unmasked
    
    def test_mask_ecliptic_poles_default(self):
        """Test mask_ecliptic_poles with default parameters."""
        masker = Masker(self.mock_density_map, 'ecliptic')
        
        initial_unmasked = np.sum(masker.mask_map)
        masker.mask_ecliptic_poles()
        final_unmasked = np.sum(masker.mask_map)
        
        # Should mask some pixels
        assert final_unmasked < initial_unmasked
        assert len(masker.masked_pixel_indices) > 0
    
    def test_coordinate_system_validation(self):
        """Test that valid coordinate systems are accepted."""
        valid_systems = ['equatorial', 'galactic', 'ecliptic']
        
        for coord_sys in valid_systems:
            masker = Masker(self.mock_density_map.copy(), coord_sys)
            assert masker.coordinate_system == coord_sys
    
    def test_density_map_type_conversion(self):
        """Test that density map is properly handled for different input types."""
        # Test with different integer types
        int_map = np.ones(self.npix, dtype=np.int_) * 5
        masker = Masker(int_map, 'equatorial')
        
        masked_map = masker.get_masked_density_map()
        assert masked_map.dtype == np.float64
        assert np.all(masked_map == 5.0)  # Should preserve values
