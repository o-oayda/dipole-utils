import json
import pytest
import numpy as np
import healpy as hp
from astropy.table import Table
from unittest.mock import Mock, patch, MagicMock
from numpy.typing import NDArray

from dipoleutils.utils.samples import CatalogueToMap, SimulatedMultipoleMap
from dipoleutils.utils.coordinate_parser import CoordinateSystemParser


class TestCatalogueToMapUnit:
    """
    Unit tests for the CatalogueToMap class.
    
    These tests focus on individual methods and components in isolation.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple test catalogue
        self.test_data = Table({
            'ra': np.random.uniform(0, 360, 100),
            'dec': np.random.uniform(-90, 90, 100),
            'glon': np.random.uniform(0, 360, 100),
            'glat': np.random.uniform(-90, 90, 100),
            'magnitude': np.random.uniform(15, 25, 100)
        })
        
    def test_initialization(self):
        """Test that CatalogueToMap initializes correctly."""
        mapper = CatalogueToMap(self.test_data)
        
        # Check that basic attributes are set
        assert hasattr(mapper, 'catalogue')
        assert hasattr(mapper, 'coordinate_systems')
        assert hasattr(mapper, 'coordinate_columns')
        assert hasattr(mapper, 'parser')
        
        # Check that parser is CoordinateSystemParser instance
        assert isinstance(mapper.parser, CoordinateSystemParser)
        
        # Check that coordinate parsing was called
        assert len(mapper.coordinate_systems) > 0
        
    @patch('dipoleutils.utils.samples.CoordinateSystemParser')
    def test_initialization_with_mock_parser(self, mock_parser_class):
        """Test initialization with mocked parser."""
        # Setup mock
        mock_parser = Mock()
        mock_parser.parse_coordinate_systems.return_value = {
            'equatorial': {'azimuthal': 'ra', 'polar': 'dec'}
        }
        mock_parser_class.return_value = mock_parser
        
        mapper = CatalogueToMap(self.test_data)
        
        # Check that parser was created and used
        mock_parser_class.assert_called_once()
        mock_parser.parse_coordinate_systems.assert_called_once()
        
    def test_get_coordinate_info(self):
        """Test get_coordinate_info method."""
        mapper = CatalogueToMap(self.test_data)
        info = mapper.get_coordinate_info()
        
        # Check return type and structure
        assert isinstance(info, dict)
        required_keys = ['coordinate_systems', 'systems_found', 'total_systems', 'all_coordinate_info']
        for key in required_keys:
            assert key in info
            
        # Check types
        assert isinstance(info['coordinate_systems'], dict)
        assert isinstance(info['systems_found'], list)
        assert isinstance(info['total_systems'], int)
        assert isinstance(info['all_coordinate_info'], dict)
        
        # Check consistency
        assert len(info['systems_found']) == info['total_systems']
        assert info['systems_found'] == list(info['coordinate_systems'].keys())
        
    def test_get_coordinates_existing_system(self):
        """Test get_coordinates with existing coordinate system."""
        mapper = CatalogueToMap(self.test_data)
        
        # Should find equatorial coordinates
        if 'equatorial' in mapper.coordinate_systems:
            coords = mapper.get_coordinates('equatorial')
            assert coords is not None
            assert isinstance(coords, tuple)
            assert len(coords) == 2
            azimuthal, polar = coords
            assert isinstance(azimuthal, str)
            assert isinstance(polar, str)
        
    def test_get_coordinates_nonexistent_system(self):
        """Test get_coordinates with non-existent coordinate system."""
        mapper = CatalogueToMap(self.test_data)
        
        coords = mapper.get_coordinates('nonexistent_system')
        assert coords is None
        
    def test_has_coordinate_system_existing(self):
        """Test has_coordinate_system with existing system."""
        mapper = CatalogueToMap(self.test_data)
        
        # Check each found system
        for system in mapper.coordinate_systems.keys():
            assert mapper.has_coordinate_system(system)
            
    def test_has_coordinate_system_nonexistent(self):
        """Test has_coordinate_system with non-existent system."""
        mapper = CatalogueToMap(self.test_data)
        
        assert not mapper.has_coordinate_system('nonexistent_system')
        assert not mapper.has_coordinate_system('')
        
    def test_has_valid_coordinates_with_coordinates(self):
        """Test has_valid_coordinates when coordinates are present."""
        mapper = CatalogueToMap(self.test_data)
        
        assert mapper.has_valid_coordinates()
        
    def test_has_valid_coordinates_without_coordinates(self):
        """Test has_valid_coordinates when no coordinates are present."""
        data_no_coords = Table({
            'magnitude': np.random.uniform(15, 25, 100),
            'flux': np.random.uniform(0.1, 10, 100)
        })
        
        mapper = CatalogueToMap(data_no_coords)
        assert not mapper.has_valid_coordinates()
        
    def test_get_available_systems(self):
        """Test get_available_systems method."""
        mapper = CatalogueToMap(self.test_data)
        
        systems = mapper.get_available_systems()
        assert isinstance(systems, list)
        
        # Should match coordinate_systems keys
        assert set(systems) == set(mapper.coordinate_systems.keys())
        
    def test_make_cut_minimum_only(self):
        """Test make_cut with minimum value only."""
        mapper = CatalogueToMap(self.test_data)
        
        original_length = len(mapper.catalogue)
        minimum_mag = 20.0
        
        mapper.make_cut('magnitude', minimum=minimum_mag, maximum=None)
        
        # Check that catalogue was filtered
        assert len(mapper.catalogue) <= original_length
        
        # Check that all remaining values satisfy the cut
        remaining_mags = np.asarray(mapper.catalogue['magnitude'])
        assert np.all(remaining_mags >= minimum_mag)
        
    def test_make_cut_maximum_only(self):
        """Test make_cut with maximum value only."""
        mapper = CatalogueToMap(self.test_data)
        
        original_length = len(mapper.catalogue)
        maximum_mag = 20.0
        
        mapper.make_cut('magnitude', minimum=None, maximum=maximum_mag)
        
        # Check that catalogue was filtered
        assert len(mapper.catalogue) <= original_length
        
        # Check that all remaining values satisfy the cut
        remaining_mags = np.asarray(mapper.catalogue['magnitude'])
        assert np.all(remaining_mags <= maximum_mag)
        
    def test_make_cut_both_bounds(self):
        """Test make_cut with both minimum and maximum values."""
        mapper = CatalogueToMap(self.test_data)
        
        original_length = len(mapper.catalogue)
        minimum_mag = 18.0
        maximum_mag = 22.0
        
        mapper.make_cut('magnitude', minimum=minimum_mag, maximum=maximum_mag)
        
        # Check that catalogue was filtered
        assert len(mapper.catalogue) <= original_length
        
        # Check that all remaining values satisfy both cuts
        remaining_mags = np.asarray(mapper.catalogue['magnitude'])
        assert np.all(remaining_mags >= minimum_mag)
        assert np.all(remaining_mags <= maximum_mag)
        
    def test_make_cut_no_bounds(self):
        """Test make_cut with no bounds (should not filter)."""
        mapper = CatalogueToMap(self.test_data)
        
        original_length = len(mapper.catalogue)
        
        mapper.make_cut('magnitude', minimum=None, maximum=None)
        
        # Should not change the catalogue
        assert len(mapper.catalogue) == original_length
        
    def test_make_cut_empty_result(self):
        """Test make_cut that results in empty catalogue."""
        mapper = CatalogueToMap(self.test_data)
        
        # Apply impossible cut
        mapper.make_cut('magnitude', minimum=30.0, maximum=35.0)
        
        # Should result in empty catalogue
        assert len(mapper.catalogue) == 0
        
    @patch('dipoleutils.utils.samples.angles_to_density_map')
    def test_make_density_map_basic(self, mock_angles_to_density_map):
        """Test make_density_map with default parameters."""
        mapper = CatalogueToMap(self.test_data)
        
        # Mock the density map function
        expected_map = np.random.randint(0, 100, 12*64**2)
        mock_angles_to_density_map.return_value = expected_map
        
        if mapper.has_valid_coordinates():
            result = mapper.make_density_map()
            
            # Check that function was called
            mock_angles_to_density_map.assert_called_once()
            
            # Check result
            assert result is expected_map
            
    def test_make_density_map_no_coordinates(self):
        """Test make_density_map when no coordinates are available."""
        data_no_coords = Table({
            'magnitude': np.random.uniform(15, 25, 100),
            'flux': np.random.uniform(0.1, 10, 100)
        })
        
        mapper = CatalogueToMap(data_no_coords)
        
        with pytest.raises(ValueError, match="No valid coordinate systems identified"):
            mapper.make_density_map()
            
    def test_make_density_map_invalid_system(self):
        """Test make_density_map with invalid coordinate system."""
        mapper = CatalogueToMap(self.test_data)
        
        with pytest.raises(ValueError, match="not available"):
            mapper.make_density_map(coordinate_system='invalid_system')
            
    @patch('dipoleutils.utils.samples.angles_to_density_map')
    def test_make_density_map_specific_system(self, mock_angles_to_density_map):
        """Test make_density_map with specific coordinate system."""
        mapper = CatalogueToMap(self.test_data)
        
        expected_map = np.random.randint(0, 100, 12*64**2)
        mock_angles_to_density_map.return_value = expected_map
        
        available_systems = mapper.get_available_systems()
        if available_systems:
            system_to_test = available_systems[0]
            result = mapper.make_density_map(coordinate_system=system_to_test)
            mock_angles_to_density_map.assert_called_once()
            call_args = mock_angles_to_density_map.call_args
            
            # Check that correct parameters were passed
            assert 'lonlat' in call_args.kwargs
            assert 'nest' in call_args.kwargs
            assert 'nside' in call_args.kwargs
            
    @patch('dipoleutils.utils.samples.angles_to_density_map')
    def test_make_density_map_custom_parameters(self, mock_angles_to_density_map):
        """Test make_density_map with custom parameters."""
        mapper = CatalogueToMap(self.test_data)
        
        expected_map = np.random.randint(0, 100, 12*128**2)
        mock_angles_to_density_map.return_value = expected_map
        
        if mapper.has_valid_coordinates():
            result = mapper.make_density_map(nside=128, nest=True)
            call_args = mock_angles_to_density_map.call_args
            assert call_args.kwargs['nside'] == 128
            assert call_args.kwargs['nest'] is True


class TestCatalogueToMapIntegration:
    """
    Integration tests for CatalogueToMap.
    
    These tests verify the class works correctly with real-world scenarios
    and in combination with the coordinate parser.
    """
    
    def test_end_to_end_equatorial_only(self):
        """Test complete workflow with equatorial coordinates only."""
        data = Table({
            'ra': np.random.uniform(0, 360, 1000),
            'dec': np.random.uniform(-90, 90, 1000),
            'magnitude': np.random.uniform(15, 25, 1000)
        })
        
        mapper = CatalogueToMap(data)
        
        # Check coordinate detection
        assert mapper.has_valid_coordinates()
        assert mapper.has_coordinate_system('equatorial')
        assert len(mapper.get_available_systems()) >= 1
        
        # Check coordinate access
        coords = mapper.get_coordinates('equatorial')
        assert coords == ('ra', 'dec')
        
        # Test filtering
        mapper.make_cut('magnitude', minimum=18, maximum=22)
        remaining_mags = np.asarray(mapper.catalogue['magnitude'])
        assert np.all((remaining_mags >= 18) & (remaining_mags <= 22))
        
    def test_end_to_end_multiple_systems(self):
        """Test complete workflow with multiple coordinate systems."""
        data = Table({
            'ra': np.random.uniform(0, 360, 1000),
            'dec': np.random.uniform(-90, 90, 1000),
            'l': np.random.uniform(0, 360, 1000),
            'b': np.random.uniform(-90, 90, 1000),
            'magnitude': np.random.uniform(15, 25, 1000)
        })
        
        mapper = CatalogueToMap(data)
        
        # Check that multiple systems are detected
        assert mapper.has_valid_coordinates()
        assert len(mapper.get_available_systems()) >= 2
        
        # Check both coordinate systems
        if mapper.has_coordinate_system('equatorial'):
            eq_coords = mapper.get_coordinates('equatorial')
            assert eq_coords == ('ra', 'dec')
            
        if mapper.has_coordinate_system('galactic'):
            gal_coords = mapper.get_coordinates('galactic')
            assert gal_coords == ('l', 'b')
            
    def test_realistic_survey_data(self):
        """Test with realistic survey-like data."""
        # Simulate GAIA-like data
        n_sources = 5000
        data = Table({
            'source_id': np.arange(n_sources),
            'ra': np.random.uniform(0, 360, n_sources),
            'dec': np.random.uniform(-90, 90, n_sources),
            'l': np.random.uniform(0, 360, n_sources),
            'b': np.random.uniform(-90, 90, n_sources),
            'phot_g_mean_mag': np.random.normal(18, 2, n_sources),
            'bp_rp': np.random.normal(0.5, 0.3, n_sources),
            'parallax': np.random.exponential(0.5, n_sources),
            'parallax_error': np.random.exponential(0.1, n_sources)
        })
        
        mapper = CatalogueToMap(data)
        
        # Basic functionality checks
        assert mapper.has_valid_coordinates()
        info = mapper.get_coordinate_info()
        assert info['total_systems'] >= 2
        
        # Apply realistic cuts
        mapper.make_cut('phot_g_mean_mag', minimum=16, maximum=20)
        mapper.make_cut('parallax_error', maximum=0.1, minimum=None)
        
        # Check that filtering worked
        assert len(mapper.catalogue) <= n_sources
        assert len(mapper.catalogue) > 0  # Should still have some data
        
        # Try to create density map
        if mapper.has_valid_coordinates():
            try:
                # This would create actual density map in real scenario
                # Here we just check it doesn't crash
                available_systems = mapper.get_available_systems()
                assert len(available_systems) > 0
            except ImportError:
                # Skip if healpy/tools not available
                pass
                
    def test_coordinate_parsing_edge_cases(self):
        """Test coordinate parsing with edge cases."""
        # Mixed case, special characters, etc.
        data = Table({
            'Right_Ascension': np.random.uniform(0, 360, 100),
            'DECLINATION': np.random.uniform(-90, 90, 100),
            'galactic_longitude': np.random.uniform(0, 360, 100),
            'galactic_latitude': np.random.uniform(-90, 90, 100),
            'Magnitude_V': np.random.uniform(15, 25, 100)
        })
        
        mapper = CatalogueToMap(data)
        
        assert mapper.has_valid_coordinates()
        info = mapper.get_coordinate_info()
        
        # Should detect both coordinate systems despite mixed naming
        assert info['total_systems'] >= 1
        
    def test_sequential_operations(self):
        """Test that operations can be chained correctly."""
        data = Table({
            'ra': np.random.uniform(0, 360, 1000),
            'dec': np.random.uniform(-90, 90, 1000),
            'magnitude': np.random.uniform(15, 25, 1000),
            'color': np.random.normal(0.5, 0.2, 1000)
        })
        
        mapper = CatalogueToMap(data)
        original_length = len(mapper.catalogue)
        
        # Apply multiple cuts
        mapper.make_cut('magnitude', minimum=18, maximum=22)
        after_mag_cut = len(mapper.catalogue)
        
        mapper.make_cut('color', minimum=0.2, maximum=0.8)
        after_color_cut = len(mapper.catalogue)
        
        # Each cut should reduce (or maintain) the size
        assert after_mag_cut <= original_length
        assert after_color_cut <= after_mag_cut
        
        # Coordinate info should still be accessible
        assert mapper.has_valid_coordinates()
        
    def test_empty_catalogue_handling(self):
        """Test behavior with empty catalogues after filtering."""
        data = Table({
            'ra': np.random.uniform(0, 360, 100),
            'dec': np.random.uniform(-90, 90, 100),
            'magnitude': np.random.uniform(15, 20, 100)  # All between 15-20
        })
        
        mapper = CatalogueToMap(data)
        
        # Apply impossible cut
        mapper.make_cut('magnitude', minimum=25, maximum=30)
        
        # Should handle empty catalogue gracefully
        assert len(mapper.catalogue) == 0
        assert mapper.has_valid_coordinates()  # Should still remember coordinate systems
        
        # Should be able to create density map with empty data (will be all zeros)
        density_map = mapper.make_density_map()
        assert isinstance(density_map, np.ndarray)
        assert np.all(density_map == 0)  # Should be all zeros for empty catalogue
            
    def test_custom_coordinate_systems_integration(self):
        """Test integration with custom coordinate systems."""
        # Create data with custom coordinate names
        data = Table({
            'instrument_x': np.random.uniform(-1000, 1000, 100),
            'instrument_y': np.random.uniform(-1000, 1000, 100),
            'signal': np.random.uniform(0, 100, 100)
        })
        
        mapper = CatalogueToMap(data)
        
        # Initially should find no coordinate systems
        assert not mapper.has_valid_coordinates()
        
        # Add custom coordinate system to the parser
        mapper.parser.add_coordinate_system(
            'instrument',
            azimuthal_patterns=[r'instrument_x'],
            polar_patterns=[r'instrument_y']
        )
        
        # Re-parse (would need to recreate CatalogueToMap in real usage)
        new_mapper = CatalogueToMap(data)
        
        # Add the same custom system to new parser
        new_mapper.parser.add_coordinate_system(
            'instrument',
            azimuthal_patterns=[r'instrument_x'],
            polar_patterns=[r'instrument_y']
        )
        
        # Create new instance to trigger re-parsing
        final_mapper = CatalogueToMap(data)
        final_mapper.parser.add_coordinate_system(
            'instrument',
            azimuthal_patterns=[r'instrument_x'],
            polar_patterns=[r'instrument_y']
        )
        
        # Manually update coordinate systems for this test
        final_mapper.coordinate_systems = final_mapper.parser.parse_coordinate_systems(data)
        final_mapper.coordinate_columns = {
            'systems': final_mapper.coordinate_systems,
            'count': len(final_mapper.coordinate_systems)
        }
        
        # Now should find the custom system
        assert final_mapper.has_valid_coordinates()
        assert final_mapper.has_coordinate_system('instrument')


class TestSimulatedMultipoleMap:
    def test_simulated_multipole_map_dipole_only(self):
        sim = SimulatedMultipoleMap(nside=4, ells=[1])
        params = {
            'M0': 25.0,
            'M1': 0.01,
            'phi_l1_0': 1.0,
            'theta_l1_0': np.pi / 2,
        }
        density_map = sim.make_map(parameters=params)
        assert density_map.shape == (hp.nside2npix(4),)
        assert np.all(density_map >= 0)
        assert np.isfinite(density_map).all()

    def test_simulated_multipole_map_quadrupole_degrees(self):
        sim = SimulatedMultipoleMap(nside=4, ells=[2], angles_in_degrees=True)
        params = {
            'M0': 40.0,
            'M2': 0.02,
            'phi_l2_0': 10.0,
            'theta_l2_0': 40.0,
            'phi_l2_1': 120.0,
            'theta_l2_1': 60.0,
        }
        density_map = sim.make_map(parameters=params)
        assert density_map.shape == (hp.nside2npix(4),)
        assert density_map.dtype.kind in {'i', 'u'}

    def test_simulated_multipole_save_simulation(self, tmp_path):
        sim = SimulatedMultipoleMap(nside=4, ells=[1])
        params = {
            'M0': 20.0,
            'M1': 0.01,
            'phi_l1_0': 0.5,
            'theta_l1_0': 1.0,
        }
        density_map = sim.make_map(parameters=params, poisson_seed=123)
        map_path, metadata_path = sim.save_simulation(
            density_map=density_map,
            parameters=params,
            output_prefix=tmp_path / 'unit_sim',
            poisson_seed=123,
            extra_metadata={'tag': 'unit'}
        )
        assert map_path.exists()
        assert metadata_path.exists()
        loaded = np.load(map_path)
        assert np.array_equal(loaded, density_map)
        metadata = json.loads(metadata_path.read_text())
        assert metadata['poisson_seed'] == 123
        assert metadata['parameters']['M1'] == pytest.approx(params['M1'])
        assert metadata['extra_metadata']['tag'] == 'unit'
