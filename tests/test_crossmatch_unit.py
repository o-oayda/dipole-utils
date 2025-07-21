import pytest
import numpy as np
from astropy.table import Table
from unittest.mock import Mock, patch, MagicMock
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle

from dipoleutils.utils.crossmatch import CrossMatch


class TestCrossMatchUnit:
    """
    Unit tests for the CrossMatch class.
    
    These tests focus on individual methods and components in isolation,
    testing the cross-matching logic without external dependencies.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create simple mock catalogues for testing
        self.mock_catalogueA = Table({
            'ra': [10.0, 20.0, 30.0, 40.0],
            'dec': [5.0, 15.0, 25.0, 35.0],
            'source_name': ['A1', 'A2', 'A3', 'A4']
        })
        
        self.mock_catalogueB = Table({
            'ra': [10.1, 20.1, 30.1],
            'dec': [5.1, 15.1, 25.1],
            'source_name': ['B1', 'B2', 'B3']
        })
        
        # Mock the coordinate parser
        self.mock_coords = {
            'equatorial': {
                'azimuthal': 'ra',
                'polar': 'dec'
            }
        }
    
    @patch('dipoleutils.utils.crossmatch.CoordinateSystemParser')
    def test_crossmatch_initialization(self, mock_parser_class):
        """Test that CrossMatch initializes correctly."""
        mock_parser = Mock()
        mock_parser.parse_coordinate_systems.return_value = self.mock_coords
        mock_parser_class.return_value = mock_parser
        
        crossmatch = CrossMatch(
            self.mock_catalogueA, 
            self.mock_catalogueB, 
            'equatorial'
        )
        
        # Check basic attributes
        assert crossmatch.catalogueA is self.mock_catalogueA
        assert crossmatch.catalogueB is self.mock_catalogueB
        assert crossmatch.coordinate_system == 'equatorial'
        assert crossmatch.crossmatch_catalogue is None
        assert crossmatch.n_matches == 0
        assert crossmatch.lonA_column == 'ra'
        assert crossmatch.latA_column == 'dec'
        assert crossmatch.lonB_column == 'ra'
        assert crossmatch.latB_column == 'dec'
    
    @patch('dipoleutils.utils.crossmatch.CoordinateSystemParser')
    def test_initialization_invalid_coordinate_system(self, mock_parser_class):
        """Test that initialization fails with invalid coordinate system."""
        mock_parser = Mock()
        mock_parser.parse_coordinate_systems.return_value = self.mock_coords
        mock_parser_class.return_value = mock_parser
        
        with pytest.raises(AssertionError):
            CrossMatch(
                self.mock_catalogueA, 
                self.mock_catalogueB, 
                'invalid_system'
            )
    
    def test_determine_source_name_columns_explicit(self):
        """Test source name column determination with explicit columns."""
        with patch('dipoleutils.utils.crossmatch.CoordinateSystemParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_coordinate_systems.return_value = self.mock_coords
            mock_parser_class.return_value = mock_parser
            
            crossmatch = CrossMatch(
                self.mock_catalogueA, 
                self.mock_catalogueB, 
                'equatorial'
            )
            
            crossmatch._determine_source_name_columns('source_name', 'source_name')
            
            assert crossmatch.source_name_A_column == 'source_name'
            assert crossmatch.source_name_B_column == 'source_name'
    
    def test_determine_source_name_columns_default(self):
        """Test source name column determination with default 'source_name'."""
        with patch('dipoleutils.utils.crossmatch.CoordinateSystemParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_coordinate_systems.return_value = self.mock_coords
            mock_parser_class.return_value = mock_parser
            
            crossmatch = CrossMatch(
                self.mock_catalogueA, 
                self.mock_catalogueB, 
                'equatorial'
            )
            
            crossmatch._determine_source_name_columns(None, None)
            
            assert crossmatch.source_name_A_column == 'source_name'
            assert crossmatch.source_name_B_column == 'source_name'
    
    def test_determine_source_name_columns_invalid(self):
        """Test source name column determination with invalid column names."""
        with patch('dipoleutils.utils.crossmatch.CoordinateSystemParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_coordinate_systems.return_value = self.mock_coords
            mock_parser_class.return_value = mock_parser
            
            crossmatch = CrossMatch(
                self.mock_catalogueA, 
                self.mock_catalogueB, 
                'equatorial'
            )
            
            with pytest.raises(ValueError, match="Column 'invalid_col' not found in catalogue A"):
                crossmatch._determine_source_name_columns('invalid_col', 'source_name')
            
            with pytest.raises(ValueError, match="Column 'invalid_col' not found in catalogue B"):
                crossmatch._determine_source_name_columns('source_name', 'invalid_col')
    
    def test_determine_source_name_columns_missing_default(self):
        """Test source name column determination when default 'source_name' is missing."""
        catalogueA_no_source_name = Table({
            'ra': [10.0, 20.0],
            'dec': [5.0, 15.0],
            'name': ['A1', 'A2']  # Different column name
        })
        
        with patch('dipoleutils.utils.crossmatch.CoordinateSystemParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_coordinate_systems.return_value = self.mock_coords
            mock_parser_class.return_value = mock_parser
            
            crossmatch = CrossMatch(
                catalogueA_no_source_name, 
                self.mock_catalogueB, 
                'equatorial'
            )
            
            with pytest.raises(ValueError, match="No 'source_name' column found in catalogue A"):
                crossmatch._determine_source_name_columns(None, None)
    
    def test_filter_duplicate_B_sources_no_crossmatch_table(self):
        """Test filtering when crossmatch table doesn't exist."""
        with patch('dipoleutils.utils.crossmatch.CoordinateSystemParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_coordinate_systems.return_value = self.mock_coords
            mock_parser_class.return_value = mock_parser
            
            crossmatch = CrossMatch(
                self.mock_catalogueA, 
                self.mock_catalogueB, 
                'equatorial'
            )
            
            with pytest.raises(Exception, match="Crossmatch table is None"):
                crossmatch._filter_duplicate_B_sources()
    
    def test_filter_duplicate_B_sources_no_matches(self):
        """Test filtering when there are no matches."""
        with patch('dipoleutils.utils.crossmatch.CoordinateSystemParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_coordinate_systems.return_value = self.mock_coords
            mock_parser_class.return_value = mock_parser
            
            crossmatch = CrossMatch(
                self.mock_catalogueA, 
                self.mock_catalogueB, 
                'equatorial'
            )
            
            # Create empty crossmatch table
            crossmatch.crossmatch_catalogue = Table({
                'source_idx_A': [0, 1, 2],
                'source_idx_B': [-1, -1, -1],  # No matches
                'source_name_A': ['A1', 'A2', 'A3'],
                'source_name_B': [None, None, None],
                'angular_distance_arcsec': [np.nan, np.nan, np.nan]
            })
            crossmatch.n_matches = 0
            
            crossmatch._filter_duplicate_B_sources()
            
            # Should return early without error
            assert crossmatch.n_matches == 0
    
    def test_filter_duplicate_B_sources_no_duplicates(self):
        """Test filtering when there are no duplicate B sources."""
        with patch('dipoleutils.utils.crossmatch.CoordinateSystemParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_coordinate_systems.return_value = self.mock_coords
            mock_parser_class.return_value = mock_parser
            
            crossmatch = CrossMatch(
                self.mock_catalogueA, 
                self.mock_catalogueB, 
                'equatorial'
            )
            
            # Create crossmatch table with unique matches
            crossmatch.crossmatch_catalogue = Table({
                'source_idx_A': [0, 1, 2],
                'source_idx_B': [0, 1, 2],  # All unique B sources
                'source_name_A': ['A1', 'A2', 'A3'],
                'source_name_B': ['B1', 'B2', 'B3'],
                'angular_distance_arcsec': [1.0, 2.0, 3.0]
            })
            crossmatch.n_matches = 3
            
            crossmatch._filter_duplicate_B_sources()
            
            # Should not change anything
            assert crossmatch.n_matches == 3
            assert np.all(np.asarray(crossmatch.crossmatch_catalogue['source_idx_B']) != -1)
    
    def test_filter_duplicate_B_sources_with_duplicates(self):
        """Test filtering when there are duplicate B sources - keep closest match."""
        with patch('dipoleutils.utils.crossmatch.CoordinateSystemParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_coordinate_systems.return_value = self.mock_coords
            mock_parser_class.return_value = mock_parser
            
            crossmatch = CrossMatch(
                self.mock_catalogueA, 
                self.mock_catalogueB, 
                'equatorial'
            )
            
            # Create crossmatch table with duplicate B sources
            # A1 and A2 both match to B1, but A1 is closer (1.0 vs 3.0 arcsec)
            # A3 matches to B2 (no duplicate)
            crossmatch.crossmatch_catalogue = Table({
                'source_idx_A': [0, 1, 2, 3],
                'source_idx_B': [0, 0, 1, -1],  # A1 and A2 both match B1
                'source_name_A': ['A1', 'A2', 'A3', 'A4'],
                'source_name_B': ['B1', 'B1', 'B2', None],
                'angular_distance_arcsec': [1.0, 3.0, 2.0, np.nan]
            })
            crossmatch.n_matches = 3
            
            crossmatch._filter_duplicate_B_sources()
            
            # Should keep A1->B1 (closer) and invalidate A2->B1 (farther)
            # A3->B2 should remain unchanged
            assert crossmatch.n_matches == 2
            
            # Check that A1->B1 is kept (source_idx_A = 0)
            assert crossmatch.crossmatch_catalogue['source_idx_B'][0] == 0
            
            # Check that A2->B1 is invalidated (source_idx_A = 1)
            assert crossmatch.crossmatch_catalogue['source_idx_B'][1] == -1
            assert crossmatch.crossmatch_catalogue['source_name_B'][1] is None
            distance_val = np.asarray(crossmatch.crossmatch_catalogue['angular_distance_arcsec'])[1]
            assert np.isnan(distance_val)
            
            # Check that A3->B2 is kept (source_idx_A = 2)
            assert crossmatch.crossmatch_catalogue['source_idx_B'][2] == 1
    
    def test_get_number_of_matches(self):
        """Test getting the number of matches."""
        with patch('dipoleutils.utils.crossmatch.CoordinateSystemParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_coordinate_systems.return_value = self.mock_coords
            mock_parser_class.return_value = mock_parser
            
            crossmatch = CrossMatch(
                self.mock_catalogueA, 
                self.mock_catalogueB, 
                'equatorial'
            )
            
            crossmatch.n_matches = 5
            assert crossmatch.get_number_of_matches() == 5


class TestCrossMatchDuplicateFiltering:
    """
    Focused unit tests specifically for the duplicate filtering functionality.
    """
    
    def setup_method(self):
        """Set up test fixtures for duplicate filtering tests."""
        self.mock_coords = {
            'equatorial': {
                'azimuthal': 'ra',
                'polar': 'dec'
            }
        }
    
    def test_complex_duplicate_scenario(self):
        """Test a complex scenario with multiple duplicates and edge cases."""
        mock_catalogueA = Table({
            'ra': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'dec': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'source_name': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
        })
        
        mock_catalogueB = Table({
            'ra': [1.0, 2.0, 3.0],
            'dec': [1.0, 2.0, 3.0],
            'source_name': ['B1', 'B2', 'B3']
        })
        
        with patch('dipoleutils.utils.crossmatch.CoordinateSystemParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_coordinate_systems.return_value = self.mock_coords
            mock_parser_class.return_value = mock_parser
            
            crossmatch = CrossMatch(mock_catalogueA, mock_catalogueB, 'equatorial')
            
            # Create complex crossmatch scenario:
            # - A1, A2, A3 all match to B1 (distances: 0.5, 1.5, 2.5)
            # - A4, A5 both match to B2 (distances: 1.0, 3.0)
            # - A6 has no match
            crossmatch.crossmatch_catalogue = Table({
                'source_idx_A': [0, 1, 2, 3, 4, 5],
                'source_idx_B': [0, 0, 0, 1, 1, -1],
                'source_name_A': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
                'source_name_B': ['B1', 'B1', 'B1', 'B2', 'B2', None],
                'angular_distance_arcsec': [0.5, 1.5, 2.5, 1.0, 3.0, np.nan]
            })
            crossmatch.n_matches = 5
            
            crossmatch._filter_duplicate_B_sources()
            
            # Should keep only the closest matches:
            # - A1->B1 (distance 0.5) - keep
            # - A2->B1 (distance 1.5) - remove
            # - A3->B1 (distance 2.5) - remove  
            # - A4->B2 (distance 1.0) - keep
            # - A5->B2 (distance 3.0) - remove
            # - A6 (no match) - unchanged
            
            assert crossmatch.n_matches == 2
            
            # Check A1->B1 is kept
            assert crossmatch.crossmatch_catalogue['source_idx_B'][0] == 0
            
            # Check A2->B1 is invalidated
            assert crossmatch.crossmatch_catalogue['source_idx_B'][1] == -1
            assert crossmatch.crossmatch_catalogue['source_name_B'][1] is None
            distance_val = np.asarray(crossmatch.crossmatch_catalogue['angular_distance_arcsec'])[1]
            assert np.isnan(distance_val)
            
            # Check A3->B1 is invalidated  
            assert crossmatch.crossmatch_catalogue['source_idx_B'][2] == -1
            assert crossmatch.crossmatch_catalogue['source_name_B'][2] is None
            distance_val = np.asarray(crossmatch.crossmatch_catalogue['angular_distance_arcsec'])[2]
            assert np.isnan(distance_val)
            
            # Check A4->B2 is kept
            assert crossmatch.crossmatch_catalogue['source_idx_B'][3] == 1
            
            # Check A5->B2 is invalidated
            assert crossmatch.crossmatch_catalogue['source_idx_B'][4] == -1
            assert crossmatch.crossmatch_catalogue['source_name_B'][4] is None
            distance_val = np.asarray(crossmatch.crossmatch_catalogue['angular_distance_arcsec'])[4]
            assert np.isnan(distance_val)
            
            # Check A6 (no match) is unchanged
            assert crossmatch.crossmatch_catalogue['source_idx_B'][5] == -1
    
    def test_edge_case_same_distance_duplicates(self):
        """Test edge case where duplicates have exactly the same distance."""
        mock_catalogueA = Table({
            'ra': [1.0, 2.0],
            'dec': [1.0, 2.0],
            'source_name': ['A1', 'A2']
        })
        
        mock_catalogueB = Table({
            'ra': [1.0],
            'dec': [1.0],
            'source_name': ['B1']
        })
        
        with patch('dipoleutils.utils.crossmatch.CoordinateSystemParser') as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_coordinate_systems.return_value = self.mock_coords
            mock_parser_class.return_value = mock_parser
            
            crossmatch = CrossMatch(mock_catalogueA, mock_catalogueB, 'equatorial')
            
            # Both A1 and A2 match to B1 with same distance
            crossmatch.crossmatch_catalogue = Table({
                'source_idx_A': [0, 1],
                'source_idx_B': [0, 0],
                'source_name_A': ['A1', 'A2'],
                'source_name_B': ['B1', 'B1'],
                'angular_distance_arcsec': [1.0, 1.0]  # Same distance
            })
            crossmatch.n_matches = 2
            
            crossmatch._filter_duplicate_B_sources()
            
            # Should keep one and remove the other (deterministic based on sort order)
            assert crossmatch.n_matches == 1
            
            # Check that exactly one match remains
            valid_matches = crossmatch.crossmatch_catalogue['source_idx_B'] != -1
            assert np.sum(valid_matches) == 1
