import pytest
import numpy as np
from astropy.table import Table

from dipoleutils.utils.crossmatch import CrossMatch


class TestCrossMatchIntegration:
    """
    Integration tests for the CrossMatch class.
    
    These tests verify that the full cross-matching pipeline works correctly
    with realistic astronomical catalogue data and coordinate systems.
    """
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create realistic test catalogues with equatorial coordinates
        self.catalogueA = Table({
            'ra': np.random.uniform(0, 360, 50),  # RA in degrees
            'dec': np.random.uniform(-90, 90, 50),  # Dec in degrees
            'source_name': [f'SRC_A_{i:03d}' for i in range(50)],
            'flux': np.random.exponential(1.0, 50)  # Random flux values
        })
        
        # Create catalogue B with some sources close to A sources and some different
        # First 15 sources are close to first 15 sources in A (within ~5 arcsec)
        close_ra = np.asarray(self.catalogueA['ra'][:15]) + np.random.normal(0, 0.001, 15)  # ~3.6 arcsec
        close_dec = np.asarray(self.catalogueA['dec'][:15]) + np.random.normal(0, 0.001, 15)
        
        # Next 10 sources are random (no matches expected)
        random_ra = np.random.uniform(0, 360, 10)
        random_dec = np.random.uniform(-90, 90, 10)
        
        self.catalogueB = Table({
            'ra': np.concatenate([close_ra, random_ra]),
            'dec': np.concatenate([close_dec, random_dec]),
            'source_name': [f'SRC_B_{i:03d}' for i in range(25)],
            'magnitude': np.random.normal(20, 2, 25)  # Random magnitudes
        })
    
    def test_end_to_end_crossmatch_basic(self):
        """Test complete cross-matching workflow with basic validation."""
        crossmatch = CrossMatch(self.catalogueA, self.catalogueB, 'equatorial')
        
        # Perform cross-match with 10 arcsec radius
        result = crossmatch.cross_match(radius=10.0)
        
        # Verify result structure
        assert isinstance(result, Table)
        assert len(result) == len(self.catalogueA)  # One row per A source
        
        # Check column names
        expected_columns = [
            'source_idx_A', 'source_idx_B', 'source_name_A', 
            'source_name_B', 'angular_distance_arcsec'
        ]
        for col in expected_columns:
            assert col in result.colnames
        
        # Check that some matches were found
        n_matches = crossmatch.get_number_of_matches()
        assert n_matches > 0
        assert n_matches <= 15  # Can't exceed the number of close sources
    
    def test_duplicate_filtering_with_mock_data(self):
        """
        Test the duplicate filtering functionality with a controlled scenario.
        Create a scenario where multiple A sources match the same B source.
        """
        # Create specific test catalogues for duplicate scenario
        catA = Table({
            'ra': [10.000, 10.001, 10.002, 20.000, 30.000],  # First 3 close to same B source
            'dec': [5.000, 5.001, 5.002, 15.000, 25.000],
            'source_name': ['A1', 'A2', 'A3', 'A4', 'A5']
        })
        
        catB = Table({
            'ra': [10.0005, 20.0005, 40.000],  # B1 is close to A1,A2,A3
            'dec': [5.0005, 15.0005, 35.000],
            'source_name': ['B1', 'B2', 'B3']
        })
        
        crossmatch = CrossMatch(catA, catB, 'equatorial')
        result = crossmatch.cross_match(radius=10.0)  # Large enough to catch all close matches
        
        # Check that duplicate filtering worked
        n_matches = crossmatch.get_number_of_matches()
        assert n_matches == 2  # A1->B1 and A4->B2 (A2,A3 should be filtered as duplicates)
        
        # Verify that we have the expected number of valid and invalid matches
        source_idx_B_array = np.asarray(result['source_idx_B'])
        valid_matches = np.sum(source_idx_B_array != -1)
        assert valid_matches == n_matches
        
        # Check that no B source appears twice in the matches
        valid_B_indices = source_idx_B_array[source_idx_B_array != -1]
        unique_B_indices = np.unique(valid_B_indices)
        assert len(unique_B_indices) == len(valid_B_indices)  # No duplicates
    
    def test_coordinate_system_validation(self):
        """Test that coordinate system validation works correctly."""
        
        # Create catalogue with only RA/Dec
        cat_equatorial_only = Table({
            'ra': [10.0, 20.0],
            'dec': [5.0, 15.0],
            'source_name': ['A1', 'A2']
        })
        
        # Should work with equatorial
        crossmatch = CrossMatch(cat_equatorial_only, cat_equatorial_only, 'equatorial')
        assert crossmatch.coordinate_system == 'equatorial'
        
        # Should fail with galactic (no l,b columns)
        with pytest.raises(AssertionError):
            CrossMatch(cat_equatorial_only, cat_equatorial_only, 'galactic')
    
    def test_empty_catalogue_handling(self):
        """Test handling of edge cases like empty catalogues."""
        # Test with empty catalogue B
        empty_catB = Table({
            'ra': np.array([]),
            'dec': np.array([]),
            'source_name': np.array([])
        })
        
        small_catA = Table({
            'ra': np.asarray(self.catalogueA['ra'][:5]),
            'dec': np.asarray(self.catalogueA['dec'][:5]),
            'source_name': np.asarray(self.catalogueA['source_name'][:5])
        })
        crossmatch = CrossMatch(small_catA, empty_catB, 'equatorial')
        result = crossmatch.cross_match(radius=10.0)
        
        assert crossmatch.get_number_of_matches() == 0
        assert np.all(np.asarray(result['source_idx_B']) == -1)


class TestCrossMatchDuplicateFilteringIntegration:
    """
    Focused integration tests specifically for the duplicate filtering functionality.
    """
    
    def test_many_to_one_duplicate_scenario(self):
        """
        Test a complex scenario where many A sources match to one B source.
        Verifies that only the closest match is kept.
        """
        # Create scenario: 5 A sources all close to 1 B source with known distances
        catA = Table({
            'ra': [10.000, 10.001, 10.002, 10.003, 10.004],  # Increasing distance from B1
            'dec': [5.000, 5.000, 5.000, 5.000, 5.000],
            'source_name': ['A_closest', 'A_second', 'A_third', 'A_fourth', 'A_farthest']
        })
        
        catB = Table({
            'ra': [10.000],  # B1 exactly at A1 position
            'dec': [5.000],
            'source_name': ['B_target']
        })
        
        crossmatch = CrossMatch(catA, catB, 'equatorial')
        result = crossmatch.cross_match(radius=20.0)  # Large radius to catch all
        
        # Only one match should remain (the closest one)
        assert crossmatch.get_number_of_matches() == 1
        
        # Verify that we have exactly one valid match
        source_idx_B_array = np.asarray(result['source_idx_B'])
        valid_matches = np.sum(source_idx_B_array != -1)
        assert valid_matches == 1
        
        # The valid match should correspond to A_closest (index 0)
        valid_indices = np.where(source_idx_B_array != -1)[0]
        assert len(valid_indices) == 1
        assert valid_indices[0] == 0  # First A source (closest) should be kept
        
        # All other matches should be invalid
        invalid_matches = np.sum(source_idx_B_array == -1)
        assert invalid_matches == 4
    
    def test_multiple_duplicate_groups(self):
        """
        Test scenario with multiple groups of duplicates.
        """
        # Create scenario: 
        # - A1,A2,A3 all match B1 (A1 closest)
        # - A4,A5 both match B2 (A4 closest)  
        # - A6 matches B3 (no duplicates)
        catA = Table({
            'ra': [10.000, 10.002, 10.004,   # Group 1: match B1
                   20.000, 20.003,          # Group 2: match B2
                   30.000],                 # Group 3: match B3
            'dec': [5.000, 5.000, 5.000,    # Group 1
                    15.000, 15.000,         # Group 2  
                    25.000],                # Group 3
            'source_name': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
        })
        
        catB = Table({
            'ra': [10.001, 20.001, 30.001],   # B1, B2, B3
            'dec': [5.001, 15.001, 25.001],
            'source_name': ['B1', 'B2', 'B3']
        })
        
        crossmatch = CrossMatch(catA, catB, 'equatorial')
        result = crossmatch.cross_match(radius=30.0)
        
        # Should have exactly 3 matches (A1->B1, A4->B2, A6->B3)
        assert crossmatch.get_number_of_matches() == 3
        
        # Verify the structure
        source_idx_B_array = np.asarray(result['source_idx_B'])
        valid_matches = np.sum(source_idx_B_array != -1)
        assert valid_matches == 3
        
        # Check that no B source appears twice
        valid_B_indices = source_idx_B_array[source_idx_B_array != -1]
        unique_B_indices = np.unique(valid_B_indices)
        assert len(unique_B_indices) == len(valid_B_indices)
        
        # Check that the correct A sources were kept (closest ones)
        valid_A_indices = np.where(source_idx_B_array != -1)[0]
        expected_kept_A_indices = {0, 3, 5}  # A1, A4, A6
        assert set(valid_A_indices) == expected_kept_A_indices
    
    def test_precision_and_very_close_sources(self):
        """
        Test cross-matching with very close sources to verify precision.
        """
        # Create sources separated by tiny amounts
        catA = Table({
            'ra': [10.0000, 10.0001],  # 0.36 arcsec apart
            'dec': [5.0000, 5.0000],
            'source_name': ['A1', 'A2']
        })
        
        catB = Table({
            'ra': [10.00005],  # Closer to A1 than A2
            'dec': [5.00005],
            'source_name': ['B1']
        })
        
        crossmatch = CrossMatch(catA, catB, 'equatorial')
        result = crossmatch.cross_match(radius=2.0)  # 2 arcsec radius
        
        # Both A1 and A2 should initially match B1, but only closest should remain
        assert crossmatch.get_number_of_matches() == 1
        
        # A1 should be kept (closer to B1)
        source_idx_B_array = np.asarray(result['source_idx_B'])
        valid_indices = np.where(source_idx_B_array != -1)[0]
        assert len(valid_indices) == 1
        assert valid_indices[0] == 0  # A1 should be kept
