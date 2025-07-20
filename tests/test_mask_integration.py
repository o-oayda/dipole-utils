import pytest
import numpy as np
import healpy as hp
import tempfile
import shutil
import os

from dipoleutils.utils.mask import Masker


class TestMaskerIntegration:
    """
    Integration tests for the Masker class.
    
    These tests verify that the full masking pipeline works correctly
    with realistic HEALPix maps and coordinate transformations.
    """
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create test maps with different resolutions
        self.nside_low = 16
        self.nside_high = 64
        
        self.npix_low = hp.nside2npix(self.nside_low)
        self.npix_high = hp.nside2npix(self.nside_high)
        
        # Create realistic density maps with some structure
        self.uniform_map_low = np.ones(self.npix_low, dtype=np.int_) * 50
        self.random_map_low = np.random.poisson(30, self.npix_low).astype(np.int_)
        
        self.uniform_map_high = np.ones(self.npix_high, dtype=np.int_) * 100
        self.random_map_high = np.random.poisson(75, self.npix_high).astype(np.int_)
    
    def test_end_to_end_equatorial_masking(self):
        """Test complete masking workflow in equatorial coordinates."""
        masker = Masker(self.random_map_high.copy(), 'equatorial')
        
        initial_unmasked = np.sum(masker.mask_map)
        print(f"Initial unmasked pixels: {initial_unmasked}")
        
        # Apply multiple masks in sequence
        masker.mask_galactic_plane(latitude_cut=20.0)
        after_galactic = np.sum(masker.mask_map)
        print(f"After galactic plane mask: {after_galactic}")
        
        masker.mask_equatorial_poles(latitude_cut=75.0)
        after_eq_poles = np.sum(masker.mask_map)
        print(f"After equatorial poles mask: {after_eq_poles}")
        
        masker.mask_ecliptic_poles(latitude_cut=75.0) 
        after_ecl_poles = np.sum(masker.mask_map)
        print(f"After ecliptic poles mask: {after_ecl_poles}")
        
        masker.mask_slice(0.0, 0.0, 15.0)  # Mask around (0,0)
        after_slice = np.sum(masker.mask_map)
        print(f"After slice mask: {after_slice}")
        
        # Get final masked map
        masked_density = masker.get_masked_density_map()
        
        # Verify progressive masking
        assert after_galactic <= initial_unmasked
        assert after_eq_poles <= after_galactic  
        assert after_ecl_poles <= after_eq_poles
        assert after_slice <= after_ecl_poles
        
        # Verify mask consistency
        nan_count = np.sum(np.isnan(masked_density))
        valid_count = np.sum(~np.isnan(masked_density))
        
        assert nan_count + valid_count == self.npix_high
        assert valid_count == after_slice
        assert nan_count == initial_unmasked - after_slice
    
    def test_end_to_end_galactic_masking(self):
        """Test complete masking workflow in galactic coordinates."""
        masker = Masker(self.random_map_low.copy(), 'galactic')
        
        initial_unmasked = np.sum(masker.mask_map)
        
        # Apply masks in different order
        masker.mask_ecliptic_poles(latitude_cut=70.0)
        masker.mask_equatorial_poles(latitude_cut=80.0)
        masker.mask_galactic_plane(latitude_cut=15.0)
        
        final_unmasked = np.sum(masker.mask_map)
        
        # Verify some masking occurred
        assert final_unmasked < initial_unmasked
        
        # Test the masked map
        masked_density = masker.get_masked_density_map()
        assert np.sum(~np.isnan(masked_density)) == final_unmasked
    
    def test_end_to_end_ecliptic_masking(self):
        """Test complete masking workflow in ecliptic coordinates.""" 
        masker = Masker(self.uniform_map_high.copy(), 'ecliptic')
        
        initial_unmasked = np.sum(masker.mask_map)
        
        # Apply comprehensive masking
        masker.mask_galactic_plane(latitude_cut=25.0)
        masker.mask_equatorial_poles(north_latitude_cut=70.0, south_latitude_cut=85.0)
        masker.mask_ecliptic_poles(latitude_cut=75.0)
        
        # Add several slice masks
        masker.mask_slice(0.0, 90.0, 5.0)    # North ecliptic pole region
        masker.mask_slice(180.0, -90.0, 5.0) # South ecliptic pole region
        masker.mask_slice(90.0, 0.0, 10.0)   # Ecliptic equator region
        
        final_unmasked = np.sum(masker.mask_map)
        masked_density = masker.get_masked_density_map()
        
        # Verify extensive masking
        assert final_unmasked < initial_unmasked * 0.8  # At least 20% masked
        
        # Verify all unmasked pixels have original value
        unmasked_pixels = masked_density[~np.isnan(masked_density)]
        assert np.all(unmasked_pixels == 100)  # Original uniform value
    
    def test_coordinate_transformation_consistency(self):
        """Test that masking results are consistent across coordinate transformations."""
        # Create identical maps in different coordinate systems
        test_map = self.random_map_low.copy()
        
        maskers = {
            'equatorial': Masker(test_map.copy(), 'equatorial'),
            'galactic': Masker(test_map.copy(), 'galactic'), 
            'ecliptic': Masker(test_map.copy(), 'ecliptic')
        }
        
        # Apply the same galactic plane masking to all
        for masker in maskers.values():
            masker.mask_galactic_plane(latitude_cut=20.0)
        
        # Get masked pixel counts
        masked_counts = {
            coord: np.sum(masker.mask_map) 
            for coord, masker in maskers.items()
        }
        
        # The number of unmasked pixels should vary between coordinate systems
        # because the galactic plane appears differently in each system
        counts = list(masked_counts.values())
        assert not all(c == counts[0] for c in counts), \
            "Masking should differ across coordinate systems"
        
        # But all should have some masking
        for coord, count in masked_counts.items():
            assert count < self.npix_low, f"No masking occurred in {coord} system"
    
    def test_asymmetric_pole_masking_integration(self):
        """Test asymmetric pole masking in realistic scenarios."""
        masker = Masker(self.random_map_high.copy(), 'equatorial')
        
        # Test various asymmetric configurations
        test_cases = [
            {'north_latitude_cut': 60.0, 'south_latitude_cut': 80.0},
            {'north_latitude_cut': 85.0, 'south_latitude_cut': 65.0},
            {'north_latitude_cut': 70.0, 'south_latitude_cut': 70.0},  # symmetric
        ]
        
        results = []
        for case in test_cases:
            masker.reset_mask()
            masker.mask_equatorial_poles(**case)
            unmasked_count = np.sum(masker.mask_map)
            results.append(unmasked_count)
        
        # Verify that different asymmetric settings produce different results
        assert len(set(results)) > 1, "Different asymmetric settings should produce different masking"
        
        # The case with lower thresholds should mask more pixels
        more_aggressive_idx = 0  # 60° north, 80° south
        less_aggressive_idx = 2   # 70° north, 70° south (symmetric)
        
        assert results[more_aggressive_idx] < results[less_aggressive_idx], \
            "More aggressive masking should result in fewer unmasked pixels"
    
    def test_cumulative_masking_with_reset(self):
        """Test that cumulative masking works correctly with resets."""
        masker = Masker(self.uniform_map_low.copy(), 'galactic')
        
        # Track masking progression
        progression = []
        
        # Initial state
        progression.append(('initial', np.sum(masker.mask_map)))
        
        # Add masks progressively
        masker.mask_galactic_plane(latitude_cut=30.0)
        progression.append(('galactic_plane', np.sum(masker.mask_map)))
        
        masker.mask_equatorial_poles(latitude_cut=70.0)
        progression.append(('with_eq_poles', np.sum(masker.mask_map)))
        
        masker.mask_slice(180.0, 0.0, 20.0)
        progression.append(('with_slice', np.sum(masker.mask_map)))
        
        # Verify monotonic decrease
        for i in range(1, len(progression)):
            prev_count = progression[i-1][1]
            curr_count = progression[i][1]
            assert curr_count <= prev_count, \
                f"Masking should be cumulative: {progression[i-1][0]} -> {progression[i][0]}"
        
        # Reset and verify
        masker.reset_mask()
        assert np.sum(masker.mask_map) == progression[0][1], "Reset should restore initial state"
        assert len(masker.masked_pixel_indices) == 0, "Reset should clear masked indices"
    
    def test_large_scale_masking_performance(self):
        """Test masking performance with larger maps."""
        # Create a moderately large map  
        nside_large = 128
        npix_large = hp.nside2npix(nside_large)
        large_map = np.random.poisson(50, npix_large).astype(np.int_)
        
        masker = Masker(large_map, 'equatorial')
        
        # Apply comprehensive masking and time it (basic performance check)
        import time
        start_time = time.time()
        
        masker.mask_galactic_plane(latitude_cut=25.0)
        masker.mask_equatorial_poles(north_latitude_cut=75.0, south_latitude_cut=80.0)
        masker.mask_ecliptic_poles(latitude_cut=75.0)
        
        # Add multiple slice masks
        for lon in [0, 60, 120, 180, 240, 300]:
            masker.mask_slice(lon, 0.0, 8.0)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Get final result
        masked_density = masker.get_masked_density_map()
        final_unmasked = np.sum(~np.isnan(masked_density))
        
        # Basic performance assertion (should complete in reasonable time)
        assert processing_time < 10.0, f"Masking took too long: {processing_time:.2f} seconds"
        
        # Verify substantial masking occurred
        masking_fraction = 1.0 - (final_unmasked / npix_large)
        assert masking_fraction > 0.1, f"Expected significant masking, got {masking_fraction:.2%}"
        
        print(f"Large scale masking completed in {processing_time:.3f} seconds")
        print(f"Masked {masking_fraction:.2%} of {npix_large} pixels")
    
    def test_edge_cases_and_robustness(self):
        """Test edge cases and robustness of masking operations."""
        masker = Masker(self.random_map_low.copy(), 'equatorial')
        
        # Test extreme latitude cuts
        masker.mask_equatorial_poles(latitude_cut=89.9)  # Very conservative
        very_conservative = np.sum(masker.mask_map)
        
        masker.reset_mask()
        masker.mask_equatorial_poles(latitude_cut=10.0)   # Very aggressive
        very_aggressive = np.sum(masker.mask_map)
        
        assert very_conservative > very_aggressive, "Conservative masking should leave more pixels"
        
        # Test small radius slice  
        masker.reset_mask()
        initial = np.sum(masker.mask_map)
        masker.mask_slice(0.0, 0.0, 5.0)  # 5 degree radius should definitely mask pixels
        after_small_slice = np.sum(masker.mask_map)
        
        # Should mask at least some pixels
        assert after_small_slice < initial, "Small slice should mask some pixels"
        
        # Test large radius slice
        masker.reset_mask()
        initial_large = np.sum(masker.mask_map)
        masker.mask_slice(0.0, 0.0, 90.0)  # Quarter sphere
        after_large_slice = np.sum(masker.mask_map)
        
        # Should mask a significant fraction
        masking_fraction = 1.0 - (after_large_slice / initial_large)
        assert masking_fraction > 0.2, "Large slice should mask substantial area"
    
    def test_realistic_survey_masking_scenario(self):
        """Test a realistic astronomical survey masking scenario."""
        # Simulate a realistic survey map
        masker = Masker(self.random_map_high.copy(), 'equatorial')
        
        initial_pixels = np.sum(masker.mask_map)
        
        # Typical survey masking sequence:
        # 1. Remove galactic plane contamination
        masker.mask_galactic_plane(latitude_cut=20.0)
        after_galactic = np.sum(masker.mask_map)
        
        # 2. Remove high-extinction regions near galactic center  
        masker.mask_slice(266.4, -28.9, 15.0)  # Galactic center region
        after_gc = np.sum(masker.mask_map)
        
        # 3. Remove polar regions with poor coverage
        masker.mask_equatorial_poles(north_latitude_cut=75.0, south_latitude_cut=80.0)
        after_poles = np.sum(masker.mask_map)
        
        # 4. Remove bright star regions (simulate multiple)
        bright_star_positions = [(45.0, 20.0), (120.0, -30.0), (200.0, 60.0)]
        for ra, dec in bright_star_positions:
            masker.mask_slice(ra, dec, 5.0)
        after_stars = np.sum(masker.mask_map)
        
        # Get final survey footprint
        survey_footprint = masker.get_masked_density_map()
        valid_survey_pixels = np.sum(~np.isnan(survey_footprint))
        
        # Verify realistic survey constraints
        total_masking_fraction = 1.0 - (valid_survey_pixels / initial_pixels)
        
        # Typical surveys mask 30-70% of the sky
        assert 0.2 < total_masking_fraction < 0.8, \
            f"Survey masking fraction {total_masking_fraction:.2%} seems unrealistic"
        
        # Verify progressive masking
        assert after_galactic < initial_pixels
        assert after_gc < after_galactic  
        assert after_poles < after_gc
        assert after_stars < after_poles
        assert after_stars == valid_survey_pixels
        
        print(f"Realistic survey retained {100*(1-total_masking_fraction):.1f}% of sky")
        print(f"Final survey footprint: {valid_survey_pixels} pixels")
