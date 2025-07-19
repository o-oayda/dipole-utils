import pytest
import numpy as np
import healpy as hp
import tempfile
import shutil
import os
from unittest.mock import patch

from dipoleutils.models.dipole import Dipole
from dipoleutils.models.priors import Prior


class TestDipoleNestSamplingIntegration:
    """
    Integration tests for nested sampling with the Dipole model.
    
    These tests verify that the full nested sampling pipeline works correctly
    with mock data, checking that all components work together properly.
    """
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create temporary directory for ultranest logs
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Test parameters
        self.nside = 64
        self.npix = hp.nside2npix(self.nside)
        
        # Known dipole parameters for mock data generation
        self.true_dipole_amplitude = 0.03
        self.true_dipole_longitude = np.pi  # 180 degrees
        self.true_dipole_colatitude = np.pi/2  # 90 degrees (equator)
        self.true_mean_density = 50.0
        
    def teardown_method(self):
        """Clean up after each test method."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir)
    
    def generate_mock_density_map(self):
        """
        Generate a mock HEALPix density map with a known dipole signal.
        
        :param add_noise: Whether to add Poisson noise to the map
        :return: Mock density map with dipole signal
        """
        # Get pixel vectors for the entire sky
        pixel_vectors = np.array(hp.pix2vec(self.nside, np.arange(self.npix))).T
        
        # Create dipole direction vector
        dipole_direction = hp.ang2vec(
            self.true_dipole_colatitude,
            self.true_dipole_longitude
        )
        
        # Compute dipole signal: 1 + D * cos(theta)
        cos_theta = np.dot(pixel_vectors, dipole_direction)
        dipole_signal = 1 + self.true_dipole_amplitude * cos_theta
        
        # Generate expected counts per pixel
        expected_counts = self.true_mean_density * dipole_signal
        
        return np.random.poisson(expected_counts)
    
    def test_poisson_likelihood_integration(self):
        """
        Test full nested sampling integration with Poisson likelihood.
        
        This test:
        1. Generates mock data with known dipole parameters
        2. Runs nested sampling with the Dipole model
        3. Verifies that sampling completes successfully
        4. Checks that samples are recovered with reasonable accuracy
        """
        # Generate mock data
        mock_density_map = self.generate_mock_density_map()
        
        # Create dipole model with Poisson likelihood
        dipole_model = Dipole(
            density_map=mock_density_map,
            likelihood='poisson'
        )
        
        # Verify model initialization
        assert dipole_model.likelihood == 'poisson'
        assert dipole_model.ndim == 4  # Nbar, D, phi, theta
        assert len(dipole_model.parameter_names) == 4
        assert 'Nbar' in dipole_model.parameter_names
        assert 'D' in dipole_model.parameter_names
        assert 'phi' in dipole_model.parameter_names
        assert 'theta' in dipole_model.parameter_names
        
        # Run nested sampling with minimal settings for speed
        dipole_model.run_nested_sampling(
            reactive_sampler_kwargs={
                'log_dir': 'test_ultranest_logs',
                'resume': False
            },
            run_kwargs={
                'min_num_live_points': 50,  # Minimal for speed
                'max_num_improvement_loops': 1,
                'min_ess': 20,
                'max_iters': 1000,
                'viz_callback': None,  # Disable visualization
                'show_status': False,
                'dlogz': 2.0,  # Less stringent convergence
                'dKL': 1.0
            }
        )
        
        # Verify that nested sampling completed successfully
        assert hasattr(dipole_model, 'results')
        assert dipole_model.results is not None
        assert hasattr(dipole_model, 'samples')
        assert dipole_model.samples is not None
        assert hasattr(dipole_model, 'log_bayesian_evidence')
        
        # Check sample dimensions
        samples = dipole_model.samples
        assert samples.shape[1] == 4  # 4 parameters
        assert samples.shape[0] > 0  # Should have some samples
        
        # Check that samples are within reasonable ranges
        nbar_samples = samples[:, 0]
        dipole_amplitude_samples = samples[:, 1]
        longitude_samples = samples[:, 2]
        colatitude_samples = samples[:, 3]
        
        # Basic sanity checks on parameter ranges
        assert np.all(nbar_samples > 0)  # Mean density should be positive
        assert np.all(dipole_amplitude_samples >= 0)  # Amplitude should be non-negative
        assert (
            np.all(longitude_samples >= 0)
            and
            np.all(longitude_samples <= 2*np.pi)
        )
        assert (
            np.all(colatitude_samples >= 0)
            and
            np.all(colatitude_samples <= np.pi)
        )
        
        # Check that recovered parameters are approximately correct
        # (allowing for noise and limited sampling)
        nbar_median = np.median(nbar_samples)
        amplitude_median = np.median(dipole_amplitude_samples)
        
        # These should be roughly correct within ~50% (generous for noisy data)
        assert (
            abs(nbar_median - self.true_mean_density)
            / self.true_mean_density
         ) < 0.5
        assert (
            abs(amplitude_median - self.true_dipole_amplitude)
            / self.true_dipole_amplitude
         ) < 1.0
        
        # Verify that the model can generate predictions
        test_theta = samples[:5]  # Use first 5 samples
        predictions = dipole_model.model(test_theta)
        assert predictions.shape == (dipole_model.n_unmasked, 5)
        assert np.all(predictions > 0)  # Should be positive (rate parameters)
        
    def test_point_likelihood_integration(self):
        """
        Test full nested sampling integration with point-by-point likelihood.
        
        This test verifies that the point-by-point likelihood works correctly
        and that the model dimension is reduced appropriately.
        """
        # Generate mock data (no noise needed for point-by-point)
        mock_density_map = self.generate_mock_density_map()
        
        # Create dipole model with point-by-point likelihood
        dipole_model = Dipole(
            density_map=mock_density_map,
            likelihood='point'
        )
        
        # Verify model initialization
        assert dipole_model.likelihood == 'point'
        assert dipole_model.ndim == 3  # D, phi, theta (no Nbar)
        assert len(dipole_model.parameter_names) == 3
        assert 'Nbar' not in dipole_model.parameter_names
        
        # Run nested sampling with minimal settings
        dipole_model.run_nested_sampling(
            reactive_sampler_kwargs={
                'log_dir': 'test_ultranest_logs_point',
                'resume': False
            },
            run_kwargs={
                'min_num_live_points': 50,
                'max_num_improvement_loops': 1,
                'min_ess': 20,
                'max_iters': 1000,
                'viz_callback': None,
                'show_status': False,
                'dlogz': 2.0,
                'dKL': 1.0
            }
        )
    
        # Verify successful completion
        assert hasattr(dipole_model, 'samples')
        assert dipole_model.samples is not None
        assert dipole_model.samples.shape[1] == 3  # 3 parameters
        
        # Test model predictions
        test_theta = dipole_model.samples[:3]
        predictions = dipole_model.model(test_theta)
        assert predictions.shape == (dipole_model.n_unmasked, 3)
        assert np.all(predictions > 0)
        
    def test_custom_prior_integration(self):
        """
        Test integration with a custom prior distribution.
        
        This verifies that custom priors work correctly in the full pipeline.
        """
        # Generate mock data
        mock_density_map = self.generate_mock_density_map()
        
        # Create custom prior with tighter bounds
        custom_prior_dict = {
            'Nbar': ['Uniform', 40.0, 60.0],  # Tighter around true value
            'D': ['Uniform', 0.01, 0.05],     # Tighter around true value
            'phi': ['Uniform', 2.5, 3.5],     # Around pi
            'theta': ['Polar', 1.0, 2.0]      # Around pi/2
        }
        custom_prior = Prior(choose_prior=custom_prior_dict)
        
        # Create dipole model with custom prior
        dipole_model = Dipole(
            density_map=mock_density_map,
            prior=custom_prior,
            likelihood='poisson'
        )
        
        # Verify custom prior was set
        assert dipole_model.prior is custom_prior
        
        # Run nested sampling
        with patch('matplotlib.pyplot.show'):
            dipole_model.run_nested_sampling(
                reactive_sampler_kwargs={
                    'log_dir': 'test_ultranest_logs_custom',
                    'resume': False
                },
                run_kwargs={
                    'min_num_live_points': 50,
                    'max_num_improvement_loops': 1,
                    'min_ess': 20,
                    'max_iters': 1000,
                    'viz_callback': None,
                    'show_status': False,
                    'dlogz': 2.0,
                    'dKL': 1.0
                }
            )
        
        # Verify completion and check that samples respect custom prior bounds
        samples = dipole_model.samples
        assert samples.shape[1] == 4
        
        # Check that samples respect the custom prior bounds
        nbar_samples = samples[:, 0]
        dipole_amplitude_samples = samples[:, 1]
        longitude_samples = samples[:, 2]
        colatitude_samples = samples[:, 3]
        
        assert (
            np.all(nbar_samples >= 40.0)
            and
            np.all(nbar_samples <= 60.0)
        )
        assert (
            np.all(dipole_amplitude_samples >= 0.01)
            and
            np.all(dipole_amplitude_samples <= 0.05)
        )
        assert (
            np.all(longitude_samples >= 2.5)
            and
            np.all(longitude_samples <= 3.5)
        )
        assert (
            np.all(colatitude_samples >= 1.0)
            and
            np.all(colatitude_samples <= 2.0)
        )
        
    def test_fixed_dipole_integration(self):
        """
        Test integration with a fixed dipole component.
        
        This verifies that the fixed dipole functionality works in the full pipeline.
        """
        # Generate mock data
        mock_density_map = self.generate_mock_density_map()
        
        # Define a fixed dipole component
        fixed_dipole = (0.01, np.pi/4, np.pi/3)
        
        # Create dipole model with fixed dipole
        dipole_model = Dipole(
            density_map=mock_density_map,
            likelihood='poisson',
            fixed_dipole=fixed_dipole
        )
        
        # Verify fixed dipole was set
        assert dipole_model.fixed_dipole is not None
        assert np.allclose(dipole_model.fixed_dipole, fixed_dipole)
        
        # Run nested sampling
        dipole_model.run_nested_sampling(
            reactive_sampler_kwargs={
                'log_dir': 'test_ultranest_logs_fixed',
                'resume': False
            },
            run_kwargs={
                'min_num_live_points': 50,
                'max_num_improvement_loops': 1,
                'min_ess': 20,
                'max_iters': 1000,
                'viz_callback': None,
                'show_status': False,
                'dlogz': 2.0,
                'dKL': 1.0
            }
        )
        
        # Verify completion
        assert hasattr(dipole_model, 'samples')
        assert dipole_model.samples is not None
        
        test_theta = dipole_model.samples[:3]
        predictions = dipole_model.model(test_theta)
        assert predictions.shape == (dipole_model.n_unmasked, 3)
        assert np.all(predictions > 0)
        
    def test_masked_map_integration(self):
        """
        Test integration with a partially masked density map.
        
        This verifies that the masking functionality works correctly.
        """
        # Generate mock data
        mock_density_map = self.generate_mock_density_map()
        
        # Mask out some pixels (set to NaN)
        mask = np.random.random(self.npix) < 0.3  # Mask ~30% of pixels
        mock_density_map[mask] = np.nan
        
        # Create dipole model
        dipole_model = Dipole(
            density_map=mock_density_map,
            likelihood='poisson'
        )
        
        # Verify masking was applied correctly
        assert dipole_model.n_unmasked < self.npix
        assert np.sum(dipole_model.boolean_mask) == dipole_model.n_unmasked
        assert not np.any(np.isnan(dipole_model.density_map))
        
        # Run nested sampling
        dipole_model.run_nested_sampling(
            reactive_sampler_kwargs={
                'log_dir': 'test_ultranest_logs_masked',
                'resume': False
            },
            run_kwargs={
                'min_num_live_points': 50,
                'max_num_improvement_loops': 1,
                'min_ess': 20,
                'max_iters': 1000,
                'viz_callback': None,
                'show_status': False,
                'dlogz': 2.0,
                'dKL': 1.0
            }
        )
        
        # Verify completion
        assert hasattr(dipole_model, 'samples')
        assert dipole_model.samples is not None
        
        # Test model predictions have correct shape (only unmasked pixels)
        test_theta = dipole_model.samples[:3]
        predictions = dipole_model.model(test_theta)
        assert predictions.shape == (dipole_model.n_unmasked, 3)
        
    def test_model_attributes_after_sampling(self):
        """
        Test that all expected model attributes exist after sampling.
        
        This is a comprehensive check of the model state after nested sampling.
        """
        # Generate mock data
        mock_density_map = self.generate_mock_density_map()
        
        # Create dipole model
        dipole_model = Dipole(
            density_map=mock_density_map,
            likelihood='poisson'
        )
        
        # Run nested sampling
        dipole_model.run_nested_sampling(
            reactive_sampler_kwargs={
                'log_dir': 'test_ultranest_logs_attrs',
                'resume': False
            },
            run_kwargs={
                'min_num_live_points': 50,
                'max_num_improvement_loops': 1,
                'min_ess': 20,
                'max_iters': 1000,
                'viz_callback': None,
                'show_status': False,
                'dlogz': 2.0,
                'dKL': 1.0
            }
        )
        
        # Test all expected attributes from different mixins
        
        # From InferenceMixin
        assert hasattr(dipole_model, 'ultranest_sampler')
        assert hasattr(dipole_model, 'results')
        assert hasattr(dipole_model, 'samples')
        assert hasattr(dipole_model, 'log_bayesian_evidence')
        assert hasattr(dipole_model, 'parameter_names')
        
        # From MapModelMixin
        assert hasattr(dipole_model, 'density_map')
        assert hasattr(dipole_model, 'pixel_vectors')
        assert hasattr(dipole_model, 'nside')
        assert hasattr(dipole_model, 'npix')
        assert hasattr(dipole_model, 'boolean_mask')
        assert hasattr(dipole_model, 'n_unmasked')
        
        # From LikelihoodMixin
        assert hasattr(dipole_model, 'prior')
        assert hasattr(dipole_model, 'likelihood')
        
        # From PosteriorMixin (inherited methods should work)
        assert hasattr(dipole_model, 'model')
        
        # Test that key methods work
        assert callable(dipole_model.log_likelihood)
        assert callable(dipole_model.prior_transform)
        assert callable(dipole_model.model)
        
        # Test method outputs have correct types and shapes
        samples = dipole_model.samples
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float64
        
        test_theta = samples[:3]
        log_like = dipole_model.log_likelihood(test_theta)
        assert isinstance(log_like, np.ndarray)
        assert log_like.shape == (3,)
        
        uniform_test = np.random.rand(3, 4)
        transformed = dipole_model.prior_transform(uniform_test)
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (3, 4)


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])
