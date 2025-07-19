import pytest
import numpy as np
import healpy as hp
from unittest.mock import Mock, patch, MagicMock

from dipoleutils.models.dipole import Dipole
from dipoleutils.models.priors import Prior
from dipoleutils.utils.math import compute_dipole_signal


class TestDipoleUnit:
    """
    Unit tests for the Dipole class.
    
    These tests focus on individual methods and components in isolation,
    using mocks and stubs to avoid dependencies on external components.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.nside = 64
        self.npix = hp.nside2npix(self.nside)
        self.mock_density_map = np.random.poisson(50, self.npix).astype(np.float64)
        
    def test_dipole_initialization(self):
        """Test that Dipole initializes correctly with default parameters."""
        dipole = Dipole(density_map=self.mock_density_map, likelihood='point')
        
        # Check basic attributes
        assert dipole.likelihood == 'point'
        assert dipole.ndim == 3  # D, phi, theta (no Nbar for point likelihood)
        assert dipole.nside == self.nside
        assert dipole.npix == self.npix
        assert dipole.fixed_dipole is None
        
        # Check parameter names
        expected_params = ['D', 'phi', 'theta']
        assert dipole.parameter_names == expected_params
        
    def test_dipole_initialization_poisson(self):
        """Test Dipole initialization with Poisson likelihood."""
        dipole = Dipole(density_map=self.mock_density_map, likelihood='poisson')
        
        assert dipole.likelihood == 'poisson'
        assert dipole.ndim == 4  # Nbar, D, phi, theta
        expected_params = ['Nbar', 'D', 'phi', 'theta']
        assert dipole.parameter_names == expected_params
        
    def test_dipole_initialization_custom_prior(self):
        """Test Dipole initialization with custom prior."""
        custom_prior_dict = {
            'D': ['Uniform', 0.0, 0.05],
            'phi': ['Uniform', 0.0, 2*np.pi],
            'theta': ['Polar', 0.0, np.pi]
        }
        custom_prior = Prior(choose_prior=custom_prior_dict)
        
        dipole = Dipole(
            density_map=self.mock_density_map,
            prior=custom_prior,
            likelihood='point'
        )
        
        assert dipole.prior is custom_prior
        assert dipole.ndim == 3
        
    def test_dipole_initialization_fixed_dipole(self):
        """Test Dipole initialization with fixed dipole component."""
        fixed_dipole = (0.02, np.pi/4, np.pi/3)
        dipole = Dipole(
            density_map=self.mock_density_map,
            likelihood='point',
            fixed_dipole=fixed_dipole
        )
        
        assert dipole.fixed_dipole is not None
        assert np.allclose(dipole.fixed_dipole, fixed_dipole)
        
    def test_model_method_point_likelihood(self):
        """Test the model method with point likelihood."""
        dipole = Dipole(density_map=self.mock_density_map, likelihood='point')
        
        # Create test parameters: [D, phi, theta]
        test_theta = np.array([
            [0.05, np.pi, np.pi/2],  # Strong dipole pointing in x-direction
            [0.01, 0.0, np.pi/2],    # Weak dipole pointing in z-direction
        ])
        
        result = dipole.model(test_theta)
        
        # Check output shape
        assert result.shape == (dipole.n_unmasked, 2)
        
        # Check that all values are positive (since it's 1 + D*cos(theta))
        assert np.all(result > 0)
        
        # Check that values are reasonable (should be around 1 ± D)
        assert np.all(result >= 0.95)  # 1 - max(D)
        assert np.all(result <= 1.05)  # 1 + max(D)
        
    def test_model_method_poisson_likelihood(self):
        """Test the model method with Poisson likelihood."""
        dipole = Dipole(density_map=self.mock_density_map, likelihood='poisson')
        
        # Create test parameters: [Nbar, D, phi, theta]
        test_theta = np.array([
            [50.0, 0.05, np.pi, np.pi/2],
            [30.0, 0.01, 0.0, np.pi/2],
        ])
        
        result = dipole.model(test_theta)
        
        # Check output shape
        assert result.shape == (dipole.n_unmasked, 2)
        
        # Check that all values are positive
        assert np.all(result > 0)
        
        # Check that values scale with Nbar
        assert np.all(result[:, 0] > result[:, 1])  # First sample has higher Nbar
        
    def test_model_method_with_fixed_dipole(self):
        """Test the model method with fixed dipole component."""
        fixed_dipole = (0.02, np.pi/4, np.pi/3)
        dipole = Dipole(
            density_map=self.mock_density_map,
            likelihood='point',
            fixed_dipole=fixed_dipole
        )
        
        # Create test parameters: [D, phi, theta] (free dipole)
        test_theta = np.array([[0.01, np.pi, np.pi/2]])
        
        result = dipole.model(test_theta)
        
        # Check output shape
        assert result.shape == (dipole.n_unmasked, 1)
        
        # Check that result includes contribution from both dipoles
        assert np.all(result > 0)
        
    def test_log_likelihood_point(self):
        """Test log likelihood calculation with point likelihood."""
        dipole = Dipole(density_map=self.mock_density_map, likelihood='point')
        
        # Create test parameters
        test_theta = np.array([
            [0.02, np.pi, np.pi/2],
            [0.01, 0.0, np.pi/2],
        ])
        
        log_like = dipole.log_likelihood(test_theta)
        
        # Check output shape and type
        assert log_like.shape == (2,)
        assert isinstance(log_like, np.ndarray)
        
        # Log likelihood should be negative (for non-trivial data)
        assert np.all(log_like < 0)
        
    def test_log_likelihood_poisson(self):
        """Test log likelihood calculation with Poisson likelihood."""
        dipole = Dipole(density_map=self.mock_density_map, likelihood='poisson')
        
        # Create test parameters
        test_theta = np.array([
            [50.0, 0.02, np.pi, np.pi/2],
            [30.0, 0.01, 0.0, np.pi/2],
        ])
        
        log_like = dipole.log_likelihood(test_theta)
        
        # Check output shape and type
        assert log_like.shape == (2,)
        assert isinstance(log_like, np.ndarray)
        
        # Log likelihood should be negative
        assert np.all(log_like < 0)
        
    def test_prior_transform(self):
        """Test prior transform method."""
        dipole = Dipole(density_map=self.mock_density_map, likelihood='point')
        
        # Create uniform deviates
        n_samples = 100
        uniform_deviates = np.random.rand(n_samples, 3)
        
        transformed = dipole.prior_transform(uniform_deviates)
        
        # Check output shape
        assert transformed.shape == (n_samples, 3)
        
        # Check that transformed values are in reasonable ranges
        D_samples = transformed[:, 0]
        phi_samples = transformed[:, 1]
        theta_samples = transformed[:, 2]
        
        assert np.all(D_samples >= 0)  # Amplitude should be non-negative
        assert np.all(phi_samples >= 0) and np.all(phi_samples <= 2*np.pi)
        assert np.all(theta_samples >= 0) and np.all(theta_samples <= np.pi)
        
    def test_property_density_map(self):
        """Test that density_map property returns unmasked pixels."""
        # Create map with some NaN values
        density_map = self.mock_density_map.copy()
        mask_indices = np.random.choice(self.npix, size=100, replace=False)
        density_map[mask_indices] = np.nan
        
        dipole = Dipole(density_map=density_map, likelihood='point')
        
        # Check that density_map property has no NaN values
        assert not np.any(np.isnan(dipole.density_map))
        
        # Check that length matches n_unmasked
        assert len(dipole.density_map) == dipole.n_unmasked
        
    def test_property_pixel_vectors(self):
        """Test that pixel_vectors property returns unmasked vectors."""
        # Create map with some NaN values
        density_map = self.mock_density_map.copy()
        mask_indices = np.random.choice(self.npix, size=100, replace=False)
        density_map[mask_indices] = np.nan
        
        dipole = Dipole(density_map=density_map, likelihood='point')
        
        # Check shape and properties
        assert dipole.pixel_vectors.shape == (dipole.n_unmasked, 3)
        
        # Check that all vectors are unit vectors
        norms = np.linalg.norm(dipole.pixel_vectors, axis=1)
        assert np.allclose(norms, 1.0)
        
    @patch('dipoleutils.models.dipole.compute_dipole_signal')
    def test_model_calls_compute_dipole_signal(self, mock_compute_dipole):
        """Test that model method calls compute_dipole_signal with correct arguments."""
        mock_compute_dipole.return_value = np.ones((100, 2))  # Mock return value
        
        dipole = Dipole(density_map=self.mock_density_map, likelihood='point')
        test_theta = np.array([
            [0.02, np.pi, np.pi/2],
            [0.01, 0.0, np.pi/2],
        ])
        
        result = dipole.model(test_theta)
        
        # Check that compute_dipole_signal was called
        mock_compute_dipole.assert_called_once()
        
        # Check the arguments passed to compute_dipole_signal
        call_args = mock_compute_dipole.call_args
        assert np.allclose(call_args[1]['dipole_amplitude'], test_theta[:, 0])
        assert np.allclose(call_args[1]['dipole_longitude'], test_theta[:, 1])
        assert np.allclose(call_args[1]['dipole_colatitude'], test_theta[:, 2])
        
    def test_compute_dipole_signal_standalone(self):
        """Test the compute_dipole_signal function directly."""
        # Create test data
        pixel_vectors = np.random.randn(100, 3)
        # Normalize to unit vectors
        pixel_vectors = pixel_vectors / np.linalg.norm(pixel_vectors, axis=1, keepdims=True)
        
        dipole_amplitude = np.array([0.05, 0.02])
        dipole_longitude = np.array([0.0, np.pi])
        dipole_colatitude = np.array([np.pi/2, np.pi/2])
        
        result = compute_dipole_signal(
            dipole_amplitude=dipole_amplitude,
            dipole_longitude=dipole_longitude,
            dipole_colatitude=dipole_colatitude,
            pixel_vectors=pixel_vectors
        )
        
        # Check output shape
        assert result.shape == (100, 2)
        
        # Check that values are in reasonable range
        assert np.all(np.abs(result) <= np.max(dipole_amplitude))
        
    def test_error_handling_invalid_likelihood(self):
        """Test that invalid likelihood raises appropriate error."""
        with pytest.raises(Exception):
            Dipole(density_map=self.mock_density_map, likelihood='invalid') # type: ignore
            
    def test_error_handling_invalid_density_map(self):
        """Test error handling for invalid density map."""
        # Test with wrong shape
        with pytest.raises(Exception):
            Dipole(density_map=np.array([1, 2, 3]), likelihood='point')
            
        # Test with all NaN values
        invalid_map = np.full(self.npix, np.nan)
        dipole = Dipole(density_map=invalid_map, likelihood='point')
        # Should handle this gracefully but result in no unmasked pixels
        assert dipole.n_unmasked == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
