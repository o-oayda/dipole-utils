from numpy.typing import NDArray
import numpy as np
from collections import defaultdict
from dipoleutils.utils.inference import InferenceMixin
from dipoleutils.utils.math import (
    compute_dipole_signal, multipole_pixel_product_vectorised,
    multipole_tensor_vectorised, vectorised_quadrupole_tensor,
    vectorised_spherical_to_cartesian
)
from dipoleutils.utils.posterior import PosteriorMixin
from dipoleutils.models.model_helpers import LikelihoodMixin, MapModelMixin
from dipoleutils.models.priors import Prior

class Multipole(LikelihoodMixin, InferenceMixin, MapModelMixin, PosteriorMixin):
    def __init__(self,
            density_map: NDArray[np.int_ | np.float64],
            ells: list[int],
            prior: Prior | None = None, 
    ):
        '''
        Fit an abitrary number of monopoles of different orders.

        :param density_map:
            Healpy source density map, of shape (n_pix,).
        :param prior:
            Pass either an instance of a Prior object or leave as None.

            If None is specified, the model dynamically creates a prior, taking
            into account the number of amplitudes and multipole unit vectors
            needed. Additionally, if a monopole is specified in the ells kwarg,
            the Poissonian likelihood is used and the prior on the monopole
            amplitude is automatically updated to a uniform distribuion 25%
            either side of the mean of the density map.

            NOTE: When passing a custom prior, the dictionary must obey a
            certain order and be named a certain way. The order should keep
            amplitudes first, then unit vectors second. For example, if one
            fits `ell = [0, 1, 2]`, the order of keys passed to Prior at
            instantiation should be
            `{'M0', 'M1', 'M2', 'phi_l1_0', 'theta_l1_0', 'phi_l2_0',
            'theta_l2_0', 'phi_l2_1', 'theta_l1_1'}`.
            Note that the amplitudes are first, with `M0` referring to the
            monopole amplitude, etc. Also, for each multipole unit vector,
            we have phi_lX_Y. X should be the order of the multipole for which
            this unit vector applies; l1 would refer to the single dipole
            vector. Y refers to the vector number; a quadrupole with two
            multipole unit vectors would have two vectors, indexed by Y=0 and
            Y=1.
        :param ells:
            Pass a list of multipole orders to fit, e.g. `ells = [1, 2, 3]`.
            If a monopole (0) is specified in the list, the Poissonian
            likelihood is used; else, the point-by-point likelihood is used.
        
        TODO: performance is still slower than dipole-stats implementation;
            determine why this is.
            dipole-stats: 116s
            dipole-ska: 158s
        '''
        self._get_healpy_map_attributes(density_map)
        self._construct_multipole_priors(ells, prior)
        
        # if we are fitting a monopole (ell=0), adjust the monopole prior to center
        # around the mean number density; otherwise, for point-by-point, the
        # prior will automatically lack a monopole prior, so no change needed
        if any(ell == 0 for ell in ells):
            self.monopole_is_fitted = True
            self._parse_likelihood_choice('poisson')
        else:
            self.monopole_is_fitted = False

        self._parameter_names = self.prior.parameter_names
        self.ndim = self.prior.ndim
        self.n_multipoles = len(ells)
        self._get_angle_indices()
    
    def _construct_multipole_priors(self,
            ells: list[int],
            prior: Prior | None = None
    ) -> None:
        if prior is None:
            azimuthal_priors = ['Uniform', 0., 2 * np.pi]
            polar_priors = ['Polar', 0., np.pi]
            amplitude_priors = ['Uniform', 0., 0.1]
            monopole_priors = ['Uniform', 0., 0.] # placeholder: is replaced later
            all_amplitude_priors = {}
            all_angle_priors = {}
            
            for ell in ells:    
                if ell == 0:
                    label = 'M0'
                    all_amplitude_priors[label] = monopole_priors
                
                else:
                    amplitude_label = f'M{ell}'
                    prior_list = amplitude_priors.copy()
                    prior_list[-1] = prior_list[-1] * ell ** 2 # naively scale prior range by ell^2
                    all_amplitude_priors[amplitude_label] = prior_list

                    # for each ell with amplitude M_ell, add the 2*ell unit vector priors
                    for i in range(2 * ell):
                        subscript = f'l{ell}_{i//2}'

                        if i % 2 == 0:
                            all_angle_priors[f'phi_{subscript}'] = azimuthal_priors
                        else:
                            all_angle_priors[f'theta_{subscript}'] = polar_priors
            
            all_priors = {**all_amplitude_priors, **all_angle_priors}
            self._prior = Prior(choose_prior=all_priors)
        else:
            self._prior = prior

    def _get_angle_indices(self):
        self.phi_indices = defaultdict(list)
        self.theta_indices = defaultdict(list)
        
        for i, key in enumerate(self.parameter_names):
            if 'M' in key:
                continue
            else:
                angle_type, ell, vec_number = key.split('_') # e.g. phi_l2_1

                new_key = f'{ell[1:]}'

                if angle_type == 'phi':
                    self.phi_indices[new_key].append(i)
                else:
                    self.theta_indices[new_key].append(i)

    def _parse_multipole_likelihood(self, ells: list[int]) -> None:
        pass

    @property
    def density_map(self) -> NDArray[np.int_ | np.float64]:
        '''
        Whenever the model calls the `density_map` attribute, provide only the
        unmasked pixels for inference.
        '''
        return self._density_map_masked
    
    @property
    def pixel_vectors(self) -> NDArray[np.float64]:
        '''
        Whenever the model calls the `pixel_vectors` attribute, provide only
        the vectors pointing to unmasked pixels.
        '''
        return self._pixel_vectors_masked
    
    @property
    def pixel_vectors_xyz(self) -> list[NDArray[np.float64]]:
        return self._pixel_vectors_xyz_masked

    @property
    def prior(self) -> Prior:
        return self._prior
    
    @property
    def parameter_names(self) -> list[str]:
        return self._parameter_names
    
    def log_likelihood(self,
            Theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        '''
        Log likelihood function passed to the Nested Sampler. Automatically
        adjusted depending on whether or not the user is fitting a monopole by
        passing ell = [0, ...] at instantiation.

        :param Theta: Parameter samples from the prior distribtuion, as
            generated by the Nested Sampler.
        '''
        multipole_terms = self.model(Theta)

        if self.monopole_is_fitted:
            return self.poisson_log_likelihood(
                rate_parameter=multipole_terms,
                density_map=self.density_map
            )
        else:
            return self.point_by_point_log_likelihood(
                multipole_signal=multipole_terms,
                density_map=self.density_map
            )
    
    def model(self, Theta: NDArray[np.float64]) -> NDArray[np.float64]:
        '''
        The essential idea of to iterate over each ell, computing the multipole
        signal for that particular order. For example, if a user specifies
        `ells = [1,2,3]`, we compute the dipole signal, then the quadrupole
        signal, then the octupole signal, summing them cumulatively.
        '''
        nlive = Theta.shape[0]
        if self.monopole_is_fitted: # exclude first params, which will be Nbar
            amplitude_like = Theta[:, 1:self.n_multipoles]
        else:
            amplitude_like = Theta[:, :self.n_multipoles]
        
        signal = np.ones((self.n_unmasked, nlive))
        for i, ((ell, phi_idxs), (ell, theta_idxs)) in enumerate(
            zip(self.phi_indices.items(), self.theta_indices.items())
        ):
            signal += self.compute_multipole_signal(
                multipole_amplitudes=amplitude_like[:, i],
                multipole_longitudes=Theta[:, phi_idxs],
                multipole_latitudes=Theta[:, theta_idxs]
            )
        
        if self.monopole_is_fitted:
            mean_number_density = Theta[:, 0]
            return mean_number_density[None, :] * np.ones(signal.shape) * signal
        else:
            return signal
    
    def compute_multipole_signal(self,
            multipole_amplitudes: NDArray[np.float64],
            multipole_longitudes: NDArray[np.float64],
            multipole_latitudes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        '''
        :param multipole_amplitudes: Vector of multipole amplitudes, shape (n_live,).
            For example, for an octupole, the vector would be the n_live samples of
            the octupole amplitude.
        :param multipole_longitudes: Matrix of multipole azimuthal angles,
            shape (n_live, ell). For example, for an octupole, the matrix would have
            three azimuthal (phi) angles for the three octupole unit vectors.
        :param multipole_latitudes: Matrix of multipole polar anglea,
            shape (n_live, ell). For example, for an octupole, the matrix would have
            three polar (theta) angles for the three octupole unit vectors.
        :param pixel_vectors: List of Cartesian coordinates of pixel vectors (of
            form [X, Y, X]). X, Y, and Z are vectors of shape (n_pixels,). 
        :return: Vectorised multipole signal of shape (n_pixels, n_live). For example,
            for an octupole, the output O_{ijk} p_{i} p_{j} p_{k} is determined,
            which is the inner product of the octupole tensor and pixel unit
            vectors.
        '''
        ell = multipole_longitudes.shape[1]

        if ell == 1:
            dipole_signal = compute_dipole_signal(
                dipole_amplitude=multipole_amplitudes,
                dipole_longitude=multipole_longitudes.squeeze(), # remove length 1 axes
                dipole_colatitude=multipole_latitudes.squeeze(),
                pixel_vectors=self.pixel_vectors # reshape to (n_pix, 3)
            )
            return dipole_signal

        elif ell == 2:
            cartesian_quadrupole_vectors = vectorised_spherical_to_cartesian(
                phi_like=multipole_longitudes,
                theta_like=multipole_latitudes
            )
            quadrupole_tensor = vectorised_quadrupole_tensor(
                amplitude_like=multipole_amplitudes,
                cartesian_quadrupole_vectors=cartesian_quadrupole_vectors
            )
            quadrupole_signal = multipole_pixel_product_vectorised(
                multipole_tensors=quadrupole_tensor,
                pixel_vectors=self.pixel_vectors_xyz,
                ell=2
            )
            return quadrupole_signal

        else:
            cartesian_multipole_vectors = vectorised_spherical_to_cartesian(
                phi_like=multipole_longitudes,
                theta_like=multipole_latitudes
            )
            multipole_tensor = multipole_tensor_vectorised(
                amplitude_like=multipole_amplitudes,
                cartesian_multipole_vectors=cartesian_multipole_vectors
            )
            multipole_signal = multipole_pixel_product_vectorised(
                multipole_tensors=multipole_tensor,
                pixel_vectors=self.pixel_vectors_xyz,
                ell=multipole_longitudes.shape[-1]
            )
            return multipole_signal