import healpy as hp
import numpy as np
from numpy.typing import NDArray
from typing import Literal
from scipy.stats import poisson
from abc import abstractmethod
from dipoleutils.models.priors import Prior

class LikelihoodMixin:
    @property
    @abstractmethod
    def prior(self) -> Prior:
        raise NotImplementedError('Subclass models must implement a prior property.')

    def point_by_point_log_likelihood(self,
            multipole_signal: NDArray[np.float64],
            density_map: NDArray[np.int_ | np.float64]
    ) -> NDArray[np.float64]:
        '''
        Compute the vectorised log likelihood for many dipole maps using
        the point-by-point function.

        :param multipole_signal: Array of dipole signals, defined as
            f_dipole = 1 + D cos ( theta ), or potentially multipole signals.
            The shape of the array should be (n_pixels, n_live), where
            n_live is the number of live points used in ultranest's
            vetcorised function call and n_pixels is the number of pixels
            in the healpy map.
        :param density_map: Healpy density map of shape (n_pixels,).
        :return: Log likelihood corresponding to each dipole signal of shape
            (n_live,).
        '''
        normalisation_factor = np.sum(multipole_signal, axis=0)
        likelihood_map = multipole_signal / normalisation_factor
        log_likelihood = np.einsum(
            'i,ij->j',
            density_map,
            np.log(likelihood_map)
        )
        return log_likelihood

    def poisson_log_likelihood(self,
            rate_parameter: NDArray[np.float64],
            density_map: NDArray[np.int_ | np.float64]
    ) -> NDArray[np.float64]:
        '''
        Compute the vectorised log likelihood for many dipole maps using the
        Poisson likelihood function.

        :param rate_parameter: Array of rate parameters for each cell; of
            shape (n_pixels, n_live).
        :param density_map: Healpy density map of shape (n_pixels,).
        :return: Log likelihood corresponding to each dipole signal of shape
            (n_live,).
        '''
        log_likelihood_map = poisson.logpmf(
            k=density_map[:, None],
            mu=rate_parameter
        )
        return np.sum(log_likelihood_map, axis=0)

    def prior_transform(self,
            uniform_deviates: NDArray[np.float64]
        ) -> NDArray[np.float64]:
        '''
        Meta function passed to the Nested Sampler's prior input; calls
        `Prior` object's `transform` method to turn deviates on the unit cube
        to deviates in prior space.
        '''
        return self.prior.transform(uniform_deviates)


class MapModelMixin:
    @property
    @abstractmethod
    def density_map(self) -> NDArray[np.float64 | np.int_]:
        raise NotImplementedError('Subclass models must define a density map.')

    def _get_healpy_map_attributes(self,
            density_map: NDArray[np.int_ | np.float64]
    ) -> None:
        '''
        Retrieve key properties of a healpix map.
        '''
        self.mean_density = np.nanmean(density_map)
        self.nside = hp.get_nside(density_map)
        self.npix = hp.nside2npix(self.nside)
        pixels_x, pixels_y, pixels_z = hp.pix2vec(
            self.nside,
            np.arange(self.npix)
        )
        self._pixel_vectors = np.stack([pixels_x, pixels_y, pixels_z]).T # (n_pix, 3)
        self._density_map = density_map

        # masked attributes
        self.boolean_mask = ~np.isnan(density_map)
        self.n_unmasked = np.sum(self.boolean_mask, dtype=np.int64)
        self._density_map_masked = self._density_map[self.boolean_mask]
        self._pixel_vectors_masked = self._pixel_vectors[self.boolean_mask]
        x, y, z = self._pixel_vectors[self.boolean_mask].T
        self._pixel_vectors_xyz_masked = [x, y, z]

    def _parse_prior_choice(self,
            default_prior: str,
            prior: Prior | None = None
        ) -> None:
        '''
        Switch to a default prior if the user has not specified one, or use
        the explicit one the user has provided.
        '''
        if prior is None:
            self._prior = Prior(choose_prior=default_prior)
            self.prior_is_custom = False
        else:
            self._prior = prior
            self.prior_is_custom = True

    def _parse_likelihood_choice(self,
            likelihood: Literal['point', 'poisson']
        ) -> None:
        '''
        If one specifies the point-by-point likelihood, we don't need to fit
        for the mean density. This function removes that parameter from the
        list of priors and parameter names, reducing the dimension of the model
        by 1.

        In addition, if one chooses the Poisson likelihood, we ideally
        want the choice of mean density prior to center around the mean density
        itself. This automatically makes that change without needing explicit
        input from the user.
        '''
        self.likelihood = likelihood

        if not self.prior_is_custom:
            if self.likelihood == 'point':
                # remove Nbar parameter from the default dipole priors
                self._prior.remove_prior(prior_index=0)
            elif self.likelihood == 'poisson':
                self._prior.change_prior(
                    prior_index=0,
                    new_prior=[
                        'Uniform',
                        0.75 * self.mean_density,
                        1.25 * self.mean_density
                    ]
                )
            else:
                raise Exception(
                    f'Likelihood choice ({self.likelihood}) not recognised.'
                )
        else:
            # check that custom priors have the expected number of dimensions
            if self.likelihood == 'point':
                assert self._prior.ndim == 3
            elif self.likelihood == 'poisson':
                assert self._prior.ndim == 4
            else:
                raise Exception(
                    f'Likelihood choice ({self.likelihood}) not recognised.'
                )