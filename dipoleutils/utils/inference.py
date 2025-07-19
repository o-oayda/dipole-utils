import logging
import ultranest
import sys
from abc import abstractmethod
from numpy.typing import NDArray
import ultranest.stepsampler
import numpy as np

class InferenceMixin:
    '''
    Provides methods for running Nested Sampling algorithms.
    Inherited by each model, which need to certain properties and methods.
    '''
    def __init__(self):
        self._disable_ultranest_logging()

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        raise NotImplementedError(
            'Subclass models must define parameter names property.'
        )

    @abstractmethod
    def log_likelihood(self,
            Theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        raise NotImplementedError(
            'Subclass models must define log_likelihood method.'
        )

    @abstractmethod
    def prior_transform(self,
            uniform_deviates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        raise NotImplementedError(
            'Subclass models must define prior_transform method.'
        )

    def run_nested_sampling(self,
            step: bool = False,
            n_steps: int | None = None,
            reactive_sampler_kwargs: dict = {},
            run_kwargs: dict = {}
        ) -> None:
        '''
        Perform Nested Sampling using ultranest's `ReactiveNestedSampler`.
        Results are saved in the results attribute.

        :param step: Specify whether or not to use the random step method.
        :param n_steps: If the random step method is specified, this is the
            number of steps as used by `SliceSampler`.
        :param reactive_sampler_kwargs: Keyword arguments for the
            `ReactiveNestedSampler`.
        :param run_kwargs: Keyword arguments for the sampler's run method.
        '''
        self.ultranest_sampler = ultranest.ReactiveNestedSampler(
            param_names=self.parameter_names,
            loglike=self.log_likelihood,
            transform=self.prior_transform,
            **{
                'log_dir': 'ultranest_logs',
                'resume': 'subfolder',
                'vectorized': True,
                **reactive_sampler_kwargs
            }
        )

        if step:
            self._switch_to_step_sampling(n_steps)
        
        self.results = self.ultranest_sampler.run(**run_kwargs)
        self.ultranest_sampler.print_results()

        # there is an issue with ultranest plotting when the log likelihood is
        # very negative (e.g. for the point-by-point likelihood)
        # this catches the ValueError raised
        try:
            self.ultranest_sampler.plot()
        except ValueError as e:
            print(e)
        
        if self.results is not None:
            self._samples = self.results['samples']
            self.log_bayesian_evidence = self.results['logz']
        else:
            raise Exception('Ultranest results are undefined.')

    @property
    def samples(self) -> NDArray[np.float64]:
        return self._samples # type: ignore

    def _switch_to_step_sampling(self, n_steps: int | None = None) -> None:
        if n_steps is None:
            n_steps = 2 * len(self.parameter_names)
        
        self.ultranest_sampler.stepsampler = ultranest.stepsampler.SliceSampler( # type: ignore
            nsteps=n_steps,
            generate_direction=(
                ultranest.stepsampler.generate_mixture_random_direction
            )
        )

    def _disable_ultranest_logging(self) -> None:
        '''
        Ultranest spams debug messages by default.
        Running this function should disable them.
        Taken from https://github.com/JohannesBuchner/UltraNest/issues/31.
        '''
        unest_logger = logging.getLogger("ultranest")
        unest_handler = logging.StreamHandler(sys.stdout)
        unest_handler.setLevel(logging.WARN)
        ultranest_formatter = logging.Formatter(
            "%(levelname)s:{}:%(message)s".format("ultranest")
        )
        unest_handler.setFormatter(ultranest_formatter)
        unest_logger.addHandler(unest_handler)
        unest_logger.setLevel(logging.WARN)