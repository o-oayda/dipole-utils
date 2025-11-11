from pathlib import Path

import corner
import healpy as hp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import time

from dipoleutils.models.multipole import Multipole
from dipoleutils.utils.samples import SimulatedMultipoleMap


ARTIFACT_DIR = Path(__file__).resolve().parent / 'artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _random_direction(rng: np.random.Generator) -> tuple[float, float]:
    phi = rng.uniform(0.0, 2 * np.pi)
    cos_theta = rng.uniform(-1.0, 1.0)
    theta = np.arccos(cos_theta)
    return phi, theta


class TestMultipoleIntegration:
    @pytest.mark.slow
    def test_nested_sampling_recovers_truth(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2025)
        nside = 16
        mean_density = 500.0  # ~1.5M total sources across the sky
        dipole_amplitude = 0.007
        quadrupole_amplitude = 0.014

        dip_phi, dip_theta = _random_direction(rng)
        quad_phi_0, quad_theta_0 = _random_direction(rng)
        quad_phi_1, quad_theta_1 = _random_direction(rng)

        parameters = {
            'M0': mean_density,
            'M1': dipole_amplitude,
            'M2': quadrupole_amplitude,
            'phi_l1_0': dip_phi,
            'theta_l1_0': dip_theta,
            'phi_l2_0': quad_phi_0,
            'theta_l2_0': quad_theta_0,
            'phi_l2_1': quad_phi_1,
            'theta_l2_1': quad_theta_1,
        }

        simulator = SimulatedMultipoleMap(nside=nside, ells=[1, 2])
        poisson_seed = 141
        density_map = simulator.make_map(parameters=parameters, poisson_seed=poisson_seed)
        map_prefix = ARTIFACT_DIR / f'fiducial_map_nside{nside}'
        map_path, metadata_path = simulator.save_simulation(
            density_map=density_map,
            parameters=parameters,
            output_prefix=map_prefix,
            poisson_seed=poisson_seed
        )
        assert map_path.exists()
        assert metadata_path.exists()
        assert np.sum(density_map) >= 1_500_000
        assert np.isclose(np.sum(density_map), mean_density * hp.nside2npix(nside), rtol=0.2)

        model = Multipole(density_map=density_map, ells=[0, 1, 2])
        log_dir = tmp_path / 'ultranest_logs'
        corner_path = ARTIFACT_DIR / 'multipole_corner.png'

        start = time.perf_counter()
        model.run_nested_sampling(
            step=True,
            reactive_sampler_kwargs={
                'log_dir': str(log_dir),
                'resume': 'overwrite'
            },
            run_kwargs={
                'min_num_live_points': 200,
                'min_ess': 200,
                'show_status': True
            }
        )

        samples = model.samples
        parameter_names = model.parameter_names
        assert samples.shape[1] == len(parameter_names)

        name_to_index = {name: idx for idx, name in enumerate(parameter_names)}
        truths = {
            'M0': mean_density,
            'M1': dipole_amplitude,
            'M2': quadrupole_amplitude,
            'phi_l1_0': dip_phi,
            'theta_l1_0': dip_theta,
            'phi_l2_0': quad_phi_0,
            'theta_l2_0': quad_theta_0,
            'phi_l2_1': quad_phi_1,
            'theta_l2_1': quad_theta_1,
        }

        def assert_within_three_sigma(param_name: str) -> None:
            idx = name_to_index[param_name]
            data = samples[:, idx]
            sigma = np.std(data, ddof=0)
            assert sigma > 0
            delta = np.abs(np.mean(data) - truths[param_name])
            assert delta <= 3 * sigma

        assert_within_three_sigma('M1')
        assert_within_three_sigma('M2')

        def angular_delta(values: np.ndarray, truth: float) -> float:
            deltas = (values - truth + np.pi) % (2 * np.pi) - np.pi
            sigma = np.std(deltas, ddof=0)
            assert sigma > 0
            mean_delta = np.abs(np.mean(deltas))
            assert mean_delta <= 3 * sigma
            return sigma

        angular_delta(samples[:, name_to_index['phi_l1_0']], truths['phi_l1_0'])
        angular_delta(samples[:, name_to_index['theta_l1_0']], truths['theta_l1_0'])

        truth_vector = [truths.get(name, np.nan) for name in parameter_names]
        figure = corner.corner(
            samples,
            labels=parameter_names,
            truths=truth_vector,
            show_titles=True
        )
        figure.savefig(corner_path, dpi=200)
        plt.close(figure)
        assert corner_path.exists()
        duration = time.perf_counter() - start
        print(f"Multipole integration completed in {duration:.1f}s")
