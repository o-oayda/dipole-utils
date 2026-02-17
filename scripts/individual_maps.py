from dipoleska.models.priors import Prior
import numpy as np
import argparse
from dipoleska.models.dipole import Dipole
import matplotlib.pyplot as plt


sample_to_file = {
    'racs': 'racsb_dmap.npy',
    'nvss': 'nvssb_dmap.npy',
    'planck': 'planck_galactic_counts.npy'
}

if __name__ == '__main__':
    aprser = argparse.ArgumentParser()
    aprser.add_argument(
        '--sample',
        choices=['racs', 'nvss', 'planck']
    )
    args = aprser.parse_args()
    SAMPLE = args.sample
    FILE = sample_to_file[SAMPLE]

    dmap = np.load(f'/home/oliver/Documents/sbi/maps/{FILE}')
    model = Dipole(dmap, likelihood='poisson')
    model.prior.change_prior(0, ['Uniform', 0.9 * np.nanmean(dmap), 1.1 * np.nanmean(dmap)])
    model.prior.change_prior(1, ['Uniform', 0., 4.27e-3 * 20])
    model.prior.plot_priors()
    model.run_nested_sampling()
    model.corner_plot(coordinates=['equatorial', 'galactic'])
    model.sky_direction_posterior(coordinates=['equatorial', 'galactic'])
    plt.show()
