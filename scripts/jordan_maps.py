import numpy as np
import os
from dipoleutils.models.dipole import Dipole
import matplotlib.pyplot as plt


map_idxs = range(0, 20)
maps = {}
path_to_maps = os.path.expanduser('~/Downloads/maps')
quick_load = lambda fn, i: np.loadtxt(fn, delimiter=',', skiprows=1, usecols=i).astype(np.int64)

for idx in map_idxs:
    fn = f'{path_to_maps}/healpix_map_{idx}.csv'
    dmap = quick_load(fn, 1)
    pix_idxs = quick_load(fn, 0)
    mask_map = quick_load(fn, 2)

    assert np.sum(mask_map) == len(dmap), 'Some pixels are masked!'

    # in case the order is mixed up, which I doubt
    dmap_ordered = np.empty_like(dmap)
    dmap_ordered[pix_idxs] = dmap
    
    maps[idx] = {
        'density_map': dmap_ordered,
        'pix_idxs': pix_idxs,
        'mask_map': mask_map
    }

for idx in map_idxs:
    model = Dipole(maps[idx]['density_map'], likelihood='poisson')
    model.run_nested_sampling(reactive_sampler_kwargs={'resume': 'overwrite'}, run_kwargs={'min_ess': 3000})

    model.sky_direction_posterior(save_path=f'plots/map{idx}_sky.png', show=False)
    model.corner_plot(save_path=f'plots/map{idx}_corner.png', show=False)
