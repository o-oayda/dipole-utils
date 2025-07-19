import numpy as np

prior_defaults = {
    'dipole': {
        'Nbar': ['Uniform', 0., 100.],
        'D': ['Uniform', 0., 0.1],
        'phi': ['Uniform', 0., 2*np.pi],
        'theta': ['Polar', 0., np.pi]
    }
}