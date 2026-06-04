from dipoleutils.utils import DataLoader, CatalogueToMap
import numpy as np
from dipoleska.models.dipole import Dipole
import matplotlib.pyplot as plt
import healpy as hp


DMAP_PATH = '/Users/ooay3125/Documents/catsim/catwise_S21_probably.npy'
MASK_PATH = '/Users/ooay3125/Documents/catsim/src/catsim/data/mask/S21_CatWISE_Mask_nside64.npy'


catwise = CatalogueToMap(DataLoader('catwise', '2021').load())
catwise.make_cut('w1', minimum=None, maximum=16.4)
dmap_cat = catwise.make_density_map(coordinate_system='galactic').astype('float64')

# dmap = hp.reorder(np.load(DMAP_PATH), n2r=True)
# mask = ~np.isnan(dmap)
mask = np.load(MASK_PATH).astype('bool')
# dmap[~mask] = np.nan
dmap_cat[~mask] = np.nan

model = Dipole(dmap_cat, likelihood='general_poisson')
model.run_nested_sampling()
model.corner_plot(coordinates=['galactic'])
plt.show()

