from dipoleutils.utils import DataLoader, CatalogueToMap
from dipoleutils.utils.plotting import plot_log_log_histogram
import numpy as np
from dipoleska.models.dipole import Dipole
import matplotlib.pyplot as plt
import healpy as hp


# DMAP_PATH = '/Users/ooay3125/Documents/catsim/catwise_S21_probably.npy'
# MASK_PATH = '/Users/ooay3125/Documents/catsim/src/catsim/data/mask/S21_CatWISE_Mask_nside64.npy'
SNR_MIN = 3


catwise = CatalogueToMap(DataLoader('catwise', '2021').load())
catwise.make_cut('w1', minimum=None, maximum=16.4)

cat = catwise.get_catalogue()

# flat-plane approximation
cat['pm'] = np.hypot(cat['pmra'], cat['pmdec']) # asec / yr
cat['pm'] *= 1000 # mas / yr
cat['sigpm'] = np.hypot(cat['sigpmra'], cat['sigpmdec']) # asec / yr
cat['sigpm'] *= 1000 # mas / yr
cat['pmsnr'] = cat['pm'] / cat['sigpm'] # sigma

# snr_cut
cut = cat['pmsnr'] > SNR_MIN
highpm_sources = cat[cut]

highpm = CatalogueToMap(highpm_sources).make_density_map(coordinate_system='galactic')
model = Dipole(highpm, likelihood='point')
model.prior.change_prior(0, ['Uniform', 0., 0.4])
model.run_nested_sampling()
model.corner_plot(coordinates=['galactic'])



# dmap_cat = catwise.make_density_map(coordinate_system='galactic').astype('float64')
#
# # dmap = hp.reorder(np.load(DMAP_PATH), n2r=True)
# # mask = ~np.isnan(dmap)
# mask = np.load(MASK_PATH).astype('bool')
# # dmap[~mask] = np.nan
# dmap_cat[~mask] = np.nan



# model = Dipole(dmap_cat, likelihood='general_poisson')
# model.run_nested_sampling()
# model.corner_plot(coordinates=['galactic'])
# plt.show()
#
