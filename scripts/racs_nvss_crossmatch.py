from dipoleutils.utils import DataLoader, CrossMatch, CatalogueToMap
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


# S_nu \propto nu^{-\alpha}
NU_LOW3_MHZ = 943.5
NU_JVSS_MHZ = 1400.0


loader = DataLoader('racs', 'low3-scaled')
racs = loader.load()
loader = DataLoader('nvss')
nvss = loader.load()

assert type(racs) is Table
assert type(nvss) is Table

racs = CatalogueToMap(racs)
nvss = CatalogueToMap(nvss)
racs.make_cut('Total_flux', 13, 1000)
nvss.make_cut('integrated_flux', 13, 1000)

xmatch_r_n = CrossMatch(
    racs.catalogue,
    nvss.catalogue,
    coordinate_system='equatorial'
)
xmatch_r_n.cross_match(radius=5, source_name_A_column='Name')
matched = xmatch_r_n.get_common_sources()
low3_flux = np.asarray(matched['A_Total_flux'], dtype=float)
nvss_flux = np.asarray(matched['B_integrated_flux'], dtype=float)
valid = (
    np.isfinite(low3_flux)
    & np.isfinite(nvss_flux)
    & (low3_flux > 0.0)
    & (nvss_flux > 0.0)
)
alpha = -np.log(low3_flux[valid] / nvss_flux[valid]) / np.log(
    NU_LOW3_MHZ / NU_JVSS_MHZ
)

plt.hist(alpha, bins=100, range=[-2, 2])
plt.xlabel(r'Spectral index, $\alpha$')
plt.ylabel('Counts')
plt.title(f'LOW3-NVSS spectral-index distribution, $N={len(alpha)}$')
plt.show()

duplicates = xmatch_r_n.get_duplicate_matches()

plt.hist(xmatch_r_n.get_crossmatch_distances(), bins=200)
plt.xlabel('Angular distance (asec)')
plt.ylabel('Counts')
plt.title(
    f'RACS -> NVSS cross-match distances, $N={xmatch_r_n.get_number_of_matches()}$'
)
plt.show()

xmatch_n_r = CrossMatch(
    nvss.catalogue,
    racs.catalogue,
    coordinate_system='equatorial'
)
xmatch_n_r.cross_match(radius=5, source_name_B_column='Name')

assert xmatch_r_n.get_number_of_matches() == xmatch_n_r.get_number_of_matches()

racs_to_nvss = xmatch_r_n.get_crossmatch_table(only_valid=True)[
    'source_name_A', 'source_name_B'
]
nvss_to_racs = xmatch_n_r.get_crossmatch_table(only_valid=True)[
    'source_name_A', 'source_name_B'
]

# sort by racs name
racs_to_nvss.sort(['source_name_A'])
nvss_to_racs.sort(['source_name_B'])

# force to be the same order
nvss_to_racs = Table(
    {
        'racs_name': nvss_to_racs['source_name_B'],
        'nvss_name': nvss_to_racs['source_name_A']
    }
)
racs_to_nvss['source_name_A'].name = 'racs_name'
racs_to_nvss['source_name_B'].name = 'nvss_name'

is_equal = racs_to_nvss == nvss_to_racs
assert is_equal.all()
