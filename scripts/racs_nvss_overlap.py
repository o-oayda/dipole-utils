# %%
from dipoleutils.utils import DataLoader, CrossMatch, CatalogueToMap
import matplotlib.pyplot as plt
from astropy.table import Table
import healpy as hp
import numpy as np
# %%
loader = DataLoader('racs', 'low1')
racs = loader.load()
loader = DataLoader('nvss')
nvss = loader.load()

assert type(racs) is Table
assert type(nvss) is Table

racs = CatalogueToMap(racs)
nvss = CatalogueToMap(nvss)
racs.add_coordinate_system(starting_frame='equatorial', target_frame='galactic')
nvss.add_coordinate_system(starting_frame='equatorial', target_frame='galactic')

racs.make_cut('total_flux_source', 15, 1000)
nvss.make_cut('integrated_flux', 15, 1000)
# mask a bit extra since max racs dex is 29.127 in flux range
MIN_DEC = -40; MAX_DEC = 29
GMASK_B = 5
racs.make_cut('dec', minimum=MIN_DEC, maximum=MAX_DEC)
nvss.make_cut('dec', minimum=MIN_DEC, maximum=MAX_DEC)

nvss.make_cut('b', minimum=-GMASK_B, maximum=GMASK_B, cut_outside=False)

racs_dmap = racs.make_density_map()
nvss_dmap = nvss.make_density_map()

hp.projview(racs_dmap, title='RACS Density Map', sub=211, graticule=True, graticule_labels=True)
hp.projview(nvss_dmap, title='NVSS Density Map', sub=212, graticule=True, graticule_labels=True)

plt.tight_layout()
plt.show()
# %%
# now do the cross matching
xmatch = CrossMatch(
    racs.get_catalogue(),
    nvss.get_catalogue(),
    coordinate_system='equatorial'
)
xmatch.cross_match(radius=5)

print("\nMask parameters:")
print(f"Declination (δ): {MIN_DEC}° ≤ δ ≤ {MAX_DEC}°")
print(f"Galactic latitude (b): |b| ≤ {GMASK_B}°")
print(f"\nTotal RACS sources: {racs.get_source_count()}")
print(f'Number of matches: {xmatch.get_number_of_matches()}')
print(
    f'% of RACS sources with counterpart NVSS: '
    f'{(xmatch.get_number_of_matches() / racs.get_source_count())*100:.2f}'
)