from dipoleutils.utils.crossmatch import CrossMatch
from dipoleutils.utils.data_loader import DataLoader
from dipoleutils.utils.samples import CatalogueToMap


RLOW_FREQ = 887.5e6
RMID_FREQ = 1367.5e6

data = DataLoader('racs', 'mid1').load()
rmid = CatalogueToMap(data)
data = DataLoader('racs', 'low1').load()
rlow = CatalogueToMap(data)

rmid.make_cut('total_flux', 15, 1000)
rlow.make_cut('total_flux_source', 15, 1000)

# match mid (A) to low (B)
xmatch = CrossMatch(
    rmid.catalogue,
    rlow.catalogue,
    coordinate_system='equatorial'
)
xmatch.cross_match(radius=5, source_name_A_column='name')
matches = xmatch.get_crossmatch_table()
where_match_exists = ~(matches['source_name_B'] == None)
rlow_match_names = matches['source_name_B'][where_match_exists]
rmid_match_names = matches['source_name_A'][where_match_exists]

# painfully slow
rmid.catalogue.add_index('name')
rlow.catalogue.add_index('source_name')
rlow_flux = rlow.catalogue.loc[rlow_match_names]['total_flux_source']
rmid_flux = rmid.catalogue.loc[rmid_match_names]['total_flux']

# S_nu ~ nu ** (-alpha)
