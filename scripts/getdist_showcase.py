from getdist import plots, MCSamples
from dipoleutils.utils.samples import SimulatedDipoleMap
import matplotlib.pyplot as plt
from dipoleutils.models.dipole import Dipole
from dipoleutils.utils.constants import CMB_L, CMB_B, CMB_PHI_GAL, CMB_THETA_GAL

true_amplitude = 0.007
true_longitude = CMB_PHI_GAL
true_colatitude = CMB_THETA_GAL

dmap_sim = SimulatedDipoleMap(nside=64)
dmap = dmap_sim.make_map(
    dipole_amplitude=true_amplitude,
    dipole_longitude=CMB_L,
    dipole_latitude=CMB_B
)

model = Dipole(dmap)
model.run_nested_sampling(run_kwargs={'min_ess': 1_000}) # get more posterior samples (min_ess)

samps = MCSamples(samples=model.samples, names=['D', 'phi', 'theta'], labels=['D', '\\phi', '\\theta']) 
samps.updateSettings({
    # 'fine_bins_2D': 64,
    'smooth_scale_2D': 2.0 # get around WARNING:root:fine_bins_2D issue
})
g = plots.get_subplot_plotter()
g.triangle_plot(
    samps,
    filled=True,
    markers=[true_amplitude, CMB_PHI_GAL, CMB_THETA_GAL]
)
plt.show()
