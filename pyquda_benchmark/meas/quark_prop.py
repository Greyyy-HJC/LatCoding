# %%
import os
import gvar as gv
from pyquda import init
from pyquda_utils import core, io, source, gamma
from pyquda_utils.phase import MomentumPhase
from opt_einsum import contract

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *
from lametlat.preprocess.read_raw import pt2_to_meff


if not os.path.exists(".cache"):
    os.makedirs(".cache")
    print("Created .cache directory for PyQUDA resources")

init([1, 1, 1, 1], resource_path=".cache")
N_conf = 50  # Number of configurations to process

# Lattice parameters
xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None # [[4, 4, 4, 4], [2, 2, 2, 8]]

latt_info = core.LatticeInfo([8, 8, 8, 32], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)

# Get gamma5 matrix
I = gamma.gamma(0)

# Lists to store correlation functions
wall_quark_corr = []
point_quark_corr = []

for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    gauge = io.readNERSCGauge(f"../../conf/S8T32_cg/gauge/wilson_b6.cg.1e-08.{cfg}")
    
    # Apply smearing to gauge field
    # gauge.stoutSmear(1, 0.125, 4)
    dirac.loadGauge(gauge)
    
    # Point source propagator
    point_source = source.propagator(latt_info, "point", [0, 0, 0, 0])
    point_propag = core.invertPropagator(dirac, point_source)
    
    # Contract to get correlation function
    point_quark_corr.append(
        core.gatherLattice(
            core.lexico(contract("wtzyxijaa,ji->wtzyx", point_propag.data, I).real.get(), [0,1,2,3,4]), 
            [0, 1, 2, 3]
        )[0, 0, :, 0]
    )

# Clean up resources
dirac.destroy()

print("\n>>> shape of point_quark_corr: ", point_quark_corr[0].shape)

# Print first few entries of the correlation functions
print("Point source, conf 0: ", point_quark_corr[0][:6])

# %%
print("shape of point_quark_corr: ", np.shape(point_quark_corr))

point_quark_corr_jk = jackknife(point_quark_corr)
point_quark_corr_jk_avg = jk_ls_avg(point_quark_corr_jk)
point_meff = pt2_to_meff(point_quark_corr_jk_avg, boundary="none")

fig, ax = default_plot()
ax.errorbar(np.arange(len(point_meff)), gv.mean(point_meff), yerr=gv.sdev(point_meff), label="point", **errorb)
ax.legend(ncol=2, **fs_small_p)
ax.set_xlabel(r"$t_{\mathrm{sep}}$", **fs_p)
ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
plt.tight_layout()
plt.show()
# %%


