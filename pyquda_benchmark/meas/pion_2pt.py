# %%
import os
import gvar as gv
from tqdm.auto import tqdm
from pyquda import init
from pyquda_utils import core, io, source
from pyquda_utils.phase import MomentumPhase
from opt_einsum import contract

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *
from lametlat.preprocess.read_raw import pt2_to_meff

if not os.path.exists(".cache"):
    os.makedirs(".cache")
    print("Created .cache directory for PyQUDA resources")

init([1, 1, 1, 1], resource_path=".cache")
N_conf = 20

xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None 

latt_info = core.LatticeInfo([8, 8, 8, 32], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)

wall_pion = []
point_pion = []
for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    gauge = io.readNERSCGauge(f"../../conf/S8T32/wilson_b6.{cfg}")

    # gauge.stoutSmear(1, 0.125, 4)
    dirac.loadGauge(gauge)
    
    # * add momentum phase to wall source
    mom_phase = MomentumPhase(latt_info).getPhase([0, 0, 0])
    wall_source = source.propagator(latt_info, "wall", 0, mom_phase)

    wall_propag = core.invertPropagator(dirac, wall_source)

    # wtzyxjiba are indices of the propagator, ->t means contract all indices except t
    # indices are: even/odd, t, z, y, x, spin, spin, color, color
    # [0, -1, -1, -1] means keep the t direction and sum over the other directions, 1 means gather the data, 0 means no action, -1 means sum / average
    wall_pion.append(
        core.gatherLattice(
            contract("wtzyxjiba,wtzyxjiba->t", wall_propag.data.conj(), wall_propag.data).real.get(), [0, -1, -1, -1]
        )
    )

    point_source = source.propagator(latt_info, "point", [0, 0, 0, 0])
    point_propag = core.invertPropagator(dirac, point_source)

    point_pion.append(
        core.gatherLattice(
            contract("wtzyxjiba,wtzyxjiba->t", point_propag.data.conj(), point_propag.data).real.get(), [0, -1, -1, -1]
        )
    )

print("Point source, conf 0: ", point_pion[0][:6])
print("Wall source, conf 0: ", wall_pion[0][:6])

# %%
wall_pion_jk = jackknife(wall_pion)
point_pion_jk = jackknife(point_pion)

wall_pion_jk_avg = jk_ls_avg(wall_pion_jk)
point_pion_jk_avg = jk_ls_avg(point_pion_jk)

wall_meff = pt2_to_meff(wall_pion_jk_avg, boundary="periodic")
point_meff = pt2_to_meff(point_pion_jk_avg, boundary="periodic")

fig, ax = default_plot()
ax.errorbar(np.arange(len(wall_meff)), gv.mean(wall_meff), yerr=gv.sdev(wall_meff), label="wall", **errorb)
ax.errorbar(np.arange(len(point_meff)), gv.mean(point_meff), yerr=gv.sdev(point_meff), label="point", **errorb)
ax.legend(ncol=2, **fs_small_p)
ax.set_xlabel(r"$t_{\mathrm{sep}}$", **fs_p)
ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
plt.tight_layout()
plt.savefig("../plots/pion_meff.pdf", transparent=True)
plt.show()
# %%
