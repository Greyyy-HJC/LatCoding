# %%
import os
import gvar as gv
from tqdm.auto import tqdm
from pyquda import init
from pyquda_utils import core, io, source, gamma
from pyquda_utils.phase import MomentumPhase
from opt_einsum import contract

from pyquda_plugins import pycontract
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

point_pion = []
mesonall_sink_pion = []
for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    gauge = io.readNERSCGauge(f"../../conf/S8T32/wilson_b6.{cfg}")

    # gauge.stoutSmear(1, 0.125, 4)
    dirac.loadGauge(gauge)
    
    # * add momentum phase to wall source
    mom_phase = MomentumPhase(latt_info).getPhases([[1, 1, 2]])
    mom_phase_meson = MomentumPhase(latt_info).getPhases([[1, 1, 2]])

    # wtzyxjiba are indices of the propagator, ->t means contract all indices except t
    # indices are: even/odd, t, z, y, x, spin, spin, color, color
    # [0, -1, -1, -1] means keep the t direction and sum over the other directions, 1 means gather the data, 0 means no action, -1 means sum / average
    point_source = source.propagator(latt_info, "point", [0, 0, 0, 0])
    point_propag = core.invertPropagator(dirac, point_source)

    point_pion.append(
        core.gatherLattice(
            contract("wtzyxjiba,wtzyxjiba,qwtzyx->qt", point_propag.data.conj(), point_propag.data, mom_phase).get(), [1, -1, -1, -1]
        )
    )
    
    temp = pycontract.mesonAllSinkTwoPoint(point_propag, point_propag, gamma.Gamma(15)).data
    
    mesonall_sink_pion.append(
        core.gatherLattice(
            contract("qwtzyx, gwtzyx -> qgt", mom_phase_meson, temp).get(), [2, -1, -1, -1]
        )
    )

print("Point source shape: ", np.shape(point_pion))
print("Mesonall sink source shape: ", np.shape(mesonall_sink_pion))

print("Point source, conf 0: ", point_pion[0][0][:6])
print("Mesonall sink source, conf 0: ", mesonall_sink_pion[0][0][15][:6])

# %%
