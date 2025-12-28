import numpy as np
from pyquda import init
from pyquda.field import LatticeGauge
from boosted_smearing_pyquda import boosted_smearing
from types import SimpleNamespace
from pyquda_utils import source, core, io
from opt_einsum import contract
import cupy as cp

init([1, 1, 1, 1], resource_path=".cache")

mass = -0.038888
csw_r = 1.02868
csw_t = 1.02868
xi_0 = 1.0
nu = 1.0

latt_info = core.LatticeInfo([8, 8, 8, 32], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t)

gauge = io.readNERSCGauge("/home/jinchen/git/lat-software/LatCoding/conf/S8T32/wilson_b6.0")

dirac.loadGauge(gauge)

# same random source as GPT
point_prop = source.propagator(latt_info, "point", [1,2,1,3])

# U_trafo = identity
U_data = cp.zeros((32,8,8,8,3,3), dtype=cp.complex128)
U_data[..., 0,0] = 1
U_data[..., 1,1] = 1
U_data[..., 2,2] = 1
U_py = SimpleNamespace(data=U_data, latt_info=latt_info)

point_prop_smear = boosted_smearing(U_py, point_prop, w=2.0, boost=(1,2,1))

propag = core.invertPropagator(dirac, point_prop_smear)


temp = core.gatherLattice(
            contract("wtzyxjiba,wtzyxjiba->t", propag.data.conj(), propag.data).real.get(), [0, -1, -1, -1]
        )

print(temp)
