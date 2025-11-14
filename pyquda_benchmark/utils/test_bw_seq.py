# proton_seq_pyquda.py

import cupy as cp
from opt_einsum import contract

from pyquda import init

import numpy as np


from types import SimpleNamespace

from pyquda.field import LatticePropagator, LatticeGauge
from pyquda_utils import core, source, gamma, io

from boosted_smearing_pyquda import boosted_smearing
from bw_seq_pyquda import create_bw_seq_pyquda

width = 2.0
boost_out = [1,2,1]
pf = [0,0,7,0]
t_insert = 4


init([1, 1, 1, 1], resource_path=".cache")

mass = -0.038888
csw_r = 1.02868
csw_t = 1.02868
xi_0 = 1.0
nu = 1.0

latt_info = core.LatticeInfo([8, 8, 8, 8], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t)

gauge = io.readNERSCGauge("/home/jinchen/git/lat-software/LatCoding/conf/S8T8/wilson_b6.0")

dirac.loadGauge(gauge)

# same random source as GPT
point_prop = source.propagator(latt_info, "point", [1,2,1,3])

propag = core.invertPropagator(dirac, point_prop)

U_data = cp.zeros((8,8,8,8,3,3), dtype=cp.complex128)
U_data[..., 0,0] = 1
U_data[..., 1,1] = 1
U_data[..., 2,2] = 1
U_trafo = SimpleNamespace(data=U_data, latt_info=latt_info)

smearing_input = create_bw_seq_pyquda(propag, U_trafo, origin=[0,0,0,0], sm_width=width, sm_boost=boost_out, momentum=pf, t_insert=t_insert)

print(np.shape(smearing_input.get()))
print(np.linalg.norm(smearing_input.get())**2)