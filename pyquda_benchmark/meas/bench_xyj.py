from time import perf_counter
from typing import List

import cupy
from cupy.cuda.runtime import deviceSynchronize
from opt_einsum import contract

from pyquda_utils import core, gamma, phase
from pyquda_plugins import pycontract

core.init(resource_path=".cache")

latt_info = core.LatticeInfo([16, 16, 16, 24])
propag = core.LatticePropagator(latt_info)

sequential_prop_down = propag.copy()
sequential_prop_up = propag.copy()

sequential_bw_prop_down = core.LatticePropagator(
    latt_info, contract("ij,wtzyxkjba,kl->wtzyxilab", gamma.gamma(15), sequential_prop_down.data, gamma.gamma(15))
)
sequential_bw_prop_up = core.LatticePropagator(
    latt_info, contract("ij,wtzyxkjba,kl->wtzyxilab", gamma.gamma(15), sequential_prop_up.data, gamma.gamma(15))
)


WL_index_list_CG = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 2, 0, 0],
    [0, 3, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 2, 0, 1],
    [0, 3, 0, 1],
]


def create_fw_prop_TMP_CG_pyquda(prop_f_pyq: core.LatticePropagator, W_index: List[int]):
    current_b_T = W_index[0]
    current_bz = W_index[1]
    transverse_direction = W_index[3]

    prop_shift_pyq = prop_f_pyq.shift(current_b_T, transverse_direction).shift(current_bz, core.Z)

    return prop_shift_pyq


pyquda_gamma_ls = cupy.asarray([gamma.gamma(i) for i in range(16)])
proton_TMDs_down = []
proton_TMDs_up = []

qext_xyz = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
phase_3pt_pyq = phase.MomentumPhase(latt_info).getPhases(qext_xyz)

sequential_bw_prop_down_pyq = contract("wtzyxijab,gjk->gwtzyxikab", sequential_bw_prop_down.data, pyquda_gamma_ls)
sequential_bw_prop_up_pyq = contract("wtzyxijab,gjk->gwtzyxikab", sequential_bw_prop_up.data, pyquda_gamma_ls)

for iW, WL_indices in enumerate(WL_index_list_CG):
    deviceSynchronize()
    s = perf_counter()
    tmd_forward_prop = create_fw_prop_TMP_CG_pyquda(propag, WL_indices)
    deviceSynchronize()
    core.getLogger().info(f"OLD cshift: {perf_counter() - s} secs")

    deviceSynchronize()
    s = perf_counter()
    tmp_down = contract("gwtzyxijab,wtzyxjiba->gwtzyx", sequential_bw_prop_down_pyq, tmd_forward_prop.data)
    tmp_up = contract("gwtzyxijab,wtzyxjiba->gwtzyx", sequential_bw_prop_up_pyq, tmd_forward_prop.data)
    deviceSynchronize()
    core.getLogger().info(f"OLD contract: {perf_counter() - s} secs")

    deviceSynchronize()
    s = perf_counter()
    proton_TMDs_down.append(contract("qwtzyx,gwtzyx->qgt", phase_3pt_pyq, tmp_down))
    proton_TMDs_up.append(contract("qwtzyx,gwtzyx->qgt", phase_3pt_pyq, tmp_up))
    deviceSynchronize()
    core.getLogger().info(f"OLD contract(phase): {perf_counter() - s} secs")

for iW, WL_indices in enumerate(WL_index_list_CG):
    deviceSynchronize()
    s = perf_counter()
    tmd_forward_prop = create_fw_prop_TMP_CG_pyquda(propag, WL_indices)
    deviceSynchronize()
    core.getLogger().info(f"NEW cshift: {perf_counter() - s} secs")

    deviceSynchronize()
    s = perf_counter()
    tmp_down = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop, sequential_prop_down, gamma.Gamma(0))
    tmp_up = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop, sequential_prop_up, gamma.Gamma(0))
    deviceSynchronize()
    core.getLogger().info(f"NEW pycontract.mesonAllSinkTwoPoint: {perf_counter() - s} secs")

    deviceSynchronize()
    s = perf_counter()
    proton_TMDs_down.append(contract("qwtzyx,gwtzyx->qgt", phase_3pt_pyq, tmp_down.data))
    proton_TMDs_up.append(contract("qwtzyx,gwtzyx->qgt", phase_3pt_pyq, tmp_up.data))
    deviceSynchronize()
    core.getLogger().info(f"NEW contract(phase): {perf_counter() - s} secs")
