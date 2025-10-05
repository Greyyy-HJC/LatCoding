# %%
from itertools import permutations
from pyquda import init
import numpy as np
import cupy as cp
from opt_einsum import contract
from matplotlib import pyplot as plt
from tqdm import tqdm

from pyquda_utils import core, io, gamma

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *
from lametlat.preprocess.read_raw import pt2_to_meff

init([1, 1, 1, 1], resource_path=".cache")
Ls = 8
Lt = 32

# Lattice parameters
xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None # [[4, 4, 4, 4], [2, 2, 2, 8]]

N_conf = 50

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)

# * gamma.gamma(n) is the same as QLUA setting
C = gamma.gamma(2) @ gamma.gamma(8)
G0 = gamma.gamma(0)
G4 = gamma.gamma(8)
G5 = gamma.gamma(15)
P = cp.zeros((Lt, 4, 4), "<c16")
P[:int(Lt/2)] = (G0 + G4) / 2
P[int(Lt/2):] = (G0 - G4) / 2
T = cp.ones((2 * Lt), "<f8")
T[:] = -1
T[int(Lt/2) : int(Lt/2) + Lt] = 1
t_src_list = list(range(0, Lt, int(Lt/4)))

pion_conf_list = []
proton_conf_list = []

# %%
for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    
    pion = cp.zeros((len(t_src_list), latt_info.Lt), "<c16")
    proton = cp.zeros((len(t_src_list), latt_info.Lt), "<c16")
    
    gauge = io.readNERSCGauge(f"../../conf/S8T32/wilson_b6.{cfg}")
    # gauge.stoutSmear(1, 0.125, 4)
    dirac.loadGauge(gauge)

    for t_idx, t_src in enumerate(t_src_list):
        propag = core.invert(dirac, "wall", t_src)

        pion[t_idx] += contract(
            "wtzyxjiba,jk,wtzyxklba,li->t",
            propag.data.conj(),
            G5 @ G5,
            propag.data,
            G5 @ G5,
        )

        P_ = cp.roll(P, t_src, 0)[latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
        T_ = T[Lt - t_src : 2 * Lt - t_src][latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
        for a, b, c in permutations(tuple(range(3))):
            for d, e, f in permutations(tuple(range(3))):
                sign = 1 if b == (a + 1) % 3 else -1
                sign *= 1 if e == (d + 1) % 3 else -1
                proton[t_idx] += (sign * T_) * (
                    contract(
                        "ij,kl,tmn,wtzyxik,wtzyxjl,wtzyxmn->t",
                        C @ G5,
                        C @ G5,
                        P_,
                        propag.data[:, :, :, :, :, :, :, a, d],
                        propag.data[:, :, :, :, :, :, :, b, e],
                        propag.data[:, :, :, :, :, :, :, c, f],
                    )
                    + contract(
                        "ij,kl,tmn,wtzyxik,wtzyxjn,wtzyxml->t",
                        C @ G5,
                        C @ G5,
                        P_,
                        propag.data[:, :, :, :, :, :, :, a, d],
                        propag.data[:, :, :, :, :, :, :, b, e],
                        propag.data[:, :, :, :, :, :, :, c, f],
                    )
                )
                
    pion_tmp = core.gatherLattice(pion.real.get(), [1, -1, -1, -1])
    proton_tmp = core.gatherLattice(proton.real.get(), [1, -1, -1, -1])
    
    if latt_info.mpi_rank == 0:
        for t_idx, t_src in enumerate(t_src_list):
            pion_tmp[t_idx] = np.roll(pion_tmp[t_idx], -t_src, 0)
            proton_tmp[t_idx] = np.roll(proton_tmp[t_idx], -t_src, 0)
        
        twopt_pion = pion_tmp.mean(0)
        twopt_proton = proton_tmp.mean(0)
        
        pion_conf_list.append(twopt_pion)
        proton_conf_list.append(twopt_proton)

dirac.destroy()

# %%
pion_jk = jackknife(pion_conf_list)
proton_jk = jackknife(proton_conf_list)
pion_jk_avg = jk_ls_avg(pion_jk)
proton_jk_avg = jk_ls_avg(proton_jk)

pion_meff = pt2_to_meff(pion_jk_avg, boundary="periodic")
proton_meff = pt2_to_meff(proton_jk_avg, boundary="periodic")

fig, ax = default_plot()
ax.errorbar(np.arange(Lt-2), gv.mean(pion_meff), yerr=gv.sdev(pion_meff), label="pion", **errorb)
ax.errorbar(np.arange(Lt-2), gv.mean(proton_meff), yerr=gv.sdev(proton_meff), label="proton", **errorb)
ax.legend(ncol=2, **fs_small_p)
ax.set_xlabel(r"$t_{\mathrm{sep}}$", **fs_p)
ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
plt.tight_layout()
plt.savefig("../plots/pion_proton_meff_wall.pdf", transparent=True)
plt.show()

# %%
