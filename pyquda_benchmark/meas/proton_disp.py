# %%
from itertools import permutations
from pyquda import init
import numpy as np
import cupy as cp
from opt_einsum import contract
from matplotlib import pyplot as plt
from tqdm import tqdm

from pyquda_utils import core, io, gamma, source, phase

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *
from lametlat.preprocess.read_raw import pt2_to_meff
from lametlat.utils.funcs import constant_fit

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

GZ5 = gamma.gamma(4) @ G5
GT5 = gamma.gamma(8) @ G5
GX5 = gamma.gamma(1) @ G5

GZ5pX5 = GZ5 + GX5
GZ5mX5 = GZ5 - GX5

P = cp.zeros((Lt, 4, 4), "<c16")
P[:int(Lt/2)] = (G0 + G4) / 2
P[int(Lt/2):] = (G0 - G4) / 2
T = cp.ones((2 * Lt), "<f8")
T[:] = -1
T[int(Lt/2) : int(Lt/2) + Lt] = 1

# momentum list
momentum_list = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]]

# source time position
t_src_list = list(range(0, Lt, int(Lt/4)))

# store all configurations results
proton_conf_list = []

# %%
for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    
    # create result array for different momenta
    proton = cp.zeros((len(t_src_list), len(momentum_list), latt_info.Lt), "<c16")
    
    gauge = io.readNERSCGauge(f"../../conf/S8T32/wilson_b6.{cfg}")
    # gauge.stoutSmear(1, 0.125, 4)
    dirac.loadGauge(gauge)

    for t_idx, t_src in enumerate(t_src_list):
        # create point source and compute propagator
        point_source = source.propagator(latt_info, "point", [0, 0, 0, t_src])
        propag = core.invertPropagator(dirac, point_source)

        P_ = cp.roll(P, t_src, 0)[latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
        T_ = T[Lt - t_src : 2 * Lt - t_src][latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
        
        # build proton operator for each momentum
        for p_idx, momentum in enumerate(momentum_list):
            # calculate momentum phase for this specific momentum
            momentum_phase = phase.MomentumPhase(latt_info).getPhases([momentum])
            
            for a, b, c in permutations(tuple(range(3))):
                for d, e, f in permutations(tuple(range(3))):
                    sign = 1 if b == (a + 1) % 3 else -1
                    sign *= 1 if e == (d + 1) % 3 else -1
                    
                    # add momentum projection (using only the single momentum)
                    proton[t_idx, p_idx] += (sign * T_) * contract(
                        "wtzyx,ij,kl,tmn,wtzyxik,wtzyxjl,wtzyxmn->t",
                        momentum_phase[0],  # Use index 0 since we only have one momentum
                        C @ GT5,
                        C @ GT5,
                        P_,
                        propag.data[:, :, :, :, :, :, :, a, d],
                        propag.data[:, :, :, :, :, :, :, b, e],
                        propag.data[:, :, :, :, :, :, :, c, f],
                    )
                    
                    proton[t_idx, p_idx] += (sign * T_) * contract(
                        "wtzyx,ij,kl,tmn,wtzyxik,wtzyxjn,wtzyxml->t",
                        momentum_phase[0],  # Use index 0 since we only have one momentum
                        C @ GT5,
                        C @ GT5,
                        P_,
                        propag.data[:, :, :, :, :, :, :, a, d],
                        propag.data[:, :, :, :, :, :, :, b, e],
                        propag.data[:, :, :, :, :, :, :, c, f],
                    )
                
    # collect results for each configuration
    proton_tmp = core.gatherLattice(proton.real.get(), [2, -1, -1, -1])
    
    if latt_info.mpi_rank == 0:
        # time shift
        for t_idx, t_src in enumerate(t_src_list):
            proton_tmp[t_idx] = np.roll(proton_tmp[t_idx], -t_src, 1)
        
        # average over different source positions
        twopt_proton = proton_tmp.mean(0)
        proton_conf_list.append(twopt_proton)

dirac.destroy()

# %%
# Jackknife analysis
print(np.shape(proton_conf_list)) # (N_conf, mom_ls, Lt)
proton_jk = jackknife(proton_conf_list)
proton_jk_avg = jk_ls_avg(proton_jk)

# calculate effective mass of momentum dispersion relation
proton_meff = {}
for p_idx, mom in enumerate(momentum_list):
    proton_meff[str(mom)] = pt2_to_meff(proton_jk_avg[p_idx], boundary="periodic")

# plot effective mass
fig, ax = default_plot()
for p_idx, mom in enumerate(momentum_list):
    mom_key = str(mom)
    ax.errorbar(
        np.arange(Lt-2), 
        gv.mean(proton_meff[mom_key]), 
        yerr=gv.sdev(proton_meff[mom_key]), 
        label=f"p={mom}", 
        **errorb
    )

ax.legend(ncol=3, **fs_small_p)
ax.set_xlabel(r"$t_{\mathrm{sep}}$", **fs_p)
ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
ax.set_ylim(0, 5)
plt.tight_layout()
plt.savefig("../plots/proton_dispersion.pdf", transparent=True)
plt.show()

# %%
# analyze energy-momentum relation
energy_values = {}
for p_idx, mom in enumerate(momentum_list):
    mom_key = str(mom)
    # get energy values from plateau region 
    plateau_range = range(6, 10)
    energy_values[mom_key] = constant_fit(proton_meff[mom_key][plateau_range])
    
# calculate momentum magnitude
p_values = []
for mom in momentum_list:
    p_mag = np.sqrt(sum(p_i**2 for p_i in mom)) * (2*np.pi/Ls)
    p_values.append(p_mag)

# plot energy-momentum relation
fig, ax = default_plot()
ax.errorbar(
    p_values,
    [gv.mean(energy_values[str(mom)]) for mom in momentum_list],
    yerr=[gv.sdev(energy_values[str(mom)]) for mom in momentum_list],
    **errorb
)

ax.set_xlabel(r"$p$", **fs_p)
ax.set_ylabel(r"$E$", **fs_p)
plt.tight_layout()
plt.savefig("../plots/proton_energy_momentum.pdf", transparent=True)
plt.show()

# %% 