# %%
import os
from tqdm.auto import tqdm
import numpy as np
import cupy as cp
from pyquda import init
from opt_einsum import contract
from matplotlib import pyplot as plt
from pyquda_utils import core, io, gamma, phase
from pyquda_utils.phase import MomentumPhase

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *
from lametlat.preprocess.read_raw import pt2_to_meff

if not os.path.exists(".cache"):
    os.makedirs(".cache")
    print("Created .cache directory for PyQUDA resources")

init([1, 1, 1, 1], resource_path=".cache")
N_conf = 50

xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None 

latt_info = core.LatticeInfo([16, 16, 16, 16], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)

G5 = gamma.gamma(15)
momentum_list = [[0, 0, 0], [1, 1, 1], [4, 4, 4], [6, 6, 6], [7, 7, 7]]
momentum_label = ["000", "111", "444", "666", "777"]
momentum_phases = MomentumPhase(latt_info).getPhases(momentum_list)

t_src_list = list(range(0, 16, 16))

data_ls = []
for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    gauge = io.readNERSCGauge(f"../../conf/S16T16/wilson_b6.{cfg}")

    # gauge.stoutSmear(1, 0.125, 4)
    dirac.loadGauge(gauge)
    
    pion = cp.zeros((len(t_src_list), len(momentum_list), latt_info.Lt), "<c16")

    for t_idx, t_src in enumerate(t_src_list):
        propag = core.invert(dirac, "point", [0, 0, 0, t_src])

        pion[t_idx] += contract(
            "pwtzyx,wtzyxjiba,jk,wtzyxklba,li->pt",
            momentum_phases,
            propag.data.conj(),
            G5 @ G5,
            propag.data,
            G5 @ G5,
        )
        
    tmp = core.gatherLattice(pion.real.get(), [2, -1, -1, -1])
    data_ls.append(tmp)

# dirac.destroy()

if latt_info.mpi_rank == 0:
    for idx, tmp in enumerate(data_ls):
        for t_idx, t_src in enumerate(t_src_list):
            tmp[t_idx] = np.roll(tmp[t_idx], -t_src, 1)
        data_ls[idx] = tmp
        
    temp = np.array(data_ls)[:,0,:,:]
    data_avg = jk_ls_avg(jackknife(temp))
    
    fig, ax = default_plot()
    
    for idx, data_mom in enumerate(data_avg):
        meff = pt2_to_meff(data_mom, boundary="periodic")
        ax.errorbar(np.arange(len(meff)), gv.mean(meff), yerr=gv.sdev(meff), label=f"{momentum_label[idx]}", **errorb)
    ax.legend(ncol=2, **fs_small_p)
    ax.set_xlabel(r"$t_{\mathrm{sep}}$", **fs_p)
    ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
    ax.set_ylim(0, 5)
    ax.legend(ncol=3, loc="upper right", **fs_small_p)
    plt.tight_layout()
    plt.savefig("../plots/pion_dispersion.pdf", transparent=True)
    plt.show()

# %%
