# %%
import os
from tqdm.auto import tqdm
import numpy as np
from pyquda import init
from opt_einsum import contract
from pyquda_utils import core, io, gamma
from pyquda_utils.phase import MomentumPhase

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *
from lametlat.preprocess.read_raw import pt2_to_meff

from mom_smearing import momentum_smearing_propagator, momentum_smearing_sink

if not os.path.exists(".cache"):
    os.makedirs(".cache")
    print("Created .cache directory for PyQUDA resources")

init([1, 1, 1, 1], resource_path=".cache")
N_conf = 20

xi_0, nu = 1.0, 1.0
csw_r = 1
csw_t = 1

Ls = 8
Lt = 32
kappa = 0.12575
mass = 1 / (2 * kappa) - 4

k1 = np.array([0.5,0.5,0.5])
k2 = np.array([0.5,0.5,0.5])
rho = 3
nsteps = 32
src_pos = [0, 0, 0, 0]

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-10, 10000, xi_0, csw_r, csw_t)

G5 = gamma.gamma(15)
GT5 = gamma.gamma(7)

momentum_list = [[1,1,1]]
momentum_phases = MomentumPhase(latt_info).getPhases(momentum_list, x0=src_pos)

data_ls = []
for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    gauge = io.readNERSCGauge(f"../../conf/S8T32/wilson_b6.{cfg}")

    momentum_source1 = momentum_smearing_propagator(latt_info, gauge, k1, src_pos, rho, nsteps)
    momentum_source2 = momentum_smearing_propagator(latt_info, gauge, k2, src_pos, rho, nsteps)

    dirac.loadGauge(gauge)
    propag1 = core.invertPropagator(dirac, momentum_source1)
    propag2 = core.invertPropagator(dirac, momentum_source2)

    propag1_sink_smeared = momentum_smearing_sink(latt_info, propag1, gauge, k1, rho, nsteps)
    propag2_sink_smeared = momentum_smearing_sink(latt_info, propag2, gauge, k2, rho, nsteps)

    temp = contract(
        "pwtzyx,wtzyxjiba,jk,wtzyxklba,li->pt",
        momentum_phases,
        propag1_sink_smeared.data.conj(),
        G5 @ GT5,
        propag2_sink_smeared.data,
        GT5 @ G5,
    )
    
    data = core.gatherLattice(temp.get(), [1, -1, -1, -1])
    data_ls.append(data)
    
dirac.destroy()

if latt_info.mpi_rank == 0:
    print(np.shape(data_ls))
    data_ls_jk = jackknife(data_ls)
    data_ls_jk_avg = jk_ls_avg(data_ls_jk)[0]
    data_ls_meff = pt2_to_meff(data_ls_jk_avg, boundary="periodic")

    fig, ax = default_plot()
    ax.errorbar(np.arange(len(data_ls_meff)), gv.mean(data_ls_meff), yerr=gv.sdev(data_ls_meff), label="pion, p=(1,1,1)", **errorb)
    ax.legend(ncol=2, **fs_small_p)
    ax.set_xlabel(r"$t_{\mathrm{sep}}$", **fs_p)
    ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
    ax.set_ylim(0, 5)
    plt.tight_layout()
    plt.savefig("../plots/pion_2pt_mom_smear.pdf", transparent=True)
    plt.show()

# %%
