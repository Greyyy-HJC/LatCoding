# %%
import gvar as gv
from tqdm.auto import tqdm
from pyquda import init
from pyquda_utils import core, io, source, gamma
from pyquda_utils.phase import MomentumPhase
from opt_einsum import contract
from latcoding.pyquda.utils.tools import mpi_print

from lametlat.plotting.plot_settings import *
from lametlat.correlators.resampling import *

from pathlib import Path
import numpy as np

Ls = 16
Lt = 16
root_dir = Path(__file__).resolve().parents[3]
conf_dir = root_dir / "configs" / f"S{Ls}T{Lt}_cg"
cache_dir = root_dir / ".cache" / "pyquda"
data_dir = root_dir / "examples" / "artifacts" / "data" / "quark_prop_mom"
plot_dir = root_dir / "examples" / "artifacts" / "plots"

cache_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

init([1, 1, 1, 1], resource_path=str(cache_dir))
N_conf = 20

xi_0, nu = 1.0, 1.0
mass = -0.038888  # kappa = 0.12623
csw_r = 1.0336
csw_t = 1.0336
multigrid = None

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)

I = gamma.gamma(0)

momentum_list = [[0, 0, 0], [0, 0, 2], (2, 2, 2)]
momentum_label = ["(0,0,0)", "(0,0,2)", "(2,2,2)"]
momentum_phases = MomentumPhase(latt_info).getPhases(momentum_list)

# S(t, p): source and sink on the same time slice t; shape per momentum is (N_conf, Lt)
wall_quark_corr_by_mom = {label: [] for label in momentum_label}

for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    mpi_print(latt_info, f"Processing configuration {cfg}")
    
    conf_path = conf_dir / f"wilson_b6.cg.1e-14.{cfg}"
    gauge = io.readNERSCGauge(str(conf_path))
    # gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)

    dirac.loadGauge(gauge)

    cfg_corr = {label: np.zeros(Lt, dtype=np.float64) for label in momentum_label}

    for t in range(Lt):
        for idx, p_phase in enumerate(momentum_phases):
            # Source phase: exp(-ip·x_src), sink phase: exp(+ip·x_snk) at the same t slice.
            wall_source = source.propagator(
                latt_info, "wall", t, source_phase=np.conj(p_phase)
            )
            wall_propag = core.invertPropagator(dirac, wall_source)

            corr_t = core.gatherLattice(
                contract(
                    "wtzyx,wtzyxijaa,ji->t",
                    p_phase,
                    wall_propag.data,
                    I,
                ).get(),
                [0, -1, -1, -1],
            )
            cfg_corr[momentum_label[idx]][t] = np.real(corr_t[t])

    for label in momentum_label:
        wall_quark_corr_by_mom[label].append(cfg_corr[label])

for label in momentum_label:
    wall_quark_corr = np.asarray(wall_quark_corr_by_mom[label])
    mom_tag = label.replace("(", "").replace(")", "").replace(",", "_")
    np.save(data_dir / f"I_{mom_tag}.npy", wall_quark_corr)
    print(f"Saved {data_dir / f'I_{mom_tag}.npy'}, shape = {wall_quark_corr.shape}")

print("\n>>> example S(t, p) for p=(0,0,0), conf 0: ", wall_quark_corr_by_mom["(0,0,0)"][0][:6])

# %%
fig, ax = default_plot()
for label in momentum_label:
    corr_jk_avg = jk_ls_avg(jackknife(wall_quark_corr_by_mom[label]))
    ax.errorbar(
        np.arange(Lt),
        gv.mean(corr_jk_avg),
        yerr=gv.sdev(corr_jk_avg),
        label=label,
        **ERRORBAR_STYLE,
    )

ax.legend(ncol=2, **LEGEND_SIZE)
ax.set_xlabel(r"$t$", **FONT_SIZE)
ax.set_ylabel(r"$S(t, \vec{p})$", **FONT_SIZE)
plt.tight_layout()
plt.savefig(plot_dir / "quark_prop_mom_same_t.pdf", transparent=True)
plt.show()
# %%
