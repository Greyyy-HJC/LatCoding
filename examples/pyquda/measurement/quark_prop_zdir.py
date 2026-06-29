# %%
import gvar as gv
from tqdm.auto import tqdm
from pyquda import init
from pyquda_utils import core, io, source, gamma
from opt_einsum import contract

from lametlat.plotting.plot_settings import *
from lametlat.correlators.resampling import *
from lametlat.correlators.pt2 import pt2_to_meff

from pathlib import Path

Ls = 16
Lt = 16
root_dir = Path(__file__).resolve().parents[3]
conf_dir = root_dir / "configs" / f"S{Ls}T{Lt}_cg"
cache_dir = root_dir / ".cache" / "pyquda"
plot_dir = root_dir / "examples" / "artifacts" / "plots"

cache_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

init([1, 1, 1, 1], resource_path=str(cache_dir))
N_conf = 20

xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.0336
csw_t = 1.0336
multigrid = None 

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)

# Get gammaI matrix
I = gamma.gamma(0)

# Lists to store correlation functions
point_quark_corr = []
for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    conf_path = conf_dir / f"wilson_b6.cg.1e-14.{cfg}"
    gauge = io.readNERSCGauge(str(conf_path))
    gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)

    dirac.loadGauge(gauge)
    
    # Point source propagator
    point_source = source.propagator(latt_info, "point", [0, 0, 0, 0])
    point_propag = core.invertPropagator(dirac, point_source)
    
    # Contract to get correlation function
    point_quark_corr.append(
        core.gatherLattice(
            core.lexico(contract("wtzyxijaa,ji->wtzyx", point_propag.data, I).real.get(), [0,1,2,3,4]), 
            [0, 1, 2, 3]
        )[0, 0, :, 0]
    )

print("\n>>> shape of point_quark_corr: ", point_quark_corr[0].shape)

# Print first few entries of the correlation functions
print("Point source, conf 0: ", point_quark_corr[0][:6])

# %%
print("shape of point_quark_corr: ", np.shape(point_quark_corr))

point_quark_corr_jk = jackknife(point_quark_corr)
point_quark_corr_jk_avg = jk_ls_avg(point_quark_corr_jk)
point_meff = pt2_to_meff(point_quark_corr_jk_avg, boundary="none")

fig, ax = default_plot()
ax.errorbar(np.arange(len(point_meff)), gv.mean(point_meff), yerr=gv.sdev(point_meff), label="point", **ERRORBAR_STYLE)
ax.legend(ncol=2, **LEGEND_SIZE)
ax.set_xlabel(r"$t_{\mathrm{sep}}$", **FONT_SIZE)
ax.set_ylabel(r"$m_{\mathrm{eff}}$", **FONT_SIZE)
plt.tight_layout()
plt.savefig(plot_dir / "quark_corr_meff.pdf", transparent=True)
plt.show()
# %%

