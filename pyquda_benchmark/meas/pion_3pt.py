# %%
import os
import gvar as gv
import cupy as cp
from tqdm.auto import tqdm
from pyquda import init
from pyquda_utils import core, io, source
from opt_einsum import contract

from pyquda_utils import core, io, gamma, source

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *

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

G5 = gamma.gamma(15)
G4 = gamma.gamma(8)
G0 = gamma.gamma(0)

Gamma_curr = G0

latt_info = core.LatticeInfo([8, 8, 8, 32], -1, xi_0 / nu)
Ls = latt_info.global_size[0]
Lt = latt_info.global_size[3]
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)

t_src_list = list(range(0, Lt, int(Lt/4)))
t_sep_list = [6, 8, 10]

pt2_pion = []
pt3_pion = []

for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    
    pt2_conf = cp.zeros((len(t_src_list), latt_info.Lt), "<c16")
    pt3_conf = cp.zeros((len(t_src_list), len(t_sep_list), latt_info.Lt), "<c16")
    
    gauge = io.readNERSCGauge(f"../../conf/S8T32/wilson_b6.{cfg}")
    # gauge.stoutSmear(1, 0.125, 4)
    dirac.loadGauge(gauge)
    
    for t_idx, t_src in enumerate(t_src_list):
        # wtzyxjiba are indices of the propagator, ->t means contract all indices except t
        # [0, -1, -1, -1] means keep the t direction and sum over the other directions, 1 means gather the data, 0 means no action, -1 means sum / average
        
        #! 2pt
        t_src = 0
        point_source = source.propagator(latt_info, "point", [0, 0, 0, t_src])
        point_propag = core.invertPropagator(dirac, point_source)

        pt2_conf[t_idx] += contract("wtzyxjiba,wtzyxjiba->t", point_propag.data.conj(), point_propag.data)
            
        for t_sep_idx, t_sep in enumerate(t_sep_list):
            #! 3pt
            t_snk = (t_src + t_sep) % Lt
            seq_source = core.LatticePropagator(latt_info)
            seq_source.data = contract("ij,etzyxjkab->etzyxikab", G5, point_propag.data)
            seq_source = source.sequential12(seq_source, t_snk)
            seq_propag = core.invertPropagator(dirac, seq_source)
            
            pt3_conf[t_idx, t_sep_idx] += contract(
                "etzyxjiba,jk,etzyxkiba->t",
                point_propag.data.conj(),
                G5 @ Gamma_curr,
                seq_propag.data,
            )
    pt2_tmp = core.gatherLattice(pt2_conf.real.get(), [1, -1, -1, -1])
    pt3_tmp = core.gatherLattice(pt3_conf.real.get(), [2, -1, -1, -1]) # * since the 3pt has one more dimension of tsep, 2 means the axis 2 is the latt_info.Lt that needs to be gathered
    
    if latt_info.mpi_rank == 0:
        pt2_pion.append(pt2_tmp.mean(0))
        pt3_pion.append(pt3_tmp.mean(0))
    
dirac.destroy()


# %%
print("shape of pt2_pion: ", np.shape(pt2_pion))
print("shape of pt3_pion: ", np.shape(pt3_pion))

pt2_pion_jk = jackknife(pt2_pion)
pt3_pion_jk = jackknife(pt3_pion)

pt2_pion_jk_avg = jk_ls_avg(pt2_pion_jk)
pt3_pion_jk_avg = jk_ls_avg(pt3_pion_jk)

print("shape of pt2_pion_jk_avg: ", np.shape(pt2_pion_jk_avg))
print("shape of pt3_pion_jk_avg: ", np.shape(pt3_pion_jk_avg))

# Calculate ratios for each tsep
tau_cut = 1
ratio_ls = []

for tsep_idx, tsep in enumerate(t_sep_list):
    # Ratio = 3pt / 2pt for each tau
    ratio = pt3_pion_jk_avg[tsep_idx, :] / pt2_pion_jk_avg[tsep]
    ratio_ls.append(ratio)

# Plot the results
fig, ax = default_plot()

for i, ratio in enumerate(ratio_ls):
    tsep = t_sep_list[i]
    tau_vals = np.arange(tau_cut, tsep+1-tau_cut) - tsep/2
    ratio_gv = ratio[tau_cut:tsep+1-tau_cut]
    
    ax.errorbar(tau_vals, gv.mean(ratio_gv), yerr=gv.sdev(ratio_gv), label=f"tsep={tsep}", **errorb)

ax.set_xlabel("τ (current insertion time)", **fs_p)
ax.set_ylabel("Ratio", **fs_p)
ax.legend(ncol=3, **fs_small_p)
ax.set_ylim(auto_ylim([gv.mean(ratio_gv)], [gv.sdev(ratio_gv)], 2))
plt.tight_layout()
plt.savefig("../plots/pion_3pt_ratio.pdf", transparent=True)
plt.show()

# Plot the meff
from lametlat.preprocess.read_raw import pt2_to_meff

pion_meff = pt2_to_meff(pt2_pion_jk_avg, boundary="periodic")

fig, ax = default_plot()
ax.errorbar(np.arange(len(pion_meff)), gv.mean(pion_meff), yerr=gv.sdev(pion_meff), **errorb)
ax.set_xlabel("t", **fs_p)
ax.set_ylabel("m_eff", **fs_p)
plt.tight_layout()
plt.show()
# %%
