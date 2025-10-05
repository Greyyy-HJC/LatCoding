# %%
import os
import time
import gvar as gv
import cupy as cp
from tqdm.auto import tqdm
from pyquda import init
from pyquda.field import LatticeGauge
from pyquda_utils import core, io, source, gamma
from pyquda_utils.phase import MomentumPhase
from opt_einsum import contract

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *
from lametlat.preprocess.read_raw import pt2_to_meff

os.environ["CUPY_ACCELERATORS"] = "cub,cutensor"

if not os.path.exists(".cache"):
    os.makedirs(".cache")
    print("Created .cache directory for PyQUDA resources")

init(None, [8, 8, 8, 32], resource_path=".cache")
Ls = 8
Lt = 32
N_conf = 10

xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None 

C = gamma.gamma(2) @ gamma.gamma(8)
G0 = gamma.gamma(0)
GZG5 = gamma.gamma(11)
G4 = gamma.gamma(8)
G4G5 = gamma.gamma(7)
G5 = gamma.gamma(15)

latt_info = core.LatticeInfo([8, 8, 8, 32], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)
t_src_list = list(range(0, Lt, int(Lt)))
z_list = list(range(8))

wall_pion_DA = []
point_pion_DA = []
for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    gauge = io.readNERSCGauge(f"../../conf/S8T32_cg/gauge/wilson_b6.cg.1e-08.{cfg}")
    # gauge.stoutSmear(1, 0.125, 4)
    dirac.loadGauge(gauge)
    
    wall_pion_DA_tmp = cp.zeros((len(t_src_list), len(z_list), latt_info.Lt), "<c16")
    point_pion_DA_tmp = cp.zeros((len(t_src_list), len(z_list), latt_info.Lt), "<c16")
        
    for idt, t_src in enumerate(t_src_list):
        # * add momentum phase to wall source
        mom_phase = MomentumPhase(latt_info).getPhase([0, 0, 0])
        wall_source = source.propagator(latt_info, "wall", t_src, mom_phase)

        wall_propag = core.invertPropagator(dirac, wall_source)

        # wtzyxjiba are indices of the propagator, ->t means contract all indices except t
        # [0, -1, -1, -1] means keep the t direction and sum over the other directions, 1 means gather the data, 0 means no action, -1 means sum / average
        wall_propag_shift = wall_propag.copy()
        wall_propag_backward = contract("li,wtzyxjiba,jk->wtzyxklba", G5 @ G5, wall_propag.data.conj(), G5 @ G4G5)
        for idz, z in enumerate(z_list):
            # Time the contraction
            cp.cuda.runtime.deviceSynchronize()
            start_contract = time.time()
            
            wall_pion_DA_tmp[idt, idz] += contract(
                "wtzyxklba,wtzyxklba->t",
                wall_propag_backward,
                wall_propag_shift.data
            )
            
            cp.cuda.runtime.deviceSynchronize()
            end_contract = time.time()
            print(f">>> Contraction type 1 time: {end_contract - start_contract:.3f} seconds")
            
            # Time the shift operation
            cp.cuda.runtime.deviceSynchronize()
            start_shift = time.time()
            for spin in range(4):
                for color in range(3):
                    fermion = wall_propag_shift.getFermion(spin, color)
                    fermion_shift = gauge.pure_gauge.covDev(fermion, 2)
                    # \psi'(x)=U_\mu(x)\psi(x+\hat\mu) 0,1,2,3 for x,y,z,t; 4,5,6,7 for -x,-y,-z,-t
                    wall_propag_shift.setFermion(fermion_shift, spin, color)
            cp.cuda.runtime.deviceSynchronize()
            end_shift = time.time()
            print(f">>> Shift time: {end_shift - start_shift:.3f} seconds")

        point_source = source.propagator(latt_info, "point", [0, 0, 0, t_src])
        point_propag = core.invertPropagator(dirac, point_source)
        
        point_propag_shift = point_propag.copy() # (2, Lt, Lz, Ly, Lx // 2, Ns, Ns Nc, Nc)
        for idz, z in enumerate(z_list):
            point_pion_DA_tmp[idt, idz] += contract(
                "wtzyxjiba,jk,wtzyxklba,li->t",
                point_propag.data.conj(),
                G5 @ GZG5,
                point_propag_shift.data,
                GZG5 @ G5
            )
            
            #! use covDev to shift fermion's data
            unit = LatticeGauge(latt_info)
            unit.gauge_dirac.loadGauge(unit)
            for spin in range(4):
                for color in range(3):
                    #! if CG, no Wilson link
                    fermion = point_propag_shift.getFermion(spin, color)
                    fermion_unit = unit.pure_gauge.covDev(fermion, 2)
                    point_propag_shift.setFermion(fermion_unit, spin, color)
            
            unit.gauge_dirac.loadGauge(gauge)

    gauge.pure_gauge.freeGauge()

    wall_pion_DA_tmp = core.gatherLattice(wall_pion_DA_tmp.real.get(), [2, -1, -1, -1])
    point_pion_DA_tmp = core.gatherLattice(point_pion_DA_tmp.real.get(), [2, -1, -1, -1])
    
    if latt_info.mpi_rank == 0:
        for idt, t_src in enumerate(t_src_list):
            wall_pion_DA_tmp[idt] = np.roll(wall_pion_DA_tmp[idt], -t_src, 0)
            point_pion_DA_tmp[idt] = np.roll(point_pion_DA_tmp[idt], -t_src, 0)
        
        wall_pion_DA.append(np.mean(wall_pion_DA_tmp, axis=0))
        point_pion_DA.append(np.mean(point_pion_DA_tmp, axis=0))

# average over t_src
wall_pion_DA = np.array(wall_pion_DA)
point_pion_DA = np.array(point_pion_DA)

print(np.shape(wall_pion_DA))
print(np.shape(point_pion_DA))

# %%
wall_pion_jk = jackknife(wall_pion_DA)
point_pion_jk = jackknife(point_pion_DA)

wall_pion_jk_avg = jk_ls_avg(wall_pion_jk)
point_pion_jk_avg = jk_ls_avg(point_pion_jk)

fig, ax = default_plot()
for idx, z in enumerate(z_list[:1]):
    wall_meff = pt2_to_meff(wall_pion_jk_avg[idx], boundary="periodic")
    point_meff = pt2_to_meff(point_pion_jk_avg[idx], boundary="periodic")
    ax.errorbar(np.arange(len(wall_meff)), gv.mean(wall_meff), yerr=gv.sdev(wall_meff), label=f"wall, z={z}", **errorb)
    ax.errorbar(np.arange(len(point_meff)), gv.mean(point_meff), yerr=gv.sdev(point_meff), label=f"point, z={z}", **errorb)

ax.legend(ncol=2, **fs_small_p)
ax.set_xlabel(r"$t_{\mathrm{sep}}$", **fs_p)
ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
plt.tight_layout()
plt.savefig("../plots/addon_2pt_meff.pdf", transparent=True)
plt.show()

# %%
fix_t = 10
bare_point_da = []
for z in z_list:
    bare_point_da.append(point_pion_jk_avg[z][fix_t])
bare_point_da = np.array(bare_point_da)
bare_point_da = bare_point_da / bare_point_da[0]

bare_wall_da = []
for z in z_list:
    bare_wall_da.append(wall_pion_jk_avg[z][fix_t])
bare_wall_da = np.array(bare_wall_da)
bare_wall_da = bare_wall_da / bare_wall_da[0]

fig, ax = default_plot()
ax.errorbar(z_list, gv.mean(bare_point_da), yerr=gv.sdev(bare_point_da), label="point", **errorb)
ax.errorbar(z_list, gv.mean(bare_wall_da), yerr=gv.sdev(bare_wall_da), label="wall", **errorb)
ax.legend(**fs_small_p)
ax.set_xlabel(r"$z$", **fs_p)
ax.set_ylabel(r"$h^0(z)$", **fs_p)
plt.tight_layout()
plt.savefig("../plots/addon_2pt_bare.pdf", transparent=True)
plt.show()

# %%
