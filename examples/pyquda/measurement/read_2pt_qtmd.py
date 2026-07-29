# %%
"""Read pion CG qTMD two- and three-point correlators into NumPy arrays."""

import re
from pathlib import Path

import h5py
import numpy as np
import gvar as gv

from lametlat.correlators.resampling import jackknife, jk_ls_avg, jk_dict_avg
from lametlat.correlators.pt2 import pt2_to_meff
from lametlat.correlators.pt3_ratio import get_pt3_ratio_data
from lametlat.plotting.plot_settings import *
from lametlat.plotting.corr_plots import pt3_ratio_plot
from lametlat.ground_state.pt2_fit import pt2_two_state_fit
from lametlat.ground_state.pt3_ratio_fit import pt3_ratio_two_state_fit


# --------------------------
# Data selection
# --------------------------

Ls = 16
Lt = 16
lat_tag = f"S{Ls}T{Lt}_cg"
sm_tag = f"S{Ls}T{Lt}_qtmd_debug_p6" #todo
data_dir = Path(__file__).resolve().parents[2] / "artifacts" / "data"

pt2_src_mode = "fixed"
pt2_src = "5"
pt2_snk = "5"
pt2_momentum = (0, 0, 6) #todo

pt3_src = "5"
pt3_snk = "5"
pt3_pf = (0, 0, 6)
pt3_q = (0, 0, 0)
# tsep_values = [2, 4, 6, 8] #todo
tsep_values = [8]

insertion_gammas = ["T"]
bT_direction = "b_X"
eta = 0
bT = 0
bz_values = np.arange(-8, 9)


def momentum_tag(momentum):
    return f"PX{momentum[0]}PY{momentum[1]}PZ{momentum[2]}"
def row_index(table, row, label):
    matches = np.flatnonzero(np.all(np.asarray(table) == np.asarray(row), axis=1))
    if len(matches) != 1:
        raise ValueError(f"Expected one {label} row {row}, found {len(matches)}")
    return int(matches[0])




def config_number(path, correlator):
    match = re.search(rf"\.{correlator}\.(\d+)\.", path.name)
    if match is None:
        raise ValueError(f"Could not parse configuration number from {path}")
    return int(match.group(1))


def index_files(paths, correlator):
    indexed = {}
    for path in paths:
        config_num = config_number(path, correlator)
        if config_num in indexed:
            raise ValueError(
                f"Multiple {correlator} files found for configuration {config_num}: "
                f"{indexed[config_num]} and {path}"
            )
        indexed[config_num] = path
    return indexed


if pt2_src_mode == "fixed":
    pt2_src_tag = f"src{pt2_src}"
else:
    pt2_src_tag = pt2_src_mode

pt2_momentum_tag = momentum_tag(pt2_momentum)
pt3_q_tag = momentum_tag(pt3_q)

c2pt_paths = sorted(
    (data_dir / "c2pt").glob(
        f"{lat_tag}.c2pt.*.CG.ex.x0y0z0t0.{sm_tag}.{pt2_src_tag}.h5"
    )
)
c2pt_files = index_files(c2pt_paths, "c2pt")
if not c2pt_files:
    raise FileNotFoundError("No matching 2pt files were found.")

configs = np.array(sorted(c2pt_files), dtype=int)
expected_configs = set(configs)

qtmd_files_by_tsep = {}
for tsep in tsep_values:
    pt3_pf_tag = f"{momentum_tag(pt3_pf)}dt{tsep}"
    qtmd_paths = sorted(
        (data_dir / "qTMD").glob(
            f"{lat_tag}.qTMD.*.CG.ex.*.{sm_tag}."
            f"src{pt3_src}.snk{pt3_snk}.{pt3_pf_tag}.h5"
        )
    )
    qtmd_files = index_files(qtmd_paths, "qTMD")
    actual_configs = set(qtmd_files)
    if actual_configs != expected_configs:
        missing = sorted(expected_configs - actual_configs)
        unexpected = sorted(actual_configs - expected_configs)
        raise FileNotFoundError(
            f"Incomplete qTMD data for tsep={tsep}: missing={missing}, unexpected={unexpected}"
        )
    qtmd_files_by_tsep[tsep] = qtmd_files


# --------------------------
# Read correlators
# --------------------------

c2pt_data = []
for config_num in configs:
    with h5py.File(c2pt_files[config_num], "r") as h5_file:
        c2pt_dataset = f"SS/{pt2_snk}/{pt2_momentum_tag}"
        c2pt_data.append(h5_file[c2pt_dataset][:])
c2pt = np.asarray(c2pt_data)

qtmd_by_tsep = {}

for tsep in tsep_values:
    qtmd_data = []
    for config_num in configs:
        qtmd_path = qtmd_files_by_tsep[tsep][config_num]
        config_qtmd = []
        with h5py.File(qtmd_path, "r") as h5_file:
            if h5_file.attrs.get("qtmd_hdf5_schema") != "connected_qtmd_dense_v1":
                raise ValueError(f"Unsupported qTMD schema in {qtmd_path}")
            gamma_labels = [value.decode() for value in h5_file["gamma_list"][:]]
            momentum_list = h5_file["momentum_list"][:]
            wilson_list = h5_file["wilson_index_list"][:]
            momentum_idx = row_index(momentum_list, [*pt3_q, 0], "momentum")
            transverse_direction = {"b_X": 0, "b_Y": 1}[bT_direction]
            for insertion_gamma in insertion_gammas:
                gamma_qtmd = []
                gamma_idx = gamma_labels.index(insertion_gamma)
                for bz in bz_values:
                    wilson_idx = row_index(wilson_list, [bT, bz, eta, transverse_direction], "Wilson")
                    gamma_qtmd.append(h5_file["corr"][wilson_idx, momentum_idx, gamma_idx])
                config_qtmd.append(gamma_qtmd)
        qtmd_data.append(config_qtmd)
    qtmd_by_tsep[tsep] = np.asarray(qtmd_data)[:,0,:,:]

print("c2pt shape (config, t):", c2pt.shape)
for tsep in tsep_values:
    print(
        f"tsep={tsep} qtmd shape (config, insertion_gamma, bz, tau):",
        qtmd_by_tsep[tsep].shape,
    )


# %%
c2pt_avg = jk_ls_avg(jackknife(np.real(c2pt)))
meff_avg = pt2_to_meff(c2pt_avg)

fig, ax = default_plot()
ax.errorbar(np.arange(len(meff_avg)), gv.mean(meff_avg), gv.sdev(meff_avg), **ERRORBAR_STYLE)
plt.tight_layout()
plt.show()

qtmd_real = {}
qtmd_imag = {}
for key in qtmd_by_tsep:
    qtmd_real[key] = jackknife(np.real(qtmd_by_tsep[key][:,9,:])) # z = 1
    qtmd_imag[key] = jackknife(np.imag(qtmd_by_tsep[key][:,9,:]))

ratio_real, ratio_imag = get_pt3_ratio_data(jackknife(np.real(c2pt)), jackknife(np.imag(c2pt)), qtmd_real, qtmd_imag)

print(np.shape( ratio_real[8]) )

# ratio_real_cut = {tsep: ratio_real[tsep][:, 1:tsep] for tsep in [2, 4, 6, 8]} #todo
ratio_real_cut = {tsep: ratio_real[tsep][:, 1:tsep] for tsep in [8]} 

print(np.shape( ratio_real_cut[8]) )

ratio_real_avg = jk_dict_avg(ratio_real_cut)

# tau_dict = {tsep: np.arange(1, tsep) for tsep in [2, 4, 6, 8]} #todo
tau_dict = {tsep: np.arange(1, tsep) for tsep in [8]} 

(fig_real, ax_real) = pt3_ratio_plot(tau_dict, ratio_real_avg)
plt.tight_layout()
fig_real.show()

# %%
bare_qpdf = []

pt2_fit_res = pt2_two_state_fit(c2pt_avg, tmin=3, tmax=8, Lt=Lt)
for idz in range(8, 17):
    qtmd_real = {}
    qtmd_imag = {}
    for key in qtmd_by_tsep:
        qtmd_real[key] = jackknife( np.real(qtmd_by_tsep[key][:,idz,:]) )
        qtmd_imag[key] = jackknife( np.imag(qtmd_by_tsep[key][:,idz,:]) )

    ratio_real, ratio_imag = get_pt3_ratio_data(jackknife( np.real(c2pt) ), jackknife( np.imag(c2pt) ), qtmd_real, qtmd_imag)

    fit_ratio_real = jk_dict_avg(ratio_real)
    fit_ratio_imag = jk_dict_avg(ratio_imag)
    
    ratio_fit_res = pt3_ratio_two_state_fit([4, 6], 1, fit_ratio_real, fit_ratio_imag, Lt, pt2_fit_res=pt2_fit_res)
    
    if ratio_fit_res.Q < 0.05:
        print(f"Warning: bad fit with z={idz-8}")
    
    bare_qpdf.append( ratio_fit_res.p['O00_re'] / 2 / ratio_fit_res.p['E0'] )
    
fig, ax = default_plot()
ax.errorbar(np.arange(len(bare_qpdf)), gv.mean(bare_qpdf), gv.sdev(bare_qpdf), **ERRORBAR_STYLE)
plt.tight_layout()
plt.show()
    
# %%
