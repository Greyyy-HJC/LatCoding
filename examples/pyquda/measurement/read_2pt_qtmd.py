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


# --------------------------
# Data selection
# --------------------------

Ls = 16
Lt = 16
lat_tag = f"S{Ls}T{Lt}_cg"
sm_tag = f"S{Ls}T{Lt}_qtmd"
data_dir = Path(__file__).resolve().parents[2] / "artifacts" / "data"

pt2_src_mode = "fixed"
pt2_src = "5"
pt2_snk = "5"
pt2_momentum = (0, 0, 0)

pt3_src = "5"
pt3_snk = "5"
pt3_pf = (0, 0, 0)
pt3_q = (0, 0, 0)
tsep_values = [2, 4, 6, 8]

insertion_gammas = ["T"]
bT_direction = "b_X"
eta = 0
bT = 0
bz_values = np.arange(-8, 9)


def momentum_tag(momentum):
    return f"PX{momentum[0]}PY{momentum[1]}PZ{momentum[2]}"


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
    pt2_src_tag = f"fixed_src{pt2_src}"
else:
    pt2_src_tag = pt2_src_mode

pt2_momentum_tag = momentum_tag(pt2_momentum)
pt3_q_tag = momentum_tag(pt3_q)

c2pt_paths = sorted(
    (data_dir / "c2pt").glob(
        f"{lat_tag}.c2pt.*.CG.ex.*.{sm_tag}.{pt2_src_tag}.h5"
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
    qtmd_files_by_gamma = {}
    for insertion_gamma in insertion_gammas:
        qtmd_paths = sorted(
            (data_dir / "qTMD").glob(
                f"{lat_tag}.qTMD.*.CG.ex.*.{sm_tag}."
                f"src{pt3_src}.snk{pt3_snk}.{pt3_pf_tag}.O{insertion_gamma}.h5"
            )
        )
        qtmd_files_by_gamma[insertion_gamma] = index_files(qtmd_paths, "qTMD")
        actual_configs = set(qtmd_files_by_gamma[insertion_gamma])
        if actual_configs != expected_configs:
            missing = sorted(expected_configs - actual_configs)
            unexpected = sorted(actual_configs - expected_configs)
            raise FileNotFoundError(
                f"Incomplete qTMD data for tsep={tsep}, gamma={insertion_gamma}: "
                f"missing configs={missing}, unexpected configs={unexpected}"
            )

    qtmd_files_by_tsep[tsep] = qtmd_files_by_gamma


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
        config_qtmd = []
        for insertion_gamma in insertion_gammas:
            gamma_qtmd = []
            qtmd_path = qtmd_files_by_tsep[tsep][insertion_gamma][config_num]
            with h5py.File(qtmd_path, "r") as h5_file:
                for bz in bz_values:
                    dataset = (
                        f"SS/{insertion_gamma}/{pt3_q_tag}/{bT_direction}/"
                        f"eta{eta}/bT{bT}/bz{bz}"
                    )
                    gamma_qtmd.append(h5_file[dataset][:])
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
    qtmd_real[key] = qtmd_by_tsep[key][:,8,:] # z = 0
    qtmd_imag[key] = np.zeros_like(qtmd_real[key])

ratio_real, ratio_imag = get_pt3_ratio_data(np.real(c2pt), np.imag(c2pt), qtmd_real, qtmd_imag)

print(np.shape( ratio_real[8]) )

ratio_real = {tsep: ratio_real[tsep][:, 1:tsep] for tsep in [2, 4, 6, 8]}

print(np.shape( ratio_real[8]) )

ratio_real_avg = jk_dict_avg(ratio_real)

tau_dict = {tsep: np.arange(1, tsep) for tsep in [2, 4, 6, 8]}

(fig_real, ax_real) = pt3_ratio_plot(tau_dict, ratio_real_avg)
plt.tight_layout()
fig_real.show()

# %%
