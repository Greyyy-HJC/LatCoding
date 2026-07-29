# %%
"""Read src5/snk5 2pt and a selected Gamma from all-channel qTMDWF files."""

import re
from pathlib import Path

import h5py
import numpy as np
import gvar as gv

from lametlat.correlators.resampling import jackknife, jk_ls_avg
from lametlat.correlators.pt2 import pt2_to_meff
from lametlat.plotting.plot_settings import *
from lametlat.plotting.corr_plots import qda_ratio_plot
from lametlat.ground_state.qda_fit import qda_two_state_joint_fit


Ls = 16
Lt = 16
sm_tag = "S16T16_tmdwf_debug_gz"
sink_gamma = "T5"

data_dir = Path(__file__).resolve().parents[2] / "artifacts" / "data"
z_values = np.arange(-8, 9)

c2pt_paths = sorted(
    (data_dir / "c2pt").glob(f"S{Ls}T{Lt}_cg.c2pt.*.x0y0z0t0.{sm_tag}.src5.h5")
)
qtmdwf_paths = sorted(
    (data_dir / "qTMDWF").glob(
        f"S{Ls}T{Lt}_cg.qTMDWF.*.x0y0z0t0.{sm_tag}.src5.h5"
    )
)

c2pt_files = {
    int(re.search(r"\.c2pt\.(\d+)\.", path.name).group(1)): path
    for path in c2pt_paths
}
qtmdwf_files = {
    int(re.search(r"\.qTMDWF\.(\d+)\.", path.name).group(1)): path
    for path in qtmdwf_paths
}
configs = np.array(sorted(c2pt_files.keys() & qtmdwf_files.keys()))

c2pt_data = []
qtmdwf_data = []

for config_num in configs:
    with h5py.File(c2pt_files[config_num], "r") as h5_file:
        c2pt_data.append(h5_file["SS/5/PX0PY0PZ6"][:])

    qtmdwf_z_data = []
    with h5py.File(qtmdwf_files[config_num], "r") as h5_file:
        for z in z_values:
            dataset = f"SP/{sink_gamma}/PX0PY0PZ6/b_X/eta0/bT0/bz{z}"
            qtmdwf_z_data.append(h5_file[dataset][:])
    qtmdwf_data.append(qtmdwf_z_data)

c2pt = np.array(c2pt_data)  # (config, t)
qtmdwf = np.array(qtmdwf_data)  # (config, z, t)

c2pt_avg = jk_ls_avg(jackknife(np.real(c2pt)))
qtmdwf_avg = jk_ls_avg(jackknife(np.real(qtmdwf)))

print("c2pt shape:", c2pt.shape)
print("qtmdwf shape:", qtmdwf.shape)

meff_avg = pt2_to_meff(c2pt_avg)

fig, ax = default_plot()
ax.errorbar(np.arange(len(meff_avg)), gv.mean(meff_avg), gv.sdev(meff_avg), **ERRORBAR_STYLE)
plt.tight_layout()
plt.show()

qtmdwf_z0 = qtmdwf_avg[8]
qtmdwf_z0_ratio = qtmdwf_z0 / c2pt_avg

fig_real, ax_real = qda_ratio_plot(np.arange(Lt), qtmdwf_z0_ratio)
ax_real.set_title("z=0", **FONT_SIZE)
fig_real.show()

qtmdwf_z1 = qtmdwf_avg[9]
qtmdwf_z1_ratio = qtmdwf_z1 / c2pt_avg

fig_real, ax_real = qda_ratio_plot(np.arange(Lt), qtmdwf_z1_ratio)
ax_real.set_title("z=1", **FONT_SIZE)
fig_real.show()

# %%

bare_qtmdwf = []
fit_for_plot = {}
for idz in range(8, 17):
    z = idz - 8
    pt2_trange = np.arange(1, 6)
    qda_trange = np.arange(3, 7)
    fit_res = qda_two_state_joint_fit(c2pt_avg, qtmdwf_avg[idz], None, pt2_trange, qda_trange, Lt)
    bare_qtmdwf.append(fit_res.p["O00_re"])
    if z in (0, 2, 4):
        fit_for_plot[z] = fit_res

for z, fit_res in fit_for_plot.items():
    idz = 8 + z
    fig_real, ax_real = qda_ratio_plot(
        np.arange(Lt),
        qtmdwf_avg[idz] / c2pt_avg,
        fit_result=fit_res,
        fit_trange=np.arange(3, 7),
        Lt=Lt,
        id_label={"z": z},
    )
    fig_real.show()

fig, ax = default_plot()
ax.errorbar(np.arange(len(bare_qtmdwf)), gv.mean(bare_qtmdwf), gv.sdev(bare_qtmdwf), **ERRORBAR_STYLE)
ax.set_title("bare qDA", **FONT_SIZE)
ax.set_xlabel("z", **FONT_SIZE)
plt.tight_layout()
plt.show()
# %%
