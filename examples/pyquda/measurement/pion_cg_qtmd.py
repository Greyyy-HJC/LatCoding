import argparse
import time
from pathlib import Path

import numpy as np

from pyquda import getMPIComm, init


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=0, help="Configuration number")
parser.add_argument("--tsep", type=int, default=4, help="Source-sink separation")
parser.add_argument("--mpi_geometry", type=str, default="1.1.1.1", help="MPI geometry")
args, unknown = parser.parse_known_args()
conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

# Global parameters
Ls = 16
Lt = 16
repo_root = Path(__file__).resolve().parents[3]
data_dir = repo_root / "examples/artifacts/data"
gauge_path = repo_root / f"configs/S{Ls}T{Lt}_cg/wilson_b6.cg.1e-14.{conf}"
lat_tag = f"S{Ls}T{Lt}_cg"
sm_tag = f"S{Ls}T{Lt}_qtmd"


# --------------------------
# Initiate QUDA
# --------------------------

init(mpi_geometry, enable_mps=True)

from pyquda_utils import core, io, source
from pyquda_utils.phase import MomentumPhase

from latcoding.pyquda.classes.pion_cg_qtmd_class import my_gammas, pion_TMD
from latcoding.pyquda.utils.boosted_smearing import boosted_smearing
from latcoding.pyquda.utils.bw_seq_pyquda import create_meson_bw_seq_pyquda
from latcoding.pyquda.utils.io_corr import (
    get_c2pt_file_tag,
    get_qTMD_file_tag,
    get_sample_log_tag,
    save_qTMD_pion_hdf5_noRoll,
)
from latcoding.pyquda.utils.pion_utils import gamma_from_label
from latcoding.pyquda.utils.tools import mpi_print, srcLoc_distri_eq


# --------------------------
# Setup parameters
# --------------------------

parameters = {
    "eta": [0],
    "b_T": 0,
    "b_z": 8,
    # Momenta are integer Fourier modes: p_i = 2*pi*n_i/L_i.  The fourth
    # component is kept for metadata compatibility; only x, y, z are used here.
    # qext is the qTMD insertion momentum transfer q = p_f - p_i.  It is passed
    # directly to MomentumPhase, giving exp(+i*q.(x-x_src)) at the insertion.
    # For b_T = 0, the resulting straight-z CG qTMD data are also the CG qPDF.
    "qext": [[0, 0, 0, 0]],
    # pf is the fixed final-pion (sink) momentum used to build the sequential
    # source; the later conjugation produces the usual exp(-i*p_f.(y-x_src)).
    "pf": [0, 0, 0, 0],
    # p_2pt is the pion momentum measured by the two-point correlator.  The
    # script negates these modes below, so its projection is exp(-i*p.(x-x_src)).
    "p_2pt": [[0, 0, 0, 0]],
    "width": 0,
    "pos_boost": [0, 0, 0],
    "neg_boost": [0, 0, 0],
    "t_insert": args.tsep,
    "save_propagators": False,
}
measurement = pion_TMD(parameters)
compute_2pt = True
pt2_src_mode = "fixed"
pt2_src = "5"
pt3_src = "5"
pt3_snk = "5"
n_src = 1

if pt2_src_mode == "fixed":
    pt2_src_gamma = pt2_src
    pt2_src_tag = f"fixed_src{pt2_src}"
elif pt2_src_mode in {"same_as_sink", "dagger_of_sink"}:
    pt2_src_gamma = pt2_src_mode
    pt2_src_tag = pt2_src_mode
else:
    raise ValueError(
        f"Invalid pt2_src_mode: {pt2_src_mode}. "
        "Expected one of ['fixed', 'same_as_sink', 'dagger_of_sink']."
    )


def sync_backend_array(arr):
    stream = getattr(arr, "stream", None)
    if stream is not None:
        stream.synchronize()
    queue = getattr(arr, "sycl_queue", None)
    if queue is not None:
        queue.wait()


def prepare_three_point(corr, pos, latt_info):
    if latt_info.mpi_rank == 0:
        corr = np.roll(corr, -pos[3], axis=-1)
        corr = corr[:, :, :, : parameters["t_insert"] + 2]
        corr = np.transpose(corr, (0, 2, 1, 3))
    return getMPIComm().bcast(corr, root=0)


def save_three_point(corr, pos, correlator_tag, momenta, wilson_indices, latt_info):
    corr = prepare_three_point(corr, pos, latt_info)
    pf = parameters["pf"]
    pf_tag = f"PX{pf[0]}PY{pf[1]}PZ{pf[2]}dt{parameters['t_insert']}"
    rank = latt_info.mpi_rank
    size = getMPIComm().Get_size()

    for gamma_idx in range(rank, len(my_gammas), size):
        insertion_gamma = my_gammas[gamma_idx]
        tag = get_qTMD_file_tag(
            str(data_dir),
            lat_tag,
            conf,
            correlator_tag,
            pos,
            f"{sm_tag}.src{pt3_src}.snk{pt3_snk}."
            f"{pf_tag}.O{insertion_gamma}",
        )
        mpi_print(latt_info, f"Saving {correlator_tag} insertion gamma {insertion_gamma}: {tag}")
        save_qTMD_pion_hdf5_noRoll(
            corr[:, :, gamma_idx : gamma_idx + 1, :],
            tag,
            [insertion_gamma],
            momenta,
            wilson_indices,
            parameters["t_insert"],
            latt_info,
        )


# --------------------------
# Load gauge and create inverter
# --------------------------

L = [Ls, Ls, Ls, Lt]
xi_0, nu = 1.0, 1.0
mass = -0.038888
csw_r = 1.0336
csw_t = 1.0336
multigrid = None
latt_info = core.LatticeInfo(L, -1, xi_0 / nu)

gauge = io.readNERSCGauge(str(gauge_path))
# gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)

mpi_print(latt_info, f"--gauge_path {gauge_path}")
mpi_print(latt_info, f"--data_dir {data_dir}")
mpi_print(latt_info, f"--lat_tag {lat_tag}")
mpi_print(latt_info, f"--sm_tag {sm_tag}")
mpi_print(latt_info, f"--config_num {conf}")
mpi_print(latt_info, f"--mpi_geometry {mpi_geometry}")
mpi_print(latt_info, f"--plaquette U: {gauge.plaquette()}")

dirac = core.getClover(latt_info, mass, 1e-10, 10000, xi_0, csw_r, csw_t, multigrid)
dirac.loadGauge(gauge)


# --------------------------
# Setup source positions and output directories
# --------------------------

src_shift = np.array([7, 11, 13, 23])
src_origin = np.array([int(conf) % L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin)[:n_src]

pf = parameters["pf"]
pf_tag = f"PX{pf[0]}PY{pf[1]}PZ{pf[2]}dt{parameters['t_insert']}"
sample_log_file = data_dir / "sample_log" / f"TMD_{sm_tag}_{conf}_{pf_tag}"
if latt_info.mpi_rank == 0:
    sample_log_file.parent.mkdir(parents=True, exist_ok=True)
    (data_dir / "c2pt").mkdir(parents=True, exist_ok=True)
    (data_dir / "qTMD").mkdir(parents=True, exist_ok=True)
    sample_log_file.touch(exist_ok=True)
getMPIComm().Barrier()

pt3_snk_gamma = gamma_from_label(pt3_snk)


# --------------------------
# Start measurements
# --------------------------

for pos in src_positions:
    source_start = time.time()
    sample_log_tag = get_sample_log_tag("ex", pos, f"{sm_tag}.{pf_tag}")
    mpi_print(latt_info, f"Contraction START: {sample_log_tag}")
    
    with open(sample_log_file, "a+") as f:
        f.seek(0)
        if sample_log_tag in f.read():
            mpi_print(latt_info, f"Contraction SKIP: {sample_log_tag}")
            continue #! comment for test

    # Forward propagators for the quark and antiquark source smearings
    t0 = time.time()
    src_point = source.propagator(latt_info, "point", pos)
    src_pos = boosted_smearing(
        src_point,
        w=parameters["width"],
        boost=parameters["pos_boost"],
    )
    same_source_smearing = parameters["width"] == 0 or np.array_equal(
        parameters["pos_boost"],
        parameters["neg_boost"],
    )
    src_neg = None
    if not same_source_smearing:
        src_neg = boosted_smearing(
            src_point,
            w=parameters["width"],
            boost=parameters["neg_boost"],
        )
    mpi_print(latt_info, f"TIME PyQUDA: Generating boosted sources {time.time() - t0}s")

    t0 = time.time()
    prop_pos = core.invertPropagator(dirac, src_pos, 1, 0)
    if same_source_smearing:
        prop_neg = prop_pos
        inversion_count = 1
    else:
        prop_neg = core.invertPropagator(dirac, src_neg, 1, 0)
        inversion_count = 2
    mpi_print(
        latt_info,
        f"TIME PyQUDA: Forward propagator inversion x{inversion_count} {time.time() - t0}s",
    )

    if compute_2pt:
        t0 = time.time()
        p_2pt_xyz = [[-p[0], -p[1], -p[2]] for p in parameters["p_2pt"]]
        phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)
        c2_tag = get_c2pt_file_tag(
            str(data_dir),
            lat_tag,
            conf,
            "CG.ex",
            pos,
            f"{sm_tag}.{pt2_src_tag}",
        )
        measurement.contract_2pt_pion(
            latt_info,
            prop_pos,
            prop_neg,
            phases_2pt,
            c2_tag,
            src_gamma=pt2_src_gamma,
        )
        mpi_print(latt_info, f"TIME PyQUDA: Pion 2pt contraction {time.time() - t0}s")
    else:
        mpi_print(latt_info, "SKIP: Pion 2pt contraction")

    # Fixed-sink sequential propagator
    t0 = time.time()
    prop_sink_smeared = boosted_smearing(
        prop_neg.copy(),
        w=parameters["width"],
        boost=parameters["neg_boost"],
    )
    seq_bw_prop = create_meson_bw_seq_pyquda(
        dirac,
        prop_sink_smeared,
        pos,
        parameters["pf"],
        parameters["t_insert"],
        pt3_snk_gamma,
        parameters["width"],
        parameters["pos_boost"],
    )
    mpi_print(latt_info, f"TIME PyQUDA: Pion sequential propagator {time.time() - t0}s")

    qext_xyz = [[q[0], q[1], q[2]] for q in parameters["qext"]]
    phases_TMD = MomentumPhase(latt_info).getPhases(qext_xyz, x0=pos)

    wilson_dir0, wilson_dir1 = measurement.create_TMD_Wilsonline_index_list_CG()
    wilson_TMD = wilson_dir0 + wilson_dir1

    # CG qTMD without gauge links; its b_T=0 subset is the CG qPDF.
    t0 = time.time()
    pion_TMDs = measurement.contract_qTMD_CG(
        latt_info,
        prop_pos,
        seq_bw_prop,
        phases_TMD,
        wilson_dir0,
        wilson_dir1,
        src_gamma=pt3_src,
    )
    mpi_print(
        latt_info,
        f"TIME CG qTMD: src{pt3_src}.snk{pt3_snk} "
        f"shape {np.shape(pion_TMDs)} {time.time() - t0}s",
    )
    save_three_point(
        pion_TMDs,
        pos,
        "CG.ex",
        parameters["qext"],
        wilson_TMD,
        latt_info,
    )
    del pion_TMDs

    sync_backend_array(prop_pos.data)
    if prop_neg is not prop_pos:
        sync_backend_array(prop_neg.data)
    if latt_info.mpi_rank == 0:
        with sample_log_file.open("a+") as log_file:
            log_file.write(sample_log_tag + "\n")
    mpi_print(latt_info, f"DONE: {sample_log_tag} total {time.time() - source_start}s")
