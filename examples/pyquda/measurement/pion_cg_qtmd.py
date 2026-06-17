import argparse
import os
import time
from pathlib import Path

import numpy as np

from pyquda import getMPIComm, init


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("PION_TMD_CONFIG_NUM", 0)))
parser.add_argument("--mpi_geometry", type=str, default=os.environ.get("PION_TMD_MPI_GEOMETRY", "1.1.1.1"))
parser.add_argument("--gauge_path", type=str, default=os.environ.get("PION_TMD_GAUGE_PATH", ""))
parser.add_argument("--data_dir", type=str, default=os.environ.get("PION_TMD_DATA_DIR", ""))
parser.add_argument("--num_src", type=int, default=int(os.environ.get("PION_TMD_NUM_SRC", 1)))
parser.add_argument("--qmax", type=int, default=int(os.environ.get("PION_TMD_QMAX", 1)))
parser.add_argument("--b_z", type=int, default=int(os.environ.get("PION_TMD_BZ", 1)))
parser.add_argument("--b_T", type=int, default=int(os.environ.get("PION_TMD_BT", 1)))
parser.add_argument("--t_insert", type=int, default=int(os.environ.get("PION_TMD_T_INSERT", 2)))
parser.add_argument("--width", type=float, default=float(os.environ.get("PION_TMD_WIDTH", 1.0)))
parser.add_argument("--src_interpolator", type=str, default=os.environ.get("PION_TMD_SRC_INTERPOLATOR", "fixed_g5"))
parser.add_argument("--sink_interpolator", type=str, default=os.environ.get("PION_TMD_SINK_INTERPOLATOR", "5"))
args, unknown = parser.parse_known_args()

mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]
init(mpi_geometry, enable_mps=True)

from pyquda_utils import core, gamma, io, phase, source
from pyquda_utils.phase import MomentumPhase

from latcoding.pyquda.utils.boosted_smearing import boosted_smearing
from latcoding.pyquda.utils.bw_seq_pyquda import create_meson_bw_seq_pyquda
from latcoding.pyquda.utils.io_corr import (
    get_c2pt_file_tag,
    get_qTMD_file_tag,
    get_sample_log_tag,
    save_qTMD_pion_hdf5_noRoll,
)
from latcoding.pyquda.classes.pion_cg_qtmd_class import my_gammas, pion_TMD
from latcoding.pyquda.utils.tools import mpi_print, srcLoc_distri_eq


def sync_backend_array(arr):
    stream = getattr(arr, "stream", None)
    if stream is not None:
        stream.synchronize()
    queue = getattr(arr, "sycl_queue", None)
    if queue is not None:
        queue.wait()


def gamma_from_label(label):
    gamma_map = {
        "5": 15,
        "T": 8,
        "T5": 7,
        "X": 1,
        "X5": 14,
        "Y": 2,
        "Y5": 13,
        "Z": 4,
        "Z5": 11,
        "I": 0,
        "SXT": 9,
        "SXY": 3,
        "SXZ": 5,
        "SYT": 10,
        "SYZ": 6,
        "SZT": 12,
    }
    if label not in gamma_map:
        raise ValueError(f"Invalid sink interpolator: {label}")
    return gamma.gamma(gamma_map[label])


software_root = Path(os.environ.get("SOFTWARE_ROOT", "/global/cfs/cdirs/m3760/xgao/software"))
script_dir = Path(__file__).resolve().parent
data_dir = Path(args.data_dir) if args.data_dir else script_dir / "data"
gauge_path = args.gauge_path or str(software_root / "Pyquda_Measurement/test_gauge/S8T32_wilson_b6.cg.1e-08.0")
lat_tag = os.environ.get("PION_TMD_LAT_TAG", "S8T32")
sm_tag = os.environ.get("PION_TMD_SM_TAG", f"1HYP_GSRC_W{args.width:g}_k0_{args.sink_interpolator}")
conf = args.config_num

q_range = range(-args.qmax, args.qmax + 1)
qext = [[x, y, 0, 0] for x in q_range for y in q_range]
qext_PDF = [[x, y, z, 0] for x in q_range for y in q_range for z in q_range]
p_2pt = [[x, y, z, 0] for x in q_range for y in q_range for z in q_range]
parameters = {
    "eta": [0],
    "b_z": args.b_z,
    "b_T": args.b_T,
    "qext": qext,
    "qext_PDF": qext_PDF,
    "pf": [0, 0, 0, 0],
    "p_2pt": p_2pt,
    "pos_boost": [0, 0, 0],
    "neg_boost": [0, 0, 0],
    "width": args.width,
    "t_insert": args.t_insert,
    "save_propagators": False,
}
pf = parameters["pf"]
pf_tag = f"PX{pf[0]}PY{pf[1]}PZ{pf[2]}dt{parameters['t_insert']}"
measurement = pion_TMD(parameters)

if getMPIComm().Get_rank() == 0:
    (data_dir / "sample_log_qtmd").mkdir(parents=True, exist_ok=True)
    (data_dir / "c2pt").mkdir(parents=True, exist_ok=True)
    (data_dir / "qTMD").mkdir(parents=True, exist_ok=True)
getMPIComm().Barrier()

if getMPIComm().Get_rank() == 0:
    print(f"--gauge_path {gauge_path}")
    print(f"--data_dir {data_dir}")
    print(f"--config_num {conf}")
    print(f"--mpi_geometry {args.mpi_geometry}")

gauge = io.readNERSCGauge(gauge_path.format(conf=conf))
gauge.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge.latt_info.t_boundary = -1
latt_info = gauge.latt_info
mpi_print(latt_info, f"DEBUG plaquette U_hyp: {gauge.plaquette()}")

xi_0, nu = 1.0, 1.0
mass = float(os.environ.get("PION_TMD_MASS", 0.236))
csw_r = float(os.environ.get("PION_TMD_CSW", 1.0372))
csw_t = csw_r
tol = float(os.environ.get("PION_TMD_TOL", 1e-15))
maxiter = int(os.environ.get("PION_TMD_MAXITER", 300))
multigrid = [[max(1, latt_info.global_size[0] // 1), max(1, latt_info.global_size[1] // 1), max(1, latt_info.global_size[2] // 2), max(1, latt_info.global_size[3] // 8)]]

dirac = core.getDirac(latt_info, mass, tol, maxiter, xi_0, csw_r, csw_t, multigrid)
dirac.loadGauge(gauge)

L = latt_info.global_size
src_shift = np.array([0, 0, 0, 0])
src_origin = np.array([int(conf) % L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin)[: args.num_src]

sample_log_file = data_dir / "sample_log_qtmd" / f"{conf}_{sm_tag}_{pf_tag}"
if latt_info.mpi_rank == 0:
    sample_log_file.touch(exist_ok=True)

sink_gamma = gamma_from_label(args.sink_interpolator)

for pos in src_positions:
    t0_pos = time.time()
    sample_log_tag = get_sample_log_tag(str(conf), pos, f"{sm_tag}_{pf_tag}")
    mpi_print(latt_info, f"START: {sample_log_tag}")

    t0 = time.time()
    srcD = source.propagator(latt_info, "point", pos)
    srcDp = boosted_smearing(srcD, w=parameters["width"], boost=parameters["pos_boost"])
    mpi_print(latt_info, f"TIME PyQUDA: Generating boosted source {time.time() - t0}s")

    t0 = time.time()
    prop_fw = core.invertPropagator(dirac, srcDp, 1, 0)
    mpi_print(latt_info, f"TIME PyQUDA: Forward propagator inversion {time.time() - t0}s")

    t0 = time.time()
    c2_tag = get_c2pt_file_tag(str(data_dir), lat_tag, conf, "CG.ex", pos, sm_tag)
    p_2pt_xyz = [[-v[0], -v[1], -v[2]] for v in parameters["p_2pt"]]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)
    measurement.contract_2pt_pion(latt_info, prop_fw, prop_fw, phases_2pt, c2_tag, src_gamma=args.src_interpolator)
    mpi_print(latt_info, f"TIME PyQUDA: Pion 2pt contraction {time.time() - t0}s")

    t0 = time.time()
    prop_sink_smeared = boosted_smearing(prop_fw.copy(), w=parameters["width"], boost=parameters["neg_boost"])
    seq_bw_prop = create_meson_bw_seq_pyquda(
        dirac,
        prop_sink_smeared,
        pos,
        parameters["pf"],
        parameters["t_insert"],
        sink_gamma,
        parameters["width"],
        parameters["pos_boost"],
    )
    mpi_print(latt_info, f"TIME PyQUDA: Pion meson sequential propagator {time.time() - t0}s")

    qext_xyz = [[v[0], v[1], v[2]] for v in parameters["qext"]]
    phases_TMD = phase.MomentumPhase(latt_info).getPhases(qext_xyz, pos)
    qext_pdf_xyz = [[v[0], v[1], v[2]] for v in parameters["qext_PDF"]]
    phases_PDF = MomentumPhase(latt_info).getPhases(qext_pdf_xyz, x0=pos)
    W_index_list_CG_dir0, W_index_list_CG_dir1 = measurement.create_TMD_Wilsonline_index_list_CG()
    W_index_list_CG = W_index_list_CG_dir0 + W_index_list_CG_dir1
    W_index_list_PDF = measurement.create_PDF_Wilsonline_index_list()

    t0 = time.time()
    pion_TMDs = measurement.contract_qTMD_CG(
        latt_info,
        prop_fw,
        seq_bw_prop,
        phases_TMD,
        W_index_list_CG_dir0,
        W_index_list_CG_dir1,
        src_gamma=args.src_interpolator,
    )
    mpi_print(latt_info, f"contract_TMD over: pion_TMDs.shape {np.shape(pion_TMDs)} {time.time() - t0}s")

    if latt_info.mpi_rank == 0:
        pion_TMDs = np.roll(pion_TMDs, -pos[3], axis=-1)
        pion_TMDs = pion_TMDs[:, :, :, : parameters["t_insert"] + 2]
        pion_TMDs = np.transpose(pion_TMDs, (0, 2, 1, 3))
    pion_TMDs = getMPIComm().bcast(pion_TMDs, root=0)

    tasks = list(range(len(my_gammas)))
    rank = latt_info.mpi_rank
    size = getMPIComm().Get_size()
    for gidx in tasks[rank::size]:
        gm = my_gammas[gidx]
        tag = get_qTMD_file_tag(
            str(data_dir),
            lat_tag,
            conf,
            "CG.ex",
            pos,
            f"{sm_tag}.{pf_tag}.{gm}",
        )
        mpi_print(latt_info, f"Saving pion qTMD gamma {gm}: {tag}")
        save_qTMD_pion_hdf5_noRoll(
            pion_TMDs[:, :, gidx : gidx + 1, :],
            tag,
            [gm],
            parameters["qext"],
            W_index_list_CG,
            parameters["t_insert"],
            latt_info,
        )

    for pdf_kind, gauge_invariant in [("GI_PDF", True), ("CG_PDF", False)]:
        t0 = time.time()
        pion_PDFs = measurement.contract_PDF(
            latt_info,
            gauge,
            prop_fw,
            seq_bw_prop,
            phases_PDF,
            W_index_list_PDF,
            src_gamma=args.src_interpolator,
            gauge_invariant=gauge_invariant,
        )
        mpi_print(latt_info, f"contract_{pdf_kind} over: pion_PDFs.shape {np.shape(pion_PDFs)} {time.time() - t0}s")

        if latt_info.mpi_rank == 0:
            pion_PDFs = np.roll(pion_PDFs, -pos[3], axis=-1)
            pion_PDFs = pion_PDFs[:, :, :, : parameters["t_insert"] + 2]
            pion_PDFs = np.transpose(pion_PDFs, (0, 2, 1, 3))
        pion_PDFs = getMPIComm().bcast(pion_PDFs, root=0)

        for gidx in tasks[rank::size]:
            gm = my_gammas[gidx]
            tag = get_qTMD_file_tag(
                str(data_dir),
                lat_tag,
                conf,
                f"{pdf_kind}.ex",
                pos,
                f"{sm_tag}.{pf_tag}.{gm}",
            )
            mpi_print(latt_info, f"Saving pion {pdf_kind} gamma {gm}: {tag}")
            save_qTMD_pion_hdf5_noRoll(
                pion_PDFs[:, :, gidx : gidx + 1, :],
                tag,
                [gm],
                parameters["qext_PDF"],
                W_index_list_PDF,
                parameters["t_insert"],
                latt_info,
            )
    sync_backend_array(prop_fw.data)

    if latt_info.mpi_rank == 0:
        with sample_log_file.open("a+") as f:
            f.write(sample_log_tag + "\n")
    mpi_print(latt_info, f"DONE: {sample_log_tag} total {time.time() - t0_pos}s")
