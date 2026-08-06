
# load python modules
import os
import time
from pathlib import Path

import numpy as np
import cupy as cp

from pyquda import init, getMPIComm
from pyquda_utils import core, io, source
from pyquda_utils.phase import MomentumPhase

from latcoding.pyquda.utils.boosted_smearing import boosted_smearing
from latcoding.pyquda.classes.pion_cg_qtmdwf_class import pion_TMDWF_measurement
from latcoding.pyquda.utils.tools import (
    append_sample_log_entry,
    mpi_print,
    read_sample_log_entries,
    srcLoc_distri_eq,
)
from latcoding.pyquda.utils.pion_utils import contract_pion_2pt
from latcoding.pyquda.utils.io_corr import get_sample_log_tag, get_c2pt_file_tag, get_qTMDWF_file_tag, save_qTMDWF_hdf5_noRoll


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=0, help="Configuration number")
parser.add_argument("--mpi_geometry", type=str, default="1.1.1.1", help="MPI geometry")
args, unknown = parser.parse_known_args()
conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

# Global parameters
repo_root = Path(__file__).resolve().parents[3]
data_dir = str(repo_root / "examples/artifacts/data")
lat_tag = "S8T32_cg" # NOTE
sm_tag = "S8T32_tmdwf_debug" # NOTE


# --------------------------
# initiate quda
# --------------------------

init(mpi_geometry, enable_mps=True)

# --------------------------
# Setup parameters
# --------------------------

parameters = {
    "eta" : [0],
    "b_T": 0,
    "b_z" : 8,
    "pzmin" : 0,
    "pzmax" : 6,
    "width" : 0,
    "pos_boost" : [0,0,0],
    "neg_boost" : [0,0,0],
    "save_propagators" : False
}
Measurement = pion_TMDWF_measurement(parameters)
gammalist = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
pt2_src = ["5", "T5"]

def source_tag(src_label):
    return f"src{src_label}"


n_src = 1 # number of sources

# --------------------------
# Load gauge and create inverter
# --------------------------

###################### load gauge ######################
Ls = 8
Lt = 32
L = [Ls, Ls, Ls, Lt]
xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.0336
csw_t = 1.0336
multigrid = None # [[8, 8, 4, 4]]
latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)

gauge = io.readNERSCGauge(str(repo_root / f"configs/S{Ls}T{Lt}_cg/wilson_b6.cg.1e-14.{conf}"))
gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)

mpi_print(latt_info, f"--lat_tag {lat_tag}")
mpi_print(latt_info, f"--sm_tag {sm_tag}")
mpi_print(latt_info, f"--config_num {conf}")
mpi_print(latt_info, f"--mpi_geometry {mpi_geometry}")
mpi_print(latt_info, f"--plaquette U_hyp: {gauge.plaquette()}")

###################### create multigrid inverter ######################

dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)


###################### setup source positions ######################
src_shift = np.array([0,0,0,0]) + np.array([7,11,13,23])
src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin) # create a list of source 4*4*4*4

src_production = src_positions[:n_src] # take the number of sources needed for this project NOTE


# --------------------------
# Start measurements
# --------------------------

###################### record the finished source position ######################
sample_log_file = data_dir + f"/sample_log/TMDWF_{sm_tag}_{conf}"
if latt_info.mpi_rank == 0:
    os.makedirs(os.path.dirname(sample_log_file), exist_ok=True)
    open(sample_log_file, "a+").close()
getMPIComm().Barrier()
boost_identity = f"pos{'_'.join(map(str, parameters['pos_boost']))}.neg{'_'.join(map(str, parameters['neg_boost']))}"
sample_identity = f"{sm_tag}.src{'-'.join(pt2_src)}.{boost_identity}"


completed_samples = read_sample_log_entries(sample_log_file) if latt_info.mpi_rank == 0 else None
completed_samples = getMPIComm().bcast(completed_samples, root=0)

#! Measurement
###################### loop over sources ######################
for ipos, pos in enumerate(src_production):
    
    sample_log_tag = get_sample_log_tag("ex", pos, sample_identity)
    mpi_print(latt_info, f"Contraction START: {sample_log_tag}")
    
    if sample_log_tag in completed_samples:
        mpi_print(latt_info, f"Contraction SKIP: {sample_log_tag}")
        continue

    #>>>>>>>>>>>>>>>>>>>>>>>>> Propagators <<<<<<<<<<<<<<<<<<<<<<<<<<#

    # get forward and backward propagator boosted source
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    srcD = source.propagator(latt_info, "point", pos)
    srcDp = boosted_smearing(srcD, w=parameters["width"], boost=parameters["pos_boost"])
    same_source_smearing = parameters["width"] == 0 or np.array_equal(
        parameters["pos_boost"],
        parameters["neg_boost"],
    )
    srcDm = None
    if not same_source_smearing:
        srcDm = boosted_smearing(srcD, w=parameters["width"], boost=parameters["neg_boost"])
    cp.cuda.runtime.deviceSynchronize()
    mpi_print(latt_info, f"TIME Pyquda: Generating boosted sources {time.time() - t0}")

    # Reuse one inversion unless the quark and antiquark sources differ.
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    dirac.loadGauge(gauge)
    propag_f = core.invertPropagator(dirac, srcDp, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
    if same_source_smearing:
        propag_b = propag_f
        inversion_count = 1
    else:
        propag_b = core.invertPropagator(dirac, srcDm, 1, 0)
        inversion_count = 2
    cp.cuda.runtime.deviceSynchronize()
    mpi_print(latt_info, f"TIME: Pyquda inversion * {inversion_count} {time.time() - t0}")

    #! PyQUDA: contract 2pt TMD
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    p_2pt_xyz = [[0, 0, -v] for v in range(parameters["pzmin"], parameters["pzmax"])]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)
    for src_label in pt2_src:
        tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, f"{sm_tag}.{source_tag(src_label)}")
        Measurement.contract_2pt_pion(
            latt_info,
            propag_f,
            propag_b,
            phases_2pt,
            tag,
            src_gamma=src_label,
        )

    cp.cuda.runtime.deviceSynchronize()
    mpi_print(latt_info, f"TIME Pyquda: Contraction 2pt (includes sink smearing) {time.time() - t0}")
    
    # SP TMDWF contraction
    mpi_print(latt_info, f"Contraction: Start TMDWF: CG no links")
    t0_contract = time.time()
    tmdwf_collect_by_src = {src_label: [] for src_label in pt2_src} # [WL_indices][p][gamma][tau]

    #>>>>>>>>>>>>>>>>>>>>>>>>> CG TMD <<<<<<<<<<<<<<<<<<<<<<<<<<#

    # C^{(g)}(q,t) = sum_{x} exp(i q·x) Tr_{c,s}[ γ5 S_b(x)^\dagger γ5 Γ_g F(x) interpolator ]

    # prepare the TMD separate indices for CG
    W_index_list_CG_dir0, W_index_list_CG_dir1 = Measurement.create_TMD_Wilsonline_index_list_CG()
    W_index_list_CG = W_index_list_CG_dir0 + W_index_list_CG_dir1
    
    #! PyQUDA: contract TMD
    mpi_print(latt_info, f"contract_TMD loop: CG no links")
    t0_contract = time.time()
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()



    #! PyQUDA: contract TMD +X direction
    tmd_backward_prop_dir0 = propag_b.copy()
    for iW, WL_indices in enumerate(W_index_list_CG_dir0):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        mpi_print(latt_info, f"TIME PyQUDA: contract TMD {iW+1}/{len(W_index_list_CG)} {WL_indices}")
        if iW == 0:
            WL_indices_previous = [0, 0, 0, 0]
        else:
            WL_indices_previous = W_index_list_CG_dir0[iW - 1]

        tmd_backward_prop_dir0 = Measurement.create_fw_prop_TMD_CG(tmd_backward_prop_dir0, WL_indices, WL_indices_previous) #! note here [WL_indices] is changed to WL_indices for PyQUDA, and prop_exact_f is changed to propag
        cp.cuda.runtime.deviceSynchronize()
        mpi_print(latt_info, f"TIME PyQUDA: cshift {time.time() - t0}")

        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        for src_label in pt2_src:
            corr = contract_pion_2pt(latt_info, propag_f, tmd_backward_prop_dir0, phases_2pt, src_gamma=src_label)
            if latt_info.mpi_rank == 0:
                corr = np.transpose(corr, (1, 0, 2))
            tmdwf_collect_by_src[src_label].append(corr)

        cp.cuda.runtime.deviceSynchronize()
        mpi_print(latt_info, f"TIME PyQUDA: contract TMDWF {time.time() - t0}")
    del tmd_backward_prop_dir0
        
    #! PyQUDA: contract TMD +Y direction
    tmd_backward_prop_dir1 = propag_b.copy()
    for iW, WL_indices in enumerate(W_index_list_CG_dir1):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        if latt_info.mpi_rank == 0:
            print(f"TIME PyQUDA: contract TMD {iW+1+len(W_index_list_CG_dir0)}/{len(W_index_list_CG)} {WL_indices}")
        if iW == 0:
            WL_indices_previous = [0, 0, 0, 0]
        else:
            WL_indices_previous = W_index_list_CG_dir1[iW - 1]

        tmd_backward_prop_dir1 = Measurement.create_fw_prop_TMD_CG(tmd_backward_prop_dir1, WL_indices, WL_indices_previous) #! note here [WL_indices] is changed to WL_indices for PyQUDA, and prop_exact_f is changed to propag
        cp.cuda.runtime.deviceSynchronize()
        mpi_print(latt_info, f"TIME PyQUDA: cshift {time.time() - t0}")

        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        for src_label in pt2_src:
            corr = contract_pion_2pt(latt_info, propag_f, tmd_backward_prop_dir1, phases_2pt, src_gamma=src_label)
            if latt_info.mpi_rank == 0:
                corr = np.transpose(corr, (1, 0, 2))
            tmdwf_collect_by_src[src_label].append(corr)

        cp.cuda.runtime.deviceSynchronize()
        mpi_print(latt_info, f"TIME PyQUDA: contract TMDWF {time.time() - t0}")
    del tmd_backward_prop_dir1
    
    for src_label in pt2_src:
        tmdwf_collect_by_src[src_label] = np.array(tmdwf_collect_by_src[src_label]) # shape (N_W, N_pz, N_gamma, N_t)
        mpi_print(latt_info, f"TIME contract_TMDWF: {source_tag(src_label)} shape {np.shape(tmdwf_collect_by_src[src_label])} {time.time()-t0_contract}s")

    #>>>>>>>>>>>>>>>>>>>>>>>>> Save correlators <<<<<<<<<<<<<<<<<<<<<<<<<<#
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    # reorder gamma, and cut useful tau in [src_t, src_t+tsep+2)
    for src_label, TMDWF_collect in tmdwf_collect_by_src.items():
        if latt_info.mpi_rank != 0:
            continue
        TMDWF_collect = np.roll(TMDWF_collect, -pos[3], axis=-1)
        qTMDWF_tag = get_qTMDWF_file_tag(data_dir, lat_tag, conf, "ex", pos, f"{sm_tag}.src{src_label}")
        mpi_print(latt_info, f"Saving all qTMDWF gamma channels: {qTMDWF_tag}")
        save_qTMDWF_hdf5_noRoll(
            TMDWF_collect,
            qTMDWF_tag,
            gammalist,
            [[0, 0, p, 0] for p in range(parameters["pzmin"], parameters["pzmax"])],
            W_index_list_CG,
            attrs={
                "source_interpolator": src_label,
                "spectator_boost": parameters["pos_boost"],
                "active_boost": parameters["neg_boost"],
            },
        )
    cp.cuda.runtime.deviceSynchronize()
    mpi_print(latt_info, f"TIME: save TMDs {time.time() - t0}")
    mpi_print(latt_info, "Contraction: Done TMDWF: CG no links")
    

    if latt_info.mpi_rank == 0:
        append_sample_log_entry(sample_log_file, sample_log_tag)
    completed_samples.add(sample_log_tag)

    mpi_print(latt_info, f"DONE: {sample_log_tag}")
