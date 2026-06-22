
# load python modules
import os
import time
import numpy as np
import cupy as cp
from opt_einsum import contract

from pyquda import init, getMPIComm
from pyquda_utils import core, io, source
from pyquda_utils.phase import MomentumPhase

from latcoding.pyquda.utils.boosted_smearing import boosted_smearing
from latcoding.pyquda.classes.pion_cg_qtmdwf_class import pion_TMDWF_measurement, my_pyquda_gammas
from latcoding.pyquda.utils.tools import gamma_matrix_to_backend, srcLoc_distri_eq, mpi_print
from latcoding.pyquda.utils.io_corr import get_sample_log_tag, get_c2pt_file_tag, get_qTMDWF_file_tag, save_qTMDWF_hdf5_noRoll


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=0, help="Configuration number")
parser.add_argument("--mpi_geometry", type=str, default="1.1.1.1", help="MPI geometry")
args, unknown = parser.parse_known_args()
conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

# Global parameters
data_dir="/home/jinchen/git/lat-software/LatCoding/examples/artifacts/data" # NOTE
lat_tag = "S16T16_cg" # NOTE
sm_tag = "S16T16_einsum" # NOTE


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
    "pzmax" : 1,
    "width" : 0,
    "pos_boost" : [0,0,0],
    "neg_boost" : [0,0,0],
    "save_propagators" : False
}
Measurement = pion_TMDWF_measurement(parameters)
xp = cp
gammalist = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
src_mode = "fixed"
pt2_src = ["5", "T5"]

def source_tag(src_label):
    return f"{src_mode}_src{src_label}"


def gamma_from_label(src_label):
    if src_label not in gammalist:
        raise ValueError(f"Invalid pion source interpolator: {src_label}. Expected one of {gammalist}.")
    return my_pyquda_gammas[gammalist.index(src_label)]


interpolator_by_src = {
    src_label: gamma_matrix_to_backend(gamma_from_label(src_label), xp)
    for src_label in pt2_src
}
G5 = gamma_matrix_to_backend(gamma_from_label("5"), xp)
n_src = 1 # number of sources

# --------------------------
# Load gauge and create inverter
# --------------------------

###################### load gauge ######################
Ls = 16
Lt = 16
L = [Ls, Ls, Ls, Lt]
xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.0336
csw_t = 1.0336
multigrid = None # [[8, 8, 4, 4]]
latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)

gauge = io.readNERSCGauge(f"/home/jinchen/git/lat-software/LatCoding/configs/S{Ls}T{Lt}_cg/wilson_b6.cg.1e-14.{conf}")
# gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)

mpi_print(latt_info, f"--lat_tag {lat_tag}")
mpi_print(latt_info, f"--sm_tag {sm_tag}")
mpi_print(latt_info, f"--config_num {conf}")
mpi_print(latt_info, f"--mpi_geometry {mpi_geometry}")
mpi_print(latt_info, f"--plaquette U_hyp: {gauge.plaquette()}")

###################### create multigrid inverter ######################

dirac = core.getClover(latt_info, mass, 1e-10, 10000, xi_0, csw_r, csw_t, multigrid)


###################### prepare gamma list ######################
# use the first gamma's dtype and device to allocate the container
first_gamma = gamma_matrix_to_backend(my_pyquda_gammas[0], xp)
n_gamma = len(my_pyquda_gammas)
    
pyquda_gamma_ls = xp.empty(
    (n_gamma,) + first_gamma.shape,
    dtype=first_gamma.dtype,
)       
for gamma_idx, gamma_pyq in enumerate(my_pyquda_gammas):
    pyquda_gamma_ls[gamma_idx] = gamma_matrix_to_backend(gamma_pyq, xp, dtype=first_gamma.dtype)

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
    f = open(sample_log_file, "a+")
    f.close()
time.sleep(1)

#! Measurement
###################### loop over sources ######################
for ipos, pos in enumerate(src_production):
    
    sample_log_tag = get_sample_log_tag("ex", pos, sm_tag)
    mpi_print(latt_info, f"Contraction START: {sample_log_tag}")
    # with open(sample_log_file, "a+") as f:
    #     f.seek(0)
    #     if sample_log_tag in f.read():
    #         mpi_print(latt_info, f"Contraction SKIP: {sample_log_tag}")
    #         continue

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
        interpolator = interpolator_by_src[src_label]
        tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, f"{sm_tag}.{source_tag(src_label)}")
        Measurement.contract_2pt_pion(
            latt_info,
            propag_f,
            propag_b,
            phases_2pt,
            tag,
            src_mode=src_mode,
            pion_interpolator=interpolator,
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

    #! PyQUDA: prepare the source-interpolator-dependent part of the contraction for TMDWF
    G16_fw_interpolator_by_src = {}
    for src_label in pt2_src:
        interpolator = interpolator_by_src[src_label]
        fw_interpolator = contract("wtzyxilab, lj -> wtzyxijab", propag_f.data, interpolator)
        G16_fw_interpolator_by_src[src_label] = contract("gim, wtzyxmjab -> gwtzyxijab", pyquda_gamma_ls, fw_interpolator)
        del fw_interpolator


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
        temp0 = contract("ki, wtzyxklab, jl -> wtzyxjiba", G5, tmd_backward_prop_dir0.data.conj(), G5)
        for src_label in pt2_src:
            temp1 = contract("wtzyxjiba, gwtzyxijab -> gwtzyx", temp0, G16_fw_interpolator_by_src[src_label])
            temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_2pt, temp1).get(), [2, -1, -1, -1])
            tmdwf_collect_by_src[src_label].append(temp2)
            del temp1, temp2
    
        cp.cuda.runtime.deviceSynchronize()
        mpi_print(latt_info, f"TIME PyQUDA: contract TMDWF {time.time() - t0}")
        del temp0
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
        temp0 = contract("ki, wtzyxklab, jl -> wtzyxjiba", G5, tmd_backward_prop_dir1.data.conj(), G5)
        for src_label in pt2_src:
            temp1 = contract("wtzyxjiba, gwtzyxijab -> gwtzyx", temp0, G16_fw_interpolator_by_src[src_label])
            temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_2pt, temp1).get(), [2, -1, -1, -1])
            tmdwf_collect_by_src[src_label].append(temp2)
            del temp1, temp2
        
        cp.cuda.runtime.deviceSynchronize()
        mpi_print(latt_info, f"TIME PyQUDA: contract TMDWF {time.time() - t0}")
        del temp0
    del tmd_backward_prop_dir1
    
    for src_label in pt2_src:
        tmdwf_collect_by_src[src_label] = np.array(tmdwf_collect_by_src[src_label]) # shape (N_W, N_pz, N_gamma, N_t)
        mpi_print(latt_info, f"TIME contract_TMDWF: {source_tag(src_label)} shape {np.shape(tmdwf_collect_by_src[src_label])} {time.time()-t0_contract}s")
    del G16_fw_interpolator_by_src

    #>>>>>>>>>>>>>>>>>>>>>>>>> Save correlators <<<<<<<<<<<<<<<<<<<<<<<<<<#
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    # reorder gamma, and cut useful tau in [src_t, src_t+tsep+2)
    for src_label, TMDWF_collect in tmdwf_collect_by_src.items():
        if latt_info.mpi_rank == 0:
            TMDWF_collect = np.roll(TMDWF_collect, -pos[3], axis=-1)
        TMDWF_collect = getMPIComm().bcast(TMDWF_collect, root=0)
        #! parallel the io through gamma
        tasks = list(range(len(gammalist)))
        rank = latt_info.mpi_rank
        size = getMPIComm().Get_size()
        for gidx in tasks[rank::size]:
            gm = gammalist[gidx]
            qTMDWF_tag = get_qTMDWF_file_tag(data_dir, lat_tag, conf, "ex", pos, f"{sm_tag}.{source_tag(src_label)}.snk{gm}")
            print(f"DEBUG: rank {rank}, {qTMDWF_tag}")
            data = TMDWF_collect[:, :, gidx:gidx+1, :] #! shape (N_W, N_pz, gm, N_t)
            save_qTMDWF_hdf5_noRoll(data, qTMDWF_tag, [gm], [[0, 0, p, 0] for p in range(parameters["pzmin"], parameters["pzmax"])], W_index_list_CG)
        cp.cuda.runtime.deviceSynchronize()
    mpi_print(latt_info, f"TIME: save TMDs {time.time() - t0}")
    mpi_print(latt_info, "Contraction: Done TMDWF: CG no links")
    

    with open(sample_log_file, "a+") as f:
        if latt_info.mpi_rank == 0:
            f.write(sample_log_tag+"\n")

    mpi_print(latt_info, f"DONE: {sample_log_tag}")
