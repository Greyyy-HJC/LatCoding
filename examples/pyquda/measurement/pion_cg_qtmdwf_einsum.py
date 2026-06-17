
# load python modules
import time
import numpy as np
import cupy as cp
from opt_einsum import contract

from pyquda import init, getMPIComm
from pyquda_utils import core, gamma, io, source
from pyquda_utils.phase import MomentumPhase

from latcoding.pyquda.utils.boosted_smearing import boosted_smearing
from latcoding.pyquda.classes.pion_cg_qtmdwf_class import pion_TMDWF_measurement, my_pyquda_gammas
from latcoding.pyquda.utils.tools import srcLoc_distri_eq, mpi_print
from latcoding.pyquda.utils.io_corr import get_sample_log_tag, get_c2pt_file_tag, get_qTMDWF_file_tag, save_qTMDWF_hdf5_noRoll


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=0, help="Configuration number")
parser.add_argument("--mpi_geometry", type=str, default="1.1.1.1", help="MPI geometry")
args, unknown = parser.parse_known_args()
conf = args.config_num
mpi_geometry = [int(i) for i in args.mpi_geometry.split(".")]

# Global parameters
data_dir="/ccs/home/xiangg/latwork/l64c64a076/qTMDWF_pyquda/data" # NOTE
lat_tag = "l64c64a076" # NOTE
interpolation = "5" # NOTE, new interpolation operator
sm_tag = "1HYP_M140_GSRC_W45_k6_einsum" # NOTE


# --------------------------
# initiate quda
# --------------------------

init(mpi_geometry, enable_mps=True)
G5 = gamma.gamma(15)
Gsrc = gamma.gamma(15)

# --------------------------
# Setup parameters
# --------------------------

parameters = {
    "eta" : [0],
    "b_T": 20,
    "b_z" : 20,
    "pzmin" : 4,
    "pzmax" : 11,
    "width" : 4.5,
    "pos_boost" : [0,0,6],
    "neg_boost" : [0,0,-6],
    "save_propagators" : False
}
Measurement = pion_TMDWF_measurement(parameters)
xp = cp
gammalist = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]


# --------------------------
# Load gauge and create inverter
# --------------------------

###################### load gauge ######################
Ls = 64
Lt = 64
L = [Ls, Ls, Ls, Lt]
xi_0, nu = 1.0, 1.0
mass = -0.049 # kappa = 0.12623
csw_r = 1.0372
csw_t = 1.0372
multigrid = [[8, 8, 4, 4]]
latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)

gauge = io.readNERSCGauge(f"/ccs/home/xiangg/latwork/l64c64a076/nucleon_TMD_noGPT/gauge/l6464f21b7130m00119m0322a.1050.coulomb.1e-14.HYP")
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
first_gamma = my_pyquda_gammas[0]
n_gamma = len(my_pyquda_gammas)
    
pyquda_gamma_ls = xp.empty(
    (n_gamma,) + first_gamma.shape,
    dtype=first_gamma.dtype,
)       
for gamma_idx, gamma_pyq in enumerate(my_pyquda_gammas):
    pyquda_gamma_ls[gamma_idx] = gamma_pyq

###################### setup source positions ######################
src_shift = np.array([0,0,0,0]) + np.array([7,11,13,23])
src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin) # create a list of source 4*4*4*4

src_production = src_positions[0:1] # take the number of sources needed for this project NOTE


# --------------------------
# Start measurements
# --------------------------

###################### record the finished source position ######################
sample_log_file = data_dir + f"/sample_log/TMDWF_{sm_tag}_{conf}"
if latt_info.mpi_rank == 0:
    f = open(sample_log_file, "a+")
    f.close()
time.sleep(2)

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
    srcDm = boosted_smearing(srcD, w=parameters["width"], boost=parameters["neg_boost"])
    cp.cuda.runtime.deviceSynchronize()
    mpi_print(latt_info, f"TIME Pyquda: Generatring boosted src {time.time() - t0}")

    # get forward propagator: smeared-point
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    dirac.loadGauge(gauge) #TODO: debug
    propag_f = core.invertPropagator(dirac, srcDp, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
    propag_b = core.invertPropagator(dirac, srcDm, 1, 0)
    cp.cuda.runtime.deviceSynchronize()
    mpi_print(latt_info, f"TIME: Pyquda inversion * 2 {time.time() - t0}")

    #! PyQUDA: contract 2pt TMD
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    p_2pt_xyz = [[0, 0, -v] for v in range(parameters["pzmin"], parameters["pzmax"])]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)
    Measurement.contract_2pt_pion(latt_info, propag_f, propag_b, phases_2pt, tag)

    cp.cuda.runtime.deviceSynchronize()
    mpi_print(latt_info, f"TIME Pyquda: Contraction 2pt (includes sink smearing) {time.time() - t0}")
    
    # SP TMDWF contraction
    mpi_print(latt_info, f"Contraction: Start TMDWF: CG no links")
    t0_contract = time.time()
    TMDWF_collect_src5 = [] # [WL_indices][p][gamma][tau]

    #>>>>>>>>>>>>>>>>>>>>>>>>> CG TMD <<<<<<<<<<<<<<<<<<<<<<<<<<#

    # C^{(g)}(q,t) = sum_{x} exp(i q·x) Tr_{c,s}[ γ5 S_b(x)^\dagger γ5 Γ_g F(x) G_src ]

    # prepare the TMD separate indices for CG
    W_index_list_CG_dir0, W_index_list_CG_dir1 = Measurement.create_TMD_Wilsonline_index_list_CG()
    W_index_list_CG = W_index_list_CG_dir0 + W_index_list_CG_dir1
    
    #! PyQUDA: contract TMD
    mpi_print(latt_info, f"contract_TMD loop: CG no links")
    t0_contract = time.time()
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()

    #! PyQUDA: prepare the common part of the contraction for TMDWF
    fw_Gsrc = contract("wtzyxilab, lj -> wtzyxijab", propag_f.data, Gsrc)
    G16_fw_Gsrc = contract("gim, wtzyxmjab -> gwtzyxijab", pyquda_gamma_ls, fw_Gsrc) 


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
        temp1 = contract("wtzyxjiba, gwtzyxijab -> gwtzyx", temp0, G16_fw_Gsrc)
        temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_2pt, temp1).get(), [2, -1, -1, -1])
        TMDWF_collect_src5.append(temp2)
    
        cp.cuda.runtime.deviceSynchronize()
        mpi_print(latt_info, f"TIME PyQUDA: contract TMDWF {time.time() - t0}")
        del temp0, temp1, temp2
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
        temp1 = contract("wtzyxjiba, gwtzyxijab -> gwtzyx", temp0, G16_fw_Gsrc)
        temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_2pt, temp1).get(), [2, -1, -1, -1])
        TMDWF_collect_src5.append(temp2)
        
        cp.cuda.runtime.deviceSynchronize()
        mpi_print(latt_info, f"TIME PyQUDA: contract TMDWF {time.time() - t0}")
        del temp0, temp1, temp2
    del tmd_backward_prop_dir1
    
    TMDWF_collect_src5 = np.array(TMDWF_collect_src5) # shape (N_W, N_pz, N_gamma, N_t)
    mpi_print(latt_info, f"TIME contract_TMDWF: TMDWF_collect.shape {np.shape(TMDWF_collect_src5)} {time.time()-t0_contract}s")

    #>>>>>>>>>>>>>>>>>>>>>>>>> Save correlators <<<<<<<<<<<<<<<<<<<<<<<<<<#
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    # reorder gamma, and cut useful tau in [src_t, src_t+tsep+2)
    srclist = ['src5']
    for isrc, TMDWF_collect in enumerate([TMDWF_collect_src5]):
        if latt_info.mpi_rank == 0:
            TMDWF_collect = np.roll(TMDWF_collect, -pos[3], axis=-1)
        TMDWF_collect = getMPIComm().bcast(TMDWF_collect, root=0)
        #! parallel the io through gamma
        tasks = []
        for gidx in range(len(gammalist)):
            tasks.append(gidx)
        rank = latt_info.mpi_rank
        if rank < len(tasks):
            gidx = tasks[rank]
            gm = gammalist[gidx]
            qTMDWF_tag = get_qTMDWF_file_tag(data_dir, lat_tag, conf, "ex", pos, f"{sm_tag}.{srclist[isrc]}.O{gm}")
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
