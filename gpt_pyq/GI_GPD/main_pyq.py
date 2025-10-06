
# load python modules
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
import os
import time
from mpi4py import MPI
import math

# load gpt modules
import gpt as g 
from PyQUDA_proton_qTMD_draft import proton_TMD, pyq_gamma_order, my_gammas  #! gamma basis information for PyQUDA contractions
from tools import *
from io_corr import *

# load pyquda modules
from pyquda import init, LatticeInfo, getMPIComm
from pyquda_utils import core, gpt, gamma, phase
from pyquda_plugins import pycontract #todo: for PyQUDA contraction v2
from pyquda_utils.core import X, Y, Z, T

# Global parameters
data_dir="data" # NOTE
lat_tag = "l64c64a076" # NOTE
interpolation = "T5" # NOTE, new interpolation operator
sm_tag = "1HYP_GSRC_W90_k3_"+interpolation # NOTE
GEN_SIMD_WIDTH = 64
conf = g.default.get_int("--config_num", 0)
g.message(f"--lat_tag {lat_tag}")
g.message(f"--sm_tag {sm_tag}")
g.message(f"--config_num {conf}")


# --------------------------
# initiate quda
# --------------------------
mpi_geometry = [1, 1, 1, 1]
init(mpi_geometry, resource_path=".cache")

G5 = gamma.gamma(15)

# --------------------------
# Setup parameters
# --------------------------
parameters = {
    
    # NOTE:
    "eta": [0],  # irrelavant for CG TMD
    "b_z": 2,
    "b_T": 2,

    # "qext": [[x,y,z,0] for x in [-2] for y in [-2] for z in [-1]], # momentum transfer for TMD, pf = pi + q
    "qext": [list(v + (0,)) for v in {tuple(sorted((x, y, z))) for x in [0] for y in [0] for z in [1]}], # momentum transfer for TMD, pf = pi + q
    "qext_PDF": [[x,y,z,0] for x in [1] for y in [1] for z in [2]], # momentum transfer for PDF, not used 
    "pf": [0,0,7,0],
    "p_2pt": [[x,y,z,0] for x in [1] for y in [1] for z in [7]], # 2pt momentum, should match pf & pi

    "boost_in": [0,0,3],
    "boost_out": [0,0,3],
    "width" : 9.0,

    "pol": ["PpUnpol"],
    "t_insert": 6, # time separation for TMD

    "save_propagators": False,
}
pf = parameters["pf"]
pf_tag = "PX"+str(pf[0]) + "PY"+str(pf[1]) + "PZ"+str(pf[2]) + "dt" + str(parameters["t_insert"])
gammalist = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
Measurement = proton_TMD(parameters)


#todo: test the .shift() method in PyQUDA and g.cshift() in GPT
def test_shift(prop_f_pyq):
    Xdir = 0
    Zdir = 2
    prop_shiftx_pyq = prop_f_pyq.shift(1, Xdir)
    prop_shiftz_pyq = prop_f_pyq.shift(1, Zdir)
    
    prop_f_gpt = g.mspincolor(grid)
    gpt.LatticePropagatorGPT(prop_f_gpt, GEN_SIMD_WIDTH, prop_f_pyq)
    
    prop_shiftx_gpt = g.eval(g.cshift(prop_f_gpt,Xdir,1))
    prop_shiftz_gpt = g.eval(g.cshift(prop_f_gpt,Zdir,1))
    
    prop_shiftx_gpt_pyq = gpt.LatticePropagatorGPT(prop_shiftx_gpt, GEN_SIMD_WIDTH)
    prop_shiftz_gpt_pyq = gpt.LatticePropagatorGPT(prop_shiftz_gpt, GEN_SIMD_WIDTH)
    
    diffx = prop_shiftx_gpt_pyq.data - prop_shiftx_pyq.data
    diffz = prop_shiftz_gpt_pyq.data - prop_shiftz_pyq.data
    
    g.message(f"DEBUG: Max difference in x direction: {np.max(np.abs(diffx))}")
    g.message(f"DEBUG: Max difference in z direction: {np.max(np.abs(diffz))}")
    
    return None


# --------------------------
# Load gauge and create inverter
# --------------------------

###################### load gauge ######################
Ls = 8
Lt = 32
conf = 0
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
U = g.convert( g.load(f"../../conf/S8T32_cg/gauge/wilson_b6.cg.1e-08.{conf}"), g.double )

g.mem_report(details=False)
L = U[0].grid.fdimensions
U_prime, trafo = g.gauge_fix(U, maxiter=5000, prec=1e-8) # CG fix, to get trafo

latt_info, gpt_latt, gpt_simd, gpt_prec = gpt.LatticeInfoGPT(U[0].grid, GEN_SIMD_WIDTH)
gauge = gpt.LatticeGaugeGPT(U_prime, GEN_SIMD_WIDTH)

###################### create multigrid inverter ######################
latt_info = LatticeInfo([Ls, Ls, Ls, Lt], -1, 1.0)
dirac = core.getDirac(latt_info, -0.049, 1e-10,  5000, 1.0, 1.0372, 1.0372) # remove the last two arguments for BiCGStab; S mass -0.015, U/D mass -0.049
# g.message("DEBUG plaquette U_prime:", g.qcd.gauge.plaquette(U_prime))
# g.message("DEBUG plaquette gauge:", gauge.plaquette())
# gauge.projectSU3(1e-15) #todo: modified by Jinchen, for the new version of pyquda
dirac.loadGauge(gauge)


src_pos = [1,2,3,4]

# --------------------------
# Start measurements
# --------------------------

#! Measurement
W_index_list_PDF = Measurement.create_PDF_Wilsonline_index_list(U[0].grid)

srcDp = Measurement.create_src_2pt(src_pos, trafo, U[0].grid)
b = gpt.LatticePropagatorGPT(srcDp, GEN_SIMD_WIDTH)
b.toDevice()
propag = core.invertPropagator(dirac, b, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
prop_exact_f = g.mspincolor(grid)
gpt.LatticePropagatorGPT(prop_exact_f, GEN_SIMD_WIDTH, propag)

sequential_bw_prop_down_pyq = Measurement.create_bw_seq_Pyquda_pyquda(dirac, prop_exact_f, trafo, 2, src_pos, interpolation) # NOTE, this is a list of propagators for each proton polarization
sequential_bw_prop_up_pyq = Measurement.create_bw_seq_Pyquda_pyquda(dirac, prop_exact_f, trafo, 1, src_pos, interpolation) # NOTE, this is a list of propagators for each proton polarization

qext_pdf_xyz = [[v[0], v[1], v[2]] for v in parameters["qext_PDF"]]
phases_pdf_pyq = phase.MomentumPhase(latt_info).getPhases(qext_pdf_xyz, src_pos)

cp.cuda.runtime.deviceSynchronize()
t0 = time.time()
sequential_prop_down = contract(
    "ij, pwtzyxilab, kl -> pwtzyxkjba",
    G5,
    sequential_bw_prop_down_pyq.conj(),
    G5,
)
sequential_prop_up = contract(
    "ij, pwtzyxilab, kl -> pwtzyxkjba",
    G5,
    sequential_bw_prop_up_pyq.conj(),
    G5,
)
t_seq = time.time() - t0
g.message(f"TIME PyQUDA: create_bw_seq", t_seq)

g.message("\ncontract_PDF loop: GI with links")


tmd_forward_prop_pyq = propag.copy()

for iW, WL_indices in enumerate(W_index_list_PDF):
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    temp_down = []
    for seq in sequential_prop_down:
        seq_lp = core.LatticePropagator(latt_info, seq)
        temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop_pyq, seq_lp, gamma.Gamma(0)).data
        contracted = contract("qwtzyx, gwtzyx -> qgt", phases_pdf_pyq, temp1).get()
        gathered = core.gatherLattice(contracted, [2, -1, -1, -1])
        temp_down.append(np.asarray(gathered))

    temp_up = []
    for seq in sequential_prop_up:
        seq_lp = core.LatticePropagator(latt_info, seq)
        temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop_pyq, seq_lp, gamma.Gamma(0)).data
        contracted = contract("qwtzyx, gwtzyx -> qgt", phases_pdf_pyq, temp1).get()
        gathered = core.gatherLattice(contracted, [2, -1, -1, -1])
        temp_up.append(np.asarray(gathered))
    
    cp.cuda.runtime.deviceSynchronize()
    t_contract = time.time() - t0
    g.message(f"TIME PyQUDA: contract_PDF {iW+1}/{len(W_index_list_PDF)} {WL_indices}", t_contract)
    
    
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    
    for spin in range(4):
        for color in range(3):
            fermion = tmd_forward_prop_pyq.getFermion(spin, color)
            fermion_shift = gauge.pure_gauge.covDev(fermion, 2)
            tmd_forward_prop_pyq.setFermion(fermion_shift, spin, color)

    cp.cuda.runtime.deviceSynchronize()
    t_shift = time.time() - t0
    g.message(f"TIME PyQUDA: create_fw_prop_PDF {iW+1}/{len(W_index_list_PDF)} {WL_indices}", t_shift)
    


    temp_down = np.stack(temp_down, axis=0)
    temp_up = np.stack(temp_up, axis=0)

    temp_down = temp_down[:, :, pyq_gamma_order, :]
    temp_up = temp_up[:, :, pyq_gamma_order, :]

    qtmd_tag_exact_D = get_qTMD_file_tag(data_dir, lat_tag, conf, "GI_PDF.D.ex", src_pos, sm_tag + '.' + pf_tag)
    qtmd_tag_exact_U = get_qTMD_file_tag(data_dir, lat_tag, conf, "GI_PDF.U.ex", src_pos, sm_tag + '.' + pf_tag)

    for pol_idx, pol in enumerate(Measurement.pol_list):
        pol_tag_D = qtmd_tag_exact_D + "." + pol
        pol_tag_U = qtmd_tag_exact_U + "." + pol
        corr_down = [temp_down[pol_idx]]
        corr_up = [temp_up[pol_idx]]
        if g.rank() == pol_idx:
            save_qTMD_proton_hdf5_subset(corr_down, pol_tag_D, my_gammas, Measurement.qlist, [WL_indices], iW, Measurement.t_insert)
            save_qTMD_proton_hdf5_subset(corr_up, pol_tag_U, my_gammas, Measurement.qlist, [WL_indices], iW, Measurement.t_insert)

g.message("\ncontract_PDF DONE: GI with links")
