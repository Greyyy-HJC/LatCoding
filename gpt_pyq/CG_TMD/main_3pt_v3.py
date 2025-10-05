# load python modules
import numpy as np
import cupy as cp
from opt_einsum import contract

import time
import os

# load gpt modules
import gpt as g 
from Proton_qTMD_utils import proton_TMD
from tools import *
from io_corr import *

# load pyquda modules
from pyquda import init, LatticeInfo
from pyquda_utils import core, gpt, gamma, phase
from pyquda_utils.core import X, Y, Z, T
from pyquda_plugins import pycontract

# Gobal parameters
data_dir="data" # NOTE
lat_tag = "l64c64a076" # NOTE
sm_tag = "1HYP_GSRC_W90_k3_Z5" # NOTE
interpolation = "5" # NOTE, new interpolation operator
GEN_SIMD_WIDTH = 64
g.message(f"--lat_tag {lat_tag}")
g.message(f"--sm_tag {sm_tag}")


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

    "qext": [list(v + (0,)) for v in {tuple(sorted((x, y, z))) for x in [-2] for y in [-1] for z in [-1]}], # momentum transfer for TMD, pf = pi + q
    "qext_PDF": [[x,y,z,0] for x in [0] for y in [0] for z in [0]], # momentum transfer for PDF, not used 
    "pf": [0,0,7,0],
    "p_2pt": [[x,y,z,0] for x in [1] for y in [1] for z in [7]], # 2pt momentum, should match pf & pi

    "boost_in": [0,0,3],
    "boost_out": [0,0,3],
    "width" : 9.0,

    "pol": ["PpUnpol"],
    "t_insert": 4, # time separation for TMD

    "save_propagators": False,
}
pf = parameters["pf"]
pf_tag = "PX"+str(pf[0]) + "PY"+str(pf[1]) + "PZ"+str(pf[2]) + "dt" + str(parameters["t_insert"])
gammalist = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
pyq_gammalist = [gamma.gamma(15), gamma.gamma(8), gamma.gamma(7), gamma.gamma(1), gamma.gamma(14), gamma.gamma(2), gamma.gamma(13), gamma.gamma(4), gamma.gamma(11), gamma.gamma(0), gamma.gamma(9), gamma.gamma(3), gamma.gamma(5), gamma.gamma(10), gamma.gamma(6), gamma.gamma(12)]
pyq_gamma_order = [15, 8, 7, 1, 14, 2, 13, 4, 11, 0, 9, 3, 5, 10, 6, 12]


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

Measurement = proton_TMD(parameters)


pyquda_gamma_ls = cp.zeros((16, 4, 4), "<c16")
for gamma_idx, gamma_pyq in enumerate(pyq_gammalist):
    pyquda_gamma_ls[gamma_idx] = gamma_pyq


def create_fw_prop_TMD_CG_pyquda(prop_f_pyq, W_index):
    
    current_b_T = W_index[0]
    current_bz = W_index[1]
    transverse_direction = W_index[3] # 0, 1
        
    if transverse_direction == 0:
        transverse_direction = X
    elif transverse_direction == 1:
        transverse_direction = Y
            
    prop_shift_pyq = prop_f_pyq.shift(current_b_T, transverse_direction).shift(round(current_bz), Z)
    
    return prop_shift_pyq




# --------------------------
# Load gauge and create inverter
# --------------------------

###################### load gauge ######################
Ls = 8
Lt = 32
conf = 0
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
U = g.convert( g.load(f"../../conf/S8T32_cg/gauge/wilson_b6.cg.1e-08.{conf}"), g.double )


L = U[0].grid.fdimensions
U_prime, trafo = g.gauge_fix(U, maxiter=50000, prec=1e-8) # CG fix, to get trafo



latt_info, gpt_latt, gpt_simd, gpt_prec = gpt.LatticeInfoGPT(U_prime[0].grid, GEN_SIMD_WIDTH)
gauge = gpt.LatticeGaugeGPT(U_prime, GEN_SIMD_WIDTH)
# gauge.projectSU3(2e-14)

src_pos = [1,2,3,4]


dirac = core.getDirac(latt_info, -0.049, 1e-10,  5000, 1.0, 1.0372, 1.0372)
dirac.loadGauge(gauge)

srcDp = Measurement.create_src_2pt(src_pos, trafo, U[0].grid)
b = gpt.LatticePropagatorGPT(srcDp, GEN_SIMD_WIDTH)
b.toDevice()
propag = core.invertPropagator(dirac, b, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
prop_exact_f = g.mspincolor(grid)
gpt.LatticePropagatorGPT(prop_exact_f, GEN_SIMD_WIDTH, propag)


# prepare phases for qext
phases_3pt = Measurement.make_mom_phases_3pt(U_prime[0].grid, src_pos)

# prepare the TMD separate indices for CG
W_index_list_CG = Measurement.create_TMD_Wilsonline_index_list_CG(U[0].grid)


sequential_bw_prop_down = Measurement.create_bw_seq_Pyquda(dirac, prop_exact_f, trafo, 2, src_pos, interpolation) # NOTE, this is a list of propagators for each proton polarization
sequential_bw_prop_up = Measurement.create_bw_seq_Pyquda(dirac, prop_exact_f, trafo, 1, src_pos, interpolation) # NOTE, this is a list of propagators for each proton polarization

#! gpt contract
g.message("\ncontract_TMD loop: CG no links")
proton_TMDs_down = [] # [WL_indices][pol][qext][gammalist][tau]
proton_TMDs_up = []
for iW, WL_indices in enumerate(W_index_list_CG):
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    g.message(f"TIME GPT: contract TMD {iW+1}/{len(W_index_list_CG)} {WL_indices}")
    tmd_forward_prop = Measurement.create_fw_prop_TMD_CG(prop_exact_f, [WL_indices])
    cp.cuda.runtime.deviceSynchronize()
    g.message(f"TIME GPT: cshift", time.time() - t0)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    proton_TMDs_down += [list(g.slice_trDA(sequential_bw_prop_down, tmd_forward_prop, phases_3pt,3))]
    proton_TMDs_up += [list(g.slice_trDA(sequential_bw_prop_up, tmd_forward_prop, phases_3pt,3))]
    cp.cuda.runtime.deviceSynchronize()
    g.message(f"TIME GPT: contract TMD for U and D", time.time() - t0)
    del tmd_forward_prop
proton_TMDs_down = np.array(proton_TMDs_down)
proton_TMDs_up = np.array(proton_TMDs_up)
g.message(f"contract_TMD over: proton_TMDs.shape {np.shape(proton_TMDs_down)}")


tag = "gpt"

# Save correlation function to txt file  
if g.rank() == 0:  # Only write from rank 0 process
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save proton_TMDs_down
    outfile_down = f"{data_dir}/corr_3pt_down_{tag}.txt"
    with open(outfile_down, "w") as f:
        # Write header with shape info
        f.write(f"# Correlation function shape: {np.shape(proton_TMDs_down)}\n")
        f.write("# Format: [WL_index][pol][qext][gamma][time] value\n")
        f.write("# Shape dimensions: (WL_indices, polarizations, momentum_transfers, gamma_matrices, time_slices)\n\n")
        
        # Write data for down quarks
        for wl_idx in range(proton_TMDs_down.shape[0]):  # WL_indices
            for pol_idx in range(proton_TMDs_down.shape[1]):  # polarizations  
                for qext_idx in range(proton_TMDs_down.shape[2]):  # momentum transfers
                    for gamma_idx in range(proton_TMDs_down.shape[3]):  # gamma matrices
                        f.write(f"WL{wl_idx}_P{pol_idx}_Q{qext_idx}_{gammalist[gamma_idx]}: {proton_TMDs_down[wl_idx, pol_idx, qext_idx, gamma_idx, :]}\n")
                    f.write("\n")  # Blank line between gamma groups
                f.write("\n")  # Blank line between qext groups  
            f.write("\n")  # Blank line between polarization groups
        f.write("\n")  # Blank line between WL groups

    g.message(f"Down quark TMDs saved to {outfile_down}")

    # Save proton_TMDs_up
    outfile_up = f"{data_dir}/corr_3pt_up_{tag}.txt"
    with open(outfile_up, "w") as f:
        # Write header with shape info
        f.write(f"# Correlation function shape: {np.shape(proton_TMDs_up)}\n")
        f.write("# Format: [WL_index][pol][qext][gamma][time] value\n") 
        f.write("# Shape dimensions: (WL_indices, polarizations, momentum_transfers, gamma_matrices, time_slices)\n\n")
        
        # Write data for up quarks
        for wl_idx in range(proton_TMDs_up.shape[0]):  # WL_indices
            for pol_idx in range(proton_TMDs_up.shape[1]):  # polarizations
                for qext_idx in range(proton_TMDs_up.shape[2]):  # momentum transfers  
                    for gamma_idx in range(proton_TMDs_up.shape[3]):  # gamma matrices
                        f.write(f"WL{wl_idx}_P{pol_idx}_Q{qext_idx}_{gammalist[gamma_idx]}: {proton_TMDs_up[wl_idx, pol_idx, qext_idx, gamma_idx, :]}\n")
                    f.write("\n")  # Blank line between gamma groups
                f.write("\n")  # Blank line between qext groups
            f.write("\n")  # Blank line between polarization groups  
        f.write("\n")  # Blank line between WL groups

    g.message(f"Up quark TMDs saved to {outfile_up}")


sequential_bw_prop_down = Measurement.create_bw_seq_Pyquda_pyquda(dirac, prop_exact_f, trafo, 2, src_pos, interpolation) # NOTE, this is a list of propagators for each proton polarization
sequential_bw_prop_up = Measurement.create_bw_seq_Pyquda_pyquda(dirac, prop_exact_f, trafo, 1, src_pos, interpolation) # NOTE, this is a list of propagators for each proton polarization



#! pyquda contract
g.message("\ncontract_TMD loop: CG no links")
proton_TMDs_down = [] # [WL_indices][pol][qext][gammalist][tau]
proton_TMDs_up = []

qext_xyz = [v[:3] for v in parameters["qext"]] #! [x, y, z] to be consistent with "qext"
phases_3pt_pyq = phase.MomentumPhase(latt_info).getPhases(qext_xyz, src_pos)

test_shift(propag)


sequential_prop_down = contract(
                "ij, pwtzyxilab, kl -> pwtzyxkjba",
                G5, sequential_bw_prop_down.conj(), G5
            )

sequential_prop_up = contract(
                "ij, pwtzyxilab, kl -> pwtzyxkjba",
                G5, sequential_bw_prop_up.conj(), G5
            )



# for iW, WL_indices in enumerate(W_index_list_CG):
#     cp.cuda.runtime.deviceSynchronize()
#     t0 = time.time()
#     g.message(f"TIME PyQUDA: contract TMD {iW+1}/{len(W_index_list_CG)} {WL_indices}")
#     tmd_forward_prop = create_fw_prop_TMD_CG_pyquda(propag, WL_indices)
#     cp.cuda.runtime.deviceSynchronize()
#     g.message(f"TIME PyQUDA: cshift", time.time() - t0)
#     cp.cuda.runtime.deviceSynchronize()
#     t0 = time.time()
    
#     proton_TMDs_down += [cp.asarray( [pycontract.mesonAllSinkTwoPoint(tmd_forward_prop, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)).data for seq in sequential_prop_down] )]
#     proton_TMDs_up += [cp.asarray( [pycontract.mesonAllSinkTwoPoint(tmd_forward_prop, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)).data for seq in sequential_prop_up] )]
    
#     cp.cuda.runtime.deviceSynchronize()
#     g.message(f"TIME PyQUDA: contract TMD for U and D", time.time() - t0)
#     del tmd_forward_prop
    
# proton_TMDs_down = [contract("qwtzyx, pgwtzyx -> pqgt", phases_3pt_pyq, temp).get() for temp in proton_TMDs_down]
# proton_TMDs_up = [contract("qwtzyx, pgwtzyx -> pqgt", phases_3pt_pyq, temp).get() for temp in proton_TMDs_up]
    
# proton_TMDs_down = np.array(proton_TMDs_down)[:,:,:,pyq_gamma_order,:]
# proton_TMDs_up = np.array(proton_TMDs_up)[:,:,:,pyq_gamma_order,:]
# g.message(f"contract_TMD over: proton_TMDs.shape {np.shape(proton_TMDs_down)}")



for iW, WL_indices in enumerate(W_index_list_CG):
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    g.message(f"TIME PyQUDA: contract TMD {iW+1}/{len(W_index_list_CG)} {WL_indices}")
    tmd_forward_prop = create_fw_prop_TMD_CG_pyquda(propag, WL_indices) #! note here [WL_indices] is changed to WL_indices for PyQUDA, and prop_exact_f is changed to propag
    cp.cuda.runtime.deviceSynchronize()
    g.message(f"TIME PyQUDA: cshift", time.time() - t0)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    #todo: contraction v2
    temp_down = []
    for seq in sequential_prop_down:
        temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)).data
        temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_3pt_pyq, temp1).get(), [2, -1, -1, -1])
        temp_down.append(temp2)
        
    temp_up = []
    for seq in sequential_prop_up:
        temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)).data
        temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_3pt_pyq, temp1).get(), [2, -1, -1, -1])
        temp_up.append(temp2)
        
    proton_TMDs_down.append(temp_down)
    proton_TMDs_up.append(temp_up)
    #todo
    cp.cuda.runtime.deviceSynchronize()
    g.message(f"TIME PyQUDA: contract TMD for U and D", time.time() - t0)
    del tmd_forward_prop
cp.cuda.runtime.deviceSynchronize()
t0 = time.time()
proton_TMDs_down = np.array(proton_TMDs_down)
print(proton_TMDs_down.shape)
proton_TMDs_up = np.array(proton_TMDs_up)
#todo: contraction v2
proton_TMDs_down = proton_TMDs_down[:,:,:,pyq_gamma_order,:]
proton_TMDs_up = proton_TMDs_up[:,:,:,pyq_gamma_order,:]
#todo
g.message(f"contract_TMD over: proton_TMDs.shape {np.shape(proton_TMDs_down)}")







tag = "pyquda"

# Save correlation function to txt file  
if g.rank() == 0:  # Only write from rank 0 process
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save proton_TMDs_down
    outfile_down = f"{data_dir}/corr_3pt_down_{tag}.txt"
    with open(outfile_down, "w") as f:
        # Write header with shape info
        f.write(f"# Correlation function shape: {np.shape(proton_TMDs_down)}\n")
        f.write("# Format: [WL_index][pol][qext][gamma][time] value\n")
        f.write("# Shape dimensions: (WL_indices, polarizations, momentum_transfers, gamma_matrices, time_slices)\n\n")
        
        # Write data for down quarks
        for wl_idx in range(proton_TMDs_down.shape[0]):  # WL_indices
            for pol_idx in range(proton_TMDs_down.shape[1]):  # polarizations  
                for qext_idx in range(proton_TMDs_down.shape[2]):  # momentum transfers
                    for gamma_idx in range(proton_TMDs_down.shape[3]):  # gamma matrices
                        f.write(f"WL{wl_idx}_P{pol_idx}_Q{qext_idx}_{gammalist[gamma_idx]}: {proton_TMDs_down[wl_idx, pol_idx, qext_idx, gamma_idx, :]}\n")
                    f.write("\n")  # Blank line between gamma groups
                f.write("\n")  # Blank line between qext groups  
            f.write("\n")  # Blank line between polarization groups
        f.write("\n")  # Blank line between WL groups

    g.message(f"Down quark TMDs saved to {outfile_down}")

    # Save proton_TMDs_up
    outfile_up = f"{data_dir}/corr_3pt_up_{tag}.txt"
    with open(outfile_up, "w") as f:
        # Write header with shape info
        f.write(f"# Correlation function shape: {np.shape(proton_TMDs_up)}\n")
        f.write("# Format: [WL_index][pol][qext][gamma][time] value\n") 
        f.write("# Shape dimensions: (WL_indices, polarizations, momentum_transfers, gamma_matrices, time_slices)\n\n")
        
        # Write data for up quarks
        for wl_idx in range(proton_TMDs_up.shape[0]):  # WL_indices
            for pol_idx in range(proton_TMDs_up.shape[1]):  # polarizations
                for qext_idx in range(proton_TMDs_up.shape[2]):  # momentum transfers  
                    for gamma_idx in range(proton_TMDs_up.shape[3]):  # gamma matrices
                        f.write(f"WL{wl_idx}_P{pol_idx}_Q{qext_idx}_{gammalist[gamma_idx]}: {proton_TMDs_up[wl_idx, pol_idx, qext_idx, gamma_idx, :]}\n")
                    f.write("\n")  # Blank line between gamma groups
                f.write("\n")  # Blank line between qext groups
            f.write("\n")  # Blank line between polarization groups  
        f.write("\n")  # Blank line between WL groups

    g.message(f"Up quark TMDs saved to {outfile_up}")
