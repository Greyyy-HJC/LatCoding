# load python modules
import numpy as np
import os
import time
import cupy as cp
from opt_einsum import contract
from Proton_qTMD_utils import proton_TMD

# load gpt modules
import gpt as g 
from io_corr import *

# load pyquda modules
from pyquda import init, LatticeInfo
from pyquda_utils import core, gpt, gamma, phase


# Gobal parameters
data_dir="data" # NOTE
lat_tag = "l64c64a076" # NOTE
sm_tag = "1HYP_GSRC_W90_k3_Z5" # NOTE
interpolation = "5" # NOTE, new interpolation operator
GEN_SIMD_WIDTH = 64
Ls = 8
Lt = 32

GI = gamma.gamma(0)
GZ = gamma.gamma(4)
GT = gamma.gamma(8)
G5 = gamma.gamma(15)
C = gamma.gamma(2) @ gamma.gamma(8)

epsilon= cp.zeros((3,3,3))
for a in range (3):
    b = (a+1) % 3
    c = (a+2) % 3
    epsilon[a,b,c] = 1
    epsilon[a,c,b] = -1

### Projection of nucleon states
Cg5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma[5].tensor()
CgT5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma["T"].tensor() * g.gamma[5].tensor()
CgZ5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma["Z"].tensor() * g.gamma[5].tensor()

g.message(f"--lat_tag {lat_tag}")
g.message(f"--sm_tag {sm_tag}")

# --------------------------
# initiate quda
# --------------------------
mpi_geometry = [1, 1, 1, 1]
init(mpi_geometry)



# --------------------------
# Setup parameters
# --------------------------
parameters = {
    
    # NOTE:
    "eta": [0],  # irrelavant for CG TMD
    "b_z": 1,
    "b_T": 1,

    "qext": [list(v + (0,)) for v in {tuple(sorted((x, y, z))) for x in [-2,-1,0] for y in [-2,-1,0] for z in [0]}], # momentum transfer for TMD, pf = pi + q
    "qext_PDF": [[x,y,z,0] for x in [-2,-1,0] for y in [-2,-1,0] for z in [-2,-1,0]], # momentum transfer for PDF, not used 
    "pf": [0,0,0,0],
    "p_2pt": [[x,y,z,0] for x in [0] for y in [0] for z in [3]], # 2pt momentum, should match pf & pi

    "boost_in": [0,0,0],
    "boost_out": [0,0,0],
    "width" : 9.0,

    "pol": ["PpUnpol"],
    "t_insert": 4, # time separation for TMD

    "save_propagators": False,
}
pf = parameters["pf"]
pf_tag = "PX"+str(pf[0]) + "PY"+str(pf[1]) + "PZ"+str(pf[2]) + "dt" + str(parameters["t_insert"])
gammalist = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
pyq_gammalist = [gamma.gamma(15), gamma.gamma(8), gamma.gamma(7), -gamma.gamma(1), gamma.gamma(14), gamma.gamma(2), gamma.gamma(13), -gamma.gamma(4), gamma.gamma(11), gamma.gamma(0), gamma.gamma(9), gamma.gamma(3), -gamma.gamma(5), -gamma.gamma(10), gamma.gamma(6), gamma.gamma(12)]

Measurement = proton_TMD(parameters)

P_2pt_gamma = cp.zeros((16, Lt, 4, 4), "<c16")
for gamma_idx, gamma_pyq in enumerate(pyq_gammalist):
    P_2pt = cp.zeros((Lt, 4, 4), "<c16")
    P_2pt[:] = gamma_pyq
    P_2pt_gamma[gamma_idx] = P_2pt


def contract_2pt_TMD(prop_f, phases):
        dq = g.qcd.baryon.diquark(g(prop_f * Cg5), g(Cg5 * prop_f))
        proton1 = g(g.spin_trace(dq) * prop_f + dq * prop_f)
        prop_unit = g.mspincolor(prop_f.grid)
        prop_unit = g.identity(prop_unit)
        corr = g.slice_trDA([prop_unit], [proton1], phases,3)
        corr = [[corr[0][i][j] for i in range(0, len(corr[0]))] for j in range(0, len(corr[0][0])) ]

        return corr


#!: Output: gamma insertion, momentum, t_insert
def contract_2pt_TMD_pyquda(prop_f, phases): #TODO: use pyquda function
    prop_f_pyq = gpt.LatticePropagatorGPT(prop_f, GEN_SIMD_WIDTH)
    
    corr = (
            - contract(
            "abc, def, pwtzyx, ij, kl, qtmn, wtzyxikad, wtzyxjlbe, wtzyxmncf->qpt",
            epsilon,    epsilon,    phases,    C @ G5,    C @ G5,    P_2pt_gamma,
            prop_f_pyq.data,  prop_f_pyq.data,  prop_f_pyq.data,
            ) 
            - contract(
                "abc, def, pwtzyx, ij, kl, qtmn, wtzyxikad, wtzyxjnbe, wtzyxmlcf->qpt",
                epsilon,    epsilon,    phases,    C @ G5,    C @ G5,    P_2pt_gamma,
                prop_f_pyq.data,  prop_f_pyq.data,  prop_f_pyq.data,
            )
        )
    
    return corr


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

w = g.qcd.fermion.wilson_clover(
        U_prime,
        {
            "kappa": 0.126,
            "csw_r": 0,
            "csw_t": 0,
            "xi_0": 1,
            "nu": 1,
            "isAnisotropic": False,
            "boundary_phases": [1.0, 1.0, 1.0, -1.0],
        },
    )
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-10, "maxiter": 1000})
propagator = w.propagator(inv.preconditioned(pc.eo1_ne(), cg))

src_pos = [0,2,3,0]
src = g.mspincolor(grid)
g.create.point(src, src_pos)
prop_exact_f = g.mspincolor(grid)
prop_exact_f @= propagator * src

phases_2pt = Measurement.make_mom_phases_2pt(U_prime[0].grid, src_pos)
tag = "gpt"

print(">>> phases_gpt shape: ", np.shape(phases_2pt[0][:]))
print(">>> phases_gpt: ", phases_2pt[0][:])

cp.cuda.runtime.deviceSynchronize()
gpt_contract_start = time.time()
corr_gpt = Measurement.contract_2pt_TMD(prop_exact_f, phases_2pt, trafo, tag, interpolation)
cp.cuda.runtime.deviceSynchronize()
gpt_contract_end = time.time()
corr_gpt = np.array(corr_gpt)
print(np.shape(corr_gpt))


# Save correlation function to txt file
if g.rank() == 0:  # Only write from rank 0 process
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate output filename
    outfile = f"{data_dir}/corr_2pt_{tag}.txt"
    
    # Write correlation function data
    with open(outfile, "w") as f:
        # Write header with shape info
        f.write(f"# Correlation function shape: {np.shape(corr_gpt)}\n")
        f.write("# Format: [momentum_index][timeslice] value\n")
        f.write(f"gpt_contract_time: {gpt_contract_end - gpt_contract_start:.6f} seconds\n\n")
        
        # Write data
        for gamma_idx in range(len(corr_gpt)):
            for mom_idx in range(len(corr_gpt[gamma_idx])):
                f.write(f"{gammalist[gamma_idx]} {mom_idx}: {corr_gpt[gamma_idx][mom_idx]}\n")
            f.write("\n")  # Blank line between momentum indices

g.message(f"Correlation function saved to {outfile}")


tag = "pyq"

# phases_2pt = phase.MomentumPhase(latt_info).getPhases([[0, 0, 0], [0, 0, -1]], src_pos) #TODO: use pyquda phase
p_2pt_xyz = [[-v[0], -v[1], -v[2]] for v in parameters["p_2pt"]] #! [-x, -y, -z], has an extra minus sign compared to "p_2pt" by convention
phases_2pt = phase.MomentumPhase(latt_info).getPhases(p_2pt_xyz, src_pos)
# phases_2pt = cp.asarray([phase * -1j for phase in phases_2pt])

print(">>> phases_pyq shape: ", np.shape(phases_2pt))
print(type(phases_2pt))
print(">>> phases_pyq: ", phases_2pt[0][:])

cp.cuda.runtime.deviceSynchronize()
pyq_contract_start = time.time()
corr_pyq = contract_2pt_TMD_pyquda(prop_exact_f, phases_2pt)
cp.cuda.runtime.deviceSynchronize()
pyq_contract_end = time.time()
corr_pyq = np.array(corr_pyq.get())
print(np.shape(corr_pyq))




# Save correlation function to txt file
if g.rank() == 0:  # Only write from rank 0 process
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate output filename
    outfile = f"{data_dir}/corr_2pt_{tag}.txt"
    
    # Write correlation function data
    with open(outfile, "w") as f:
        # Write header with shape info
        f.write(f"# Correlation function shape: {np.shape(corr_pyq)}\n")
        f.write("# Format: [momentum_index][timeslice] value\n")
        f.write(f"pyq_contract_time: {pyq_contract_end - pyq_contract_start:.6f} seconds\n\n")
        
        # Write data
        for gamma_idx in range(len(corr_pyq)):
            for mom_idx in range(len(corr_pyq[gamma_idx])):
                f.write(f"{gammalist[gamma_idx]} {mom_idx}: {corr_pyq[gamma_idx][mom_idx]}\n")
            f.write("\n")  # Blank line between momentum indices

g.message(f"Correlation function saved to {outfile}")

