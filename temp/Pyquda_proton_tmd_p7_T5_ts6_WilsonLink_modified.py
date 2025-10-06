
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
from qTMD.developing.PyQUDA_proton_qTMD_draft import proton_TMD, pyquda_gamma_ls, pyq_gamma_order #! import pyquda_gamma_ls and pyq_gamma_order for 3pt
from tools import *
from io_corr import *

# load pyquda modules
from pyquda import init, LatticeInfo, getMPIComm
from pyquda_utils import core, gpt, gamma, phase
from pyquda_plugins import pycontract #todo: for PyQUDA contraction v2

# Global parameters
data_dir="/lustre/orion/nph158/proj-shared/xgao/l64c64a076/nucleon_TMD_pyquda/data" # NOTE
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
mpi_geometry = [2, 2, 2, 4]
init(mpi_geometry, enable_mps=True, grid_map="shared")
G5 = gamma.gamma(15)

# --------------------------
# Setup parameters
# --------------------------
parameters = {
    
    # NOTE:
    "eta": [0],  # irrelavant for CG TMD
    "b_z": 20,
    "b_T": 2,

    "qext": [[x,y,z,0] for x in [-2,-1,0,1,2] for y in [-2,-1,0,1,2] for z in [-2,-1,0]], # momentum transfer for TMD, pf = pi + q
    #"qext": [list(v + (0,)) for v in {tuple(sorted((x, y, z))) for x in [-2,-1,0] for y in [-2,-1,0] for z in [0]}], # momentum transfer for TMD, pf = pi + q
    "qext_PDF": [[x,y,z,0] for x in [-2,-1,0] for y in [-2,-1,0] for z in [-2,-1,0]], # momentum transfer for PDF, not used 
    "pf": [0,0,9,0],
    "p_2pt": [[x,y,z,0] for x in [-2,-1,0,1,2] for y in [-2,-1,0,1,2] for z in [5, 6, 7, 8, 9]], # 2pt momentum, should match pf & pi

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
Ls = 64
Lt = 64
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
U = g.convert( g.load(f"/lustre/orion/nph158/proj-shared/jinchen/debug/nucleon_TMD/fixed_GLU/l6464f21b7130m00119m0322a.{conf}.coulomb.1e-14"), g.double )

g.mem_report(details=False)
L = U[0].grid.fdimensions
U_prime, trafo = g.gauge_fix(U, maxiter=5000, prec=1e-12) # CG fix, to get trafo
del U_prime
U_hyp = g.qcd.gauge.smear.hyp(U, alpha = np.array([0.75, 0.6, 0.3])) # hyp smearing
latt_info, gpt_latt, gpt_simd, gpt_prec = gpt.LatticeInfoGPT(U[0].grid, GEN_SIMD_WIDTH)
gauge = gpt.LatticeGaugeGPT(U_hyp, GEN_SIMD_WIDTH)
g.mem_report(details=False)

###################### setup source positions ######################
src_shift = np.array([0,0,0,0]) + np.array([7,11,13,23])
src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin) # create a list of source 4*4*4*4
src_production = src_positions[0:32] # take the number of sources needed for this project NOTE

###################### create multigrid inverter ######################
latt_info = LatticeInfo([Ls, Ls, Ls, Lt], -1, 1.0)
dirac = core.getDirac(latt_info, -0.049, 1e-10,  5000, 1.0, 1.0372, 1.0372, [[8, 8, 4, 4]]) # remove the last two arguments for BiCGStab; S mass -0.015, U/D mass -0.049
g.message("DEBUG plaquette U_hyp:", g.qcd.gauge.plaquette(U_hyp))
g.message("DEBUG plaquette gauge:", gauge.plaquette())
# gauge.projectSU3(1e-15) #todo: modified by Jinchen, for the new version of pyquda
dirac.loadGauge(gauge)
g.message("Multigrid inverter ready.")
g.mem_report(details=False)


# --------------------------
# Start measurements
# --------------------------

###################### record the finished source position ######################
sample_log_file = data_dir + "/sample_log_qtmd/" + str(conf) + '_' + sm_tag + "_" + pf_tag
if g.rank() == 0:
    f = open(sample_log_file, "a+")
    f.close()

#! Measurement
###################### loop over sources ######################
for ipos, pos in enumerate(src_production):
    
    sample_log_tag = get_sample_log_tag(str(conf), pos, sm_tag + "_" + pf_tag)
    g.message(f"START: {sample_log_tag}")
    with open(sample_log_file, "a+") as f:
        f.seek(0)
        if sample_log_tag in f.read():
            g.message("SKIP: " + sample_log_tag)
            continue # NOTE comment this out for test otherwise it will skip all the sources that are already done



    #>>>>>>>>>>>>>>>>>>>>>>>>> Propagators <<<<<<<<<<<<<<<<<<<<<<<<<<#

    # get forward propagator boosted source
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)
    b = gpt.LatticePropagatorGPT(srcDp, GEN_SIMD_WIDTH)
    b.toDevice()
    cp.cuda.runtime.deviceSynchronize()
    g.message("TIME GPT-->Pyquda: Generatring boosted src", time.time() - t0)

    # get forward propagator: smeared-point
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    propag = core.invertPropagator(dirac, b, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
    prop_exact_f = g.mspincolor(grid)
    gpt.LatticePropagatorGPT(prop_exact_f, GEN_SIMD_WIDTH, propag)
    cp.cuda.runtime.deviceSynchronize()
    g.message("TIME Pyquda-->GPT: Forward propagator inversion", time.time() - t0)
    
    #todo: test the .shift() method in PyQUDA and g.cshift() in GPT
    if ipos == 0:
        test_shift(propag)

    #! GPT: contract 2pt TMD
    #cp.cuda.runtime.deviceSynchronize()
    #t0 = time.time()
    #tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    #phases_2pt = Measurement.make_mom_phases_2pt(U[0].grid, pos)
    #Measurement.contract_2pt_TMD(prop_exact_f, phases_2pt, trafo, tag, interpolation) # NOTE, new interpolation operator
    #cp.cuda.runtime.deviceSynchronize()
    #g.message("TIME GPT: Contraction 2pt (includes sink smearing)", time.time() - t0)
    
    #! PyQUDA: get backward propagator through sequential source for U and D
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    sequential_bw_prop_down_pyq = Measurement.create_bw_seq_Pyquda_pyquda(dirac, prop_exact_f, trafo, 2, pos, interpolation) # NOTE, this is a list of propagators for each proton polarization
    sequential_bw_prop_up_pyq = Measurement.create_bw_seq_Pyquda_pyquda(dirac, prop_exact_f, trafo, 1, pos, interpolation) # NOTE, this is a list of propagators for each proton polarization
    cp.cuda.runtime.deviceSynchronize()
    g.message("TIME GPT-->Pyquda: Backward propagator through sequential source for U and D", time.time() - t0)

    #! PyQUDA: prepare phases for qext
    qext_xyz = [[v[0], v[1], v[2]] for v in parameters["qext"]]
    phases_3pt_pyq = phase.MomentumPhase(latt_info).getPhases(qext_xyz, pos)

    phases_PDF = Measurement.make_mom_phases_PDF(U[0].grid, pos)



    #>>>>>>>>>>>>>>>>>>>>>>>>> CG TMD <<<<<<<<<<<<<<<<<<<<<<<<<<#

    W_index_list_PDF = Measurement.create_PDF_Wilsonline_index_list(U[0].grid)

    # prepare the TMD separate indices for CG
    W_index_list_CG_dir0, W_index_list_CG_dir1 = Measurement.create_TMD_Wilsonline_index_list_CG_pyquda()
    W_index_list_CG = W_index_list_CG_dir0 + W_index_list_CG_dir1
    
    #! PyQUDA: contract TMD
    g.message("\ncontract_TMD loop: CG no links")
    t0_contract = time.time()
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    proton_TMDs_down = [] # [WL_indices][pol][qext][gammalist][tau]
    proton_TMDs_up = []
    
    sequential_bw_prop_down_contracted_pyq = contract(
                "ij, pwtzyxilab, kl -> pwtzyxkjba",
                G5, sequential_bw_prop_down_pyq.conj(), G5
            )

    sequential_bw_prop_up_contracted_pyq = contract(
                "ij, pwtzyxilab, kl -> pwtzyxkjba",
                G5, sequential_bw_prop_up_pyq.conj(), G5
            )
    
    cp.cuda.runtime.deviceSynchronize()
    g.message(f"TIME PyQUDA: contract bw prop with gamma_ls for U and D", time.time() - t0)
   
    #! PyQUDA: contract TMD +X direction
    tmd_forward_prop_dir0 = propag.copy()
    for iW, WL_indices in enumerate(W_index_list_CG_dir0):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        g.message(f"TIME PyQUDA: contract TMD {iW+1}/{len(W_index_list_CG)} {WL_indices}")
        if iW == 0:
            WL_indices_previous = [0, 0, 0, 0]
        else:
            WL_indices_previous = W_index_list_CG_dir0[iW - 1]
            
        tmd_forward_prop_dir0 = Measurement.create_fw_prop_TMD_CG_pyquda(tmd_forward_prop_dir0, WL_indices, WL_indices_previous) #! note here [WL_indices] is changed to WL_indices for PyQUDA, and prop_exact_f is changed to propag
        cp.cuda.runtime.deviceSynchronize()
        g.message(f"TIME PyQUDA: cshift", time.time() - t0)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        temp_down = []
        for seq in sequential_bw_prop_down_contracted_pyq:
            temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop_dir0, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)).data # loop over 16 gamma structure
            temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_3pt_pyq, temp1).get(), [2, -1, -1, -1])
            temp_down.append(temp2)
        proton_TMDs_down.append(temp_down)
        
        temp_up = []
        for seq in sequential_bw_prop_up_contracted_pyq:
            temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop_dir0, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)).data # loop over 16 gamma structure
            temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_3pt_pyq, temp1).get(), [2, -1, -1, -1])
            temp_up.append(temp2)
        proton_TMDs_up.append(temp_up)
        cp.cuda.runtime.deviceSynchronize()
        g.message(f"TIME PyQUDA: contract TMD for U and D", time.time() - t0)
    del tmd_forward_prop_dir0
        
    #! PyQUDA: contract TMD +Y direction
    tmd_forward_prop_dir1 = propag.copy()
    for iW, WL_indices in enumerate(W_index_list_CG_dir1):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        g.message(f"TIME PyQUDA: contract TMD {iW+1+len(W_index_list_CG_dir0)}/{len(W_index_list_CG)} {WL_indices}")
        if iW == 0:
            WL_indices_previous = [0, 0, 0, 0]
        else:
            WL_indices_previous = W_index_list_CG_dir1[iW - 1]
        tmd_forward_prop_dir1 = Measurement.create_fw_prop_TMD_CG_pyquda(tmd_forward_prop_dir1, WL_indices, WL_indices_previous) #! note here [WL_indices] is changed to WL_indices for PyQUDA, and prop_exact_f is changed to propag
        cp.cuda.runtime.deviceSynchronize()
        g.message(f"TIME PyQUDA: cshift", time.time() - t0)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        temp_down = []
        for seq in sequential_bw_prop_down_contracted_pyq:
            temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop_dir1, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)).data
            temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_3pt_pyq, temp1).get(), [2, -1, -1, -1])
            temp_down.append(temp2)
        proton_TMDs_down.append(temp_down)
        
        temp_up = []
        for seq in sequential_bw_prop_up_contracted_pyq:
            temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop_dir1, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)).data
            temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_3pt_pyq, temp1).get(), [2, -1, -1, -1])
            temp_up.append(temp2)
        proton_TMDs_up.append(temp_up)
        cp.cuda.runtime.deviceSynchronize()
        g.message(f"TIME PyQUDA: contract TMD for U and D", time.time() - t0)
    del tmd_forward_prop_dir1
    
    proton_TMDs_down = np.array(proton_TMDs_down)
    proton_TMDs_up = np.array(proton_TMDs_up)
    g.message(f"contract_TMD over: proton_TMDs.shape {np.shape(proton_TMDs_down)} {time.time()-t0_contract}s")

    # save the TMD correlators
    for i, pol in enumerate(parameters["pol"]):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()

        # reorder gamma, and cut useful tau in [src_t, src_t+tsep+2)
        if g.rank() == 0:
            proton_TMDs_down = np.roll(proton_TMDs_down, -pos[3], axis=-1)
            proton_TMDs_up = np.roll(proton_TMDs_up, -pos[3], axis=-1)
            proton_TMDs_down = proton_TMDs_down[:,:,:,pyq_gamma_order,:parameters["t_insert"]+2]
            proton_TMDs_up = proton_TMDs_up[:,:,:,pyq_gamma_order,:parameters["t_insert"]+2]
        proton_TMDs_down = getMPIComm().bcast(proton_TMDs_down, root=0)
        proton_TMDs_up = getMPIComm().bcast(proton_TMDs_up, root=0)

        #! parallel the io through flavor and gamma
        tasks = []
        for gidx in range(len(gammalist)):
            tasks.append((gidx, 'D'))  # Down
            tasks.append((gidx, 'U'))  # Up
        rank = g.rank()
        if rank < len(tasks):
            gidx, flavor = tasks[rank]
            gm = gammalist[gidx]
            tag = get_qTMD_file_tag(data_dir, lat_tag, conf, f"CG.{flavor}.ex", pos, f"{sm_tag}.{pf_tag}.{pol}.{gm}")
            print(f"DEBUG: rank {rank}, {tag}")
            data = proton_TMDs_down[:, i, :, gidx:gidx+1, :] if flavor == 'D' else proton_TMDs_up[:, i, :, gidx:gidx+1, :]
            save_qTMD_proton_hdf5_noRoll(data, tag, [gm], parameters["qext"], W_index_list_CG, parameters["t_insert"])
        cp.cuda.runtime.deviceSynchronize()
        g.message(f"TIME: save TMDs for {pol}", time.time() - t0)
    g.message("\ncontract_TMD DONE: CG no links")

    #>>>>>>>>>>>>>>>>>>>>>>>>> GI GPD <<<<<<<<<<<<<<<<<<<<<<<<<<#

    #! GPT: bw prop
    # del propag
    # sequential_bw_prop_down = []
    # for seq in sequential_bw_prop_down_pyq:
    #     seq_prop = g.mspincolor(grid)
    #     gpt.LatticePropagatorGPT(seq_prop, GEN_SIMD_WIDTH, core.LatticePropagator(latt_info, seq))
    #     sequential_bw_prop_down.append(seq_prop)
    # del sequential_bw_prop_down_pyq, sequential_bw_prop_down_contracted_pyq
    # sequential_bw_prop_up = []
    # for seq in sequential_bw_prop_up_pyq:
    #     seq_prop = g.mspincolor(grid)
    #     gpt.LatticePropagatorGPT(seq_prop, GEN_SIMD_WIDTH, core.LatticePropagator(latt_info, seq))
    #     sequential_bw_prop_up.append(seq_prop)
    # del sequential_bw_prop_up_pyq, sequential_bw_prop_up_contracted_pyq
    
    
    #! Addons for PyQUDA code
    import h5py as h5
    
    pyq_gamma_order = [15, 8, 7, 1, 14, 2, 13, 4, 11, 0, 9, 3, 5, 10, 6, 12]
    my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
    
    def save_qTMD_proton_hdf5_subset(corr, tag, gammalist, plist, W_index_list, i_sub, tsep):

        roll = -int(tag.split(".")[6].split('t')[1]) # 6: xyzt
        bT_list = ['b_X', 'b_Y']

        g.message("-->>",W_index_list)

        save_h5 = tag + ".h5"
        if i_sub == 0:
            f = h5.File(save_h5, 'w')
        else:
            f = h5.File(save_h5, 'a')

        sm = f.require_group("SS")
        for ig, gm in enumerate(gammalist):
            g_gm = sm.require_group(gm)
            for ip, p in enumerate(plist):
                p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
                g_p = g_gm.require_group(p_tag)
                for i, idx in enumerate(W_index_list):
                    path = bT_list[idx[3]] + '/' + 'eta'+str(idx[2]) + '/' + 'bT'+str(idx[0])
                    g_data = g_p.require_group(path)
                    # g.message("Want to save", path+'bz'+str(idx[1]))
                    g_data.create_dataset('bz'+str(idx[1]), data=np.roll(corr[i][ip][ig], roll, axis=0)[:tsep+2])
        f.close()
    
    
    
    #! PyQUDA: prepare phases for qext
    qext_pdf_xyz = [[v[0], v[1], v[2]] for v in parameters["qext_PDF"]]
    phases_pdf_pyq = phase.MomentumPhase(latt_info).getPhases(qext_pdf_xyz, pos)
    
    #! PyQUDA: bw prop
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
    
    tmd_forward_prop_pyq = propag.copy()
    g.message("\ncontract_PDF loop: GI with links")
    for iW, WL_indices in enumerate(W_index_list_PDF):
        #! GPT: forward prop
        # W = Measurement.create_PDF_Wilsonline(U_hyp, WL_indices)

        # tmd_forward_prop = Measurement.create_fw_prop_PDF(prop_exact_f, [W], [WL_indices])
        
        # g.message("TMD forward prop done")

        qtmd_tag_exact_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI_PDF.D.ex", pos, sm_tag+'.'+pf_tag)
        qtmd_tag_exact_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI_PDF.U.ex", pos, sm_tag+'.'+pf_tag)
        g.message("Starting TMD contractions")

        #! GPT: contract
        # proton_TMDs_down = Measurement.contract_PDF(tmd_forward_prop, sequential_bw_prop_down, phases_PDF, WL_indices, qtmd_tag_exact_D, iW)
        # proton_TMDs_up = Measurement.contract_PDF(tmd_forward_prop, sequential_bw_prop_up, phases_PDF, WL_indices, qtmd_tag_exact_U, iW)

        # del tmd_forward_prop
        
        #! PyQUDA: contract
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
            
        temp_down = np.stack(temp_down, axis=0)
        temp_up = np.stack(temp_up, axis=0)

        temp_down = temp_down[:, :, pyq_gamma_order, :]
        temp_up = temp_up[:, :, pyq_gamma_order, :]
        
        #! PyQUDA: forward prop
        for spin in range(4):
            for color in range(3):
                fermion = tmd_forward_prop_pyq.getFermion(spin, color)
                fermion_shift = gauge.pure_gauge.covDev(fermion, 2)
                tmd_forward_prop_pyq.setFermion(fermion_shift, spin, color)
                
        #! PyQUDA: save
        for pol_idx, pol in enumerate(Measurement.pol_list):
            pol_tag_D = qtmd_tag_exact_D + "." + pol
            pol_tag_U = qtmd_tag_exact_U + "." + pol
            corr_down = [temp_down[pol_idx]]
            corr_up = [temp_up[pol_idx]]
            if g.rank() == pol_idx:
                save_qTMD_proton_hdf5_subset(corr_down, pol_tag_D, my_gammas, Measurement.qlist, [WL_indices], iW, Measurement.t_insert)
                save_qTMD_proton_hdf5_subset(corr_up, pol_tag_U, my_gammas, Measurement.qlist, [WL_indices], iW, Measurement.t_insert)

        
        
    g.message("\ncontract_PDF DONE: GI with links")

    with open(sample_log_file, "a+") as f:
        if g.rank() == 0:
            f.write(sample_log_tag+"\n")
    g.message("DONE: " + sample_log_tag)
