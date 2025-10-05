from cmath import phase
from math import gamma
import gpt as g
import cupy as cp
from io_corr import *
import numpy as np
from gpt_proton_qTMD_utils import proton_measurement

# load pyquda modules
from pyquda import init, LatticeInfo
from pyquda_utils import core, gpt, gamma
import subprocess
from opt_einsum import contract
from itertools import permutations

GEN_SIMD_WIDTH = 64

G5 = gamma.gamma(15)
GZ5 = gamma.gamma(4) @ G5
GT5 = gamma.gamma(8) @ G5

"""
================================================================================
                Gamma structures and Projection of nucleon states
================================================================================
"""
### Gamma structures
my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]

### Projection of nucleon states
Cg5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma[5].tensor()
CgT5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma["T"].tensor() * g.gamma[5].tensor()
CgZ5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma["Z"].tensor() * g.gamma[5].tensor()

Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
Szp = (g.gamma["I"].tensor() - 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Szm = (g.gamma["I"].tensor() + 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Sxp = (g.gamma["I"].tensor() - 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
Sxm = (g.gamma["I"].tensor() + 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
PpSzp = Pp * Szp
PpSzm = Pp * Szm
PpSxp = Pp * Sxp
PpSxm = Pp * Sxm
#my_projections=["PpSzp", "PpSxp", "PpSxm"]
#my_projections=["PpSzp", "PpSzm", "PpSxp"]
#PolProjections = [PpSzp, PpSxp, PpSxm]
#PolProjections = [PpSzp, PpSzm, PpSxp]
PolProjections = {
    "PpSzp": PpSzp,
    "PpSzm": PpSzm,
    "PpSxp": PpSxp,
    "PpSxm": PpSxm,  
    "PpUnpol": Pp,  
}

"""
================================================================================
                Used for proton two-point function contraction
================================================================================
"""
ordered_list_of_gammas = [g.gamma[5], g.gamma["T"], g.gamma["T"]*g.gamma[5],
                                      g.gamma["X"], g.gamma["X"]*g.gamma[5], 
                                      g.gamma["Y"], g.gamma["Y"]*g.gamma[5],
                                      g.gamma["Z"], g.gamma["Z"]*g.gamma[5], 
                                      g.gamma["I"], g.gamma["SigmaXT"], 
                                      g.gamma["SigmaXY"], g.gamma["SigmaXZ"], 
                                      g.gamma["SigmaYT"], g.gamma["SigmaYZ"], 
                                      g.gamma["SigmaZT"]
                            ]
def uud_two_point(Q1, Q2, kernel):
    dq = g.qcd.baryon.diquark(g(Q1 * kernel), g(kernel * Q2))
    return g(g.color_trace(g.spin_trace(dq) * Q1 + dq * Q1))

def proton_contr(Q1, Q2):
    C = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
    Gamma = C * g.gamma[5].tensor()
    #Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
    corr = []
    for ig, gm in enumerate(ordered_list_of_gammas):
        Pp = gm
        corr += [g(g.trace(uud_two_point(Q1, Q2, Gamma) * Pp))]
    return corr
    #return g(g.trace(uud_two_point(Q1, Q2, Gamma) * Pp))

"""
================================================================================
                                proton_TMD
================================================================================
"""
class proton_TMD(proton_measurement):

    def __init__(self, parameters):

        self.eta = parameters["eta"] # list of eta
        self.b_z = parameters["b_z"] # largest b_z
        self.b_T = parameters["b_T"] # largest b_T

        self.pf = parameters["pf"] # momentum of final nucleon state; pf = pi + q
        self.plist = parameters["qext"]
        self.qlist = parameters["qext_PDF"]
        #self.plist = [list(v + (0,)) for v in {tuple(sorted((x, y, z))) for x in parameters["qext"] for y in parameters["qext"] for z in [0]}]
        #self.plist = [[x,y,z,0] for x in parameters["qext"] for y in parameters["qext"] for z in parameters["qext"]] # generating momentum transfers for TMD
        #self.qlist = [[x,y,z,0] for x in parameters["qext_PDF"] for y in parameters["qext_PDF"] for z in parameters["qext_PDF"]] # generating momentum transfers for PDF
        #self.pilist = [[parameters["pf"][0]-x,parameters["pf"][1]-y,parameters["pf"][2]-z,0] for x in parameters["qext"] for y in parameters["qext"] for z in parameters["qext"]] # generating pi = pf - q
        self.pilist = parameters["p_2pt"]  # 2pt momentum

        self.width = parameters["width"] # Gaussian smearing width
        self.boost_in = parameters["boost_in"] # ?? Forward propagator boost smearing
        self.boost_out = parameters["boost_out"] # ?? Backward propagator boost smearing
        self.pos_boost = self.boost_in # Forward propagator boost smearing for 2pt

        self.pol_list = parameters["pol"] # projection of nucleon state
        self.t_insert = parameters["t_insert"] # time separation of three point function

        self.save_propagators = parameters["save_propagators"] # if save propagators
    
    ############## make list of complex phases for momentum proj.
    def make_mom_phases_2pt(self, grid, origin=None):    
        one = g.identity(g.complex(grid))
        pp = [-2 * np.pi * np.array(pi) / grid.fdimensions for pi in self.pilist] # pilist is the pf-q

        P = g.exp_ixp(pp, origin)
        mom = [g.eval(pp*one) for pp in P]
        return mom
    def make_mom_phases_3pt(self, grid, origin=None):    
        one = g.identity(g.complex(grid))
        pp = [2 * np.pi * np.array(p) / grid.fdimensions for p in self.plist] # plist is the q for TMD

        P = g.exp_ixp(pp, origin)
        mom = [g.eval(pp*one) for pp in P]
        return mom
    def make_mom_phases_PDF(self, grid, origin=None):    
        one = g.identity(g.complex(grid))
        pp = [2 * np.pi * np.array(p) / grid.fdimensions for p in self.qlist] # qlist is the q for PDF

        P = g.exp_ixp(pp, origin)
        mom = [g.eval(pp*one) for pp in P]
        return mom
    
    #function that does the contractions for the smeared-smeared pion 2pt function
    def contract_2pt_TMD(self, prop_f, phases, trafo, tag, interpolation = "5"):

        # g.message("Begin sink smearing")
        # tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        # prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        # g.message("Sink smearing completed")

        #TODO: Jinchen, new interpolation operator
        if interpolation == "5":
            dq = g.qcd.baryon.diquark(g(prop_f * Cg5), g(Cg5 * prop_f))
        elif interpolation == "T5":
            dq = g.qcd.baryon.diquark(g(prop_f * CgT5), g(CgT5 * prop_f)) 
        elif interpolation == "Z5":
            dq = g.qcd.baryon.diquark(g(prop_f * CgZ5), g(CgZ5 * prop_f)) 
        else:
            raise ValueError("Invalid interpolation operator")
        
        proton1 = g(g.spin_trace(dq) * prop_f + dq * prop_f)
        prop_unit = g.mspincolor(prop_f.grid)
        prop_unit = g.identity(prop_unit)
        corr = g.slice_trDA([prop_unit], [proton1], phases,3)
        corr = [[corr[0][i][j] for i in range(0, len(corr[0]))] for j in range(0, len(corr[0][0])) ]

        return corr

    def create_fw_prop_TMD(self, prop_f, W, W_index_list):
        g.message("Creating list of W*prop_f with shift bT and 2*bz")
        prop_list = []
        
        for i, idx in enumerate(W_index_list):

            current_b_T = idx[0]
            current_bz = idx[1]
            current_eta = idx[2]
            transverse_direction = idx[3]
            #prop_list.append(g.eval(g.gamma[5]*g.adj(g.gamma[5]*g.eval(W[i] * g.cshift(g.cshift(prop_f,transverse_direction,current_b_T),2,round(2*current_bz)))*g.gamma[5])))
            prop_list.append(g.eval(W[i] * g.cshift(g.cshift(prop_f,transverse_direction,current_b_T),2,round(2*current_bz)))) 
        return prop_list

    def create_fw_prop_TMD_CG(self, prop_f, W_index_list):
        g.message("Creating list of prop_f with shift bT and bz")
        prop_list = []
        
        for i, idx in enumerate(W_index_list):

            current_b_T = idx[0]
            current_bz = idx[1]
            current_eta = idx[2]
            transverse_direction = idx[3]

            prop_list.append(g.eval(g.cshift(g.cshift(prop_f,transverse_direction,current_b_T),2,round(current_bz)))) 

        return prop_list

    def create_fw_prop_PDF(self, prop_f, W, W_index_list):
        g.message("Creating list of W*prop_f")
        prop_list = []
        
        for i, idx in enumerate(W_index_list):

            current_b_T = idx[0]
            current_bz = idx[1]
            current_eta = idx[2]
            transverse_direction = idx[3]
            assert current_b_T == 0
            assert current_eta == 0
            assert transverse_direction == 0

            prop_list.append(g.eval(W[i] * g.cshift(g.cshift(prop_f,0,0),2,round(current_bz)))) 
        return prop_list

    def create_bw_seq_Pyquda(self, dirac, prop, trafo, flavor, origin=None, interpolation = "5"):
        tmp_trafo = g.convert(trafo, prop.grid.precision) #Need later for mixed precision solver
        
        prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.boost_out)
        
        pp = 2.0 * np.pi * np.array(self.pf) / prop.grid.fdimensions
        P = g.exp_ixp(pp, origin)
        
        src_seq = [g.mspincolor(prop.grid) for i in range(len(self.pol_list))]
        dst_seq = []
        dst_tmp = g.mspincolor(prop.grid)
        
        #g.qcd.baryon.proton_seq_src(prop, src_seq, self.t_insert, flavor)
        for i, pol in enumerate(self.pol_list):

            if (flavor == 1): 
                g.message("starting diquark contractions for up quark insertion and Polarization ", pol)

                #TODO: Jinchen, new interpolation operator
                if interpolation == "5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, Cg5, PolProjections[pol])
                elif interpolation == "T5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, CgT5, PolProjections[pol]) 
                elif interpolation == "Z5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, CgZ5, PolProjections[pol]) 
                else:
                    raise ValueError("Invalid interpolation operator")
                
            elif (flavor == 2):
                g.message("starting diquark contractions for down quark insertion and Polarization ", pol)

                #TODO: Jinchen, new interpolation operator
                if interpolation == "5":
                    src_seq[i] = self.down_quark_insertion(prop, Cg5, PolProjections[pol])
                elif interpolation == "T5":
                    src_seq[i] = self.down_quark_insertion(prop, CgT5, PolProjections[pol]) 
                elif interpolation == "Z5":
                    src_seq[i] = self.down_quark_insertion(prop, CgZ5, PolProjections[pol]) 
                else:
                    raise ValueError("Invalid interpolation operator")
            else: 
                raise Exception("Unknown flavor for backward sequential src construction")
        
            # sequential solve through t=t_insert
            src_seq_t = g.lattice(src_seq[i])
            src_seq_t[:] = 0
            src_seq_t[:, :, :, (origin[3]+self.t_insert)%prop.grid.fdimensions[3]] = src_seq[i][:, :, :, (origin[3]+self.t_insert)%prop.grid.fdimensions[3]]

            g.message("diquark contractions for Polarization ", i, pol, " done")
        
            smearing_input = g.eval(g.gamma[5]*P*g.adj(src_seq_t))

            tmp_prop = g.create.smear.boosted_smearing(trafo, smearing_input,w=self.width, boost=self.boost_out)

            src_pyquda = gpt.LatticePropagatorGPT(tmp_prop, GEN_SIMD_WIDTH)
            prop_pyquda = core.invertPropagator(dirac, src_pyquda, 1, 0) # NOTE or "prop_pyquda = core.invertPropagator(dirac, src_pyquda, 0)" depends on the quda version
            dst_tmp = g.mspincolor(prop.grid)
            gpt.LatticePropagatorGPT(dst_tmp, GEN_SIMD_WIDTH, prop_pyquda)
            del src_pyquda, prop_pyquda

            dst_seq.append(g.eval(g.adj(dst_tmp) * g.gamma[5]))

        return dst_seq
    
    def create_bw_seq_Pyquda_pyquda(self, dirac, prop, trafo, flavor, origin=None, interpolation = "5"):
        tmp_trafo = g.convert(trafo, prop.grid.precision) #Need later for mixed precision solver
        
        prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.boost_out)
        
        pp = 2.0 * np.pi * np.array(self.pf) / prop.grid.fdimensions
        P = g.exp_ixp(pp, origin)
        
        src_seq = [g.mspincolor(prop.grid) for i in range(len(self.pol_list))]
        # dst_seq = cp.zeros((len(self.pol_list), 2, prop.grid.fdimensions[3], prop.grid.fdimensions[2], prop.grid.fdimensions[1], int(prop.grid.fdimensions[0]/2), 4, 4, 3, 3), "<c16") # even/odd, t, z, y, x/2, spin, spin, color, color
        
        dst_seq = []
        
        #g.qcd.baryon.proton_seq_src(prop, src_seq, self.t_insert, flavor)
        for i, pol in enumerate(self.pol_list):

            if (flavor == 1): 
                g.message("starting diquark contractions for up quark insertion and Polarization ", pol)

                #TODO: Jinchen, new interpolation operator
                if interpolation == "5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, Cg5, PolProjections[pol])
                elif interpolation == "T5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, CgT5, PolProjections[pol]) 
                elif interpolation == "Z5":
                    src_seq[i] = self.up_quark_insertion(prop, prop, CgZ5, PolProjections[pol]) 
                else:
                    raise ValueError("Invalid interpolation operator")
                
            elif (flavor == 2):
                g.message("starting diquark contractions for down quark insertion and Polarization ", pol)

                #TODO: Jinchen, new interpolation operator
                if interpolation == "5":
                    src_seq[i] = self.down_quark_insertion(prop, Cg5, PolProjections[pol])
                elif interpolation == "T5":
                    src_seq[i] = self.down_quark_insertion(prop, CgT5, PolProjections[pol]) 
                elif interpolation == "Z5":
                    src_seq[i] = self.down_quark_insertion(prop, CgZ5, PolProjections[pol]) 
                else:
                    raise ValueError("Invalid interpolation operator")
            else: 
                raise Exception("Unknown flavor for backward sequential src construction")
        
            # sequential solve through t=t_insert
            src_seq_t = g.lattice(src_seq[i])
            src_seq_t[:] = 0
            src_seq_t[:, :, :, (origin[3]+self.t_insert)%prop.grid.fdimensions[3]] = src_seq[i][:, :, :, (origin[3]+self.t_insert)%prop.grid.fdimensions[3]]

            g.message("diquark contractions for Polarization ", i, pol, " done")
        
            smearing_input = g.eval(g.gamma[5]*P*g.adj(src_seq_t))

            tmp_prop = g.create.smear.boosted_smearing(trafo, smearing_input,w=self.width, boost=self.boost_out)

            src_pyquda = gpt.LatticePropagatorGPT(tmp_prop, GEN_SIMD_WIDTH)
            prop_pyquda = core.invertPropagator(dirac, src_pyquda, 1, 0) # NOTE or "prop_pyquda = core.invertPropagator(dirac, src_pyquda, 0)" depends on the quda version
            
            prop_pyquda_contracted = contract( "wtzyxijfc, ik -> wtzyxjkcf", prop_pyquda.data.conj(), G5 )
            
            dst_seq.append(prop_pyquda_contracted)

        dst_seq = cp.asarray(dst_seq)

        return dst_seq

    
    def contract_TMD(self, prop_f, prop_bw_seq, phases, W_index, tag, iW):
        
        corr = g.slice_trDA(prop_bw_seq, prop_f, phases,3)

        for pol_index in range(len(prop_bw_seq)):
            pol_tag = tag + "." + self.pol_list[pol_index]
            
            corr_write = [corr[pol_index]]  
            
            if g.rank() == pol_index:
                #print('g.rank():',g.rank(), ', pol_tag:', pol_tag)
                save_qTMD_proton_hdf5_subset(corr_write, pol_tag, my_gammas, self.plist, [W_index], iW, self.t_insert)

    def contract_PDF(self, prop_f, prop_bw_seq, phases, W_index, tag, iW):
        
        corr = g.slice_trDA(prop_bw_seq, prop_f, phases,3)

        for pol_index in range(len(prop_bw_seq)):
            pol_tag = tag + "." + self.pol_list[pol_index]
            
            corr_write = [corr[pol_index]]  
            
            if g.rank() == pol_index:
                #print('g.rank():',g.rank(), ', pol_tag:', pol_tag)
                save_qTMD_proton_hdf5_subset(corr_write, pol_tag, my_gammas, self.qlist, [W_index], iW, self.t_insert)
    
    def create_PDF_Wilsonline_index_list(self, grid):
        index_list = []
        
        for current_bz in range(0, grid.fdimensions[0]//4+1):
            # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
            index_list.append([0, current_bz, 0, 0])
            
            # create Wilson lines from all to all - (eta+bz) + b_perp - (eta-b_z)
            #if current_bz != 0:
            #    index_list.append([0, -current_bz, 0, 0])
                    
        return index_list
    
    def create_TMD_Wilsonline_index_list(self):
        index_list = []
        
        for transverse_direction in [0,1]:
            for current_eta in self.eta:
                
                if current_eta <= 12:
                    for current_bz in range(0, min([self.b_z+1, current_eta+1])):
                        for current_b_T in range(0, min([self.b_T+1, current_eta+1])):
                            
                            # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
                            index_list.append([current_b_T, current_bz, current_eta, transverse_direction])
                            
                            # create Wilson lines from all to all - (eta+bz) + b_perp - (eta-b_z)
                            index_list.append([current_b_T, -current_bz, -current_eta, transverse_direction])
                else:
                    # create Wilson lines from all to all + (eta+0) + b_perp - (eta-0)
                    for current_b_T in range(0, min([self.b_T+1, current_eta+1])):
                        index_list.append([current_b_T, 0, current_eta, transverse_direction])
                    
        return index_list
        
    def create_TMD_Wilsonline_index_list_CG(self, grid):
        index_list = []
        
        for transverse_direction in [0,1]:
            for current_bz in range(0, self.b_z+1):
                for current_b_T in range(0, self.b_T+1):
            
                    # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
                    index_list.append([current_b_T, current_bz, 0, transverse_direction])
                    
                    # create Wilson lines from all to all - (eta+bz) + b_perp - (eta-b_z)
                    #if current_bz != 0:
                    #    index_list.append([current_b_T, -current_bz, 0, transverse_direction])
                    
        return index_list
    
    def create_PDF_Wilsonline(self, U, index_set):

        assert len(index_set) == 4
        bt_index = index_set[0]
        bz_index = index_set[1]
        eta_index = index_set[2]
        transverse_dir = index_set[3]
        assert bt_index == 0
        assert eta_index == 0
        assert transverse_dir == 0
        
        prv_link = g.qcd.gauge.unit(U[2].grid)[0]
        WL = prv_link

        if bz_index >= 0:
            for dz in range(0, bz_index):
                WL = g.eval(prv_link * g.cshift(U[2], 2, dz))
                prv_link = WL
        else:
            for dz in range(0, abs(bz_index)):
                WL = g.eval(prv_link * g.adj(g.cshift(U[2],2, -dz-1)))
                prv_link = WL

        return WL
    
    def create_TMD_Wilsonline(self, U, index_set):

        assert len(index_set) == 4
        bt_index = index_set[0]
        bz_index = index_set[1]
        eta_index = index_set[2]
        transverse_dir = index_set[3]
        
        prv_link = g.qcd.gauge.unit(U[2].grid)[0]
        WL = prv_link

        if eta_index+bz_index >= 0:
            for dz in range(0, eta_index+bz_index):
                WL = g.eval(prv_link * g.cshift(U[2], 2, dz))
                prv_link = WL
        else:
            for dz in range(0, abs(eta_index+bz_index)):
                WL = g.eval(prv_link * g.adj(g.cshift(U[2],2, -dz-1)))
                prv_link = WL
        
        # dx and bt_index are >=0
        for dx in range(0, bt_index):
            WL=g.eval(prv_link * g.cshift(g.cshift(U[transverse_dir], 2, eta_index+bz_index),transverse_dir, dx))
            prv_link=WL

        if eta_index-bz_index >= 0:
            for dz in range(0, eta_index-bz_index):
                WL=g.eval(prv_link * g.adj(g.cshift(g.cshift(g.cshift(U[2], 2, eta_index+bz_index-1), transverse_dir, bt_index),2,-dz)))
                prv_link=WL
        else:
            for dz in range(0, abs(eta_index-bz_index)):
                WL=g.eval(prv_link * g.cshift(g.cshift(g.cshift(U[2], 2, eta_index+bz_index), transverse_dir, bt_index),2,dz))
                prv_link=WL

        return WL

    def create_TMD_Wilsonline_CG(self, U, index_set):

        assert len(index_set) == 4
        bt_index = index_set[0]
        bz_index = index_set[1]
        eta_index = index_set[2]
        transverse_dir = index_set[3]

        return g.qcd.gauge.unit(U[2].grid)[0]
            
    def create_TMD_Wilsonline_CG_Tlink(self, U, index_set):

        assert len(index_set) == 4
        bt_index = index_set[0]
        bz_index = index_set[1]
        eta_index = index_set[2]
        transverse_dir = index_set[3]
        
        prv_link = g.qcd.gauge.unit(U[2].grid)[0]
        WL = prv_link
        
        # dx and bt_index are >=0
        for dx in range(0, bt_index):
            WL=g.eval(prv_link * g.cshift(g.cshift(U[transverse_dir], 2, eta_index+bz_index),transverse_dir, dx))
            prv_link=WL

        return WL
    
    def down_quark_insertion(self, Q, Gamma, P):
        #eps_abc eps_a'b'c'Gamma_{beta alpha}Gamma_{beta'alpha'}P_{gamma gamma'}
        # * ( Q^beta'beta_b'b Q^gamma'gamma_{c'c} -  Q^beta'gamma_b'c Q^gamma'beta_{c'b} )
        
        eps = g.epsilon(Q.otype.shape[2])
        
        R = g.lattice(Q)
        
        PDu = g(g.spin_trace(P*Q))

        GtDG = g.eval(g.transpose(Gamma)*Q*Gamma)

        GtDG = g.separate_color(GtDG)
        PDu = g.separate_color(PDu)
        
        GtD = g.eval(g.transpose(Gamma)*Q)
        PDG = g.eval(P*Q*Gamma)
        
        GtD = g.separate_color(GtD)
        PDG = g.separate_color(PDG)
        
        D = {x: g.lattice(GtDG[x]) for x in GtDG}

        for d in D:
            D[d][:] = 0
            
        for i1, sign1 in eps:
            for i2, sign2 in eps:
                D[i1[0], i2[0]] += -sign1 * sign2 * g.transpose((PDu[i2[2], i1[2]] * GtDG[i2[1], i1[1]] - GtD[i2[1],i1[2]] * PDG[i2[2], i1[1]]))
                
        g.merge_color(R, D)
        return R

    #Qlua definition, reproduce the results as Chroma difinition
    def up_quark_insertion(self, Qu, Qd, Gamma, P):

        eps = g.epsilon(Qu.otype.shape[2])
        R = g.lattice(Qu)

        Du_sep = g.separate_color(Qu)
        GDd = g.eval(Gamma * Qd)
        GDd = g.separate_color(GDd)

        PDu = g.eval(P*Qu)
        PDu = g.separate_color(PDu)

        # ut
        DuP = g.eval(Qu * P)
        DuP = g.separate_color(DuP)
        TrDuP = g(g.spin_trace(Qu * P))
        TrDuP = g.separate_color(TrDuP)
        
        # s2ds1b
        GtDG = g.eval(g.transpose(Gamma)*Qd*Gamma)
        GtDG = g.separate_color(GtDG)

        #sum color indices
        D = {x: g.lattice(GDd[x]) for x in GDd}
        for d in D:
            D[d][:] = 0

        for i1, sign1 in eps:
            for i2, sign2 in eps:
                D[i2[2], i1[2]] += -sign1 * sign2 * (P * g.spin_trace(GtDG[i1[1],i2[1]]*g.transpose(Du_sep[i1[0],i2[0]]))
                                    + g.transpose(TrDuP[i1[0],i2[0]] * GtDG[i1[1],i2[1]])
                                    + PDu[i1[0],i2[0]] * g.transpose(GtDG[i1[1],i2[1]])
                                    + g.transpose(GtDG[i1[0],i2[0]]) * DuP[i1[1],i2[1]])
        
        g.merge_color(R, D)

        return R
