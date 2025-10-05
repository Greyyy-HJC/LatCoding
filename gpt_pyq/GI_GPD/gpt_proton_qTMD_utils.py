from cmath import phase
from math import gamma
import gpt as g
import numpy as np
from io_corr import *

#ordered list of gamma matrix identifiers, needed for the tag in the correlator output
my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
#my_proton_proj = ["P+","P+_Sz+","P+_Sx+","P+_Sx-"]
#my_proton_proj = ["P+"]

ordered_list_of_gammas = [g.gamma[5], g.gamma["T"], g.gamma["T"]*g.gamma[5],
                                      g.gamma["X"], g.gamma["X"]*g.gamma[5], 
                                      g.gamma["Y"], g.gamma["Y"]*g.gamma[5],
                                      g.gamma["Z"], g.gamma["Z"]*g.gamma[5], 
                                      g.gamma["I"], g.gamma["SigmaXT"], 
                                      g.gamma["SigmaXY"], g.gamma["SigmaXZ"], 
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

class proton_measurement:
    def __init__(self, parameters):
        self.plist = parameters["plist"]
        self.pol_list = ["P+_Sz+","P+_Sx+","P+_Sx-"]
        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]

    def set_input_facilities(self, corr_file):
        self.input_correlator = g.corr_io.reader(corr_file)

    def set_output_facilities(self, corr_file, prop_file):
        self.output_correlator = g.corr_io.writer(corr_file)
        
        if(self.save_propagators):
            self.output = g.gpt_io.writer(prop_file)

    def propagator_input(self, prop_file):
        g.message(f"Reading propagator file {prop_file}")
        read_props = g.load(prop_file)
        return read_props

    def propagator_output_k0(self, tag, prop_f):

        g.message("Saving forward propagator")
        prop_f_tag = "%s/%s" % (tag, str(self.pos_boost))
        self.output.write({prop_f_tag: prop_f})
        self.output.flush()
        g.message("Propagator IO done")

    def propagator_output(self, tag, prop_f, prop_b):

        g.message("Saving forward propagator")
        prop_f_tag = "%s/%s" % (tag, str(self.pos_boost)) 
        self.output.write({prop_f_tag: prop_f})
        self.output.flush()
        g.message("Saving backward propagator")
        prop_b_tag = "%s/%s" % (tag, str(self.neg_boost))
        self.output.write({prop_b_tag: prop_b})
        self.output.flush()
        g.message("Propagator IO done")

    def make_24D_inverter(self, U, evec_file):

        l_exact = g.qcd.fermion.zmobius(
            #g.convert(U, g.single),
            U,
            {
                "mass": 0.00107,
                "M5": 1.8,
                "b": 1.0,
                "c": 0.0,
                "omega": [
                    1.0903256131299373,
                    0.9570283702230611,
                    0.7048886040934104,
                    0.48979921782791747,
                    0.328608311201356,
                    0.21664245377015995,
                    0.14121112711957107,
                    0.0907785101745156,
                    0.05608303440064219 - 0.007537158177840385j,
                    0.05608303440064219 + 0.007537158177840385j,
                    0.0365221637144842 - 0.03343945161367745j,
                    0.0365221637144842 + 0.03343945161367745j,
                ],
                "boundary_phases": [1.0, 1.0, 1.0, -1.0],
            },
        )

        l_sloppy = l_exact.converted(g.single)
        g.message(f"Loading eigenvectors from {evec_file}")
        g.mem_report(details=False)
        eig = g.load(evec_file, grids=l_sloppy.F_grid_eo)

        g.mem_report(details=False)
        pin = g.pin(eig[1], g.accelerator)
        g.message("creating deflated solvers")

        g.message("creating deflated solvers")
        light_innerL_inverter = g.algorithms.inverter.preconditioned(
           g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
           g.algorithms.inverter.sequence(
               g.algorithms.inverter.coarse_deflate(
                   eig[1],
                   eig[0],
                   eig[2],
                   block=400,
                   fine_block=4,
                   linear_combination_block=32,
               ),
               g.algorithms.inverter.split(
                   g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 200}),
                   mpi_split=g.default.get_ivec("--mpi_split", None, 4),
               ),
           ),
        )

        light_innerH_inverter = g.algorithms.inverter.preconditioned(
            g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
            g.algorithms.inverter.sequence(
               g.algorithms.inverter.coarse_deflate(
                   eig[1],
                   eig[0],
                   eig[2],
                   block=400,
                   fine_block=4,
                   linear_combination_block=32,
               ),
               g.algorithms.inverter.split(
                   g.algorithms.inverter.cg({"eps": 1e-4, "maxiter": 200}),
                   mpi_split=g.default.get_ivec("--mpi_split", None, 4),
               ),
           ),
        )

        g.mem_report(details=False)
        light_exact_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=12,
        )

        light_sloppy_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
            eps=1e-4,
            maxiter=12,
        )


        ############### final inverter definitions
        prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(4)
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(4)

        return prop_l_exact, prop_l_sloppy, pin
            

    def make_64I_inverter(self, U, evec_file):
        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #64I params
                "mass": 0.000678,
                "M5": 1.8,
                "b": 1.5,
                "c": 0.5,
                "Ls": 12,
                "boundary_phases": [1.0, 1.0, 1.0, 1.0],
                },

        )

        l_sloppy = l_exact.converted(g.single)
        g.message(f"Loading eigenvectors from {evec_file}")
        g.mem_report(details=False)
        eig = g.load(evec_file, grids=l_sloppy.F_grid_eo)

        g.mem_report(details=False)
        pin = g.pin(eig[1], g.accelerator)
        g.message("creating deflated solvers")

        light_innerL_inverter = g.algorithms.inverter.preconditioned(
           g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
           g.algorithms.inverter.sequence(
               g.algorithms.inverter.coarse_deflate(
                   eig[1],
                   eig[0],
                   eig[2],
                   block=400,
                   fine_block=4,
                   linear_combination_block=32,
               ),
               g.algorithms.inverter.split(
                   g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 200}),
                   mpi_split=g.default.get_ivec("--mpi_split", None, 4),
               ),
           ),
        )

        light_innerH_inverter = g.algorithms.inverter.preconditioned(
            g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
            g.algorithms.inverter.sequence(
               g.algorithms.inverter.coarse_deflate(
                   eig[1],
                   eig[0],
                   eig[2],
                   block=400,
                   fine_block=4,
                   linear_combination_block=32,
               ),
               g.algorithms.inverter.split(
                   g.algorithms.inverter.cg({"eps": 1e-4, "maxiter": 200}),
                   mpi_split=g.default.get_ivec("--mpi_split", None, 4),
               ),
           ),
        )

        g.mem_report(details=False)
        light_exact_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=12,
        )

        light_sloppy_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
            eps=1e-4,
            maxiter=12,
        )


        ############### final inverter definitions
        prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(4)
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(4)

        return prop_l_exact, prop_l_sloppy, pin

    def make_debugging_inverter(self, U):

        
        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #96I params
                #"mass": 0.00054,
                #"M5": 1.8,
                #"b": 1.5,
                #"c": 0.5,
                #"Ls": 12,
                #"boundary_phases": [1.0, 1.0, 1.0, -1.0],},
        #MDWF_2+1f_64nt128_IWASAKI_b2.25_ls12b+c2_M1.8_ms0.02661_mu0.000678_rhmc_HR_G
                # 64I params
                "mass": 0.0006203,
                "M5": 1.8,
                "b": 1.5,
                "c": 0.5,
                "Ls": 12,
                "boundary_phases": [1.0, 1.0, 1.0, 1.0],},
                #48I params
                # "mass": 0.00078,
                # "M5": 1.8,
                # "b": 1.5,
                # "c": 0.5,
                # "Ls": 24,
                # "boundary_phases": [1.0, 1.0, 1.0, -1.0],},
        )
        
        # l_exact = g.qcd.fermion.zmobius(
        #     #g.convert(U, g.single),
        #     U,
        #     {
        #         "mass": 0.00107,
        #         "M5": 1.8,
        #         "b": 1.0,
        #         "c": 0.0,
        #         "omega": [
        #             1.0903256131299373,
        #             0.9570283702230611,
        #             0.7048886040934104,
        #             0.48979921782791747,
        #             0.328608311201356,
        #             0.21664245377015995,
        #             0.14121112711957107,
        #             0.0907785101745156,
        #             0.05608303440064219 - 0.007537158177840385j,
        #             0.05608303440064219 + 0.007537158177840385j,
        #             0.0365221637144842 - 0.03343945161367745j,
        #             0.0365221637144842 + 0.03343945161367745j,
        #         ],
        #         "boundary_phases": [1.0, 1.0, 1.0, -1.0],
        #     },
        # )
        
        l_sloppy = l_exact.converted(g.single)

        light_innerL_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo2_ne(), g.algorithms.inverter.cg(eps = 1e-4, maxiter = 10000))
        light_innerH_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo2_ne(), g.algorithms.inverter.cg(eps = 1e-4, maxiter = 200))

        prop_l_sloppy = l_exact.propagator(light_innerH_inverter).grouped(6)
        prop_l_exact = l_exact.propagator(light_innerL_inverter).grouped(6)
        return prop_l_exact, prop_l_sloppy

    ############## make list of complex phases for momentum proj.
    def make_mom_phases(self, grid, origin=None):    
        one = g.identity(g.complex(grid))
        pp = [-2 * np.pi * np.array(p) / grid.fdimensions for p in self.plist]
       
        P = g.exp_ixp(pp, origin)
        mom = [g.eval(pp*one) for pp in P]
        return mom

    # create Wilson lines from all --> all + dz for all dz in 0,zmax
    def create_WL(self, U):
        W = []
        W.append(g.qcd.gauge.unit(U[2].grid)[0])
        for dz in range(0, self.zmax):
            W.append(g.eval(W[dz] * g.cshift(U[2], 2, dz)))
                
        return W

    '''
    #function that does the contractions for the smeared-smeared pion 2pt function
    def contract_2pt(self, prop_f, phases, trafo, tag):

        #g.message("Begin sink smearing")
        #tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        #prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        #g.message("Sink smearing completed")

        corr = g.slice_proton(prop_f, phases, 3) 
        
        if g.rank() == 0:
            save_proton_c2pt_hdf5(corr, tag, my_proton_proj, self.plist)
        del corr 
    '''



    #function that does the contractions for the smeared-smeared pion 2pt function
    def contract_2pt_SRC(self, prop_f, phases, trafo, tag):

        g.message("Begin sink smearing")
        tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        g.message("Sink smearing completed")

        proton1 = proton_contr(prop_f, prop_f)
        corr = [[g.slice(g.eval(gm*pp),3) for pp in phases] for gm in proton1]
        
        if g.rank() == 0:
            save_proton_c2pt_hdf5(corr, tag, my_gammas, self.plist)
        del corr 

    '''
    def contract_proton_2pt(self,prop_f,phases,trafo):
        proton1 = proton_contr(prop_f, prop_f)
        
        corr = [g.slice(g.eval(proton1*pp),3) for pp in phases]
        
        return corr
    '''

    #function that creates boosted, smeared src.
    def create_src_2pt(self, pos, trafo, grid):
        
        srcD = g.mspincolor(grid)
        g.create.point(srcD, pos)
        srcDp = g.create.smear.boosted_smearing(trafo, srcD, w=self.width, boost=self.pos_boost)
        
        return srcDp


class proton_qpdf_measurement(proton_measurement):
   
    def __init__(self, parameters):
        self.zmax = parameters["zmax"]
        self.q = parameters["q"]

        self.pzmin = parameters["pzmin"]
        self.pzmax = parameters["pzmax"]
        self.plist = [ [0,0, pz, 0] for pz in range(self.pzmin,self.pzmax)]

        self.pol_list = ["P+_Sz+","P+_Sx+","P+_Sx-"]
        #self.Gamma = parameters["gamma"]
        self.t_insert = parameters["t_insert"]
        self.width = parameters["width"]
        self.boost_in = parameters["boost_in"]
        self.boost_out = parameters["boost_out"]
        self.pos_boost = self.boost_in
        self.save_propagators = parameters["save_propagators"]



    def create_fw_prop_QPDF(self, prop_f, W):
        g.message("Creating list of W*prop_f for all z")
        prop_list = [prop_f,]

        for z in range(1,self.zmax):
            prop_list.append(g.eval(W[z]*g.cshift(prop_f,2,z)))
        
        return prop_list  

    def create_bw_seq(self, inverter, prop, trafo):
        tmp_trafo = g.convert(trafo, prop.grid.precision)

        #Make SS propagator
        prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.boost_out)

        pp = 2.0 * np.pi * np.array(self.p) / prop.grid.fdimensions
        P = g.exp_ixp(pp)

        # sequential solve through t=insertion_time for all 3 proton polarizations
        src_seq = [g.mspincolor(prop.grid) for i in range(3)]
        dst_seq = []
        #g.mem_report(details=True)
        g.message("starting diquark contractions")
        g.qcd.baryon.proton_seq_src(prop, src_seq, self.t_insert)
        g.message("diquark contractions done")
        dst_tmp = g.mspincolor(prop.grid)
        for i in range(3):

            dst_tmp @= inverter * g.create.smear.boosted_smearing(tmp_trafo, g.eval(g.gamma[5]* P* g.conj(src_seq[i])), w=self.width, boost=self.boost_out)
            #del src_seq[i]
            dst_seq.append(g.eval(g.gamma[5] * g.conj( dst_tmp)))
        g.message("bw. seq propagator done")
        return dst_seq            


    def contract_QPDF(self, prop_f, prop_bw, phases, tag):
 
        #This and the IO still need work

        for pol in self.pol_list:
            corr = g.slice_trQPDF(prop_bw, prop_f, phases, 3)

            corr_tag = f"{tag}/QPDF/Pol{pol}"
            for z, corr_p in enumerate(corr):
                for i, corr_mu in enumerate(corr_p):
                    p_tag = f"{corr_tag}/pf{self.p}/q{self.q}"
                    for j, corr_t in enumerate(corr_mu):
                        out_tag = f"{p_tag}/{my_gammas[j]}"
                        self.output_correlator.write(out_tag, corr_t)
