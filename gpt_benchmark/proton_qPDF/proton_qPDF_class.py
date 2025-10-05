import gpt as g
import numpy as np

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

Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
Szp = (g.gamma["I"].tensor() - 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Szm = (g.gamma["I"].tensor() + 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Sxp = (g.gamma["I"].tensor() - 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
Sxm = (g.gamma["I"].tensor() + 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
PpSzp = Pp * Szp
PpSzm = Pp * Szm
PpSxp = Pp * Sxp
PpSxm = Pp * Sxm

PolProjections = {
    "PpSzp": PpSzp,
    "PpSzm": PpSzm,
    "PpSxp": PpSxp,
    "PpSxm": PpSxm,  
}

"""
================================================================================
                                proton_qPDF
================================================================================
"""
class proton_qPDF:

    def __init__(self, parameters):
        # PDF-specific parameters only
        self.hadron_momentum = parameters["hadron_momentum"] # single hadron momentum state

        self.width = parameters["width"] # Gaussian smearing width
        self.boost_in = parameters["boost_in"] # Forward propagator boost smearing
        self.boost_out = parameters["boost_out"] # Backward propagator boost smearing
        self.pos_boost = self.boost_in # Forward propagator boost smearing for 2pt

        self.pol_list = parameters["pol"] # projection of nucleon state
        self.tsep_list = parameters["tsep_list"] # time separation list for three point function

        self.z_max = parameters["z_max"] # maximum z separation for PDF calculation
    
    def make_mom_phases_2pt(self, grid, origin=None):    
        one = g.identity(g.complex(grid))
        pp = 2 * np.pi * np.array(self.hadron_momentum) / grid.fdimensions

        P = g.exp_ixp([pp], origin)
        mom = [g.eval(pp*one) for pp in P]
        return mom
    
    def contract_2pt_PDF(self, prop_f, phases, trafo):
        # g.message("Begin sink smearing")
        # tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        # prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        # g.message("Sink smearing completed")
        
        dq = g.qcd.baryon.diquark(g(prop_f * Cg5), g(Cg5 * prop_f))
        # dq = g.qcd.baryon.diquark(g(prop_f * CgT5), g(CgT5 * prop_f))
        proton1 = g(g.spin_trace(dq) * prop_f + dq * prop_f)
        prop_unit = g.mspincolor(prop_f.grid)
        prop_unit = g.identity(prop_unit)
        corr = g.slice_trDA([prop_unit], [proton1], phases, 3)
        
        return corr
        
    
    def create_fw_prop_PDF(self, prop_f, W, W_index_list):
        g.message("Creating list of W*prop_f for PDF")
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
    
    def create_bw_seq(self, inverter, prop, trafo, flavor, tsep, hadron_momentum, origin=None):
        tmp_trafo = g.convert(trafo, prop.grid.precision)
        
        # prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.boost_out)
        
        pp = 2.0 * np.pi * np.array(hadron_momentum) / prop.grid.fdimensions
        P = g.exp_ixp(pp, origin)
        
        src_seq = [g.mspincolor(prop.grid) for i in range(len(self.pol_list))]
        dst_seq = []
        dst_tmp = g.mspincolor(prop.grid)
        
        for i, pol in enumerate(self.pol_list):

            if (flavor == 1): 
                g.message("starting diquark contractions for up quark insertion and Polarization ", i)
                src_seq[i] = self.up_quark_insertion(prop, prop, Cg5, PolProjections[pol])
                # src_seq[i] = self.up_quark_insertion(prop, prop, CgT5, PolProjections[pol])
            elif (flavor == 2):
                g.message("starting diquark contractions for down quark insertion and Polarization ", i)
                src_seq[i] = self.down_quark_insertion(prop, Cg5, PolProjections[pol])
                # src_seq[i] = self.down_quark_insertion(prop, CgT5, PolProjections[pol])
            else: 
                raise Exception("Unknown flavor for backward sequential src construction")
        
            # sequential solve through t=tsep
            src_seq_t = g.lattice(src_seq[i])
            src_seq_t[:] = 0
            src_seq_t[:, :, :, (origin[3]+tsep)%prop.grid.fdimensions[3]] = src_seq[i][:, :, :, (origin[3]+tsep)%prop.grid.fdimensions[3]]

            g.message("diquark contractions for Polarization ", i, " done")
        
            smearing_input = g.eval(g.gamma[5]*P*g.adj(src_seq_t))
            # tmp_prop = g.create.smear.boosted_smearing(trafo, smearing_input, w=self.width, boost=self.boost_out)
            tmp_prop = smearing_input
            dst_tmp = g.eval(inverter * tmp_prop)           
            dst_seq.append(g.eval(g.adj(dst_tmp) * g.gamma[5]))

        g.message("bw. seq propagator done")
        return dst_seq

    def contract_PDF(self, prop_f, prop_bw_seq, phases):
        """Contract PDF and return correlation data"""
        corr = g.slice_trDA(prop_bw_seq, prop_f, phases, 3)
        return corr

    def create_PDF_Wilsonline_index_list(self):
        """Create Wilson line index list for PDF calculation"""
        index_list = []
        
        for current_bz in range(0, self.z_max + 1):
            # For PDF, create Wilson lines with different z separations
            index_list.append([0, current_bz, 0, 0])  # [bT, bz, eta, T direction]
                    
        return index_list

    def get_z_separations(self):
        """Get list of z separations that will be calculated"""
        return list(range(0, self.z_max + 1))

    def get_momentum_label(self, momentum):
        """Generate momentum label for HDF5 keys"""
        px, py, pz, pt = momentum
        return f"px{px}_py{py}_pz{pz}_pt{pt}"

    def print_calculation_info(self):
        """Print information about the PDF calculation"""
        z_seps = self.get_z_separations()
        g.message(f"PDF Calculation Setup:")
        g.message(f"  - Z separations: {z_seps} (total: {len(z_seps)} points)")
        g.message(f"  - Nucleon polarizations: {self.pol_list}")
        g.message(f"  - Hadron momentum: {self.hadron_momentum}")
        g.message(f"  - Time insertion: {self.tsep_list}")

    def create_PDF_Wilsonline(self, U, index_set):
        """Create Wilson line for PDF calculation"""
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

    def down_quark_insertion(self, Q, Gamma, P):
        """Down quark insertion for sequential source"""
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

    def up_quark_insertion(self, Qu, Qd, Gamma, P):
        """Up quark insertion for sequential source"""
        eps = g.epsilon(Qu.otype.shape[2])
        R = g.lattice(Qu)

        Du_sep = g.separate_color(Qu)
        GDd = g.eval(Gamma * Qd)
        GDd = g.separate_color(GDd)

        PDu = g.eval(P*Qu)
        PDu = g.separate_color(PDu)

        DuP = g.eval(Qu * P)
        DuP = g.separate_color(DuP)
        TrDuP = g(g.spin_trace(Qu * P))
        TrDuP = g.separate_color(TrDuP)
        
        GtDG = g.eval(g.transpose(Gamma)*Qd*Gamma)
        GtDG = g.separate_color(GtDG)

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

    def create_src_2pt(self, pos, trafo, grid):
        """Create 2pt source for propagator calculation"""
        src = g.mspincolor(grid)
        g.create.point(src, pos)
        trafo = g.convert(trafo, grid.precision)
        # src = g.create.smear.boosted_smearing(trafo, src, w=self.width, boost=self.boost_in)
        return src 