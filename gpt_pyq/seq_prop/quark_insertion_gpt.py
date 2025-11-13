import numpy as np

import gpt as g

def down_quark_insertion_gpt(Q, Gamma, P):
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


def up_quark_insertion(Qu, Qd, Gamma, P):

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


lat = g.grid([8,8,8,8], g.double)
U = g.convert(g.load("/home/jinchen/git/lat-software/LatCoding/conf/S8T8/wilson_b6.0"), g.double)

U_prime, U_trafo = g.gauge_fix(U, maxiter=1000, prec=1e-2)

U_trafo = g.identity(U_trafo)

# random source (spin-color field)
src = g.mspincolor(lat)
g.create.point(src, [1, 2, 1, 3])

p = {
    "mass": -0.038888,
    "csw_r": 1.02868,
    "csw_t": 1.02868,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [1, 1, 1, -1],
}

w = g.qcd.fermion.wilson_clover(U, p)

# build solver using eo prec. and cg
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-8, "maxiter": 1000})

slv = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

# propagator
prop = g.mspincolor(lat)
prop @= slv * src

Cg5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma[5].tensor()

Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
Szp = (g.gamma["I"].tensor() - 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Szm = (g.gamma["I"].tensor() + 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Sxp = (g.gamma["I"].tensor() - 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
Sxm = (g.gamma["I"].tensor() + 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
PpSzp = Pp * Szp

src_seq_gpt_down = down_quark_insertion_gpt(prop, Cg5, PpSzp)

print(np.shape(src_seq_gpt_down[:]))
print(np.linalg.norm(src_seq_gpt_down[:])**2)

src_seq_gpt_up = up_quark_insertion(prop, prop, Cg5, PpSzp)

print(np.shape(src_seq_gpt_up[:]))
print(np.linalg.norm(src_seq_gpt_up[:])**2)

