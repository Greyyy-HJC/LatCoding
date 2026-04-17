import gpt as g
import numpy as np

# lattice   
lat = g.grid([8,8,8,32], g.double)
U = g.convert(g.load("/home/jinchen/git/lat-software/LatCoding/conf/S8T32/wilson_b6.0"), g.double)

U_prime, U_trafo = g.gauge_fix(U, maxiter=1000, prec=1e-2)

U_trafo = g.identity(U_trafo)

# random source (spin-color field)
src = g.mspincolor(lat)
g.create.point(src, [1, 2, 1, 3])

# apply GPT boosted smearing
sm_src = g.create.smear.boosted_smearing(U_trafo, src, w=2.0, boost=(1,2,1))

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
dst = g.mspincolor(lat)
dst @= slv * sm_src

# pi -pi corr:
corr_pion = g.slice(g.trace(g.adj(dst) * dst), 3)
print(np.real(corr_pion))