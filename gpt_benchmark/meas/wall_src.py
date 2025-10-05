# %%
""" 
To build wall source and add momentum phase.
Compare with point source results without gauge fixing.
Checked with PyQuDA.
"""


import numpy as np
import gpt as g
import gvar as gv

def apply_phase(grid, src, mom):
    one = g.identity(g.complex(grid))
    pp = 2 * np.pi * np.array(mom) / grid.fdimensions
    phase = g.exp_ixp(pp)
    phase_eval = g.eval(phase*one)
    src = src * phase_eval
    return src

def create_wall_src(grid, mom):
    coor_ls = g.coordinates(grid)
    
    src = g.identity(g.mspincolor(grid))
    src = apply_phase(grid, src, mom)
    
    mask = g.complex(grid)
    g.coordinate_mask(mask, np.array([1 if i[3] == 0 else 0 for i in coor_ls])) # only set src at (x, y, z, t=0)
    
    src = g(src * mask)
    
    return src

conf_path = "../../conf/S8T32"
conf_n_ls = np.arange(5)

# Main loop
corr_pt_ls = []
corr_wall_ls = []

mom = [3, 0, 0, 0]

for conf_n in conf_n_ls:
    U_read = g.convert(g.load(f"{conf_path}/wilson_b6.{conf_n}"), g.double) # load configuration

    U_hyp = g.qcd.gauge.smear.hyp(U_read, alpha=(np.array([0.75, 0.6, 0.3]))) # smearing
    plaq_hyp = g.qcd.gauge.plaquette(U_hyp)

    grid = U_hyp[0].grid

    p = {
        "kappa": 0.12623,
        "csw_r": 1.02868,
        "csw_t": 1.02868,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1, 1, 1, -1],
    }

    w = g.qcd.fermion.wilson_clover(U_hyp, p)
    
    # build solver using eo prec. and cg
    inv = g.algorithms.inverter
    pc = g.qcd.fermion.preconditioner
    cg = inv.cg({"eps": 1e-8, "maxiter": 10000})
    slv = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))
    
    #! point source part
    # create point source
    src = g.mspincolor(grid)
    g.create.point(src, pos=[0, 0, 0, 0])
    
    # apply momentum phase
    src = apply_phase(grid, src, mom)

    # propagator
    dst_pt = g.mspincolor(grid)
    dst_pt = g.eval(slv * src)
    
    corr_pion_pt = g.slice(g.trace(g.adj(dst_pt) * dst_pt), 3)
    corr_pt_ls.append(np.real(corr_pion_pt))
    
    
    #! wall source part
    # create wall source
    src_wall = create_wall_src(grid, mom)
    
    # solve for wall source propagator
    dst_wall = g.mspincolor(grid)
    dst_wall = g.eval(slv * src_wall)
    
    # calculate correlation function
    corr_wall = g.slice(g.trace(g.adj(dst_wall) * dst_wall), 3)
    corr_wall_ls.append(np.real(corr_wall))

gv.dump(corr_pt_ls, f"../dump/pion_mass_pt_src.dat")
gv.dump(corr_wall_ls, f"../dump/pion_mass_wall_src.dat")

# %%
