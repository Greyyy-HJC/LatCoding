"""
This is a standard script to measure pion mass.
"""
# %%
import gpt as g
import gvar as gv
import numpy as np
from pathlib import Path

Ls = 8
Lt = 8
root_dir = Path(__file__).resolve().parents[3]
conf_dir = root_dir / "configs" / f"S{Ls}T{Lt}"
dump_dir = Path(__file__).resolve().parents[1] / "dump"
conf_n_ls = np.arange(0, 3)


# Main loop
corr_conf_ls = []
for conf_n in conf_n_ls:
    conf_path = conf_dir / f"wilson_b6.{conf_n}"
    U_read = g.convert(g.load(str(conf_path)), g.double) # load configuration

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

    # create point source
    src = g.mspincolor(grid)
    g.create.point(src, [0, 0, 0, 0])

    # build solver using eo prec. and cg
    inv = g.algorithms.inverter
    pc = g.qcd.fermion.preconditioner
    cg = inv.cg({"eps": 1e-8, "maxiter": 10000})

    slv = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

    # propagator
    dst = g.mspincolor(grid)
    dst @= slv * src

    # pi -pi corr:
    corr_pion = g.slice(g.trace(g.adj(dst) * dst), 3)
    corr_conf_ls.append(np.real(corr_pion))

dump_dir.mkdir(parents=True, exist_ok=True)
gv.dump(corr_conf_ls, str(dump_dir / "pion_mass_conf_ls.dat"))
# %%
