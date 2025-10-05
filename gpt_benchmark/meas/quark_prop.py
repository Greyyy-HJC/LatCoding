"""
This is a script to measure quark propagator on the coulomb gauge configs.
"""
# %%
import gpt as g
import numpy as np
import gvar as gv

# Configuration
rng = g.random("T")
gamma_idx = "I"

conf_path = "../../conf/S8T8"
conf_n_ls = np.arange(0, 3)

# Main loop
corr_conf_ls = []
for conf_n in conf_n_ls:
    U_fixed = g.convert(g.load(f"{conf_path}/wilson_b6.{conf_n}"), g.double)

    # Quark and solver setup (same for all source positions)
    grid = U_fixed[0].grid
    L = np.array(grid.fdimensions)

    w = g.qcd.fermion.wilson_clover(
        U_fixed,
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

    # momentum
    # p = 2.0 * np.pi * np.array([1, 0, 0, 0]) / L
    # P = g.exp_ixp(p)

    # Source positions
    src = g.mspincolor(grid)
    g.create.point(src, [0,0,0,0])
    dst = g.mspincolor(grid)
    dst @= propagator * src
    correlator = g(g.trace(dst * g.gamma[gamma_idx]))[0, 0, :, 0].flatten()

    corr_conf_ls.append(np.real(correlator))

# gv.dump(corr_conf_ls, f"../dump/quark_prop_conf_ls.dat")
print(np.mean( corr_conf_ls, axis=0 ))
# %%