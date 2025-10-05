#!/usr/bin/env python3
'''
This is a script to run gauge fixing without using g.gauge_fix function.
'''

import gpt as g
import numpy as np
import sys

# Parameters
gauge_file = "../conf/S8T8/wilson_b6.0"
precision = 1e-10

# Optimizer parameters
p_maxiter_cg = 500
p_maxiter_gd = 2500
p_eps = precision
p_step = 0.03
p_gd_step = 0.05
p_max_abs_step = 0.3
p_theta_eps = 1e-14

g.message(f"""
  Coulomb gauge fixer run with:
    maxiter_cg    = {p_maxiter_cg}
    maxiter_gd    = {p_maxiter_gd}
    eps           = {p_eps}
    step          = {p_step}
    gd_step       = {p_gd_step}
    max_abs_step  = {p_max_abs_step}
    theta_eps     = {p_theta_eps}
""")

# Load configuration and convert to double precision
U = g.convert(g.load(gauge_file), g.double)

# Apply random gauge transformation first
rng = g.random("T")
V0 = g.identity(U[1])
rng.element(V0)
U = g.qcd.gauge.transformed(U, V0)

# split in time
Nt = U[0].grid.gdimensions[3]
g.message(f"Separate {Nt} time slices")
Usep = [g.separate(u, 3) for u in U[0:3]]
Vt = [g.mcolor(Usep[0][0].grid) for t in range(Nt)]

# optimizer
opt = g.algorithms.optimize
cg = opt.non_linear_cg(
    maxiter=p_maxiter_cg,
    eps=p_eps,
    step=p_step,
    line_search=opt.line_search_quadratic,
    beta=opt.polak_ribiere,
    max_abs_step=p_max_abs_step,
)
gd = opt.gradient_descent(maxiter=p_maxiter_gd, eps=p_eps, step=p_gd_step)

# Coulomb functional on each time-slice
g.message(f"Start gauge fixing on {Nt} time slices")
for t in range(Nt):
    f = g.qcd.gauge.fix.landau([Usep[mu][t] for mu in range(3)])
    fa = opt.fourier_accelerate.inverse_phat_square(Vt[t].grid, f)

    g.message(f"Run time slice {t} / {Nt}")
    Vt[t] @= g.identity(Vt[t])

    if not cg(fa)(Vt[t], Vt[t]):
        gd(fa)(Vt[t], Vt[t])

    group_defect = g.group.defect(Vt[t])
    g.message(f"Distance to group manifold: {group_defect}")
    if group_defect > 1e-12:
        g.message(f"Time slice {t} has group_defect = {group_defect}")
        sys.exit(1)

g.message("Project to group (should only remove rounding errors)")
Vt = [g.project(vt, "defect") for vt in Vt]

# test results and print functional values
g.message(">>> Final Functional Values per time slice:")
for t in range(Nt):
    f = g.qcd.gauge.fix.landau([Usep[mu][t] for mu in range(3)])
    dfv = f.gradient(Vt[t], Vt[t])
    theta = g.norm2(dfv).real / Vt[t].grid.gsites / dfv.otype.Nc
    g.message(f"theta[{t}] = {theta}")
    if theta > p_theta_eps or np.isnan(theta):
        g.message(f"Time slice {t} did not converge: {theta} >= {p_theta_eps}")
        sys.exit(1)

# merge time slices and transform gauge field
V = g.merge(Vt, 3)
U_fixed = g.qcd.gauge.transformed(U, V)

# remove rounding errors on U_fixed
U_fixed = [g.project(u, "defect") for u in U_fixed]

# Print final gradient value
c = g.qcd.gauge.fix.landau(U_fixed[0:3])
g.message(">>> Final Gradient Value: ")
g.message(g.norm2(c.gradient(V, V)))

# save results
g.save("../conf/S8T8/conf_fixed", U_fixed, g.format.nersc())
g.save("../conf/S8T8/V_trans", V)