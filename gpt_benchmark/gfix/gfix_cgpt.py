'''
This is a script to run the CGPT code to do the gauge fixing.
'''

# %%
import gpt as g
from coulomb_gauge import coulomb

gauge_file = f"../conf/S8T8/wilson_b6.0"
gauge_todo = "coulomb"
precision = 1e-10


#! gauge fixing

U_read = g.convert( g.load(gauge_file), g.double )

rng = g.random("T")
V0 = g.identity(U_read[1])

rng.element(V0)
U_read = g.qcd.gauge.transformed(U_read, V0) # random gauge transformation
c = coulomb(U_read)

U_fixed, V_trans = g.gauge_fix(U_read, maxiter=12000, prec=precision, use_fourier=False, orthog_dir=3) # Coulomb gauge

g.message(">>> Final Functional Value: ")
g.message(c([V_trans]))
g.message(">>> Final Gradient Value: ")
g.message(g.norm2(c.gradient(V_trans, V_trans)))

g.save("../conf/S8T8/conf_fixed", U_fixed, g.format.nersc())

g.save("../conf/S8T8/V_trans", V_trans)

# %%