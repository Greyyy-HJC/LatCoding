# %%
import gpt as g
import numpy as np
###################### load gauge ######################
Ls = 8
Lt = 32
conf = 0
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
U = g.convert( g.load(f"../../conf/S8T32_cg/gauge/wilson_b6.cg.1e-08.{conf}"), g.double )


L = U[0].grid.fdimensions
U_prime, trafo = g.gauge_fix(U, maxiter=50000, prec=1e-8) # CG fix, to get trafo

trafoI = g.identity(trafo)

print(np.shape(trafo[:]))
print(np.shape(trafoI[:]))
print(trafoI[:])

# %%
