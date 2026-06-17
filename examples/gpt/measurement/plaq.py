"""
This is a script to measure plaquette.
"""
# %%
import gpt as g
import numpy as np
import gvar as gv
from pathlib import Path

Ls = 8
Lt = 32
root_dir = Path(__file__).resolve().parents[3]
conf_dir = root_dir / "configs" / f"S{Ls}T{Lt}"
conf_n_ls = np.arange(0, 5)
Nd = 4

def plaquette(U):
    # U[mu](x)*U[nu](x+mu)*adj(U[mu](x+nu))*adj(U[nu](x))
    tr = 0.0
    vol = float(U[0].grid.fsites)
    Nd = len(U)
    ndim = U[0].otype.shape[0]
    for mu in range(Nd):
        for nu in range(mu):
            tr += g.sum(
                g.trace(
                    U[mu] * g.cshift(U[nu], mu, 1) * g.adj(g.cshift(U[mu], nu, 1)) * g.adj(U[nu])
                )
            )
    return 2.0 * tr.real / vol / Nd / (Nd - 1) / ndim

# Main loop on configurations
plaq_ls = []

for conf_n in conf_n_ls:
    conf_path = conf_dir / f"wilson_b6.{conf_n}"
    U_read = g.load(str(conf_path))
    plaq = plaquette(U_read)
    plaq_ls.append(plaq)

plaq_gv = gv.dataset.avg_data(plaq_ls) # average over configurations using gvar

g.message("This is the plaquette on each configuration:")
g.message(plaq_ls)
g.message("This is the plaquette after averaging over configurations:")
g.message("Mean: ", plaq_gv.mean)
g.message("Error: ", plaq_gv.sdev)
# %%
