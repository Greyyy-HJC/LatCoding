"""
This is a script to measure glueball mass.
"""
# %%
import gpt as g
import numpy as np

conf_path = "../../conf/S8T32" # path to configurations
conf_n_ls = np.arange(0, 5) # 5 configurations
Nd = 4 # number of dimensions
Lt = 32 # temporal size
Ls = 8 # spatial size


# Main loop on configurations
corr_conf_ls = []
plaq_t = []
plaq_0 = []
for conf_n in conf_n_ls:
    U_read = g.load(f"{conf_path}/wilson_b6.{conf_n}")
    vol = Ls ** 3
    ndim = U_read[0].otype.shape[0] # 3, means SU(3)
    
    val = np.zeros((Lt), dtype=complex)
    for mu in range(Nd):
        for nu in range(mu):
            temp = g.slice(
                g.trace(
                    U_read[mu] * g.cshift(U_read[nu], mu, 1) * g.adj(g.cshift(U_read[mu], nu, 1)) * g.adj(U_read[nu]) # this is a plaquette
                ), 3
            )
            val += np.array(temp)

    plaq = 2.0 * val.real / vol / Nd / (Nd - 1) / ndim
    corr = plaq * plaq[0]

    plaq_t.append(plaq) # plaquette at each time slice
    plaq_0.append(plaq[0]) # plaquette at t=0
    corr_conf_ls.append(corr) # correlation function

g.message( np.mean(corr_conf_ls, axis=0) - np.mean(plaq_0, axis=0) * np.mean(plaq_t, axis=0) ) # subtract the vacuum contribution
# %%

