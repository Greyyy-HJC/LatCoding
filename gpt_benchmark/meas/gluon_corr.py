"""
This is a script to measure gluon propagator on the coulomb gauge configs.
"""
# %%
import itertools
import gpt as g
import numpy as np
import gvar as gv

# Configuration
rng = g.random("T")

conf_path = "../../conf/S8T8"
conf_n_ls = np.arange(0, 1)
Ls = Lt = 8 # spatial and temporal lattice size

corr_length = 3 # correlation length between source and sink


# Main loop
corr_conf_ls = []
for conf_n in conf_n_ls:
    U_fixed = g.convert(g.load(f"{conf_path}/wilson_b6.{conf_n}"), g.double)

    fs1 = g.qcd.gauge.field_strength(U_fixed, mu=0, nu=2) # first F_\mu\nu 
    fs2 = g.qcd.gauge.field_strength(U_fixed, mu=0, nu=2) # second F_\mu\nu 

    # * if point source to point sink
    src_x, src_y, src_z, src_t = 0, 0, 0, 0
    snk_x, snk_y, snk_z, snk_t = src_x, src_y, (src_z+corr_length)%Ls, src_t

    g_corr = g(g.trace(fs1[src_x, src_y, src_z, src_t] * fs2[snk_x, snk_y, snk_z, snk_t]))

    g.message( "Gluon corr of point source: ", g_corr )

    # * if average over all source positions
    range_Ls = range(Ls)
    range_Lt = range(Lt)

    g_corr_ls = []
    for src_x, src_y, src_z, src_t in itertools.product(range_Ls, range_Ls, range_Ls, range_Lt):
        snk_x, snk_y, snk_z, snk_t = src_x, src_y, (src_z+corr_length)%Ls, src_t
        temp = g(g.trace(fs1[src_x, src_y, src_z, src_t] * fs2[snk_x, snk_y, snk_z, snk_t]))
        g_corr_ls.append(temp)

    g.message( "Gluon corr avg over all source positions: ", np.mean(g_corr_ls) )

# %%
