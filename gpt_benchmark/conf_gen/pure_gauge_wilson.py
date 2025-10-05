'''
This is a test script to generate a set of pure gauge configurations with wilson action. Using heatbath, including heat balance and config save.

Two parts:
1. Markov chain to generate the configurations;
2. Save the configurations to a file.
'''

# %%
import gpt as g
import os

Ls = 32
Lt = 32

# grid
lattice = [Ls, Ls, Ls, Lt]
grid = g.grid(lattice, g.double)
grid_eo = g.grid(lattice, g.double, g.redblack)

# hot start
g.default.push_verbose("random", False)
rng = g.random("T")
U_it = g.qcd.gauge.unit(grid)
Nd = len(U_it)

# red/black mask
mask_rb = g.complex(grid_eo)
mask_rb[:] = 1

# full mask
mask = g.complex(grid)

# action
w = g.qcd.gauge.action.wilson(6.0) # beta = 6.0

# heatbath sweeps
g.default.push_verbose("su2_heat_bath", False)
markov = g.algorithms.markov.su2_heat_bath(rng)

# %%
#! heat balance
if True:
    conf_dir = f"../../conf/S{Ls}T{Lt}"
    os.makedirs(conf_dir, exist_ok=True)
    
    for it in range(50):
        plaq = g.qcd.gauge.plaquette(U_it)
        R_2x1 = g.qcd.gauge.rectangle(U_it, 2, 1)
        g.message(f"SU(2)-subgroup heatbath {it} has P = {plaq}, R_2x1 = {R_2x1}")
        for cb in [g.even, g.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)

            for mu in range(Nd):
                markov(U_it[mu], w.staple(U_it, mu), mask)

    g.save(f"../../conf/S{Ls}T{Lt}/wilson_b6.balance", U_it, g.format.nersc())
    U_check = g.load(f"../../conf/S{Ls}T{Lt}/wilson_b6.balance")

    g.message( g.norm2(U_check[1] - U_it[1]) )


# %%
#! save configs
U_it = g.load(f"../../conf/S{Ls}T{Lt}/wilson_b6.balance")

for n_conf in range(5):
    for gap in range(40):
        it = n_conf * 40 + gap

        plaq = g.qcd.gauge.plaquette(U_it)
        R_2x1 = g.qcd.gauge.rectangle(U_it, 2, 1)
        g.message(f"SU(2)-subgroup heatbath {it} has P = {plaq}, R_2x1 = {R_2x1}")
        for cb in [g.even, g.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)

            for mu in range(Nd):
                markov(U_it[mu], w.staple(U_it, mu), mask)

    g.save(f"../../conf/S{Ls}T{Lt}/wilson_b6.{n_conf}", U_it, g.format.nersc())

# %%
