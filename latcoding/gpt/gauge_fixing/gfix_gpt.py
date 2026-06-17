#!/usr/bin/env python3
"""Coulomb gauge fixing via per-time-slice optimization."""

from pathlib import Path
import argparse
import sys

import gpt as g
import numpy as np


def _default_paths() -> tuple[Path, Path, Path]:
    root_dir = Path(__file__).resolve().parents[3]
    conf_dir = root_dir / "configs" / "S8T8"
    return conf_dir / "wilson_b6.0", conf_dir / "conf_fixed", conf_dir / "V_trans"


def parse_args() -> argparse.Namespace:
    default_in, default_out_conf, default_out_v = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge-file", default=str(default_in), help="Input gauge configuration")
    parser.add_argument("--out-conf", default=str(default_out_conf), help="Output fixed gauge configuration")
    parser.add_argument("--out-transform", default=str(default_out_v), help="Output gauge transformation")
    parser.add_argument("--precision", type=float, default=1e-10, help="Convergence precision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    p_maxiter_cg = 500
    p_maxiter_gd = 2500
    p_eps = args.precision
    p_step = 0.03
    p_gd_step = 0.05
    p_max_abs_step = 0.3
    p_theta_eps = 1e-14

    g.message(
        f"""
  Coulomb gauge fixer run with:
    gauge_file    = {args.gauge_file}
    out_conf      = {args.out_conf}
    out_transform = {args.out_transform}
    maxiter_cg    = {p_maxiter_cg}
    maxiter_gd    = {p_maxiter_gd}
    eps           = {p_eps}
    step          = {p_step}
    gd_step       = {p_gd_step}
    max_abs_step  = {p_max_abs_step}
    theta_eps     = {p_theta_eps}
"""
    )

    U = g.convert(g.load(args.gauge_file), g.double)

    rng = g.random("T")
    V0 = g.identity(U[1])
    rng.element(V0)
    U = g.qcd.gauge.transformed(U, V0)

    nt = U[0].grid.gdimensions[3]
    g.message(f"Separate {nt} time slices")
    Usep = [g.separate(u, 3) for u in U[0:3]]
    Vt = [g.mcolor(Usep[0][0].grid) for _ in range(nt)]

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

    g.message(f"Start gauge fixing on {nt} time slices")
    for t in range(nt):
        f = g.qcd.gauge.fix.landau([Usep[mu][t] for mu in range(3)])
        fa = opt.fourier_accelerate.inverse_phat_square(Vt[t].grid, f)

        g.message(f"Run time slice {t} / {nt}")
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

    g.message(">>> Final Functional Values per time slice:")
    for t in range(nt):
        f = g.qcd.gauge.fix.landau([Usep[mu][t] for mu in range(3)])
        dfv = f.gradient(Vt[t], Vt[t])
        theta = g.norm2(dfv).real / Vt[t].grid.gsites / dfv.otype.Nc
        g.message(f"theta[{t}] = {theta}")
        if theta > p_theta_eps or np.isnan(theta):
            g.message(f"Time slice {t} did not converge: {theta} >= {p_theta_eps}")
            sys.exit(1)

    V = g.merge(Vt, 3)
    U_fixed = g.qcd.gauge.transformed(U, V)
    U_fixed = [g.project(u, "defect") for u in U_fixed]

    c = g.qcd.gauge.fix.landau(U_fixed[0:3])
    g.message(">>> Final Gradient Value:")
    g.message(g.norm2(c.gradient(V, V)))

    Path(args.out_conf).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_transform).parent.mkdir(parents=True, exist_ok=True)
    g.save(args.out_conf, U_fixed, g.format.nersc())
    g.save(args.out_transform, V)


if __name__ == "__main__":
    main()