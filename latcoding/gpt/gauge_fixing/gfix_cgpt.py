"""Run Coulomb gauge fixing with the cgpt backend."""

from pathlib import Path
import argparse

import gpt as g

from latcoding.gpt.gauge_fixing.coulomb_gauge import coulomb


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
    parser.add_argument("--precision", type=float, default=1e-10, help="Gauge-fix precision")
    parser.add_argument("--maxiter", type=int, default=12000, help="Maximum gauge-fix iterations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    U_read = g.convert(g.load(args.gauge_file), g.double)

    rng = g.random("T")
    V0 = g.identity(U_read[1])
    rng.element(V0)
    U_read = g.qcd.gauge.transformed(U_read, V0)
    c = coulomb(U_read)

    U_fixed, V_trans = g.gauge_fix(
        U_read,
        maxiter=args.maxiter,
        prec=args.precision,
        use_fourier=False,
        orthog_dir=3,
    )

    g.message(">>> Final Functional Value:")
    g.message(c([V_trans]))
    g.message(">>> Final Gradient Value:")
    g.message(g.norm2(c.gradient(V_trans, V_trans)))

    Path(args.out_conf).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_transform).parent.mkdir(parents=True, exist_ok=True)
    g.save(args.out_conf, U_fixed, g.format.nersc())
    g.save(args.out_transform, V_trans)


if __name__ == "__main__":
    main()