# LatCoding

LatCoding is a benchmark and consistency-check workspace for lattice gauge theory software stacks.

Current focus:
- [GPT (Grid Python Toolkit)](https://github.com/dbollweg/gpt): Grid-enabled Python toolkit for lattice gauge theory
- [PyQUDA](https://github.com/CLQCD/PyQUDA): Python interface for QUDA
- [GLU](https://github.com/RJHudspith/GLU): Grid Lattice Utility for lattice gauge theory, mainly for gauge fixing.

## Repository structure

- `gpt/` GPT software area with `benchmarks/`, `tests/`, and `addons/`
- `pyquda/` PyQUDA software area with `benchmarks/`, `tests/`, and `addons/`
- `glu/` GLU software area with benchmark-only content in `benchmark/`
- `consistency/` cross-software normalization and comparison pipeline
- `configs/` run configuration inputs

## Environment

Python is managed with repository-root `.venv` (Python 3.8, pinned to match the prebuilt GPT/PyQUDA stack).

The compiled stack (`pyquda`, `cupy`, `cgpt.so`, and `pyquda_plugins.pycontract`) is built for CPython 3.8 ABI, so `.venv` is created from the existing Python 3.8 interpreter with system site-packages to reuse those artifacts without rebuilding any non-Python software:

```bash
/home/jinchen/miniconda3/envs/pygpt/bin/python -m venv .venv --system-site-packages
```

`gpt` and `cgpt` are made importable via `.venv/lib/python3.8/site-packages/gpt.pth`, which mirrors `gpt/lib/cgpt/build/source.sh`.

Non-Python software is reused from existing local installations:
- `source /home/jinchen/env/gpt.env`
- `export QUDA_PATH=/home/jinchen/git/lat-software/quda/build`

## Quick start

1. Activate `.venv` (`source .venv/bin/activate`); `pyquda`, `pyquda_utils`, `cupy`, `gpt`, and `cgpt` are already available.
2. `export QUDA_PATH=/home/jinchen/git/lat-software/quda/build` if building/measuring with QUDA.
3. `gpt/`, `pyquda/`, and `glu/benchmark/` host the benchmark/addon scripts in this repository.

## License

This project is licensed under the MIT License. See `LICENSE`.
