# LatCoding

LatCoding is a benchmark and consistency-check workspace for lattice gauge theory software stacks.

Current focus:
- GPT (Grid Python Toolkit)
- PyQUDA
- GLU

## Repository structure

- `gpt/` GPT software area with `benchmarks/`, `tests/`, and `addons/`
- `pyquda/` PyQUDA software area with `benchmarks/`, `tests/`, and `addons/`
- `glu/` GLU software area with benchmark-only content in `benchmark/`
- `consistency/` cross-software normalization and comparison pipeline
- `configs/` run configuration inputs
- `temp/legacy_snapshot/` archived pre-refactor repository content

## Environment

Python is managed with repository-root `.venv` (Python 3.10+).

Non-Python software is reused from existing local installations:
- `source /home/jinchen/env/gpt.env`
- `export QUDA_PATH=/home/jinchen/git/lat-software/quda/build`

## Quick start

1. Create and activate `.venv`.
2. Install `requirements.txt`.
3. Source GPT environment and export `QUDA_PATH`.
4. `gpt/` and `pyquda/` are still directory skeletons for future scripts.
5. `glu/benchmark/` is populated from `temp/legacy_snapshot/GLU_gfix`.

## License

This project is licensed under the MIT License. See `LICENSE`.
