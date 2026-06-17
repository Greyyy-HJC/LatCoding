# LatCoding

LatCoding is an installable addon package plus example and consistency-check workspace for lattice gauge theory software stacks.

Current focus:
- [GPT (Grid Python Toolkit)](https://github.com/dbollweg/gpt): Grid-enabled Python toolkit for lattice gauge theory
- [PyQUDA](https://github.com/CLQCD/PyQUDA): Python interface for QUDA
- [GLU](https://github.com/RJHudspith/GLU): Grid Lattice Utility for lattice gauge theory, mainly for gauge fixing

## Repository structure

- `latcoding/` installable package for reusable addon code owned by this repository
  - `latcoding.gpt` GPT-side helper modules
  - `latcoding.pyquda` PyQUDA-side helper modules
  - `latcoding.pyquda.classes` class-based PyQUDA measurement building blocks
  - `latcoding.common` shared helpers
- `examples/` runnable single-software examples and drafts
  - `examples/gpt/` GPT examples
  - `examples/pyquda/` PyQUDA runnable or draft driver scripts
  - `examples/glu/` GLU examples
- `checks/consistency/` cross-software checks and comparison workflows
- `configs/` gauge/configuration inputs

The repository package deliberately uses the `latcoding` namespace so it does not shadow upstream `gpt` or `pyquda` imports.

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

The Python packages used by current scripts are provided by the system site-packages visible to `.venv`; do not reinstall GPT/PyQUDA/QUDA from this repository. The current environment expects:

- `numpy`
- `scipy`
- `h5py`
- `mpi4py`
- `opt_einsum`
- `tqdm`
- `matplotlib`
- `gvar`
- `pyyaml`

## Quick start

1. Activate `.venv` (`source .venv/bin/activate`); `pyquda`, `pyquda_utils`, `cupy`, `gpt`, and `cgpt` are already available.
2. Install this repository package in editable mode: `.venv/bin/pip install -e .`
3. Import reusable code through `latcoding`, for example:

```python
from latcoding.pyquda.utils.boosted_smearing import boosted_smearing
from latcoding.pyquda.classes.pion_cg_qtmd_class import pion_TMD
from latcoding.pyquda.utils.pion_utils import gamma_stack
from latcoding.gpt.proton_qPDF.proton_qPDF_class import proton_qPDF
```

4. Run single-stack examples from `examples/` and cross-stack checks from `checks/consistency/`.

## License

This project is licensed under the MIT License. See `LICENSE`.
