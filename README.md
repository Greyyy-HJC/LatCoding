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
- `configs/` gauge configuration inputs (local symlink; ignored by git). Point it at your ensembles tree, e.g. `ln -s /path/to/ensembles configs`.

The repository package deliberately uses the `latcoding` namespace so it does not shadow upstream `gpt` or `pyquda` imports.

## Environment

Python is managed with repository-root `.venv` (Python 3.12), created from the local `qcd` conda env with system site-packages so PyQUDA/CuPy and related packages are reused without reinstalling the compiled stack:

```bash
/home/jinchen/software/miniconda3/envs/qcd/bin/python -m venv .venv --system-site-packages
```

`pyquda` / `libquda` are built against OpenMPI (`/home/jinchen/software/openmpi`). The `qcd` conda env ships Intel MPI `mpi4py`, which conflicts with that stack, so `.venv` installs its own OpenMPI-backed `mpi4py` that shadows the conda one. When rebuilding `mpi4py`, temporarily hide conda `libmpi*` during the build so the extension links to `libmpi.so.40`.

Optional GPT: if a local GPT tree is available, add a `.venv/lib/python3.12/site-packages/gpt.pth` pointing at that tree's `lib/` and `lib/cgpt/build/` (mirroring GPT's `source.sh`). GPT is not currently installed on this machine.

Do not reinstall PyQUDA/QUDA from this repository. Expected packages (mostly from `qcd` site-packages):

- `numpy`
- `scipy`
- `h5py`
- `mpi4py` (OpenMPI build inside `.venv`)
- `opt_einsum`
- `tqdm`
- `matplotlib`
- `gvar`
- `pyyaml`
- `cupy`
- `pyquda` / `pyquda_utils`

## Quick start

1. Activate `.venv` (`source .venv/bin/activate`); `pyquda`, `pyquda_utils`, and `cupy` are available via system site-packages.
2. Install this repository package in editable mode if needed: `.venv/bin/pip install -e .`
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
