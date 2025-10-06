# LatCoding

LatCoding collects small-scale benchmarks and code snippets that compare lattice QCD workflows across GPT and PyQUDA. The goal is to keep a set of reproducible reference calculations that demonstrate feature parity between the two software stacks.

## Repository Layout
- `gpt_benchmark/` – stand-alone GPT benchmarks (timings, strong-scaling checks, reference inputs)
- `pyquda_benchmark/` – PyQUDA benchmarks mirroring the GPT scenarios
- `gpt_pyq/` – cross-checks where the same observable is computed with both GPT and PyQUDA back-ends
- `conf/` – example gauge configurations used by the sandbox jobs
- `agent.md` – ground rules for contributors working in this repo

## Prerequisites
- Python 3.10 with `mpi4py`, `cupy`, and `opt_einsum`
- A working GPT installation with GPU support
- PyQUDA (built against the same MPI/CUDA stack as GPT)
- Access to NVIDIA GPUs and an MPI runtime

## Running the GI_GPD Cross-Check
The GI_GPD example demonstrates a three-point function cross-check. The GPT reference driver is `gpt_pyq/GI_GPD/main_gpt.py`, while the PyQUDA contraction path is implemented in `gpt_pyq/GI_GPD/main_pyq.py`.

Typical usage (from the repository root):

```bash
# GPT-based contraction
python gpt_pyq/GI_GPD/main_gpt.py --config_num 0

# PyQUDA-based contraction (shares the same CLI arguments)
python gpt_pyq/GI_GPD/main_pyq.py --config_num 0
```

Both scripts write HDF5 outputs under `gpt_pyq/GI_GPD/data/` using the naming convention defined in `io_corr.py`. Match the run-time options (`--config_num`, smearing tags, etc.) across both runs before comparing observables.

## Development Notes
- Always branch off and push to `dev`; never merge or open PRs into `main` without explicit approval
- Treat the benchmark folders as read-only inputs when working on cross-check code
- Document any new workflows or command-line switches in `README.md` and keep `agent.md` in sync
- Validate changes by running the relevant driver with `python -m compileall <file>` when the full software stack is unavailable
