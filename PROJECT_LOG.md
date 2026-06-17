# PROJECT_LOG.md

## 2026-06-17

- Updated GPT measurement examples to use repository-root `configs/` paths and script-anchored dump output paths, matching `examples/gpt/conf_gen/pure_gauge_wilson.py`.
- Moved the pion CG qTMD/qTMDWF class modules from `examples/pyquda/measurement/` into `latcoding/pyquda/classes/` as importable package modules.
- Removed GI qTMD/PDF paths from `latcoding/pyquda/classes/pion_cg_qtmd_class.py`, leaving only CG qTMD/PDF contractions in that example.
- Removed `AGENTS.md`, `CLAUDE.md`, `SPEC.md`, `requirements.txt`, and the temporary `setup.py`; current structure and environment notes now live in `README.md` plus `pyproject.toml`.
- Flattened the follow-up layout by removing `examples/pyquda/drafts/`, removing the `addons/` layer under `latcoding/gpt` and `latcoding/pyquda`, and restoring `examples/glu/gauge_fixing/GLU` as a local symlink to `/home/jinchen/local/glu/bin/GLU`.
- Reorganized the repository around the installable `latcoding` package to avoid shadowing upstream `gpt` and `pyquda` imports.
- Moved reusable GPT addon modules into `latcoding/gpt/` and PyQUDA utility modules into `latcoding/pyquda/utils/`.
- Moved GPT conf-generation and GLU gauge-fixing scripts into `examples/`; moved current PyQUDA qTMD measurement files into `examples/pyquda/measurement/`.
- Moved consistency placeholder into `checks/consistency/` and removed obsolete top-level `gpt/` and `pyquda/` placeholders.
- Added `pyproject.toml` for editable installs.
- Upgraded repository-root `.venv` pip from 21.1.1 to 25.0.1, removed the temporary `setup.py` compatibility shim, and folded the former `requirements.txt` environment notes into `README.md`.
- Updated docs for the package/examples/checks layout.

## 2026-05-30

- Reorganized repository from `software/` wrapper layout to root-level `gpt/`, `pyquda/`, and `glu/` software directories with standardized `benchmarks/`, `tests/`, and `addons/` subdirectories.
- Initialized full-refactor baseline for LatCoding.
- Archived pre-refactor repository snapshot during migration (kept `.git` in place).
- Recreated required bootstrap files from `INIT.md`: `README.md`, `SPEC.md`, `AGENTS.md`, `CLAUDE.md`, `.gitignore`, `requirements.txt`, `LICENSE`, `PROJECT_LOG.md`.
- Set project direction to software-first layout for PyQUDA, GPT, and GLU with dedicated benchmark and consistency layers.
- Created software-first runtime tree: `software/`, `benchmarks/`, `consistency/`, `configs/`, `scripts/`, `docs/`.
- Added benchmark entry scripts for `conf_gen`, `measurement`, and `gauge_fixing`.
- Migrated first-batch core assets for GPT, PyQUDA, and GLU from the legacy snapshot into the new layout and documented mapping in `docs/migration_map.md`.
- Copied legacy configurations into `configs/legacy_conf`.
- Implemented minimal consistency pipeline in `consistency/run_consistency.py` with unified schema and sample inputs.
- Verified consistency pipeline by comparing sample GPT vs PyQUDA plaquette outputs and generating `consistency/report.json` with a match result.
- Confirmed Python baseline is 3.10+ (`.venv` created with Python 3.10.15).
- Finalized directory model per latest requirement: removed root `software/`, `scripts/`, `docs/`, and top-level `benchmarks/`; now uses root `gpt/`, `pyquda/`, `glu/` each with `benchmarks/`, `tests/`, `addons/`, plus parallel `consistency/`.
- Removed all prefilled scripts from `gpt/`, `pyquda/`, `glu/`, and `consistency/` to keep a pure directory skeleton; future scripts will be restored selectively from migration sources.
- Updated `glu/` to benchmark-only layout by removing `addons/`, `tests/`, and `benchmarks/`, then copying GLU gauge-fixing benchmark contents into `glu/benchmark/`; added ignore rule for benchmark symlink `glu/benchmark/GLU` (with `*.log` already ignored).
- Synced root `.gitignore` config rules with legacy `conf/` handling by mapping to `configs/legacy_conf/` (explicit ensemble directories plus `wilson_b6.[5-9]` and `wilson_b6.[1-9][0-9]*` patterns).
- Flattened `configs/` by removing the `legacy_conf/` layer (moved all ensemble directories directly under `configs/`) and updated `.gitignore`/`SPEC.md` paths accordingly.
- Renamed GLU benchmark defaults to explicit S24T24 names: `glu/benchmark/gfix.log` -> `glu/benchmark/S24T24_gfix.log` and `glu/benchmark/input.txt` -> `glu/benchmark/input_S24T24.txt`.
- Updated `gpt/benchmarks/conf_gen/pure_gauge_wilson.py` to write/read ensembles from root `configs/S{Ls}T{Lt}/` using script-anchored paths instead of legacy `../../conf/`.
- Recreated `.venv` on Python 3.8.12 (replacing the prior 3.10.15 venv) to match the prebuilt CPython 3.8 ABI of the working stack: `pyquda`/`pyquda_comm`/`pyquda_plugins.pycontract`, `cupy`, and `gpt/lib/cgpt/build/cgpt.so` are all `cpython-38` artifacts that a 3.10 interpreter cannot import. This intentionally deviates from the "Python 3.10+" guideline because the existing compiled stack is unavoidably 3.8.
- Based the new venv on the existing Python 3.8 interpreter with `--system-site-packages` so it reuses `pyquda`, `pyquda_utils`, `cupy`, `mpi4py`, `numpy 1.24.4`, `gvar`, `opt_einsum`, `tqdm`, `h5py`, etc. without reinstalling any non-Python software; `lametlat` intentionally excluded.
- Added `.venv/lib/python3.8/site-packages/gpt.pth` mirroring `gpt/lib/cgpt/build/source.sh` so `import gpt` and `import cgpt` work from the venv without sourcing the env script.
- Verified in the venv: `pyquda_utils.{core,io,source,gamma}`, `pyquda_utils.phase.MomentumPhase`, `pyquda_plugins.pycontract`, `gvar`, `tqdm`, `opt_einsum`, `cupy`, `mpi4py`, `gpt`, and `cgpt` all import (`libquda.so` resolves via RPATH).
- Polished newly added GPT addon scripts to match the refactored repository layout: removed notebook-cell leftovers, added `main()`/CLI entry points for gauge-fixing scripts, and replaced legacy `../conf` path assumptions with repository-root `configs/` path resolution; also updated qPDF output writing to a deterministic addon-local output path.
- Updated `gpt/addons/proton_qPDF` to current config layout by removing obsolete `Vtrans` dependency (unused in runtime path), switching config lookup to `configs/{ensemble}` without `gauge/` and `Vtrans/` subdirectories, and simplifying related class method signatures.
- Cleaned project docs to remove stale references to `temp/legacy_snapshot` so repository documentation reflects only current tracked structure.
- Fixed qPDF config loading mismatch for current `configs/` file naming by preferring `wilson_b6.{conf_n}` and keeping backward compatibility fallback to the older `wilson_b6.cg.{precision}.{conf_n}` pattern.
- Switched qPDF loading to strict CG-fixed input only: read exclusively from `configs/S8T32_cg/wilson_b6.cg.1e-14.{conf_n}` and fail fast when files are missing; removed compatibility fallback behavior.
- Added an explicit repository rule in `AGENTS.md` to avoid compatibility fallback logic unless explicitly requested.
