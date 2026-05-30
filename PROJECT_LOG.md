# PROJECT_LOG.md

## 2026-05-30

- Reorganized repository from `software/` wrapper layout to root-level `gpt/`, `pyquda/`, and `glu/` software directories with standardized `benchmarks/`, `tests/`, and `addons/` subdirectories.
- Initialized full-refactor baseline for LatCoding.
- Archived pre-refactor repository into `temp/legacy_snapshot` (kept `.git` in place).
- Recreated required bootstrap files from `INIT.md`: `README.md`, `SPEC.md`, `AGENTS.md`, `CLAUDE.md`, `.gitignore`, `requirements.txt`, `LICENSE`, `PROJECT_LOG.md`.
- Set project direction to software-first layout for PyQUDA, GPT, and GLU with dedicated benchmark and consistency layers.
- Created software-first runtime tree: `software/`, `benchmarks/`, `consistency/`, `configs/`, `scripts/`, `docs/`.
- Added benchmark entry scripts for `conf_gen`, `measurement`, and `gauge_fixing`.
- Migrated first-batch core assets for GPT, PyQUDA, and GLU from `temp/legacy_snapshot` into the new layout and documented mapping in `docs/migration_map.md`.
- Copied legacy configurations from `temp/legacy_snapshot/conf` into `configs/legacy_conf`.
- Implemented minimal consistency pipeline in `consistency/run_consistency.py` with unified schema and sample inputs.
- Verified consistency pipeline by comparing sample GPT vs PyQUDA plaquette outputs and generating `consistency/report.json` with a match result.
- Confirmed Python baseline is 3.10+ (`.venv` created with Python 3.10.15).
- Finalized directory model per latest requirement: removed root `software/`, `scripts/`, `docs/`, and top-level `benchmarks/`; now uses root `gpt/`, `pyquda/`, `glu/` each with `benchmarks/`, `tests/`, `addons/`, plus parallel `consistency/`.
- Removed all prefilled scripts from `gpt/`, `pyquda/`, `glu/`, and `consistency/` to keep a pure directory skeleton; future scripts will be hand-picked from `temp/legacy_snapshot/`.
- Updated `glu/` to benchmark-only layout by removing `addons/`, `tests/`, and `benchmarks/`, then copying `temp/legacy_snapshot/GLU_gfix` contents into `glu/benchmark/`; added ignore rule for benchmark symlink `glu/benchmark/GLU` (with `*.log` already ignored).
- Synced root `.gitignore` config rules with legacy `conf/` handling by mapping to `configs/legacy_conf/` (explicit ensemble directories plus `wilson_b6.[5-9]` and `wilson_b6.[1-9][0-9]*` patterns).
- Flattened `configs/` by removing the `legacy_conf/` layer (moved all ensemble directories directly under `configs/`) and updated `.gitignore`/`SPEC.md` paths accordingly.
- Renamed GLU benchmark defaults to explicit S24T24 names: `glu/benchmark/gfix.log` -> `glu/benchmark/S24T24_gfix.log` and `glu/benchmark/input.txt` -> `glu/benchmark/input_S24T24.txt`.
- Updated `gpt/benchmarks/conf_gen/pure_gauge_wilson.py` to write/read ensembles from root `configs/S{Ls}T{Lt}/` using script-anchored paths instead of legacy `../../conf/`.
