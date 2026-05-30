# SPEC.md

## Top-level map

- `gpt/` GPT-side benchmark, test, and addon code
- `pyquda/` PyQUDA-side benchmark, test, and addon code
- `glu/` GLU-side benchmark-only code in `benchmark/`
- `consistency/` cross-stack schema, adapters, compare runner
- `configs/` gauge/configuration inputs for runs
- `temp/legacy_snapshot/` full archive of pre-refactor repository state

## Core responsibilities

- Benchmark generation and measurement:
  - GPT and PyQUDA for `conf_gen` and `measurement`
  - GLU for `gauge_fixing`
- Consistency checks:
  - normalize software outputs into one schema
  - compare observables with tolerances
  - produce machine-readable and human-readable summaries

## Current state

- Directory skeleton is kept under `gpt/`, `pyquda/`, and `consistency/`.
- `glu/benchmark/` is populated from `temp/legacy_snapshot/GLU_gfix`.
- GPT/PyQUDA scripts should be selected and restored later from `temp/legacy_snapshot/`.

## Legacy mapping

- `temp/legacy_snapshot/gpt_benchmark` -> `gpt`
- `temp/legacy_snapshot/pyquda_benchmark` -> `pyquda`
- `temp/legacy_snapshot/GLU_gfix` -> `glu`
- `temp/legacy_snapshot/gpt_pyq` -> `consistency/legacy_inputs`
- `temp/legacy_snapshot/conf` -> `configs/`
