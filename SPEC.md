# SPEC.md

## Top-level map

- `gpt/` GPT-side benchmark, test, and addon code
- `pyquda/` PyQUDA-side benchmark, test, and addon code
- `glu/` GLU-side benchmark-only code in `benchmark/`
- `consistency/` cross-stack schema, adapters, compare runner
- `configs/` gauge/configuration inputs for runs

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
- `glu/benchmark/` contains the current GLU gauge-fixing benchmark scripts.
