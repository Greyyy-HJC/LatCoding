# AGENTS.md

Project-specific instructions for coding agents working in this repository.

## Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State assumptions explicitly.
- If multiple interpretations exist, present them instead of picking silently.
- If something is unclear, ask before implementing.
- If a simpler approach exists, say so.

## Simplicity First

Write the minimum code that solves the requested problem.

- No features beyond what was asked.
- No speculative abstractions.
- No unnecessary configurability.
- Prefer locally understandable logic.

Ask: would a strong engineer consider this overcomplicated? If yes, simplify it.

## Surgical Changes

Touch only what is required for the task.

- Do not refactor unrelated code unless asked.
- Match the existing style of the repository.
- Keep changes easy to trace to the current task.

## Goal-Driven Execution

Turn tasks into verifiable outcomes.

- Define what success looks like before changing code.
- Prefer checks when they are appropriate.
- For multi-step work, keep a short plan and verify each step.
- Do not stop at implementation; verify the result.

## Workflow Hygiene

- Before each `git add` and `git commit`, check whether `.gitignore` needs updates.
- After each meaningful implementation pass, check whether `PROJECT_LOG.md` needs updates.

## Project-Specific Rules

- Keep code concise and readable.
- Prefer direct flow in scripts; avoid extra helper functions unless clearly needed.
- Keep exception handling minimal; only keep checks required by the main runtime path.
- Do not add compatibility fallbacks for old paths/formats unless explicitly requested; fail fast with explicit errors so misconfigurations are visible.
- Use repository-root `.venv` for Python dependencies.
- Keep Python runtime at 3.8 or newer.
- Reuse existing non-Python software stack from local environment:
  - `source /home/jinchen/env/gpt.env`
  - `export QUDA_PATH=/home/jinchen/git/lat-software/quda/build`
- Do not reinstall QUDA or GPT system stack unless explicitly requested.

## Documentation Maintenance

- Keep `SPEC.md` aligned with structure changes.
- Update `PROJECT_LOG.md` when meaningful changes are made.
- Keep `README.md`, `AGENTS.md`, and `CLAUDE.md` consistent.
