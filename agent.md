# Agent Notes

- Work exclusively on the `dev` branch. Pushes are fine, but never merge or open PRs into `main` without an explicit request.
- Treat the benchmark folders (`gpt_benchmark/`, `pyquda_benchmark/`) as read-only. Copy inputs before experimenting and keep the original data intact.
- When modifying cross-check code under `gpt_pyq/`, update both GPT and PyQUDA drivers as needed and document the workflow in `README.md`.
- New scripts must save outputs into the existing `data/` hierarchy and reuse the helpers in `io_corr.py` so file formats stay consistent.
- Synchronize docs: any workflow or interface change should be reflected here and in `README.md` to reduce context switching for collaborators.
